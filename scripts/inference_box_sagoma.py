import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Usiamo la pipeline standard e il processore di maschere
from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import IPAdapterMaskProcessor

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# --- CLASSI MODELLO ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    emb = emb * 0.95
    return emb
import torch.nn.functional as F
from PIL import ImageFilter

import torch.nn.functional as F
from PIL import ImageFilter
import numpy as np

def extract_custom_shape_mask(depth_flat_tensor, threshold=0.6):
    """
    Abbandona il rigido Bounding Box. Usa la "costruzione contigua 
    di quadrati" reale estratta dalla V1 per creare una maschera organica.
    """
    # 1. Reshape a 64x64 e min-max scaling
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    # 2. PRE-PULIZIA: Taglia il rumore sotto soglia
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    
    # 3. EROSIONE: Pulisce i pixel spuri scollegati dalla massa principale
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    
    # 4. LA TUA IDEA: Invece del box, teniamo i pixel esatti del cervello!
    # Tutti i quadratini 64x64 rimasti accesi diventano bianco solido (1.0).
    binary_shape = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    
    # Se il segnale è completamente morto (solo rumore eroso), fallback a maschera totale
    if binary_shape.sum() < 5:
        binary_shape = torch.ones_like(binary_shape)
    
    # 5. Upscale a 1024x1024 per SDXL
    # Mode='bilinear' smussa un po' i "quadrati", se usassi 'nearest' vedresti i "blocchi" netti
    shape_1024 = F.interpolate(binary_shape, size=(1024, 1024), mode='bilinear', align_corners=False)
    
    # 6. Converti in immagine PIL e applica la sfumatura per evitare i bordi duri
    shape_np = (shape_1024[0, 0].cpu().numpy() * 255).astype(np.uint8)
    shape_pil = Image.fromarray(shape_np)
    
    # Sfumatura fondamentale per fondere la figura con lo sfondo naturale (es. prato)
    shape_pil = shape_pil.filter(ImageFilter.GaussianBlur(radius=25))
    
    return shape_pil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_custom_box")
    parser.add_argument("--guidance_scale", type=float, default=6.0) 
    parser.add_argument("--mask_threshold", type=float, default=0.5) 
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16 
    print(f"--- Inference BRAIN-ONLY (GLIGEN-Style Masked IP) SDXL S{args.subject} | {args.rois} ---")

    # 1. Setup Dimensioni
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # 2. Carica Decoder Semantico (fMRI -> SDXL IP-Adapter 1280)
    aligner_sem = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs_mse/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    if not os.path.exists(path_sem):
        print(f"ERRORE: Manca il decoder semantico: {path_sem}")
        return
    aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
    aligner_sem.eval()
    
    # 3. Carica Decoder Strutturale (fMRI -> Saliency Map 4096)
    aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
    path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth"
    if not os.path.exists(path_struct):
        print(f"ERRORE: Manca il decoder strutturale: {path_struct}")
        return
    aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
    aligner_struct.eval()

    # 4. Pipeline SDXL + IP-Adapter (NIENTE ControlNet!)
    print("Caricamento SDXL Base e IP-Adapter (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0) 
    
    # Inizializza il processore per le maschere spaziali
    mask_processor = IPAdapterMaskProcessor()

    # 5. Dati Test
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )
    
    grouped = defaultdict(list)
    img_map = {}
    for i, vec in enumerate(raw_data['fmri']):
        p = raw_data['image_paths'][i]
        fname = os.path.basename(p)
        grouped[fname].append(vec)
        img_map[fname] = p
    
    unique_files = sorted(list(grouped.keys()))
    save_dir = os.path.join(args.output_dir, args.rois, str(args.subject))
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generazione su {len(unique_files)} immagini...")

    # --- INFERENCE LOOP ---
    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        
        trials_np = np.array(grouped[fname]) 
        fmri_input = torch.tensor(trials_np, dtype=torch.float32).to(device).mean(dim=0, keepdim=True)

        with torch.no_grad():
            # A. DECODE SEMANTICA (Il "Cosa")
            emb_sem = aligner_sem(fmri_input)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
            # B. DECODE STRUTTURA E BOUNDING BOX (Il "Dove")
            depth_flat = aligner_struct(fmri_input)
            box_pil = extract_custom_shape_mask(depth_flat, threshold=args.mask_threshold)
            
            # Preprocessiamo la maschera per l'Attenzione dell'IP-Adapter
            ip_mask = mask_processor.preprocess([box_pil])
            
        seed = 42
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # C. GENERAZIONE
        generated_image = pipe(
            prompt="", # IL TUO SALVAGENTE!
            negative_prompt="",
            ip_adapter_image_embeds=[ip_adapter_embeds],
            cross_attention_kwargs={"ip_adapter_masks": ip_mask}, # GLIGEN MAGIC HERE!
            num_inference_steps=20,
            guidance_scale=args.guidance_scale, 
            height=1024, width=1024, 
            generator=generator
        ).images[0]

        gen_thumb = generated_image.resize((512, 512))
        
        # --- NUOVO: Salva solo l'immagine generata pura per le metriche ---
        single_save_dir = os.path.join(save_dir, "singles")
        os.makedirs(single_save_dir, exist_ok=True)
        gen_thumb.save(os.path.join(single_save_dir, f"{real_key}_generated.png"))
        # ------------------------------------------------------------------
        
        # 1. Prepariamo l'immagine Ground Truth (in RGBA per gestire le trasparenze)
        if os.path.exists(img_map[fname]):
            gt_img = Image.open(img_map[fname]).convert("RGBA").resize((512, 512))
        else:
            gt_img = Image.new("RGBA", (512, 512))

        # 2. Adattiamo la maschera organica (box_pil)
        mask_resized = box_pil.resize((512, 512)).convert("L")
        
        # Creiamo un livello di colore ciano semi-trasparente (Opacità al 50% = 128)
        cyan_layer = Image.new("RGBA", (512, 512), (0, 255, 255, 128))
        
        # Incolliamo il livello ciano usando la mappa fMRI come "stencil"
        gt_img.paste(cyan_layer, (0, 0), mask_resized)
        
        # Riconvertiamo in RGB per salvare il JPG/PNG standard
        gt_img = gt_img.convert("RGB")

        # 3. Creiamo la griglia finale a 2 pannelli
        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(gen_thumb, (512, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth + Brain Shape Overlay", fill="white")
        d.text((522, 10), "Brain Reconstruction", fill="lime")
        
        # Salviamo la griglia completa per la visualizzazione umana
        grid.save(os.path.join(save_dir, f"{real_key}_organic_mask.png"))

    print(f"Finito. Output in {save_dir}")

if __name__ == "__main__":
    main()