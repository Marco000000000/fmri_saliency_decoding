import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from diffusers import KandinskyV22ControlnetPipeline

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# --- CLASSI MODELLO (Lineari Semplici) ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
    """
    Normalizzazione avanzata per evitare artefatti viola ad alta guidance.
    """
    # 1. Clipping dei valori estremi (evita che singoli voxel rumorosi dominino)
    # Gli embedding CLIP solitamente stanno tra -0.1 e 0.1 dopo la proiezione
    emb = torch.clamp(emb, -1.0, 1.0)
    
    # 2. Normalizzazione L2 rigorosa
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    
    # 3. Scaling fattore correttivo
    # Kandinsky è tarato su embedding che hanno una certa 'energia'. 
    # Moltiplicare per un fattore piccolo (0.9 - 0.95) a volte aiuta la stabilità.
    emb = emb * 0.95
    
    return emb.to(dtype=torch.float32)
from torchvision.transforms import GaussianBlur

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

def prepare_soft_hint(depth_flat_tensor, device, dtype, threshold=0.5, blur_kernel=51, scale=0.4):
    """
    Versione anti-aliasing: elimina i bordi fantasma e la quadratizzazione
    usando soft-thresholding e upscale preventivo.
    """
    # 1. Reshape e Normalizzazione (rimaniamo in float per precisione)
    depth_small = depth_flat_tensor.view(1, 1, 64, 64)
    d_min, d_max = depth_small.min(), depth_small.max()
    depth_norm = (depth_small - d_min) / (d_max - d_min + 1e-6)

    # 2. SOFT-THRESHOLDING (Invece di binarizzare 0/1)
    # Usiamo una sigmoide per creare una transizione fluida già a bassa ris.
    # 'steepness' controlla quanto è netto il bordo senza diventare un gradino di pixel.
    steepness = 15 
    soft_mask = torch.sigmoid(steepness * (depth_norm - threshold))

    # 3. UPSCALE BILINEARE (Più sicuro del bicubico per le maschere)
    # L'interp. bilineare non crea "valori fantasma" (overshoot) oltre l'1.0
    depth_big = F.interpolate(soft_mask, size=(768, 768), mode='bilinear', align_corners=False)
    
    # 4. PULIZIA MORFOLOGICA (Opzionale, su scala grande per precisione)
    # Se vuoi ancora unire i pezzi, fallo qui, ma con kernel piccoli.
    # Un max_pool qui agisce come una dilatazione "sub-pixel"
    depth_big = F.max_pool2d(depth_big, kernel_size=3, stride=1, padding=1)

    # 5. GAUSSIAN BLUR (Smoothing finale)
    # Il sigma deve essere proporzionato al kernel per evitare aloni random.
    sigma = blur_kernel / 4.0
    blurrer = GaussianBlur(kernel_size=blur_kernel, sigma=sigma)
    depth_soft = blurrer(depth_big)
    
    # 6. SCALING & CLAMP (Fondamentale)
    # Moltiplichiamo per scale PRIMA del clamp per assicurarci che 
    # i bordi chiari non vengano esasperati.
    depth_final = torch.clamp(depth_soft * scale, 0.0, 1.0)
    
    # Ritorna in RGB
    depth_rgb = depth_final.repeat(1, 3, 1, 1)
    
    return depth_rgb.to(device=device, dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_binary_control")
    parser.add_argument("--guidance_scale", type=float, default=1.5) # Forza Semantica fMRI
    parser.add_argument("--mask_threshold", type=float, default=0.5) # Soglia per silhouette (0.4 = più grassa, 0.6 = più magra)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference BRAIN-ONLY (Binary/Silhouette) S{args.subject} | {args.rois} ---")

    # 1. Setup Dimensioni
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # 2. Carica Decoder Semantico (fMRI -> CLIP 1280)
    # Usa il modello addestrato in precedenza (L-BFGS Sparse)
    aligner_sem = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    
    if not os.path.exists(path_sem):
        print(f"ERRORE: Manca il decoder semantico: {path_sem}")
        return
    aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
    aligner_sem.eval()
    
    # 3. Carica Decoder Strutturale (fMRI -> Binary Mask 4096)
    # Usa il modello che hai addestrato sui target binari
    aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
    # NOTA: Assicurati che il path punti alla cartella dove hai salvato il decoder binario
    path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth"
    
    if not os.path.exists(path_struct):
        print(f"ERRORE: Manca il decoder strutturale/binario: {path_struct}")
        print("Hai eseguito 'train_depth_decoder_lbfgs.py' puntando a 'precomputed_binary_masks'?")
        return
    aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
    aligner_struct.eval()
    
    print("Modelli fMRI caricati.")

    # 4. Pipeline ControlNet (FLOAT32 per evitare artefatti viola)
    print("Caricamento Kandinsky ControlNet (FP32)...")
    pipe = KandinskyV22ControlnetPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-controlnet-depth", 
        torch_dtype=torch.float32 
    ).to(device)
    # Doppia sicurezza per il VAE
    pipe.movq.to(dtype=torch.float32)

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
        
        # Prepara Input fMRI (Media dei trial per pulizia)
        trials_np = np.array(grouped[fname]) 
        fmri_input = torch.tensor(trials_np, dtype=torch.float32).to(device).mean(dim=0, keepdim=True)

        with torch.no_grad():
            # A. DECODE SEMANTICA (fMRI -> CLIP)
            emb_sem = aligner_sem(fmri_input)
            emb_sem = normalize_and_cast(emb_sem)
            
            # Se la guidance è molto alta, "calmiamo" l'embedding condizionato
            # per evitare che la differenza (cond - uncond) esploda.
            if args.guidance_scale > 5.0:
                emb_sem = emb_sem * (5.0 / args.guidance_scale)
            
            neg_emb = torch.zeros_like(emb_sem, dtype=torch.float32)
            # B. DECODE STRUTTURA (fMRI -> Binary Mask)
            depth_flat = aligner_struct(fmri_input)
            
            hint_tensor = prepare_soft_hint(
                depth_flat, 
                device, 
                torch.float32, 
                threshold=0.5,
                blur_kernel=1, # Sfoca i bordi
                scale=0.3     # Riduci intensità
            )
        seed = abs(hash(fname)*12345) % (2**32)
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # C. GENERAZIONE
        generated_image = pipe(
            image_embeds=emb_sem,
            negative_image_embeds=neg_emb,
            hint=hint_tensor,
            num_inference_steps=25,
            guidance_scale=args.guidance_scale, # Importante: CLIP deve essere forte
            height=768, width=768,
            generator=generator
        ).images[0]

        # D. GRIGLIA DI CONFRONTO
        gen_thumb = generated_image.resize((512, 512))
        
        # Recupera GT
        if os.path.exists(img_map[fname]):
            gt_img = Image.open(img_map[fname]).convert("RGB").resize((512, 512))
        else:
            gt_img = Image.new("RGB", (512, 512))
            
        # Visualizza la Maschera Binaria usata
        # Hint tensor è (1, 3, 768, 768), valori 0.0 o 1.0
        mask_vis = hint_tensor[0,0].cpu().numpy() # Prendi 1 canale
        mask_vis = (mask_vis * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_vis).resize((512, 512))

        # Griglia 3 colonne: GT | Maschera Decodificata | Risultato Finale
        grid = Image.new("RGB", (512 * 3, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(mask_pil, (512, 0))
        grid.paste(gen_thumb, (1024, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth", fill="white")
        d.text((522, 10), "Brain-Decoded Silhouette", fill="cyan")
        d.text((1034, 10), "Brain Reconstruction", fill="lime")
        
        grid.save(os.path.join(save_dir, f"{real_key}_binary.png"))

    print(f"Finito. Output in {save_dir}")

if __name__ == "__main__":
    main()