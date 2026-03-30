import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import GaussianBlur
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

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
    # Scaling factor leggero per stabilità
    emb = emb * 0.95
    return emb
def prepare_sharp_silhouette(depth_flat_tensor, device, dtype, threshold=0.5):
    """
    Crea una silhouette netta, contigua e senza buchi applicando 
    operazioni morfologiche leggere, evitando espansioni eccessive.
    """
    # 1. Reshape a 64x64 e min-max scaling
    depth_small = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = depth_small.min(), depth_small.max()
    depth_norm = (depth_small - d_min) / (d_max - d_min + 1e-6)

    # 2. PRE-PULIZIA: Ignora il rumore di fondo prima di dilatare.
    # Tutto ciò che è debole (sotto una frazione della soglia) diventa 0.
    clean_mask = torch.where(depth_norm > (threshold * 0.7), depth_norm, torch.zeros_like(depth_norm))

    # 3. CLOSING MORFOLOGICO LEGGERO (Kernel 3 invece di 5)
    # a) Dilatazione: fonde solo i pixel molto vicini
    dilated = F.max_pool2d(clean_mask, kernel_size=3, stride=1, padding=1)
    # b) Erosione: restringe i bordi
    eroded = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
    
    # 4. Smoothing a bassa risoluzione (Kernel 3)
    blurrer_small = GaussianBlur(kernel_size=3, sigma=1.0)
    smoothed = blurrer_small(eroded)

    # 5. Threshold netto (crea il blocco binario solido)
    steepness = 100.0
    hard_mask = torch.sigmoid(steepness * (smoothed - threshold))

    # 6. Upscale bilineare a 1024x1024 (applica un anti-aliasing naturale)
    depth_big = F.interpolate(hard_mask, size=(1024, 1024), mode='bilinear', align_corners=False)

    # 7. Smoothing finale meno aggressivo per non sfocare troppo i bordi
    blurrer_large = GaussianBlur(kernel_size=11, sigma=2.0)
    depth_smooth = blurrer_large(depth_big)

    # 8. Clamp e conversione a RGB
    depth_final = torch.clamp(depth_smooth, 0.0, 1.0)
    depth_rgb = depth_final.repeat(1, 3, 1, 1)
    
    return depth_rgb.to(device=device, dtype=dtype)
def prepare_soft_hint(depth_flat_tensor, device, dtype, threshold=0.5, blur_kernel=51, scale=0.4):
    depth_small = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = depth_small.min(), depth_small.max()
    depth_norm = (depth_small - d_min) / (d_max - d_min + 1e-6)

    steepness = 15 
    soft_mask = torch.sigmoid(steepness * (depth_norm - threshold))
    
    # SDXL lavora idealmente a 1024x1024
    depth_big = F.interpolate(soft_mask, size=(1024, 1024), mode='bilinear', align_corners=False)
    depth_big = F.max_pool2d(depth_big, kernel_size=3, stride=1, padding=1)

    sigma = blur_kernel / 4.0
    blurrer = GaussianBlur(kernel_size=blur_kernel, sigma=sigma)
    depth_soft = blurrer(depth_big)
    
    depth_final = torch.clamp(depth_soft, 0.0, 1.0)
    depth_rgb = depth_final.repeat(1, 3, 1, 1)
    
    return depth_rgb.to(device=device, dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_binary_control_sdxl_controlnet")
    parser.add_argument("--guidance_scale", type=float, default=10.0) # SDXL scala meglio intorno a 5-7
    parser.add_argument("--mask_threshold", type=float, default=0.5) 
    parser.add_argument("--scale", type=float, default=0.0)
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, f"_{args.scale}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16 # SDXL in FP16
    print(f"--- Inference BRAIN-ONLY (Silhouette) SDXL + IP-Adapter S{args.subject} | {args.rois} ---")

    # 1. Setup Dimensioni
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # 2. Carica Decoder Semantico (fMRI -> SDXL IP-Adapter 1280)
    aligner_sem = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    if not os.path.exists(path_sem):
        print(f"ERRORE: Manca il decoder semantico: {path_sem}")
        return
    aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
    aligner_sem.eval()
    
    # 3. Carica Decoder Strutturale (fMRI -> Binary Mask 4096)
    aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
    path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth"
    if not os.path.exists(path_struct):
        print(f"ERRORE: Manca il decoder strutturale: {path_struct}")
        return
    aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
    aligner_struct.eval()

    # 4. Pipeline SDXL ControlNet + IP-Adapter
    print("Caricamento SDXL ControlNet (FP16)...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=dtype_inf
    ).to(device)
    
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=dtype_inf
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0) # Potenza del condizionamento immagine

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
            # A. DECODE SEMANTICA (fMRI -> SDXL IP-Adapter)
            emb_sem = aligner_sem(fmri_input)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            
            # Formattiamo [Batch, 1, 1280]
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            
            # CORREZIONE: Creiamo l'embedding incondizionato (zeri) per il Classifier-Free Guidance
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            
            # Concateniamo lungo il batch per avere [2, 1, 1280] (prima negativo, poi positivo)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
            # B. DECODE STRUTTURA
            depth_flat = aligner_struct(fmri_input)
            hint_tensor = prepare_sharp_silhouette(
                depth_flat, device, dtype_inf, threshold=args.mask_threshold
            )
            
        seed = 42
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # C. GENERAZIONE
        generated_image = pipe(
            prompt="",
            negative_prompt="unrealistic, blurry",
            ip_adapter_image_embeds=[ip_adapter_embeds],
            image=hint_tensor, # In SDXL si usa control_image invece di hint
            num_inference_steps=20,
            guidance_scale=args.guidance_scale, 
            controlnet_conditioning_scale=args.scale, # Forza della maschera ControlNet
            height=1024, width=1024, # SDXL è ottimizzato per 1024
            generator=generator
        ).images[0]

        # D. GRIGLIA DI CONFRONTO
        gen_thumb = generated_image.resize((512, 512))
        
        if os.path.exists(img_map[fname]):
            gt_img = Image.open(img_map[fname]).convert("RGB").resize((512, 512))
        else:
            gt_img = Image.new("RGB", (512, 512))
            
        mask_vis = hint_tensor[0,0].cpu().to(torch.float32).numpy() 
        mask_vis = (mask_vis * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_vis).resize((512, 512))

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