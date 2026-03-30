import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# --- Pipeline SDXL ---
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

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
    emb = emb * 0.95  # Riduce il rischio di artefatti cromatici
    return emb

def get_spatial_mask_pil(depth_flat_tensor, spatial_type, threshold=0.6, target_size=(1024, 1024)):
    """Estrae la maschera spaziale dal segnale fMRI."""
    if spatial_type == "none":
        return None # Non serve maschera per il run 'none'
        
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)

    if spatial_type == "mask":
        mask_pil = Image.fromarray(bin_64_np * 255).resize(target_size, Image.NEAREST)
        return mask_pil
        
    elif spatial_type == "box":
        coords = np.argwhere(bin_64_np > 0)
        mask_np = np.zeros(target_size, dtype=np.uint8)
        if len(coords) > 5:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
            x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            mask_np[y1:y2, x1:x2] = 255
        else:
            mask_np[:, :] = 255
        return Image.fromarray(mask_np)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_sdxl_coarse_to_fine_mse")
    parser.add_argument("--guidance_scale", type=float, default=3.5) 
    parser.add_argument("--mask_threshold", type=float, default=0.6) 
    parser.add_argument("--spatial", type=str, default="none", choices=["none", "mask", "box"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16 
    print(f"--- Inference SDXL Coarse-To-Fine S{args.subject} | Spatial: {args.spatial} ---")

    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # 1. Carica Modelli fMRI
    aligner_sem = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs_mse/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
    aligner_sem.eval()
    
    if args.spatial in ["mask", "box"]:
        aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
        path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth"
        aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
        aligner_struct.eval()

    # 2. Carica le Pipeline SDXL
    print("Caricamento SDXL BASE Pipeline...")
    pipe_base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf
    ).to(device)
    pipe_base.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_base.set_ip_adapter_scale(1.0) 
    
    pipe_inpaint = None
    if args.spatial in ["mask", "box"]:
        print("Caricamento SDXL INPAINT Pipeline...")
        # Usiamo il modello SDXL specializzato per l'inpainting
        pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtype_inf
        ).to(device)
        pipe_inpaint.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe_inpaint.set_ip_adapter_scale(1.0)

    # 3. Data Load
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"), test_h5, rois_dict
    )
    
    grouped = defaultdict(list)
    img_map = {}
    for i, vec in enumerate(raw_data['fmri']):
        p = raw_data['image_paths'][i]
        fname = os.path.basename(p)
        grouped[fname].append(vec)
        img_map[fname] = p
    
    unique_files = sorted(list(grouped.keys()))
    save_dir = os.path.join(args.output_dir, f"S{args.subject}_{args.rois}_{args.spatial}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "single"), exist_ok=True)

    print(f"Generazione su {len(unique_files)} immagini...")

    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        fmri_input = torch.tensor(np.array(grouped[fname]), dtype=torch.float32).to(device).mean(dim=0, keepdim=True)

        with torch.no_grad():
            # Semantica
            emb_sem = aligner_sem(fmri_input)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
            # Struttura
            depth_flat = aligner_struct(fmri_input) if args.spatial != "none" else None

        seed = 42
        generator = torch.Generator(device=device).manual_seed(seed)
        
        coarse_image = None
        final_image = None

        if args.spatial == "none":
            final_image = pipe_base(
                prompt="", negative_prompt="",
                ip_adapter_image_embeds=[ip_adapter_embeds],
                num_inference_steps=25, guidance_scale=args.guidance_scale, 
                height=1024, width=1024, generator=generator
            ).images[0]
            
        else:
            # STEP 1: Generazione Globale
            coarse_image = pipe_base(
                prompt="", negative_prompt="",
                ip_adapter_image_embeds=[ip_adapter_embeds],
                num_inference_steps=25, guidance_scale=args.guidance_scale, 
                height=1024, width=1024, generator=generator
            ).images[0]
            
            # STEP 2: Inpainting Locale
            mask_pil = get_spatial_mask_pil(depth_flat, args.spatial, threshold=args.mask_threshold)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=12)) # Sfocatura bordi

            final_image = pipe_inpaint(
                prompt="", negative_prompt="",
                image=coarse_image,          # Immagine di background
                mask_image=mask_pil,         # Dove inpaintare (255 = inpaint, 0 = keep)
                ip_adapter_image_embeds=[ip_adapter_embeds],
                num_inference_steps=25, guidance_scale=args.guidance_scale, 
                height=1024, width=1024, generator=generator
            ).images[0]

        # Salvataggio Singola
        final_thumb = final_image.resize((512, 512))
        final_thumb.save(os.path.join(save_dir, "single", f"{real_key}.png"))

        # --- PLOTTING GRIGLIA ---
        gt_img = Image.open(img_map[fname]).convert("RGB").resize((512, 512)) if os.path.exists(img_map[fname]) else Image.new("RGB", (512, 512))

        if args.spatial == "none":
            grid = Image.new("RGB", (512 * 2, 512), "black")
            grid.paste(gt_img, (0, 0))
            grid.paste(final_thumb, (512, 0))
            d = ImageDraw.Draw(grid)
            d.text((10, 10), "Ground Truth", fill="white")
            d.text((522, 10), "SDXL Recon (None)", fill="lime")
            
        else:
            grid = Image.new("RGB", (512 * 3, 512), "black")
            grid.paste(gt_img, (0, 0))
            grid.paste(coarse_image.resize((512, 512)), (512, 0))
            grid.paste(final_thumb, (1024, 0))
            
            d = ImageDraw.Draw(grid)
            d.text((10, 10), "Ground Truth", fill="white")
            d.text((522, 10), "Step 1: Coarse (Context)", fill="cyan")
            d.text((1034, 10), f"Step 2: Fine (Inpaint {args.spatial})", fill="lime")
            
        grid.save(os.path.join(save_dir, f"{real_key}_grid.png"))

    print(f"Finito. Output in {save_dir}")

if __name__ == "__main__":
    main()