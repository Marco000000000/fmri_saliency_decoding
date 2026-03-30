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

# --- Pipeline Kandinsky ---
from diffusers import KandinskyV22Pipeline, KandinskyV22InpaintPipeline

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# --- 1. RIDEFINIZIONE MODELLO LINEARE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, clip_dim) 

    def forward(self, x):
        return self.linear(x)

# --- 2. HELPER FUNZIONI MASCHERA ---
def get_bbox_mask(mask_2d, target_size=(512, 512)):
    """Trova i confini (Bounding Box) di una maschera 2D 64x64 e la scala a target_size."""
    coords = np.argwhere(mask_2d > 0)
    bbox_mask = np.zeros(target_size, dtype=np.uint8)
    if len(coords) > 5:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
        x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
        if x2 <= x1: x2 = x1 + 5
        if y2 <= y1: y2 = y1 + 5
        bbox_mask[y1:y2, x1:x2] = 1
    else:
        bbox_mask[:, :] = 1 
    return np.stack([bbox_mask]*3, axis=-1).astype(np.float32)

def predict_masks(fmri_tensor, model, threshold=0.6):
    """Passa fMRI nel modello V1, pulisce il rumore e restituisce Sagoma(512) e Box(512)."""
    with torch.no_grad(): 
        depth_flat = model(fmri_tensor)
    mask_64 = depth_flat.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)
    
    mask_pil = Image.fromarray(bin_64_np * 255).resize((512, 512), Image.NEAREST)
    mask_shape_512 = np.stack([np.array(mask_pil) / 255.0]*3, axis=-1).astype(np.float32)
    mask_box_512 = get_bbox_mask(bin_64_np, target_size=(512, 512))
    
    return mask_shape_512, mask_box_512

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_kandinsky_mse")
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--spatial", type=str, default="none", choices=["none", "mask", "box"])
    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.output_dir, f"S{args.subject}_{args.rois}_{args.spatial}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference Triple Plot Kandinsky S{args.subject} | Spatial: {args.spatial} ---")

    # 1. Setup Modelli fMRI
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    aligner_sem = SimpleLinearAligner(input_dim, clip_dim=1280).to(device)
    path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs_mse/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    if not os.path.exists(path_sem): return print(f"❌ ERRORE: Modello semantico non trovato in {path_sem}")
    aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
    aligner_sem.eval()

    if args.spatial in ["mask", "box"]:
        aligner_struct = SimpleLinearAligner(input_dim, clip_dim=4096).to(device)
        path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth"
        aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
        aligner_struct.eval()

    # 2. Setup Pipeline Kandinsky
    # Carichiamo SEMPRE la base per lo step coarse
    print("Caricamento Kandinsky BASE Pipeline...")
    pipe_base = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
    ).to(device)
    
    pipe_inpaint = None
    if args.spatial in ["mask", "box"]:
        # Carichiamo ANCHE l'inpaint per lo step fine
        print("Caricamento Kandinsky INPAINT Pipeline...")
        pipe_inpaint = KandinskyV22InpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        ).to(device)

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
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "single"), exist_ok=True)
    generator= torch.Generator(device).manual_seed(42)
    # 4. Generazione
    for fname in tqdm(unique_files, desc=f"Reconstructing ({args.spatial})"):
        real_key = os.path.splitext(fname)[0]
        fmri_mean = torch.tensor(np.array(grouped[fname]).mean(axis=0, keepdims=True)).float().to(device)
        
        # A. Predizione Semantica
        with torch.no_grad():
            emb_sem = aligner_sem(fmri_mean)
            norm = emb_sem.norm(p=2, dim=-1, keepdim=True)
            if norm.item() > 1e-6: emb_sem = emb_sem / norm
            emb_sem = emb_sem.to(dtype=pipe_base.dtype)
            neg_emb = torch.zeros_like(emb_sem)

        # B. Generazione
        coarse_image = None
        final_image = None

        if args.spatial == "none":
            # Generazione Singola (No Inpainting)
            final_image = pipe_base(
                image_embeds=emb_sem, negative_image_embeds=neg_emb,
                height=768, width=768, num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            
        else:
            # COARSE-TO-FINE (2 Step)
            # Step 1: Genera il contesto globale (Sfondo coerente)
            coarse_image = pipe_base(
                image_embeds=emb_sem, negative_image_embeds=neg_emb,
                height=768, width=768, num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            
            # Step 2: Prepara la maschera e Inpaint
            pred_shape_512, pred_box_512 = predict_masks(fmri_mean, aligner_struct, threshold=0.6)
            mask_np = pred_shape_512 if args.spatial == "mask" else pred_box_512
            
            mask_pil = Image.fromarray((mask_np[:,:,0] * 255).astype(np.uint8)).resize((768, 768), Image.NEAREST)
            # Sfocatura per blending
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=10)) 
            
            final_image = pipe_inpaint(
                image=coarse_image,      # <-- Passiamo l'immagine base come sfondo!
                mask_image=mask_pil,     # <-- Inpaintiamo solo dentro la box
                image_embeds=emb_sem, negative_image_embeds=neg_emb,
                height=768, width=768, num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator

            ).images[0]

        # C. Salvataggio Triple Plot
        W, H = 512, 512 
        final_image = final_image.resize((W, H))
        final_image.save(os.path.join(args.output_dir, "single", f"{real_key}_recon.png"))
        
        # --- CREAZIONE GRIGLIA TRIPLA O DOPPIA ---
        if args.spatial == "none":
            # Griglia Doppia Standard: GT vs Final
            grid = Image.new("RGB", (W*2, H), "black")
            d = ImageDraw.Draw(grid)
            
            # Paste GT
            if os.path.exists(img_map[fname]):
                try: grid.paste(Image.open(img_map[fname]).convert("RGB").resize((W,H)), (0, 0))
                except: pass
            
            # Paste Final
            grid.paste(final_image, (W, 0))
            
            # Labels
            d.text((10, 10), "Ground Truth", fill="white")
            d.text((W+10, 10), f"Kandinsky Recon (None)", fill="yellow")
            
        else:
            # Griglia Tripla: GT vs Coarse vs Final
            grid = Image.new("RGB", (W*3, H), "black")
            d = ImageDraw.Draw(grid)
            
            # Paste GT (W=0)
            if os.path.exists(img_map[fname]):
                try: grid.paste(Image.open(img_map[fname]).convert("RGB").resize((W,H)), (0, 0))
                except: pass
            
            # Paste Coarse (W=W)
            coarse_resized = coarse_image.resize((W, H))
            grid.paste(coarse_resized, (W, 0))
            
            # Paste Final (W=2*W)
            grid.paste(final_image, (W*2, 0))
            
            # Labels
            d.text((10, 10), "Ground Truth", fill="white")
            d.text((W+10, 10), "Step 1: Coarse (Context)", fill="cyan")
            d.text((W*2+10, 10), f"Step 2: Fine (Inpaint {args.spatial})", fill="lime")
            
        # Salvataggio Griglia
        grid.save(os.path.join(args.output_dir, f"{real_key}_triple_recon.png"))

    print(f"Finito. Risultati tripli in: {args.output_dir}")

if __name__ == "__main__":
    main()