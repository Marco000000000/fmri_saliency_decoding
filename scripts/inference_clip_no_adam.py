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

# --- Pipeline Kandinsky (Decoder) ---
from diffusers import KandinskyV22Pipeline 

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# --- 1. RIDEFINIZIONE MODELLO LINEARE (Deve essere identico al training) ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, clip_dim) 

    def forward(self, x):
        return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_linear_sparse_S3")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference Linear Sparse (Decoder Only) S{args.subject} | ROI: {args.rois} ---")

    # 1. Setup Modello Lineare
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    target_dim = 1280 
    
    print(f"Creazione Modello Lineare: Input {input_dim} -> Output {target_dim}")
    aligner = SimpleLinearAligner(input_dim, clip_dim=target_dim).to(device)
    
    # Path del modello addestrato con L-BFGS + L1
    model_path = f"trained_fmri_decoders/linear_sparse_lbfgs/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello non trovato in {model_path}")
        return
    
    # Caricamento Pesi
    aligner.load_state_dict(torch.load(model_path, map_location=device))
    aligner.eval()

    # --- HARD THRESHOLDING (Post-Processing per Sparsità Reale) ---
    # L-BFGS lascia pesi tipo 0.000001. Noi li vogliamo a 0.0 spaccato.
    threshold = 1e-4
    cleaned_count = 0
    with torch.no_grad():
        mask = aligner.linear.weight.abs() < threshold
        cleaned_count = mask.sum().item()
        aligner.linear.weight.data[mask] = 0.0
    
    total_params = aligner.linear.weight.numel()
    sparsity_perc = (cleaned_count / total_params) * 100
    print(f"🧹 Pulizia Pesi (Hard Thresholding < {threshold}):")
    print(f"   - Parametri azzerati: {cleaned_count}/{total_params} ({sparsity_perc:.2f}% Sparsità Reale)")

    # 2. Setup Kandinsky Pipeline
    print("Caricamento Kandinsky 2.2 Pipeline...")
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # 3. Data Load (Test Set)
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    
    print("Caricamento Dataset Test...")
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )

    # Raggruppa i trial
    grouped = defaultdict(list)
    img_map = {}
    for i, vec in enumerate(raw_data['fmri']):
        p = raw_data['image_paths'][i]
        fname = os.path.basename(p)
        grouped[fname].append(vec)
        img_map[fname] = p
    
    unique_files = sorted(list(grouped.keys()))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Inizio ricostruzione su {len(unique_files)} immagini...")

    for fname in tqdm(unique_files, desc="Reconstructing"):
        real_key = os.path.splitext(fname)[0]
        
        # A. Media Trial fMRI
        trials = np.array(grouped[fname]) 
        fmri_mean = torch.tensor(trials.mean(axis=0, keepdims=True)).float().to(device)
        
        # B. Predizione Lineare
        with torch.no_grad():
            predicted_embedding = aligner(fmri_mean) # [1, 1280]
            
            # Normalizzazione
            norm = predicted_embedding.norm(p=2, dim=-1, keepdim=True)
            if norm.item() > 1e-6:
                predicted_embedding = predicted_embedding / norm
            
            predicted_embedding = predicted_embedding.to(dtype=pipe.dtype)
            negative_embedding = torch.zeros_like(predicted_embedding)

        # C. Generazione Kandinsky
        gen_image = pipe(
            image_embeds=predicted_embedding,
            negative_image_embeds=negative_embedding,
            height=768,
            width=768,
            num_inference_steps=25, 
            guidance_scale=args.guidance_scale
        ).images[0]

        # D. Salvataggio
        W, H = 512, 512 
        gen_image = gen_image.resize((W, H))
        
        grid = Image.new("RGB", (W*2, H), "black")
        
        if os.path.exists(img_map[fname]):
            try:
                gt_img = Image.open(img_map[fname]).convert("RGB").resize((W,H))
                grid.paste(gt_img, (0, 0))
            except: pass
        
        grid.paste(gen_image, (W, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth", fill="white")
        d.text((W+10, 10), f"Linear Sparse Recon ({sparsity_perc:.0f}%)", fill="yellow")
        
        grid.save(os.path.join(args.output_dir, f"{real_key}_recon.png"))

    print(f"Finito. Risultati in: {args.output_dir}")

if __name__ == "__main__":
    main()