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
from diffusers import KandinskyV22Pipeline 

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# --- MODELLO LINEARE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, clip_dim) 

    def forward(self, x):
        return self.linear(x)

def normalize_and_cast(emb):
    """
    Funzione helper per normalizzare l'embedding.
    Ritorna SEMPRE float32 per evitare artefatti.
    """
    # 1. Normalizzazione L2
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    
    # 2. Casting esplicito a FLOAT32
    return emb.to(dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_fp32") # Cambio nome cartella
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Confronto Strategie S{args.subject} | ROI: {args.rois} (PRECISIONE FLOAT32) ---")

    # 1. Setup Modello Lineare
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    aligner = SimpleLinearAligner(input_dim, clip_dim=1280).to(device)
    
    model_path = f"trained_fmri_decoders/linear_sparse_lbfgs/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    if not os.path.exists(model_path):
        print(f"Modello non trovato: {model_path}")
        return
        
    # Carichiamo i pesi (che sono float32 di default)
    aligner.load_state_dict(torch.load(model_path, map_location=device))
    aligner.eval()

    # Hard Thresholding
    threshold = 1e-4
    with torch.no_grad():
        mask = aligner.linear.weight.abs() < threshold
        aligner.linear.weight.data[mask] = 0.0
    print("Modello fMRI caricato.")

    # 2. Pipeline in FLOAT 32 (NO FP16)
    # Rimuoviamo qualsiasi riferimento a torch.float16
    print("Caricamento Kandinsky Pipeline in Float32...")
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", 
        torch_dtype=torch.float32 # <--- QUI LA MAGIA ANTI-VIOLA
    ).to(device)

    # 3. Dati
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
    save_path_root = os.path.join(args.output_dir, args.rois, str(args.subject))
    os.makedirs(save_path_root, exist_ok=True)

    print(f"Generazione su {len(unique_files)} immagini...")

    # Loop Inference
    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        
        trials_np = np.array(grouped[fname]) 
        # Assicuriamoci che l'input fMRI sia float32
        trials_tensor = torch.tensor(trials_np, dtype=torch.float32).to(device)
        
        embedding_variants = {}

        with torch.no_grad():
            # A: Single Trial
            single_fmri = trials_tensor[0:1] 
            emb_single = aligner(single_fmri) 
            embedding_variants["Single Trial"] = normalize_and_cast(emb_single)

            # B: Input Avg
            input_mean = trials_tensor.mean(dim=0, keepdim=True)
            emb_input_avg = aligner(input_mean)
            embedding_variants["Input Avg"] = normalize_and_cast(emb_input_avg)

        # Generazione Immagini
        generated_imgs = {}
        seed = abs(hash(fname)*12345) % (2**32)
        
        for name, emb in embedding_variants.items():
            # Negative embedding esplicito float32
            neg_emb = torch.zeros_like(emb, dtype=torch.float32)
            
            generator = torch.Generator(device=device).manual_seed(seed)
            
            img = pipe(
                image_embeds=emb,
                negative_image_embeds=neg_emb,
                height=768, width=768,
                num_inference_steps=20,
                guidance_scale=args.guidance_scale,
                generator=generator 
            ).images[0]
            
            generated_imgs[name] = img.resize((512, 512))

        # Creazione Griglia
        W, H = 512, 512
        grid = Image.new("RGB", (W * 3, H), "black")
        
        if os.path.exists(img_map[fname]):
            try:
                gt_img = Image.open(img_map[fname]).convert("RGB").resize((W,H))
                grid.paste(gt_img, (0, 0))
            except: pass
        
        grid.paste(generated_imgs["Single Trial"], (W, 0))
        grid.paste(generated_imgs["Input Avg"], (W*2, 0))

        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth", fill="white")
        d.text((W + 10, 10), "Single Trial", fill="yellow")
        d.text((W*2 + 10, 10), "Input Avg (FP32)", fill="lime")

        grid.save(os.path.join(save_path_root, f"{real_key}_compare.png"))

    print(f"Finito. Output in {save_path_root}")

if __name__ == "__main__":
    main()