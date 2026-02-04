import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from diffusers import KandinskyV22Pipeline 

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.clip_aligner import FmriToClipAligner

def normalize_embeds(emb):
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    norm[norm < 1e-6] = 1.0 
    return emb / norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_deterministic_S3")
    parser.add_argument("--seed", type=int, default=42, help="Seed fisso per determinismo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Deterministic Inference S{args.subject} | ROI: {args.rois} | Seed: {args.seed} ---")

    # 1. SETUP MODELLI
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    target_dim = 1280 
    
    print(f"Caricamento fMRI Aligner...")
    aligner = FmriToClipAligner(input_dim, clip_dim=target_dim).to(device)
    model_path = f"trained_fmri_decoders/kandinsky_aligner/S{args.subject}_{args.rois}/best_kandinsky_aligner.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello non trovato in {model_path}")
        return
    aligner.load_state_dict(torch.load(model_path, map_location=device))
    aligner.eval()

    print("Caricamento Kandinsky 2.2 Decoder...")
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", 
        torch_dtype=torch.float32 # Float32 per qualità massima
    ).to(device)

    # 2. CARICAMENTO DATI
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
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Inizio generazione su {len(unique_files)} immagini...")

    # Funzione helper per generazione deterministica
    def generate_deterministic(emb, seed_val):
        # Creiamo un generatore fresco ogni volta. 
        # Questo assicura che MeanInput, MeanEmb e SingleTrial usino ESATTAMENTE
        # lo stesso rumore iniziale. Le differenze saranno solo dovute all'embedding.
        generator = torch.Generator(device=device).manual_seed(seed_val)
        
        neg_emb = torch.zeros_like(emb)
        
        return pipe(
            image_embeds=emb,
            negative_image_embeds=neg_emb,
            prompt="", 
            height=768, width=768,
            num_inference_steps=20,
            guidance_scale=1.0, # Nessuna forzatura
            generator=generator # <--- LA CHIAVE DEL DETERMINISMO
        ).images[0].resize((512, 512))

    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        trials = np.array(grouped[fname]) # [N_trials, Input_Dim]

        # --- PREPARAZIONE TENSORI ---
        fmri_mean_input = torch.tensor(trials.mean(axis=0, keepdims=True)).float().to(device)
        fmri_single_input = torch.tensor(trials[0:1]).float().to(device)
        fmri_all_trials = torch.tensor(trials).float().to(device)

        with torch.no_grad():
            # A. Mean Input
            emb_from_mean_input = normalize_embeds(aligner(fmri_mean_input)).to(dtype=torch.float32)

            # B. Single Trial
            emb_from_single = normalize_embeds(aligner(fmri_single_input)).to(dtype=torch.float32)

            # C. Mean Embedding
            all_embs = aligner(fmri_all_trials)
            emb_mean_of_embs = torch.mean(all_embs, dim=0, keepdim=True)
            emb_mean_of_embs = normalize_embeds(emb_mean_of_embs).to(dtype=torch.float32)

        # --- GENERAZIONE ---
        # Usiamo args.seed per tutti e tre. 
        # Se rilanci lo script domani, otterrai gli stessi pixel.
        img_A = generate_deterministic(emb_from_mean_input, args.seed)
        img_B = generate_deterministic(emb_mean_of_embs, args.seed)
        img_C = generate_deterministic(emb_from_single, args.seed)

        # --- GRIGLIA ---
        W, H = 512, 512
        grid = Image.new("RGB", (W*4, H), "black")

        # 1. GT
        if os.path.exists(img_map[fname]):
            try:
                gt_img = Image.open(img_map[fname]).convert("RGB").resize((W,H))
                grid.paste(gt_img, (0, 0))
            except: pass
        
        # 2. Incolla Risultati
        grid.paste(img_A, (W, 0))
        grid.paste(img_B, (W*2, 0))
        grid.paste(img_C, (W*3, 0))

        # 3. Labels Pulite (Senza testo predetto)
        d = ImageDraw.Draw(grid)
        
        def draw_label(text, x):
            d.rectangle([x+10, 10, x+220, 40], fill="black")
            d.text((x+15, 15), text, fill="white")

        draw_label("Ground Truth", 0)
        draw_label("Mean Input (Best)", W)
        draw_label("Mean Embedding", W*2)
        draw_label("Single Trial (Noisy)", W*3)

        grid.save(os.path.join(args.output_dir, f"{real_key}_det.png"))

    print(f"Finito. Risultati deterministici in {args.output_dir}")

if __name__ == "__main__":
    main()