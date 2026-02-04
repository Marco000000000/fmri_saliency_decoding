import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
# --- MODIFICA 1: Usiamo la pipeline specifica del Decoder ---
from diffusers import KandinskyV22Pipeline 
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]
# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.clip_aligner import FmriToClipAligner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_kandinsky_S3_2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference Kandinsky (Decoder Only) S{args.subject} | ROI: {args.rois} ---")

    # 1. Setup fMRI Aligner (1280 dim per Kandinsky)
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    target_dim = 1280 
    
    print(f"Caricamento fMRI Aligner (Input: {input_dim} -> Output: {target_dim})...")
    aligner = FmriToClipAligner(input_dim, clip_dim=target_dim).to(device)
    
    # Path del modello addestrato
    model_path = f"trained_fmri_decoders/kandinsky_aligner/S{args.subject}_{args.rois}/best_kandinsky_aligner.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello non trovato in {model_path}")
        print("Esegui prima 'src/train_clip_aligner.py'!")
        return
    
    aligner.load_state_dict(torch.load(model_path, map_location=device))
    aligner.eval()

    # 2. Setup Kandinsky Pipeline (Decoder)
    # Questa pipeline accetta 'image_embeds' direttamente
    print("Caricamento Kandinsky 2.2 Decoder...")
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", 
        torch_dtype=torch.float32
    ).to(device)
    # Opzionale: per risparmiare memoria
    # pipe.enable_model_cpu_offload() 

    # 3. Data Load
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
    
    print(f"Inizio ricostruzione su {len(unique_files)} immagini...")

    # Limita per test rapido se vuoi
    # unique_files = unique_files[:10]
    i=0
    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        
        # A. fMRI -> Embedding
        trials = np.array(grouped[fname]) 
        fmri_mean = torch.tensor(trials.mean(axis=0, keepdims=True)).float().to(device)
        
        def normalize_and_cast(emb, target_dtype):
            # 1. Casting a Float32 (Massima precisione prima dei calcoli)
            emb = emb.to(dtype=torch.float32)
            
            # 2. Normalizzazione Sfera Unitaria
            norm = emb.norm(p=2, dim=-1, keepdim=True)
            if norm.item() > 1e-6:
                emb = emb / norm
    
            # 3. Casting finale per la pipeline
            return emb.to(dtype=target_dtype)
            # Normalizzazione (CRUCIALE per Kandinsky)
        with torch.no_grad():
            # Predizione raw
            raw_embedding = aligner(fmri_mean) 
            
            # Normalizzazione pulita usando la funzione helper
            predicted_embedding = normalize_and_cast(raw_embedding, pipe.dtype)
            
            # Negative embedding
            negative_embedding = torch.zeros_like(predicted_embedding)
            

        # B. Generazione
        # KandinskyV22Pipeline accetta 'image_embeds'
        gen_image = pipe(
            prompt="", #TEST_CLASSES[i],
            image_embeds=predicted_embedding,
            negative_image_embeds=negative_embedding,
            height=768,
            width=768,
            num_inference_steps=25,
            guidance_scale=1.0
        ).images[0]
        i+=1
        # C. Salvataggio
        W, H = 768, 768
        gen_image = gen_image.resize((W, H))
        
        grid = Image.new("RGB", (W*2, H), "black")
        
        if os.path.exists(img_map[fname]):
            try:
                gt_img = Image.open(img_map[fname]).convert("RGB").resize((W,H))
                grid.paste(gt_img, (0, 0))
            except: pass
        
        grid.paste(gen_image, (W, 0))
        
        from PIL import ImageDraw
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "GT", fill="white")
        d.text((W+10, 10), "Reconstruction (Kandinsky)", fill="yellow")
        
        grid.save(os.path.join(args.output_dir, f"{real_key}_recon.png"))

    print("Finito.")

if __name__ == "__main__":
    main()