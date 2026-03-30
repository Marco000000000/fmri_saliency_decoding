import argparse
import os
import sys
import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Usiamo la pipeline standard (senza ControlNet)
from diffusers import StableDiffusionXLPipeline

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# Importiamo la libreria Kamitani per il bias
from bdpy.ml import add_bias

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    emb = emb * 0.95
    return emb

def predict_with_kamitani_slir(fmri_input_np, model_dict):
    """Applica i modelli SLiR univariati all'input fMRI"""
    norm_mean_x = model_dict['norm_mean_x']
    norm_scale_x = model_dict['norm_scale_x']
    
    # Normalizzazione con metriche di addestramento
    X_test = (fmri_input_np - norm_mean_x) / norm_scale_x
    
    target_dim = len(model_dict['models'])
    y_preds = []
    
    for i in range(target_dim):
        m_info = model_dict['models'][i]
        x_unit = X_test[:, m_info['voxel_index']]
        x_unit_biased = add_bias(x_unit, axis=1)
        
        y_pred = m_info['model'].predict(x_unit_biased)
        y_pred = y_pred * m_info['norm_scale_y'] + m_info['norm_mean_y']
        y_preds.append(y_pred)
        
    return np.vstack(y_preds).T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_kamitani_semantic")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    
    # Puntiamo direttamente al file che hai appena addestrato!
    parser.add_argument("--sem_decoder_path", type=str, default="trained_fmri_decoders/kamitani_slir/S3_VC/kamitani_slir_models.pkl")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    print(f"--- Inference KAMITANI SLiR (Semantic Only) -> SDXL S{args.subject} | {args.rois} ---")

    # 1. Carica Decoder SLiR
    if not os.path.exists(args.sem_decoder_path):
        print(f"ERRORE: Manca il decoder SLiR in: {args.sem_decoder_path}")
        return
    with open(args.sem_decoder_path, 'rb') as f:
        kamitani_sem_dict = pickle.load(f)

    # 2. Pipeline SDXL + IP-Adapter (No ControlNet)
    print("Caricamento SDXL Base e IP-Adapter (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0)

    # 3. Dati Test
    roi_keys = parse_roi_keys(args.rois)
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
        
        # Array di tutti i trial fMRI per questa immagine
        trials_np = np.array(grouped[fname]) 

        with torch.no_grad():
            # 1. Decode Semantica tramite SLiR per ogni trial separatamente
            emb_sem_np_trials = predict_with_kamitani_slir(trials_np, kamitani_sem_dict)
            
            # 2. Media delle predizioni (metodo Kamitani)
            emb_sem_np = emb_sem_np_trials.mean(axis=0, keepdims=True)
            
            # Conversione per PyTorch IP-Adapter
            emb_sem = torch.tensor(emb_sem_np, dtype=torch.float32).to(device)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
        generator = torch.Generator(device=device).manual_seed(42)
        
        # 3. GENERAZIONE IMMAGINE (Solo Semantica)
        generated_image = pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image_embeds=[ip_adapter_embeds],
            num_inference_steps=20,
            guidance_scale=args.guidance_scale, 
            height=1024, width=1024, 
            generator=generator
        ).images[0]

        # 4. GRIGLIA DI CONFRONTO
        gen_thumb = generated_image.resize((512, 512))
        
        if os.path.exists(img_map[fname]):
            gt_img = Image.open(img_map[fname]).convert("RGB").resize((512, 512))
        else:
            gt_img = Image.new("RGB", (512, 512))

        # Creiamo un'immagine doppia: Ground Truth vs Predetta
        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(gen_thumb, (512, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth", fill="white")
        d.text((522, 10), "Brain Reconstruction (Kamitani + IP-Adapter)", fill="lime")
        
        grid.save(os.path.join(save_dir, f"{real_key}.png"))

    print(f"Finito. Output in {save_dir}")

if __name__ == "__main__":
    main()