import argparse
import os
import sys
import pickle
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path

# Usiamo la pipeline standard (senza ControlNet)
from diffusers import StableDiffusionXLPipeline

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    emb = emb * 0.95
    return emb

# Helper per parsare gli ID alla Kamitani e recuperare il file
def get_image_id(image_path):
    basename = os.path.basename(image_path)
    cat_id = int(basename.split('_')[0][1:])
    img_id = int(basename.split('_')[1].split('.')[0])
    return float('%d.%06d' % (cat_id, img_id))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--roi", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_kamitani_semantic")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    
    # Ora puntiamo al file dei risultati uniti di Kamitani
    parser.add_argument("--results_file", type=str, default="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/kamitani_method/results/GenericObjectDecoding.pkl")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    print(f"--- Inference KAMITANI Results -> SDXL S{args.subject} | {args.roi} ---")

    # 1. Carica i risultati uniti di Kamitani
    if not os.path.exists(args.results_file):
        print(f"ERRORE: Manca il file dei risultati in: {args.results_file}")
        return
        
    with open(args.results_file, 'rb') as f:
        results_df = pickle.load(f)

    # 2. Filtra il DataFrame per il Soggetto e la ROI scelti
    sbj_str = f"Subject{args.subject}"
    filtered_df = results_df[(results_df['subject'] == sbj_str) & (results_df['roi'] == args.roi)]
    
    if filtered_df.empty:
        print(f"ERRORE: Nessun risultato trovato per {sbj_str} e ROI {args.roi}.")
        return
        
    # Estraiamo gli array dalla riga (dovrebbe essercene solo una)
    row = filtered_df.iloc[0]
    predicted_features = row['predicted_feature_averaged_percept'] # (50, 1280)
    test_labels = row['test_label_set_percept'] # (50,) ID delle immagini
    
    # 3. Mappiamo gli ID di Kamitani ai percorsi reali delle immagini di Test
    test_img_dir = os.path.join(args.dataset_root, "images", "test")
    img_paths_map = {}
    if os.path.exists(test_img_dir):
        for fname in os.listdir(test_img_dir):
            if fname.endswith(".JPEG") or fname.endswith(".png"):
                full_path = os.path.join(test_img_dir, fname)
                img_id = get_image_id(full_path)
                img_paths_map[img_id] = full_path

    # 4. Pipeline SDXL + IP-Adapter (No ControlNet)
    print("Caricamento SDXL Base e IP-Adapter (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0)

    # Setup cartella output
    save_dir = os.path.join(args.output_dir, args.roi, str(args.subject))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir,"single"), exist_ok=True)
    print(f"Generazione su {len(test_labels)} immagini pre-calcolate...")

    # --- INFERENCE LOOP SUI RISULTATI ---
    for i, img_id in enumerate(tqdm(test_labels)):
        # Recuperiamo la feature predetta per questa specifica immagine
        emb_sem_np = predicted_features[i] # Forma: (1280,)
        emb_sem_np = np.expand_dims(emb_sem_np, axis=0) # Forma: (1, 1280)

        # Cerchiamo il percorso reale dell'immagine
        real_img_path = img_paths_map.get(img_id)
        if real_img_path:
            real_key = os.path.splitext(os.path.basename(real_img_path))[0]
        else:
            real_key = f"unknown_image_{img_id}"

        with torch.no_grad():
            # Conversione per PyTorch IP-Adapter
            emb_sem = torch.tensor(emb_sem_np, dtype=torch.float32).to(device)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
        generator = torch.Generator(device=device).manual_seed(42)
        
        # GENERAZIONE IMMAGINE
        generated_image = pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image_embeds=[ip_adapter_embeds],
            num_inference_steps=20,
            guidance_scale=args.guidance_scale, 
            height=1024, width=1024, 
            generator=generator
        ).images[0]

        # GRIGLIA DI CONFRONTO
        gen_thumb = generated_image.resize((512, 512))
       
        gen_thumb.save(os.path.join(save_dir,"single", f"{real_key}_pred.png"))
        if real_img_path and os.path.exists(real_img_path):
            gt_img = Image.open(real_img_path).convert("RGB").resize((512, 512))
        else:
            gt_img = Image.new("RGB", (512, 512))

        # Creiamo un'immagine doppia: Ground Truth vs Predetta
        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(gen_thumb, (512, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth", fill="white")
        d.text((522, 10), f"Brain Recon. (Kamitani + IP-Adapter) [ID: {img_id}]", fill="lime")
        
        grid.save(os.path.join(save_dir, f"{real_key}.png"))

    print(f"Finito. Output in {save_dir}")

if __name__ == "__main__":
    main()