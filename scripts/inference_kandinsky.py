import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import bdpy

# Importiamo Kandinsky 2.2 Decoder
from diffusers import KandinskyV22Pipeline

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
     
    return emb

def get_image_id(image_path):
    basename = os.path.basename(image_path)
    cat_id = int(basename.split('_')[0][1:])
    img_id = int(basename.split('_')[1].split('.')[0])
    return float('%d.%06d' % (cat_id, img_id))

def main():
    parser = argparse.ArgumentParser()
    # Usa il percorso reale del tuo dataset
    parser.add_argument("--dataset_root", type=str, default="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/GOD_Dataset")
    # Usa il percorso del file H5 che hai generato tu
    parser.add_argument("--features_file", type=str, default="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/ImageFeatures_SDXL.h5")
    parser.add_argument("--output_dir", type=str, default="reconstructions_ground_truth_kandinsky")
    parser.add_argument("--guidance_scale", type=float, default=4.0) # Kandinsky lavora bene con guidance più basse (es. 4.0)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16

    print("--- UPPER BOUND: Ricostruzione da GT con KANDINSKY 2.2 (unCLIP) ---")

    # 1. Caricamento delle feature estratte (Ground Truth)
    if not os.path.exists(args.features_file):
        print(f"ERRORE: File feature non trovato in: {args.features_file}")
        return
        
    bdata = bdpy.BData(args.features_file)
    
    # Filtriamo SOLO il test set della percezione visiva (FeatureType == 2)
    idx_test = (bdata.select('FeatureType') == 2).flatten()
    gt_features = bdata.select('sdxl_ip_adapter')[idx_test, :]
    test_image_ids = bdata.select('ImageID')[idx_test, :].flatten()

    # 2. Mappiamo gli ID ai percorsi reali delle immagini di Test
    test_img_dir = os.path.join(args.dataset_root, "images", "test")
    img_paths_map = {}
    if os.path.exists(test_img_dir):
        for fname in os.listdir(test_img_dir):
            if fname.endswith(".JPEG") or fname.endswith(".png"):
                full_path = os.path.join(test_img_dir, fname)
                img_id = get_image_id(full_path)
                img_paths_map[img_id] = full_path

    # 3. Pipeline Kandinsky 2.2 Decoder
    print("Caricamento Kandinsky 2.2 Decoder (FP16)...")
    # Nota: Non ci serve il Prior, perché abbiamo già l'embedding dell'immagine!
    pipe = KandinskyV22Pipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=dtype_inf
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generazione Upper Bound Kandinsky su {len(test_image_ids)} immagini...")

    # --- INFERENCE LOOP SUI GROUND TRUTH ---
    for i, img_id in enumerate(tqdm(test_image_ids)):
        emb_sem_np = gt_features[i] # Forma: (1280,)

        real_img_path = img_paths_map.get(img_id)
        if real_img_path:
            real_key = os.path.splitext(os.path.basename(real_img_path))[0]
        else:
            real_key = f"unknown_image_{img_id}"

        with torch.no_grad():
            # 1. Nessun clamp! Usiamo i valori crudi (Ground Truth puro)
            emb_sem = torch.tensor(emb_sem_np, dtype=dtype_inf, device=device).unsqueeze(0)
            
            # 2. Creiamo il negative embedding
            negative_emb = torch.zeros_like(emb_sem)
            
        generator = torch.Generator(device=device).manual_seed(42)
        
        # GENERAZIONE IMMAGINE
        generated_image = pipe(
            image_embeds=emb_sem,
            negative_image_embeds=negative_emb,
            num_inference_steps=25,
            guidance_scale=1.0, # <-- Abbassato da 4.0 a 2.5 per ridurre la saturazione artefatta
            height=512, width=512, 
            generator=generator
        ).images[0]

        # GRIGLIA DI CONFRONTO (Tutto a 512x512)
        if real_img_path and os.path.exists(real_img_path):
            gt_img = Image.open(real_img_path).convert("RGB").resize((512, 512))
        else:
            gt_img = Image.new("RGB", (512, 512))

        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(generated_image, (512, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Immagine Originale", fill="white")
        d.text((522, 10), "UPPER BOUND (Kandinsky 2.2)", fill="yellow")
        
        grid.save(os.path.join(args.output_dir, f"{real_key}_UB_kandinsky.png"))

    print(f"Finito. Output in {args.output_dir}")

if __name__ == "__main__":
    main()