import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import bdpy

# Usiamo la pipeline standard
from diffusers import StableDiffusionXLPipeline

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    emb = emb * 0.95
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
    parser.add_argument("--output_dir", type=str, default="reconstructions_ground_truth")
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16

    print("--- UPPER BOUND: Ricostruzione dalle Feature SDXL Originali (Ground Truth) ---")

    # 1. Caricamento delle feature estratte (Ground Truth)
    if not os.path.exists(args.features_file):
        print(f"ERRORE: File feature non trovato in: {args.features_file}")
        return
        
    bdata = bdpy.BData(args.features_file)
    
    # Filtriamo SOLO il test set della percezione visiva (FeatureType == 2)
    # in modo da generare esattamente le stesse 50 immagini che stiamo decodificando
    idx_test = (bdata.select('FeatureType') == 2).flatten()
    gt_features = bdata.select('sdxl_ip_adapter')[idx_test, :]
    test_image_ids = bdata.select('ImageID')[idx_test, :].flatten()

    # 2. Mappiamo gli ID ai percorsi reali delle immagini di Test (per fare la griglia)
    test_img_dir = os.path.join(args.dataset_root, "images", "test")
    img_paths_map = {}
    if os.path.exists(test_img_dir):
        for fname in os.listdir(test_img_dir):
            if fname.endswith(".JPEG") or fname.endswith(".png"):
                full_path = os.path.join(test_img_dir, fname)
                img_id = get_image_id(full_path)
                img_paths_map[img_id] = full_path

    # 3. Pipeline SDXL + IP-Adapter
    print("Caricamento SDXL Base e IP-Adapter (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generazione Upper Bound su {len(test_image_ids)} immagini...")

    # --- INFERENCE LOOP SUI GROUND TRUTH ---
    for i, img_id in enumerate(tqdm(test_image_ids)):
        emb_sem_np = gt_features[i] # Forma: (1280,)
        emb_sem_np = np.expand_dims(emb_sem_np, axis=0) # Forma: (1, 1280)

        real_img_path = img_paths_map.get(img_id)
        if real_img_path:
            real_key = os.path.splitext(os.path.basename(real_img_path))[0]
        else:
            real_key = f"unknown_image_{img_id}"

        with torch.no_grad():
            emb_sem = torch.tensor(emb_sem_np, dtype=torch.float32).to(device)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
        generator = torch.Generator(device=device).manual_seed(42)
        
        # GENERAZIONE IMMAGINE (Da Embedding Perfetto)
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
        
        if real_img_path and os.path.exists(real_img_path):
            gt_img = Image.open(real_img_path).convert("RGB").resize((512, 512))
        else:
            gt_img = Image.new("RGB", (512, 512))

        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(gen_thumb, (512, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Immagine Originale", fill="white")
        d.text((522, 10), "UPPER BOUND (Generata da Feature Reale)", fill="yellow")
        
        grid.save(os.path.join(args.output_dir, f"{real_key}_UB.png"))

    print(f"Finito. Output in {args.output_dir}")

if __name__ == "__main__":
    main()