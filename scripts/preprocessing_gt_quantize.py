import os
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from pathlib import Path
import multiprocessing

# Configurazione
DATASET_ROOT = "data/GOD_Dataset"
INPUT_DIR = os.path.join(DATASET_ROOT, "derived_maps", "depth", "training")
# Salviamo in una nuova cartella per non confonderci
OUTPUT_DIR = os.path.join(DATASET_ROOT, "derived_maps", "depth_quantized_visible", "training")

def quantize_image_visible(file_info):
    fname, input_path, output_path = file_info
    
    try:
        # 1. Carica e Normalizza
        img = Image.open(input_path).convert("L")
        arr = np.array(img).astype(np.float32) / 255.0
        
        # Skip immagini piatte
        if arr.std() < 0.01:
            Image.new("L", img.size, 0).save(output_path)
            return
            
        # 2. K-Means
        h, w = arr.shape
        pixels = arr.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=3)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_.flatten()
        
        # 3. Ordina i centri (0=BG, 1=MID, 2=FG)
        sorted_indices = np.argsort(centers)
        map_label = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
        
        # Mappa labels a 0, 1, 2
        mapped_labels = np.array([map_label[l] for l in labels]).reshape(h, w)
        
        # 4. CONVERSIONE IN GRIGI VISIBILI (La parte che volevi)
        # 0 -> 0   (Nero)
        # 1 -> 127 (Grigio)
        # 2 -> 255 (Bianco)
        final_img_arr = np.zeros_like(mapped_labels, dtype=np.uint8)
        final_img_arr[mapped_labels == 1] = 127
        final_img_arr[mapped_labels == 2] = 255
        
        # Salva
        res_img = Image.fromarray(final_img_arr)
        res_img.save(output_path)
        
    except Exception as e:
        print(f"Errore su {fname}: {e}")

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Errore: Input {INPUT_DIR} non trovato.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.JPEG'))]
    print(f"Quantizzazione VISIBILE per {len(files)} file...")
    
    tasks = []
    for f in files:
        in_p = os.path.join(INPUT_DIR, f)
        # Assicuriamoci che l'estensione sia .png
        out_name = os.path.splitext(f)[0] + ".png"
        out_p = os.path.join(OUTPUT_DIR, out_name)
        tasks.append((f, in_p, out_p))
    
    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm(pool.imap(quantize_image_visible, tasks), total=len(tasks)))
        
    print(f"Finito! Immagini salvate in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()