import sys
import os
import torch
import numpy as np
import bdpy
from pathlib import Path
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, parse_roi_keys

def main():
    print("--- DATA DEBUGGING ---")
    dataset_root = "data/GOD_Dataset"
    subject = 3
    
    # 1. Load h5
    h5_path = os.path.join(dataset_root, "fmri_files", "training", f"Subject{subject}_ImageNetTraining.h5")
    dat = bdpy.BData(h5_path)
    labels = dat.get_labels('stimulus_name')
    
    print(f"Totale campioni nel dataset: {len(labels)}")
    
    # 2. Check Depth Maps
    depth_dir = os.path.join(dataset_root, "derived_maps", "depth", "training")
    
    missing = 0
    zeros = 0
    valid = 0
    
    print(f"Controllo path immagini in: {depth_dir}")
    
    # Controlliamo i primi 50 e un campione casuale
    indices = list(range(50)) + list(np.random.randint(0, len(labels), 50))
    
    for i in indices:
        raw_label = str(labels[i]).strip()
        fname_base = os.path.splitext(raw_label)[0]
        fname = fname_base + ".png"
        path = os.path.join(depth_dir, fname)
        # print(f"Controllo: {path}")
        if not os.path.exists(path):
            missing += 1
            print(f"MANCANTE: {fname}")
            continue
            
        # Load
        img = Image.open(path).convert("L")
        arr = np.array(img)
        
        if arr.max() == 0:
            zeros += 1
            print(f"VUOTA (Tutto Nero): {fname}")
        else:
            valid += 1
            # Analisi Distribuzione Classi per un'immagine valida
            # Soglie: 0.35 * 255 = ~89, 0.65 * 255 = ~166
            bg_pixels = (arr < 89).sum()
            mid_pixels = ((arr >= 89) & (arr < 166)).sum()
            fg_pixels = (arr >= 166).sum()
            total = arr.size
            print(f"OK {fname}: BG={bg_pixels/total:.2f}, MID={mid_pixels/total:.2f}, FG={fg_pixels/total:.2f}")

    print("-" * 20)
    print(f"Report Campione:")
    print(f"Valid: {valid}")
    print(f"Missing Files: {missing}")
    print(f"All Zeros (Black Images): {zeros}")
    
    if valid == 0:
        print("CRITICO: Non stiamo caricando nessuna depth map corretta. Controlla i percorsi!")

if __name__ == "__main__":
    main()