import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path
import sys

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

def main():
    dataset_root = "data/GOD_Dataset"
    output_json = "data/GOD_Dataset/captions_generated.json"
    subject = 3 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Generazione Caption Ground Truth con BLIP su {device} ---")

    # 1. Carica BLIP (Modello di Captioning State-of-the-Art)
    print("Caricamento BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    model.eval()

    # 2. Raccogli tutte le immagini (Training e Test)
    # Usiamo il loader esistente per ottenere i path corretti
    print("Scansione immagini dataset...")
    
    # Carichiamo dummy data solo per avere i path delle immagini
    # (Non ci interessa l'fMRI ora)
    rois_dict = {"VC": ALL_ROI_MAPPINGS["VC"]} 
    
    # Train set
    train_data = load_fmri_image_pairs(
        os.path.join(dataset_root, "fmri_files", "training"),
        os.path.join(dataset_root, "images", "training"),
        f"Subject{subject}_ImageNetTraining.h5", rois_dict
    )
    
    # Test set
    test_data = load_fmri_image_pairs(
        os.path.join(dataset_root, "fmri_files", "test"),
        os.path.join(dataset_root, "images", "test"),
        f"Subject{subject}_ImageNetTest.h5", rois_dict
    )

    all_paths = sorted(list(set(train_data['image_paths'] + test_data['image_paths'])))
    print(f"Trovate {len(all_paths)} immagini uniche.")

    # 3. Generazione Loop
    captions_dict = {}
    batch_size = 16 # Aumenta se hai VRAM
    
    for i in tqdm(range(0, len(all_paths), batch_size)):
        batch_paths = all_paths[i:i+batch_size]
        images = []
        valid_paths = []
        
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except:
                pass
        
        if not images: continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=5)
            caps = processor.batch_decode(out, skip_special_tokens=True)
        
        for p, c in zip(valid_paths, caps):
            fname = os.path.basename(p)
            captions_dict[fname] = c

    # 4. Salvataggio
    with open(output_json, "w") as f:
        json.dump(captions_dict, f, indent=4)
    
    print(f"Finito! Caption salvate in {output_json}")
    print("Esempio:", list(captions_dict.items())[0])

if __name__ == "__main__":
    main()