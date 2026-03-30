import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

def main():
    dataset_root = "data/GOD_Dataset" 
    output_dir = "data/precomputed_binary_masks" # Sovrascriviamo quelli vecchi
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Caricamento Modello Salient Object Detection (RMBG-1.4)...")
    # Questo modello è fatto apposta per "Subject vs Background" senza semantica
    model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
    model.to(device)
    model.eval()

    train_img_dir = os.path.join(dataset_root, "images", "test")
    images_list = sorted([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Generazione Silhouette Salienti 64x64 per {len(images_list)} immagini...")

    # Dimensioni specifiche richieste da RMBG-1.4 per l'input
    model_input_size = (1024, 1024)
    target_res = (64, 64)

    # Funzione di preprocessamento immagine
    def preprocess_image(im):
        im = im.resize(model_input_size, Image.BILINEAR)
        im_tensor = transforms.ToTensor()(im)
        im_tensor = transforms.Normalize([0.5,0.5,0.5], [1.0,1.0,1.0])(im_tensor)
        return im_tensor.unsqueeze(0).to(device)

    # Funzione di postprocessamento maschera
    def postprocess_mask(pred, orig_size):
        # Interpolazione alla dimensione originale non serve, andiamo diretti a 64x64
        # pred shape: (1, 1, 1024, 1024)
        mask = F.interpolate(pred, size=target_res, mode='bilinear')
        # Clamp tra 0 e 1
        mask = torch.clamp(mask, 0, 1)
        return mask.squeeze().cpu().numpy() # (64, 64)

    batch_size = 1 # RMBG è veloce, facciamolo uno alla volta per semplicità
    
    for fname in tqdm(images_list):
        img_path = os.path.join(train_img_dir, fname)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # 1. Inferenza
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                preds = model(input_tensor)[0][0] # Output grezzo
            
            # 2. Post-processing a 64x64
            mask_arr = postprocess_mask(preds, image.size)
            
            # 3. Binarizzazione Netta
            # RMBG dà un alpha matte sfumato sui bordi. Noi vogliamo binario netto.
            binary_mask = (mask_arr > 0.5).astype(np.float32)
            
            # 4. Flatten e Salvataggio
            target_flat = binary_mask.flatten()
            
            save_name = os.path.splitext(fname)[0] + ".npy"
            np.save(os.path.join(output_dir, save_name), target_flat)
            
        except Exception as e:
            print(f"Errore su {fname}: {e}")
            # Salva maschera nera in caso di errore per non rompere il training
            save_name = os.path.splitext(fname)[0] + ".npy"
            np.save(os.path.join(output_dir, save_name), np.zeros(64*64, dtype=np.float32))

    print("Finito. Maschere salvate in", output_dir)

if __name__ == "__main__":
    main()