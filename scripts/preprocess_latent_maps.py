import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, SegformerImageProcessor, SegformerForSemanticSegmentation
from pathlib import Path

# Setup path per i tuoi moduli
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.config.roi_config import ALL_ROI_MAPPINGS

def main():
    dataset_root = "data/GOD_Dataset"
    output_dir = "data/GOD_Dataset/latent_targets"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"

    print("--- 1. Caricamento Oracoli (Encoder) ---")
    
    # Depth Anything Small
    depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
    depth_model.eval()
    
    # SegFormer B0
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade20k-512-512")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade20k-512-512").to(device)
    seg_model.eval()

    print("--- 2. Caricamento Dataset GOD ---")
    # Carichiamo solo per avere la lista dei file immagine corretti
    raw_data = load_fmri_image_pairs(
        os.path.join(dataset_root, "fmri_files", "training"),
        os.path.join(dataset_root, "images", "training"),
        "Subject3_ImageNetTraining.h5", 
        {"VC": ALL_ROI_MAPPINGS["VC"]} 
    )

    print(f"Estrazione Latent Embeddings su {len(raw_data['image_paths'])} immagini...")

    for img_path in tqdm(raw_data['image_paths']):
        fname = os.path.basename(img_path)
        save_name = os.path.splitext(fname)[0] + ".pt"
        save_path = os.path.join(output_dir, save_name)
        
        if os.path.exists(save_path): continue

        try:
            image = Image.open(img_path).convert("RGB")

            with torch.no_grad():
                # A. DEPTH LATENT
                # DepthAnything vuole size specifiche (es 518x518), lo fa il processor
                inputs_d = depth_processor(images=image, return_tensors="pt").to(device)
                # Estraiamo l'ultima feature map dal backbone
                out_d = depth_model.backbone(**inputs_d).feature_maps[-1] # [1, 384, H, W]
                # Forziamo pooling a 16x16 per coerenza col modello lineare
                latent_d = F.adaptive_avg_pool2d(out_d, (16, 16)).cpu()

                # B. SEMANTIC LATENT
                inputs_s = seg_processor(images=image, return_tensors="pt").to(device)
                out_s = seg_model.segformer.encoder(**inputs_s).last_hidden_state # [1, 256, 16, 16]
                latent_s = out_s.cpu() # Già 16x16 solitamente con input 512

                torch.save({"depth": latent_d, "seg": latent_s}, save_path)
                
        except Exception as e:
            print(f"Errore su {fname}: {e}")

    print("Preprocessing finito.")

if __name__ == "__main__":
    main()