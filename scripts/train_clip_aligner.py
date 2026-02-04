import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from tqdm import tqdm
from pathlib import Path

# Aggiungi root al path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.models.clip_aligner import FmriToClipAligner
from src.datasets.multimodal_dataset import MultimodalFmriDataset 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Stop training if validation loss doesn't improve for N epochs")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/clip_aligner")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training fMRI -> CLIP Aligner (S{args.subject} {args.rois}) ---")
    print(f"Early Stopping Patience: {args.patience}")

    # 1. Setup CLIP (Frozen Target)
    print("Caricamento CLIP Vision Model...")
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # 2. Data
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}

    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    
    if raw_data is None:
        print("Errore nel caricamento dati.")
        return

    # Split
    train_dict, val_dict = split_data_for_class(raw_data, 0.9)
    train_ds = MultimodalFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.dataset_root, load_maps=False)
    val_ds = MultimodalFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.dataset_root, load_maps=False)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Model
    aligner = FmriToClipAligner(input_dim, clip_dim=clip_model.config.projection_dim).to(device)
    optimizer = optim.AdamW(aligner.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CosineEmbeddingLoss() 

    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)

    # 4. Loop con Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for ep in range(args.epochs):
        aligner.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {ep+1}", leave=False):
            fmri = batch['fmri'].to(device)
            img_paths = batch['image_path']
            
            # Estrai CLIP Features on-the-fly
            from PIL import Image
            imgs = [Image.open(p).convert("RGB") for p in img_paths]
            # Nota: spostare le immagini su GPU dentro il processor può essere lento se fatto batch per batch,
            # ma per questo dataset va bene.
            inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
            
            with torch.no_grad():
                targets = clip_model(**inputs).image_embeds 
            
            optimizer.zero_grad()
            preds = aligner(fmri)
            
            y = torch.ones(preds.shape[0]).to(device)
            loss = criterion(preds, targets, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validate
        aligner.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                fmri = batch['fmri'].to(device)
                imgs = [Image.open(p).convert("RGB") for p in batch['image_path']]
                inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
                targets = clip_model(**inputs).image_embeds
                
                preds = aligner(fmri)
                y = torch.ones(preds.shape[0]).to(device)
                val_loss += criterion(preds, targets, y).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Ep {ep+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}", end="")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_clip_aligner.pth"))
            print(" [SAVED *]")
        else:
            patience_counter += 1
            print(f" (Pat: {patience_counter}/{args.patience})")
            
        if patience_counter >= args.patience:
            print(f"Early Stopping attivato. Miglior Val Loss: {best_val_loss:.4f}")
            break

if __name__ == "__main__":
    main()