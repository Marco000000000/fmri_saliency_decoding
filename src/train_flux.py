import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.models.flux_aligner import FmriToFluxSequenceAligner

class FluxFmriDataset(Dataset):
    def __init__(self, fmri_data, image_paths, targets_dir):
        self.fmri_data = torch.tensor(np.array(fmri_data)).float()
        self.image_paths = image_paths
        self.targets_dir = targets_dir
        
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        fmri = self.fmri_data[idx]
        img_path = self.image_paths[idx]
        
        fname = os.path.basename(img_path)
        target_name = os.path.splitext(fname)[0] + ".pt"
        target_path = os.path.join(self.targets_dir, target_name)
        
        try:
            target_data = torch.load(target_path, map_location="cpu")
            
            # --- MODIFICA CRUCIALE PER SEQUENCE ALIGNMENT ---
            
            # CLIP: [1, 768] -> [768]
            clip_vec = target_data["clip"].squeeze(0).float()
            
            # T5: Il file originale è [1, 512, 4096].
            # Noi ne prendiamo solo i primi 77 token per risparmiare memoria e parametri.
            # 77 è la lunghezza standard di CLIP, sufficiente per descrivere l'immagine.
            t5_full = target_data["t5"].squeeze(0) # [512, 4096]
            t5_seq = t5_full[:77, :].float()       # [77, 4096]
            
        except Exception as e:
            clip_vec = torch.zeros(768).float()
            t5_seq = torch.zeros(77, 4096).float()
            
        return {
            "fmri": fmri,
            "t5": t5_seq, # Ora ritorniamo una SEQUENZA
            "clip": clip_vec
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--batch_size", type=int, default=32) # Batch piccolo per stabilità
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--targets_dir", type=str, default="data/GOD_Dataset/flux_targets")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/flux_aligner")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training MindEye3 (Flux Aligner) S{args.subject} {args.rois} ---")

    # 1. Carica Dati fMRI
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}

    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    
    # Split Train/Val
    train_dict, val_dict = split_data_for_class(raw_data, 0.9)
    
    # Crea Dataset e Loader
    train_ds = FluxFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.targets_dir)
    val_ds = FluxFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.targets_dir)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Modello
    aligner = FmriToFluxSequenceAligner(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(aligner.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    
    # Loss: MSE è standard per regressione di embedding
    criterion = nn.MSELoss()

    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)

    best_val_loss = float('inf')

    # 3. Loop
    for ep in range(args.epochs):
        aligner.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {ep+1}", leave=False):
            fmri = batch['fmri'].to(device)
            target_t5 = batch['t5'].to(device)
            target_clip = batch['clip'].to(device)
            
            optimizer.zero_grad()
            
            pred_clip, pred_t5 = aligner(fmri)
            
            # Loss combinata
            # Diamo più peso a T5 perché è più grande e contiene la semantica
            loss_t5 = criterion(pred_t5, target_t5)
            loss_clip = criterion(pred_clip, target_clip)
            
            loss = loss_t5 + loss_clip
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        # Validazione
        aligner.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                fmri = batch['fmri'].to(device)
                target_t5 = batch['t5'].to(device)
                target_clip = batch['clip'].to(device)
                
                pred_clip, pred_t5 = aligner(fmri)
                
                l_t5 = criterion(pred_t5, target_t5)
                l_clip = criterion(pred_clip, target_clip)
                val_loss += (l_t5 + l_clip).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Ep {ep+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}", end="")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_flux_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

if __name__ == "__main__":
    main()