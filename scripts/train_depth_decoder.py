import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

class FmriDepthDataset(Dataset):
    def __init__(self, fmri_data, image_paths, depth_root):
        self.fmri = fmri_data
        self.image_paths = image_paths
        self.depth_root = depth_root
    
    def __len__(self):
        return len(self.fmri)
    
    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri[idx]).float()
        fname = os.path.basename(self.image_paths[idx])
        npy_name = os.path.splitext(fname)[0] + ".npy"
        depth_path = os.path.join(self.depth_root, npy_name)
        if os.path.exists(depth_path):
            target = np.load(depth_path)
            target = torch.tensor(target).float()
        else:
            target = torch.zeros(4096).float()
        return fmri, target

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC") 
    parser.add_argument("--epochs", type=int, default=50) # Aumentato epoche dato che abbiamo early stopping
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--depth_targets_dir", type=str, default="data/precomputed_binary_masks")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/binary_decoder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training DEPTH DECODER (With Validation) S{args.subject} {args.rois} ---")

    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}

    # 1. Caricamento Dati
    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    if raw_data is None: return

    # Carichiamo tutto in un dataset temporaneo
    full_dataset = FmriDepthDataset(raw_data['fmri'], raw_data['image_paths'], args.depth_targets_dir)
    
    # 2. Split Train / Validation (Manuale per avere tensori contigui in GPU)
    print("Caricamento dati in VRAM e Splitting...")
    loader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False) # Carica tutto
    X_all, Y_all = next(iter(loader))
    
    # Shuffle manuale degli indici prima dello split (importante!)
    indices = torch.randperm(len(X_all))
    X_all = X_all[indices]
    Y_all = Y_all[indices]

    # Split 90% Train - 10% Val
    split_idx = int(0.9 * len(X_all))
    
    X_train = X_all[:split_idx].to(device)
    Y_train = Y_all[:split_idx].to(device)
    
    X_val = X_all[split_idx:].to(device)
    Y_val = Y_all[split_idx:].to(device)
    
    print(f"Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

    # 3. Modello
    target_dim = 64 * 64 
    model = SimpleLinearAligner(input_dim, output_dim=target_dim).to(device)
    
    # Regolarizzazione Bassa (come stabilito)
    l2_lambda = 1e-4
    l1_lambda = 1e-5

    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss() 

    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)

    print(f"Start Training (L1: {l1_lambda}, L2: {l2_lambda})...")
    
    best_val_loss = float('inf')
    patience_counter = 0 # Opzionale: per fermarsi se non migliora

    for ep in range(args.epochs):
        # --- TRAIN STEP ---
        model.train()
        def closure():
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds, Y_train)
            
            # ElasticNet light
            reg_loss = 0
            for param in model.parameters():
                l2 = torch.sum(param ** 2)
                l1 = torch.sum(torch.abs(param)) 
                reg_loss += (0.5 * l2_lambda * l2) + (l1_lambda * l1)
            
            loss += reg_loss
            loss.backward()
            return loss
        
        train_loss = optimizer.step(closure)
        
        # --- VALIDATION STEP ---
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, Y_val) # MSE puro su validation
            
            # Check Sparsity
            weights = model.linear.weight
            sparsity = (weights.abs() < 1e-4).sum().item() / weights.numel() * 100

        print(f"Ep {ep+1:03d} | Train Loss: {train_loss.item():.4f} | Val MSE: {val_loss.item():.4f} | Sparsity: {sparsity:.2f}%", end="")

        # --- SAVE BEST MODEL ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, "best_depth_decoder.pth"))
            print(" [BEST SAVED *]")
            patience_counter = 0
        else:
            print("")
            patience_counter += 1
            
        # Opzionale: Early Stopping
        if patience_counter > 10:
            print("Early Stopping triggerato.")
            break

    print(f"Training finito. Best Val MSE: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()