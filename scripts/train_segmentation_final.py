import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import bdpy
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_segmentation_dataset import FmriSegmentationDataset
from src.models.segmentation_expert import MultiClassExpert

# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, C, H, W], targets: [N, H, W]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100) # Abbondiamo, tanto c'è Early Stopping
    parser.add_argument("--patience", type=int, default=15) # Stop se non migliora
    parser.add_argument("--lr", type=float, default=2e-4) 
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--save_dir", type=str, default="trained_segmentation_focal")
    return parser.parse_args()

def load_data_local(fmri_dir, image_dir, h5_filename, rois_dict):
    print(f"Caricamento dati da {h5_filename}...")
    file_path = os.path.join(fmri_dir, h5_filename)
    if not os.path.exists(file_path): raise FileNotFoundError(f"File non trovato: {file_path}")
    dat = bdpy.BData(file_path)
    x = None
    for roi_name, roi_select_str in rois_dict.items():
        try:
            roi_data = dat.select(roi_select_str)
            x = roi_data if x is None else np.hstack([x, roi_data])
        except Exception as e:
            print(f"Errore caricamento ROI {roi_name}: {e}")
            return None
    labels = dat.get_labels('stimulus_name')
    image_paths = [os.path.join(image_dir, str(lbl).strip()) for lbl in labels]
    return {'fmri': x, 'image_paths': np.array(image_paths)}

def validate(model, loader, criterion, device):
    """
    Valida il modello. Usiamo la stessa loss del training (Focal) 
    per assicurarci che stiamo migliorando sull'obiettivo reale.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            fmri = batch['fmri'].to(device)
            targets = batch['mask_target'].to(device)
            
            logits = model(fmri)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            
    return total_loss / len(loader)

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Segmentation FOCAL + Early Stopping (S{args.subject}) ---")
    
    # 1. Load Data
    roi_keys = parse_roi_keys("VC")
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_data_local(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    
    # 2. Split Train/Val (90/10)
    num_samples = len(raw_data['fmri'])
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(0.9 * num_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    
    print(f"Samples: {num_samples} | Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    # Dataset
    train_ds = FmriSegmentationDataset(raw_data['fmri'][train_idx], raw_data['image_paths'][train_idx], args.dataset_root)
    val_ds = FmriSegmentationDataset(raw_data['fmri'][val_idx], raw_data['image_paths'][val_idx], args.dataset_root)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Model & Loss
    model = MultiClassExpert(input_dim, num_classes=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # FOCAL LOSS SETUP
    # Pesi statici per focalizzare l'attenzione
    alpha_weights = torch.tensor([0.2, 1.5, 8.0]).to(device) 
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Tracking
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 4. Loop
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False)
        for batch in pbar:
            fmri = batch['fmri'].to(device)
            targets = batch['mask_target'].to(device) 
            
            optimizer.zero_grad()
            logits = model(fmri) 
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
        
        # 5. Validation
        avg_train_loss = ep_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)
        
        log_msg = f"Ep {epoch+1:03d} | Focal Train: {avg_train_loss:.4f} | Focal Val: {avg_val_loss:.4f}"
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Salva il migliore
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"S{args.subject}_seg_focal.pth"))
            log_msg += " [SAVED *]"
        else:
            patience_counter += 1
            log_msg += f" (Pat: {patience_counter}/{args.patience})"
            
        print(log_msg)
        
        if patience_counter >= args.patience:
            print(f"Early Stopping! Best Val Loss: {best_val_loss:.4f}")
            break

if __name__ == "__main__":
    main()