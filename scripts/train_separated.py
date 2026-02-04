import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner

# --- LOSSES (Invariate) ---
class FeatureTargetLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, f, t): return 1 - F.cosine_similarity(f.view(f.size(0),-1), t.view(t.size(0),-1), dim=-1).mean()

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        k_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kx', k_x); self.register_buffer('ky', k_y)
    def forward(self, p, g):
        if p.shape[1]==3: p=p.mean(1,keepdim=True); g=g.mean(1,keepdim=True)
        return torch.abs(F.conv2d(p,self.kx,padding=1)-F.conv2d(g,self.kx,padding=1)).mean() + \
               torch.abs(F.conv2d(p,self.ky,padding=1)-F.conv2d(g,self.ky,padding=1)).mean()

class TVLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return (torch.pow(x[:,:,1:,:]-x[:,:,:-1,:],2).sum() + torch.pow(x[:,:,:,1:]-x[:,:,:,:-1],2).sum()) / x.numel()

def l1_penalty(m): return sum(torch.sum(torch.abs(p)) for p in m.parameters() if p.requires_grad)

# --- TRAINING LOGIC ---

def train_epoch(model, loader, opt, criteria, args, device, task):
    model.train()
    total_loss = 0.0
    count = 0
    
    pbar = tqdm(loader, desc=f"Train [{task}]", leave=False)
    for batch in pbar:
        fmri = batch['fmri'].to(device)
        opt.zero_grad()
        loss = 0.0

        # --- BRANCH DEPTH ---
        if task == 'depth':
            gt = batch['depth'].to(device)
            pred_depth, _ = model(fmri) # Ignoriamo output seg
            loss = (args.lambda_pixel * criteria['pixel'](pred_depth, gt)) + \
                   (args.lambda_grad * criteria['grad'](pred_depth, gt))
        
        # --- BRANCH SEG ---
        elif task == 'seg':
            gt = batch['seg'].to(device)
            _, pred_seg = model(fmri) # Ignoriamo output depth
            loss = (args.lambda_pixel * criteria['pixel'](pred_seg, gt)) + \
                   (args.lambda_tv * criteria['tv'](pred_seg))

        # --- BRANCH ALIGNER ---
        elif task == 'aligner':
            if 'alexnet' not in batch: continue
            gt = batch['alexnet'].to(device)
            pred_feat, _ = model(fmri)
            loss = criteria['align'](pred_feat, gt) + (1e-5 * l1_penalty(model))

        loss.backward()
        opt.step()
        total_loss += loss.item()
        count += 1
        pbar.set_postfix({'loss': total_loss/count})

    return total_loss / count if count > 0 else 0.0

def validate(model, loader, criteria, args, device, task):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            fmri = batch['fmri'].to(device)
            loss = 0.0
            
            if task == 'depth':
                gt = batch['depth'].to(device)
                pd, _ = model(fmri)
                loss = (args.lambda_pixel * criteria['pixel'](pd, gt)) + \
                       (args.lambda_grad * criteria['grad'](pd, gt))
            
            elif task == 'seg':
                gt = batch['seg'].to(device)
                _, ps = model(fmri)
                loss = (args.lambda_pixel * criteria['pixel'](ps, gt)) + \
                       (args.lambda_tv * criteria['tv'](ps))
            
            elif task == 'aligner':
                if 'alexnet' not in batch: continue
                gt = batch['alexnet'].to(device)
                pf, _ = model(fmri)
                loss = criteria['align'](pf, gt)

            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else float('inf')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default="3")
    parser.add_argument("--rois", type=str, default="VC")
    # TASK SPECIFICO: Uno alla volta per massima separazione
    parser.add_argument("--task", type=str, required=True, choices=["depth", "seg", "aligner"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    parser.add_argument("--lambda_pixel", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=5.0)
    parser.add_argument("--lambda_tv", type=float, default=0.5)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Totally Separated: Task [{args.task}] ---")

    # Load Features solo se serve
    alex_features = None
    if args.task == 'aligner':
        path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f: alex_features = pickle.load(f)
        else:
            print("ERR: AlexNet features not found."); return

    subjects = [int(s) for s in args.subjects.split(",")]
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in parse_roi_keys(args.rois)}
    roi_str = "_".join(rois_dict.keys())

    for sub in subjects:
        input_dim = sum([subject_dims[k][sub-1] for k in rois_dict.keys()])
        
        # Dataloading
        train_h5 = f"Subject{sub}_ImageNetTraining.h5"
        raw_data = load_fmri_image_pairs(
            os.path.join(args.dataset_root, "fmri_files", "training"),
            os.path.join(args.dataset_root, "images", "training"),
            train_h5, rois_dict
        )
        if not raw_data: continue

        tr_d, val_d = split_data_for_class(raw_data, 0.9)
        train_ds = MultimodalFmriDataset(tr_d['fmri'], tr_d['image_paths'], args.dataset_root, "training", True, alex_features)
        val_ds = MultimodalFmriDataset(val_d['fmri'], val_d['image_paths'], args.dataset_root, "training", True, alex_features)
        
        tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Init Specific Model
        model = None
        if args.task in ['depth', 'seg']:
            # Usiamo lo stesso MultiTaskDecoder architetturalmente, ma pesi diversi verranno salvati
            model = MultiTaskDecoder(input_dim).to(device)
        else:
            model = ParallelAlexNetAligner(input_dim).to(device)

        opt = optim.AdamW(model.parameters(), lr=args.lr)
        
        criteria = {
            'pixel': nn.L1Loss(), 'grad': GradientLoss().to(device),
            'tv': TVLoss().to(device), 'align': FeatureTargetLoss().to(device)
        }

        save_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{sub}", roi_str)
        os.makedirs(save_dir, exist_ok=True)
        
        # Nome file specifico per task
        save_name = f"best_{args.task}_decoder.pth"
        if args.task == 'aligner': save_name = "best_aligner.pth"
        
        print(f"Saving to: {os.path.join(save_dir, save_name)}")

        best_loss = float('inf')
        patience = 0
        
        for ep in range(args.epochs):
            tr_loss = train_epoch(model, tr_loader, opt, criteria, args, device, args.task)
            val_loss = validate(model, val_loader, criteria, args, device, args.task)
            
            msg = f"Ep {ep+1:03d} | Tr: {tr_loss:.4f} | Val: {val_loss:.4f}"
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), os.path.join(save_dir, save_name))
                msg += " *"
            else:
                patience += 1
                msg += f" (Pat: {patience})"
            
            print(msg)
            if patience >= args.patience:
                print("Early Stopping.")
                break

if __name__ == "__main__":
    main()