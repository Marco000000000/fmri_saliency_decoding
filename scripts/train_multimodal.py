import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Imposta la GPU specifica come richiesto
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

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner

# --- LOSS FUNCTIONS ---

class FeatureTargetLoss(nn.Module):
    """
    Cosine Similarity Loss: L = 1 - cos_sim(pred, target)
    """
    def __init__(self):
        super(FeatureTargetLoss, self).__init__()

    def forward(self, features, targets):
        features = features.view(features.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        cosine_sim = F.cosine_similarity(features, targets, dim=-1)
        loss = 1 - cosine_sim.mean()
        return loss

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, gt):
        if pred.shape[1] == 3: 
            pred = pred.mean(dim=1, keepdim=True)
            gt = gt.mean(dim=1, keepdim=True)
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)
        return torch.abs(pred_grad_x - gt_grad_x).mean() + torch.abs(pred_grad_y - gt_grad_y).mean()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

def l1_penalty(model):
    return sum(torch.sum(torch.abs(p)) for p in model.parameters() if p.requires_grad)

# --- TRAINING LOOPS ---

def train_one_epoch(decoder, aligner, loader, opt_decoder, opt_aligner, 
                    crit_pixel, crit_grad, crit_tv, crit_align, args, device):
    decoder.train()
    aligner.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        fmri = batch['fmri'].to(device)
        gt_depth = batch['depth'].to(device)
        gt_seg = batch['seg'].to(device)
        gt_alex = batch['alexnet'].to(device) if 'alexnet' in batch else None

        # 1. Train Decoder (Maps)
        opt_decoder.zero_grad()
        pred_depth, pred_seg = decoder(fmri)
        
        l_px = crit_pixel(pred_depth, gt_depth) + crit_pixel(pred_seg, gt_seg)
        l_grad = crit_grad(pred_depth, gt_depth)
        l_tv = crit_tv(pred_seg)
        
        loss_maps = (args.lambda_pixel * l_px) + (args.lambda_grad * l_grad) + (args.lambda_tv * l_tv)
        loss_maps.backward()
        opt_decoder.step()
        
        # 2. Train Aligner (AlexNet Features)
        loss_a_val = 0
        if gt_alex is not None:
            opt_aligner.zero_grad()
            pred_feat, _ = aligner(fmri)
            
            # Cosine Loss + L1 Reg
            l_cos = crit_align(pred_feat, gt_alex)
            l_reg = 1e-5 * l1_penalty(aligner)
            
            loss_a = l_cos + l_reg
            loss_a.backward()
            opt_aligner.step()
            loss_a_val = loss_a.item()

        total_loss += loss_maps.item() + loss_a_val
        pbar.set_postfix({'MapL': loss_maps.item(), 'AlexL': loss_a_val})

    return total_loss / len(loader)

def validate(decoder, aligner, loader, crit_pixel, crit_grad, crit_tv, crit_align, args, device):
    decoder.eval()
    aligner.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            fmri = batch['fmri'].to(device)
            gt_depth = batch['depth'].to(device)
            gt_seg = batch['seg'].to(device)
            gt_alex = batch['alexnet'].to(device) if 'alexnet' in batch else None

            # Maps
            pred_depth, pred_seg = decoder(fmri)
            l_px = crit_pixel(pred_depth, gt_depth) + crit_pixel(pred_seg, gt_seg)
            l_grad = crit_grad(pred_depth, gt_depth)
            l_tv = crit_tv(pred_seg)
            loss_maps = (args.lambda_pixel * l_px) + (args.lambda_grad * l_grad) + (args.lambda_tv * l_tv)
            
            # Align
            loss_a = 0
            if gt_alex is not None:
                pred_feat, _ = aligner(fmri)
                loss_a = crit_align(pred_feat, gt_alex) # Valutiamo senza L1 penalty per la metrica pura
            
            total_loss += loss_maps + loss_a

    return total_loss.item() / len(loader)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default="3")
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    
    # Pesi Loss
    parser.add_argument("--lambda_pixel", type=float, default=10.0)
    parser.add_argument("--lambda_grad", type=float, default=5.0)
    parser.add_argument("--lambda_tv", type=float, default=0.5)
    parser.add_argument("--lambda_alex", type=float, default=1.0)
    
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    alex_feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    if not os.path.exists(alex_feat_path):
        print(f"ERRORE: {alex_feat_path} non trovato! Esegui extract_alexnet_spatial.py")
        return
    with open(alex_feat_path, "rb") as f:
        alex_features = pickle.load(f)

    subjects = [int(s) for s in args.subjects.split(",")]
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in parse_roi_keys(args.rois)}
    roi_str = "_".join(rois_dict.keys())

    for sub in subjects:
        print(f"\n--- Training S{sub} {roi_str} ---")
        input_dim = sum([subject_dims[k][sub-1] for k in rois_dict.keys()])
        
        train_h5 = f"Subject{sub}_ImageNetTraining.h5"
        raw_data = load_fmri_image_pairs(
            os.path.join(args.dataset_root, "fmri_files", "training"),
            os.path.join(args.dataset_root, "images", "training"),
            train_h5, rois_dict
        )
        if not raw_data: continue

        train_dict, val_dict = split_data_for_class(raw_data, 0.9)
        train_ds = MultimodalFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.dataset_root, "training", True, alex_features)
        val_ds = MultimodalFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.dataset_root, "training", True, alex_features)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        map_decoder = MultiTaskDecoder(input_dim).to(device)
        aligner = ParallelAlexNetAligner(input_dim).to(device)

        opt_dec = optim.AdamW(map_decoder.parameters(), lr=args.lr)
        opt_aln = optim.AdamW(aligner.parameters(), lr=args.lr)

        c_px = nn.L1Loss()
        c_gr = GradientLoss().to(device)
        c_tv = TVLoss().to(device)
        c_al = FeatureTargetLoss().to(device)

        best_loss = float('inf')
        patience_counter = 0
        save_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{sub}", roi_str)
        os.makedirs(save_dir, exist_ok=True)

        for ep in range(args.epochs):
            t_loss = train_one_epoch(map_decoder, aligner, train_loader, opt_dec, opt_aln, c_px, c_gr, c_tv, c_al, args, device)
            v_loss = validate(map_decoder, aligner, val_loader, c_px, c_gr, c_tv, c_al, args, device)
            
            print(f"Ep {ep+1:03d} | Tr: {t_loss:.4f} | Val: {v_loss:.4f}", end="")

            if v_loss < best_loss:
                best_loss = v_loss
                patience_counter = 0
                torch.save(map_decoder.state_dict(), os.path.join(save_dir, "best_map_decoder.pth"))
                torch.save(aligner.state_dict(), os.path.join(save_dir, "best_aligner.pth"))
                print(" *")
            else:
                patience_counter += 1
                print(f" (Pat: {patience_counter})")
                
            if patience_counter >= args.patience:
                print("Early Stopping.")
                break

if __name__ == "__main__":
    main()