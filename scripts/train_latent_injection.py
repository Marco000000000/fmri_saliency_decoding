import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import AutoencoderKL
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset
from src.models.latent_projector import LinearLatentProjector

def train_epoch(projector, vae, loader, opt, device, task):
    projector.train()
    total_loss = 0
    # Fattore di scala magico di SD (serve per mantenere la varianza corretta)
    scaler = 0.18215 

    for batch in tqdm(loader, desc=f"Train {task}", leave=False):
        fmri = batch['fmri'].to(device)
        
        # 1. PREPARA IL TARGET (La mappa GT creata dal preprocessing)
        if task == 'depth':
            # La depth è 1 canale, il VAE ne vuole 3 (RGB). Replichiamo.
            gt_map = batch['depth'].to(device) # [B, 1, 224, 224]
            gt_map = gt_map.repeat(1, 3, 1, 1)
        else:
            gt_map = batch['seg'].to(device)   # [B, 3, 224, 224]

        # 2. RIDIMENSIONA A 512x512 (Risoluzione nativa di SD VAE)
        # Usiamo 'nearest' per la segmentazione per non mischiare i colori (indici)
        mode = 'nearest' if task == 'seg' else 'bilinear'
        gt_map = torch.nn.functional.interpolate(gt_map, size=(512, 512), mode=mode)
        
        # Normalizza in [-1, 1] come vuole il VAE (i dati arrivano in [0, 1])
        gt_map = gt_map * 2.0 - 1.0

        # 3. ESTRAI IL LATENTE TARGET (ORACOLO)
        with torch.no_grad():
            # Encoder VAE -> Distribuzione -> Sample -> Scaling
            target_latents = vae.encode(gt_map).latent_dist.sample()
            target_latents = target_latents * scaler 
            
        # 4. TRAINING (Allineamento Lineare)
        opt.zero_grad()
        pred_latents = projector(fmri) # fMRI -> Latente Predetto
        
        loss = nn.MSELoss()(pred_latents, target_latents)
        loss.backward()
        opt.step()
        
        total_loss += loss.item()

    return total_loss / len(loader)

def validate(projector, vae, loader, device, task):
    projector.eval()
    total_loss = 0
    scaler = 0.18215
    
    with torch.no_grad():
        for batch in loader:
            fmri = batch['fmri'].to(device)
            if task == 'depth':
                gt = batch['depth'].to(device).repeat(1, 3, 1, 1)
            else:
                gt = batch['seg'].to(device)
            
            mode = 'nearest' if task == 'seg' else 'bilinear'
            gt = torch.nn.functional.interpolate(gt, size=(512, 512), mode=mode)
            gt = gt * 2.0 - 1.0
            
            target_latents = vae.encode(gt).latent_dist.sample() * scaler
            pred_latents = projector(fmri)
            
            total_loss += nn.MSELoss()(pred_latents, target_latents).item()
            
    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--task", type=str, required=True, choices=['depth', 'seg'])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="trained_latent_models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Latent Projection: {args.task} (S{args.subject}) ---")

    # 1. Carica l'Oracolo (VAE)
    # Questo è il modello che usiamo per "comprimere" le mappe e poi "decomprimerle"
    print("Loading VAE Oracle...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    # 2. Dataset
    roi_keys = ["VC"]
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        f"Subject{args.subject}_ImageNetTraining.h5", rois_dict
    )
    tr_d, val_d = split_data_for_class(raw_data, 0.9)
    
    # Assicurati che MultimodalFmriDataset carichi le derived_maps se esistono
    train_ds = MultimodalFmriDataset(tr_d['fmri'], tr_d['image_paths'], args.dataset_root, "training")
    val_ds = MultimodalFmriDataset(val_d['fmri'], val_d['image_paths'], args.dataset_root, "training")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 3. Modello Studente (Lineare)
    projector = LinearLatentProjector(input_dim).to(device)
    opt = optim.Adam(projector.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    early_stop_counter = 0
    for ep in range(args.epochs):
        t_loss = train_epoch(projector, vae, train_loader, opt, device, args.task)
        v_loss = validate(projector, vae, val_loader, device, args.task)
        
        print(f"Ep {ep+1} | T: {t_loss:.4f} | V: {v_loss:.4f}", end="")
        
        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(projector.state_dict(), os.path.join(args.output_dir, f"best_latent_{args.task}.pth"))
            print(" * Saved")
        else:
            early_stop_counter += 1
            print("")
        if early_stop_counter >= 3:
            print("Early stopping triggered.")
            break
if __name__ == "__main__":
    main()