import os
import glob
import tarfile
import io
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from tqdm import tqdm

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def get_train_stats(train_dir):
    """Estrae tutti i fMRI di train solo per calcolare Media e Std per lo Z-Score."""
    print("⚖️ Calcolo Z-Score dal Train Set...")
    X_train = []
    for tf in tqdm(glob.glob(os.path.join(train_dir, "*.tar")), desc="Lettura fMRI Train"):
        with tarfile.open(tf, "r") as tar:
            fmri_member = next((m for m in tar.getmembers() if m.name.endswith('.voxel.pyd')), None)
            if fmri_member:
                X_train.append(np.load(io.BytesIO(tar.extractfile(fmri_member).read()), allow_pickle=True).flatten())
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    return X_train.mean(dim=0, keepdim=True), X_train.std(dim=0, keepdim=True), X_train.shape[1]

def get_midas_gt(img, processor, model, device):
    """Genera la vera Mappa di Profondità (Ground Truth) al volo per il plot."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
        depth_64 = F.interpolate(depth.unsqueeze(1), size=(64, 64), mode="bicubic", align_corners=False).squeeze()
        d_min, d_max = depth_64.min(), depth_64.max()
        norm_depth = (depth_64 - d_min) / (d_max - d_min + 1e-8)
    return norm_depth.cpu().numpy()

def main():
    subject = "CSI1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_data_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/WAVE-BOLD5000"
    
    train_dir = os.path.join(base_data_dir, subject, "train")
    test_dir = os.path.join(base_data_dir, subject, "test")

    # 1. Z-Scoring
    x_mean, x_std, voxel_dim = get_train_stats(train_dir)
    x_mean, x_std = x_mean.to(device), x_std.to(device)

    # 2. Caricamento Modelli
    print("\n🧠 Caricamento Decodificatore Depth...")
    aligner = SimpleLinearAligner(voxel_dim, output_dim=4096).to(device)
    aligner.load_state_dict(torch.load(f"trained_fmri_decoders/bold5000_depth_tar/{subject}_depth_aligner.pth", map_location=device))
    aligner.eval()

    print("📏 Caricamento MiDaS (DPT) per calcolo Ground Truth...")
    depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()

    # ==========================================
    # ELABORAZIONE TRAIN SET (Senza media, 4 sample random)
    # ==========================================
    print("\n🔍 Campionamento Train Set...")
    train_tars = random.sample(glob.glob(os.path.join(train_dir, "*.tar")), 4)
    train_results = []
    
    for tf in train_tars:
        with tarfile.open(tf, "r") as tar:
            img_m = next(m for m in tar.getmembers() if m.name.endswith('.png'))
            fmri_m = next(m for m in tar.getmembers() if m.name.endswith('.voxel.pyd'))
            
            img = Image.open(tar.extractfile(img_m)).convert("RGB")
            gt_depth = get_midas_gt(img, depth_processor, depth_model, device)
            
            fmri_raw = np.load(io.BytesIO(tar.extractfile(fmri_m).read()), allow_pickle=True).flatten()
            fmri_vec = torch.tensor(fmri_raw, dtype=torch.float32).to(device).unsqueeze(0)
            fmri_z = (fmri_vec - x_mean) / (x_std + 1e-6)
            
            with torch.no_grad():
                pred_depth = aligner(fmri_z).cpu().numpy().reshape(64, 64)
                
            train_results.append((img, gt_depth, pred_depth))

    # ==========================================
    # ELABORAZIONE TEST SET (Con Media delle ripetizioni!)
    # ==========================================
    print("🔍 Raggruppamento e Media del Test Set...")
    test_dict = {} # img_name -> {'fmris': [], 'img': PIL}
    
    for tf in tqdm(glob.glob(os.path.join(test_dir, "*.tar")), desc="Lettura Test"):
        with tarfile.open(tf, "r") as tar:
            json_m = next(m for m in tar.getmembers() if m.name.endswith('.json'))
            img_m = next(m for m in tar.getmembers() if m.name.endswith('.png'))
            fmri_m = next(m for m in tar.getmembers() if m.name.endswith('.voxel.pyd'))
            
            config = json.loads(tar.extractfile(json_m).read().decode('utf-8'))
            img_name = config['img_name']
            
            fmri_raw = np.load(io.BytesIO(tar.extractfile(fmri_m).read()), allow_pickle=True).flatten()
            
            if img_name not in test_dict:
                test_dict[img_name] = {'fmris': [], 'img': Image.open(tar.extractfile(img_m)).convert("RGB")}
            test_dict[img_name]['fmris'].append(fmri_raw)

    # Seleziona 4 immagini dal test set (preferibilmente quelle con più ripetizioni)
    sorted_test_keys = sorted(test_dict.keys(), key=lambda k: len(test_dict[k]['fmris']), reverse=True)
    sampled_test_keys = sorted_test_keys[4:8]
    test_results = []

    for key in sampled_test_keys:
        data = test_dict[key]
        img = data['img']
        num_trials = len(data['fmris'])
        
        gt_depth = get_midas_gt(img, depth_processor, depth_model, device)
        
        # 🚨 LA MEDIA DEI TRIAL 🚨
        fmri_avg = np.mean(data['fmris'], axis=0)
        fmri_vec = torch.tensor(fmri_avg, dtype=torch.float32).to(device).unsqueeze(0)
        fmri_z = (fmri_vec - x_mean) / (x_std + 1e-6)
        
        with torch.no_grad():
            pred_depth = aligner(fmri_z).cpu().numpy().reshape(64, 64)
            
        test_results.append((img, gt_depth, pred_depth, num_trials))

    # ==========================================
    # PLOTTING
    # ==========================================
    fig, axes = plt.subplots(8, 3, figsize=(12, 24))
    plt.subplots_adjust(hspace=0.4)
    
    # Plot Train
    for i, (img, gt, pred) in enumerate(train_results):
        axes[i, 0].imshow(img); axes[i, 0].set_title(f"Train {i+1} Originale"); axes[i, 0].axis('off')
        axes[i, 1].imshow(gt, cmap='inferno', vmin=0, vmax=1); axes[i, 1].set_title("GT (MiDaS)"); axes[i, 1].axis('off')
        axes[i, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1); axes[i, 2].set_title("Predizione fMRI"); axes[i, 2].axis('off')

    # Plot Test
    for i, (img, gt, pred, trials) in enumerate(test_results):
        row = i + 4
        axes[row, 0].imshow(img); axes[row, 0].set_title(f"Test {i+1} Orig (Media {trials} trial)"); axes[row, 0].axis('off')
        axes[row, 1].imshow(gt, cmap='inferno', vmin=0, vmax=1); axes[row, 1].set_title("GT (MiDaS)"); axes[row, 1].axis('off')
        axes[row, 2].imshow(pred, cmap='inferno', vmin=0, vmax=1); axes[row, 2].set_title("Predizione fMRI (Mediata)"); axes[row, 2].axis('off')

    plt.suptitle("Check Profondità: Train (Single-Trial) vs Test (Averaged fMRI)", fontsize=16, fontweight='bold', y=0.92)
    save_path = "check_depth_predictions.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n✅ Dashboard salvata! Apri: {save_path}")

if __name__ == "__main__":
    main()