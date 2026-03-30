import argparse
import os
import glob
import tarfile
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import DPTImageProcessor, DPTForDepthEstimation

# Modello Lineare (Sempre senza Bias per robustezza)
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def process_tars_for_depth(tar_files, depth_cache_dir, depth_processor, depth_model, device, desc):
    """Estrae fMRI e genera le Mappe di Profondità con MiDaS (DPT)."""
    X_fmri, Y_depth = [], []

    for tf in tqdm(tar_files, desc=desc):
        base_name = os.path.basename(tf).replace('.tar', '')
        cache_path = os.path.join(depth_cache_dir, f"{base_name}.npy")
        
        try:
            with tarfile.open(tf, "r") as tar:
                members = tar.getmembers()
                img_member = next((m for m in members if m.name.endswith('.png')), None)
                fmri_member = next((m for m in members if m.name.endswith('.voxel.pyd')), None)
                
                if img_member and fmri_member:
                    # 1. Carica fMRI
                    fmri_file = tar.extractfile(fmri_member)
                    fmri_vec = np.load(io.BytesIO(fmri_file.read()), allow_pickle=True).flatten()
                    
                    # 2. Carica o Genera Depth Map
                    if os.path.exists(cache_path):
                        depth_flat = np.load(cache_path)
                    else:
                        img_file = tar.extractfile(img_member)
                        img = Image.open(img_file).convert("RGB")
                        
                        inputs = depth_processor(images=img, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            outputs = depth_model(**inputs)
                            predicted_depth = outputs.predicted_depth
                            
                            # Resize morbido a 64x64
                            depth_64 = F.interpolate(
                                predicted_depth.unsqueeze(1),
                                size=(64, 64),
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze()
                            
                            # Normalizzazione Min-Max rigorosa (da 0.0 a 1.0)
                            d_min, d_max = depth_64.min(), depth_64.max()
                            norm_depth = (depth_64 - d_min) / (d_max - d_min + 1e-8)
                            
                            depth_flat = norm_depth.cpu().numpy().flatten()
                            
                            # Salva in cache
                            np.save(cache_path, depth_flat)
                            
                    X_fmri.append(fmri_vec)
                    Y_depth.append(torch.tensor(depth_flat, dtype=torch.float32))
                    
        except Exception as e:
            print(f"⚠️ Errore su {tf}: {e}")
            
    return torch.tensor(np.array(X_fmri), dtype=torch.float32), torch.stack(Y_depth)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--subject", type=str, default="CSI1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_data_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/WAVE-BOLD5000"
    subj_dir = os.path.join(base_data_dir, args.subject)
    
    save_dir = "trained_fmri_decoders/bold5000_depth_tar"
    depth_cache_dir = os.path.join(subj_dir, "precomputed_depths")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(depth_cache_dir, exist_ok=True)

    print("\n📏 Inizializzazione modello MiDaS (DPT-Large) per la Profondità...")
    depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()

    # --- TRAIN / VAL SPLIT RIGOROSO ---
    train_dir = os.path.join(subj_dir, "train")
    all_train_tars = sorted(glob.glob(os.path.join(train_dir, "*.tar")))
    
    np.random.seed(42) 
    shuffled_tars = np.random.permutation(all_train_tars)
    split_idx = int(0.9 * len(shuffled_tars))
    
    train_tars = shuffled_tars[:split_idx].tolist()
    val_tars = shuffled_tars[split_idx:].tolist()
    print(f"✅ Split completato: {len(train_tars)} Training | {len(val_tars)} Validation.")

    X_train, Y_train = process_tars_for_depth(train_tars, depth_cache_dir, depth_processor, depth_model, device, "Estrazione Training")
    X_val, Y_val = process_tars_for_depth(val_tars, depth_cache_dir, depth_processor, depth_model, device, "Estrazione Validation")

    del depth_model
    torch.cuda.empty_cache()

    print("\n⚖️ Normalizzazione Z-Score sui dati cerebrali...")
    x_mean, x_std = X_train.mean(dim=0, keepdim=True), X_train.std(dim=0, keepdim=True)
    X_train = (X_train - x_mean) / (x_std + 1e-6)
    X_val = (X_val - x_mean) / (x_std + 1e-6)
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)
    
    Y_train = Y_train.view(Y_train.size(0), -1)
    Y_val = Y_val.view(Y_val.size(0), -1)
    
    voxel_dim = X_train.shape[1]
    aligner = SimpleLinearAligner(voxel_dim, output_dim=4096).to(device)
    
    # Torniamo all'ottimizzatore che ha dominato Shen!
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss() # La MSELoss è perfetta per la profondità continua

    l2_lambda, l1_lambda = 1e-4, 1e-5 
    best_val_loss = float('inf')

    print(f"\n🔥 Training Depth Aligner {args.subject}...")
    for ep in range(args.epochs):
        aligner.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(aligner(X_train), Y_train)
            reg_loss = sum((0.5 * l2_lambda * torch.sum(p ** 2)) + (l1_lambda * torch.sum(torch.sqrt(p ** 2 + 1e-6))) for p in aligner.parameters())
            (loss + reg_loss).backward()
            return loss + reg_loss
        train_loss = optimizer.step(closure)
        
        aligner.eval()
        with torch.no_grad():
            val_loss = criterion(aligner(X_val), Y_val)

        print(f"Ep {ep+1:02d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
        
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_dir, f"{args.subject}_depth_aligner.pth"))
            print("   -> [SAVED *]")

if __name__ == "__main__":
    main()