import argparse
import os
import glob
import tarfile
import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def process_tars_for_clip(tar_files, clip_cache_dir, clip_processor, clip_model, device, desc):
    """Estrae fMRI e carica/salva i CLIP embeddings in cache."""
    X_fmri, Y_clip = [], []

    for tf in tqdm(tar_files, desc=desc):
        base_name = os.path.basename(tf).replace('.tar', '')
        cache_path = os.path.join(clip_cache_dir, f"{base_name}.npy")
        
        try:
            with tarfile.open(tf, "r") as tar:
                members = tar.getmembers()
                fmri_member = next((m for m in members if m.name.endswith('.voxel.pyd')), None)
                
                if not fmri_member:
                    continue
                
                # 1. Carica fMRI (Veloce, lo leggiamo direttamente dal tar)
                fmri_file = tar.extractfile(fmri_member)
                fmri_vec = np.load(io.BytesIO(fmri_file.read()), allow_pickle=True).flatten()
                
                # 2. Carica CLIP dalla cache o calcolalo
                if os.path.exists(cache_path):
                    clip_emb = torch.tensor(np.load(cache_path), dtype=torch.float32)
                else:
                    img_member = next((m for m in members if m.name.endswith('.png')), None)
                    if not img_member:
                        continue
                        
                    img_file = tar.extractfile(img_member)
                    img = Image.open(img_file).convert("RGB")
                    inputs = clip_processor(images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        clip_emb = clip_model(**inputs).image_embeds.cpu().squeeze(0)
                        
                    # Salva in cache per i futuri training!
                    np.save(cache_path, clip_emb.numpy())
                    
                X_fmri.append(fmri_vec)
                Y_clip.append(clip_emb)
                
        except Exception as e:
            print(f"⚠️ Errore lettura {tf}: {e}")
            
    return torch.tensor(np.array(X_fmri), dtype=torch.float32), torch.stack(Y_clip)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--subject", type=str, default="CSI1")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Percorsi puliti senza ../
    save_dir = "trained_fmri_decoders/bold5000_clip_tar"
    os.makedirs(save_dir, exist_ok=True)

    base_data_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/WAVE-BOLD5000"
    subj_dir = os.path.join(base_data_dir, args.subject)
    
    # Nuova cartella per la cache di CLIP
    clip_cache_dir = os.path.join(subj_dir, "precomputed_clip")
    os.makedirs(clip_cache_dir, exist_ok=True)

    print("\n🎨 Inizializzazione modello CLIP (LAION 1280D)...")
    clip_model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    # --- TRAIN / VAL SPLIT ---
    train_dir = os.path.join(subj_dir, "train")
    all_train_tars = sorted(glob.glob(os.path.join(train_dir, "*.tar")))
    
    np.random.seed(42) 
    shuffled_tars = np.random.permutation(all_train_tars)
    split_idx = int(0.9 * len(shuffled_tars))
    
    train_tars = shuffled_tars[:split_idx].tolist()
    val_tars = shuffled_tars[split_idx:].tolist()
    print(f"✅ Split: {len(train_tars)} Training | {len(val_tars)} Validation.")

    X_train, Y_train = process_tars_for_clip(train_tars, clip_cache_dir, clip_processor, clip_model, device, "Estrazione Training")
    X_val, Y_val = process_tars_for_clip(val_tars, clip_cache_dir, clip_processor, clip_model, device, "Estrazione Validation")

    del clip_model
    torch.cuda.empty_cache()

    print("\n⚖️ Normalizzazione Z-Score sui dati cerebrali...")
    x_mean = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, keepdim=True)
    
    X_train = (X_train - x_mean) / (x_std + 1e-6)
    X_val = (X_val - x_mean) / (x_std + 1e-6)
    
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    X_val, Y_val = X_val.to(device), Y_val.to(device)

    voxel_dim = X_train.shape[1]
    aligner = SimpleLinearAligner(voxel_dim, output_dim=1280).to(device)
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss()

    l2_lambda, l1_lambda = 1e-4, 1e-5
    best_val_loss = float('inf')

    print(f"\n🔥 Training CLIP Aligner {args.subject}...")
    for ep in range(args.epochs):
        aligner.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(aligner(X_train), Y_train)
            reg_loss = 0
            for param in aligner.parameters():
                reg_loss += (0.5 * l2_lambda * torch.sum(param ** 2)) + (l1_lambda * torch.sum(torch.sqrt(param ** 2 + 1e-6)))
            loss += reg_loss
            loss.backward()
            return loss
        
        train_loss = optimizer.step(closure)
        
        aligner.eval()
        with torch.no_grad():
            val_loss = criterion(aligner(X_val), Y_val)
            sparsity = (aligner.linear.weight.abs() < 1e-4).sum().item() / aligner.linear.weight.numel() * 100

        print(f"Ep {ep+1:02d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Sparsity: {sparsity:.2f}%", end="")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_dir, f"{args.subject}_clip_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

if __name__ == "__main__":
    main()