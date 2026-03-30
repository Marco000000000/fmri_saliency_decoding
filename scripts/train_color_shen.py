import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import bdpy
from PIL import Image
from tqdm import tqdm

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
        self.sigmoid = nn.Sigmoid() # FIX 1: Forza i valori strettamente tra 0.0 (Nero) e 1.0 (Luce massima)
        
    def forward(self, x): 
        return self.sigmoid(self.linear(x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--epochs", type=int, default=50) # Possiamo alzare le epoche dato che è leggerissimo
    
    # 512 / 64 = 8. Creiamo una palette 8x8 come in T2I-Adapter.
    parser.add_argument("--palette_size", type=int, default=8) 
    
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    args = parser.parse_args()

    h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTraining_fmriprep_volume_native.h5"
    tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTraining.tsv"
    img_dir = f"{base_dir}/data/GOD_Dataset/images/training"
    
    save_dir = f"{base_dir}/trained_fmri_decoders/shen_color_palette"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    # 1. Caricamento Dati e Mappatura TSV
    print(f"🧠 Caricamento fMRI S{args.subject} e TSV...")
    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    id_to_filename = {int(row[3]): (str(row[0]) + ".JPEG" if not str(row[0]).endswith('.JPEG') else str(row[0])) for _, row in df_tsv.iterrows()}

    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    stim_key = 'image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id'
    labels = bdata.select(stim_key)[:, 0]

    valid_ids = np.unique(labels[labels > 0])
    
    # --- TRAIN / VAL SPLIT ---
    np.random.seed(42)
    shuffled_ids = np.random.permutation(valid_ids)
    split_idx = int(0.9 * len(shuffled_ids))
    train_ids = set(shuffled_ids[:split_idx])
    val_ids = set(shuffled_ids[split_idx:])

    # 2. Creazione Color Palette (8x8)
    print(f"🎨 Estrazione Color Palette ({args.palette_size}x{args.palette_size}) per T2I-Adapter style...")
    color_targets = {}
    missing = 0
    
    for img_id in tqdm(valid_ids):
        img_id_int = int(img_id)
        if img_id_int in id_to_filename:
            img_path = os.path.join(img_dir, id_to_filename[img_id_int])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                
                # BOX Resampling calcola la media esatta dei pixel nel blocco (perfetto per le palette)
                img_resized = img.resize((args.palette_size, args.palette_size), Image.Resampling.BOX)
                
                # Normalizziamo tra 0 e 1
                img_arr = np.array(img_resized, dtype=np.float32) / 255.0
                
                # Flatten -> array 1D di dimensione 8*8*3 = 192
                color_targets[img_id_int] = torch.tensor(img_arr.flatten()).float()
            else:
                missing += 1

    if missing > 0:
        print(f"⚠️ Attenzione: mancano {missing} immagini.")

    # 3. Costruzione Tensori
    X_tr, Y_tr, X_va, Y_va = [], [], [], []
    for i, lbl in enumerate(labels):
        lbl_int = int(lbl)
        if lbl_int in color_targets:
            if lbl_int in train_ids:
                X_tr.append(X_raw[i])
                Y_tr.append(color_targets[lbl_int].unsqueeze(0))
            elif lbl_int in val_ids:
                X_va.append(X_raw[i])
                Y_va.append(color_targets[lbl_int].unsqueeze(0))

    X_train = torch.tensor(np.array(X_tr), dtype=torch.float32).to(device)
    Y_train = torch.cat(Y_tr).to(device)
    X_val = torch.tensor(np.array(X_va), dtype=torch.float32).to(device)
    Y_val = torch.cat(Y_va).to(device)

    print(f"✅ Split completato: Train {X_train.shape[0]} trial | Val {X_val.shape[0]} trial")

    # 4. Training
    target_dim = args.palette_size * args.palette_size * 3 # 8 * 8 * 3 = 192
    aligner = SimpleLinearAligner(X_train.shape[1], target_dim).to(device)
    
    # Essendo un output di soli 192 valori, L-BFGS con history_size=100 non intaserà minimamente la VRAM
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss()

    l2_lambda, l1_lambda = 1e-2, 1e-4
    best_val_loss = float('inf')

    print(f"🔥 Training Color Palette Aligner (Target dim: {target_dim})...")
    for ep in range(args.epochs):
        aligner.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(aligner(X_train), Y_train)
            reg_loss = 0
            # FIX BIAS: Applichiamo L1/L2 SOLO ai pesi (weight)
            for name, param in aligner.named_parameters():
                if 'bias' not in name:
                    reg_loss += (0.5 * l2_lambda * torch.sum(param ** 2)) + (l1_lambda * torch.sum(torch.abs(param)))
            loss += reg_loss
            loss.backward()
            return loss
        
        train_loss = optimizer.step(closure)
        
        aligner.eval()
        with torch.no_grad():
            val_loss = criterion(aligner(X_val), Y_val)
            sparsity = (aligner.linear.weight.abs() < 1e-4).sum().item() / aligner.linear.weight.numel() * 100

        print(f"Ep {ep+1:02d} | Train MSE: {train_loss.item():.5f} | Val MSE: {val_loss.item():.5f} | Sparsity: {sparsity:.2f}%", end="")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_dir, f"S{args.subject}_color_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

if __name__ == "__main__":
    main()