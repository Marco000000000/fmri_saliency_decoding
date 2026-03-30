import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import bdpy
from tqdm import tqdm

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=4096):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--epochs", type=int, default=100)
    
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    args = parser.parse_args()

    h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTraining_fmriprep_volume_native.h5"
    tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTraining.tsv"
    
    # QUI DOVREBBERO ESSERCI I .npy GENERATI DA RMBG
    depth_targets_dir = f"{base_dir}/data/precomputed_binary_masks" 
    save_dir = f"{base_dir}/trained_fmri_decoders/shen_depth_sparse"
    
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento e TSV
    print(f"🧠 Caricamento fMRI S{args.subject} e Mappatura Maschere...")
    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    # Mapping ID -> Nome_File.npy
    id_to_filename = {int(row[3]): (str(row[0]) + ".npy") for _, row in df_tsv.iterrows()}

    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    stim_key = 'image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id'
    labels = bdata.select(stim_key)[:, 0]

    valid_ids = np.unique(labels[labels > 0])
    
    # --- TRAIN / VAL SPLIT RIGOROSO ---
    np.random.seed(42)
    shuffled_ids = np.random.permutation(valid_ids)
    split_idx = int(0.9 * len(shuffled_ids))
    train_ids = set(shuffled_ids[:split_idx])
    val_ids = set(shuffled_ids[split_idx:])

    # 2. Caricamento Target (Maschere)
    print("✂️ Caricamento Maschere di Segmentazione (64x64)...")
    depth_targets = {}
    missing = 0
    for img_id in tqdm(valid_ids):
        img_id_int = int(img_id)
        if img_id_int in id_to_filename:
            mask_path = os.path.join(depth_targets_dir, id_to_filename[img_id_int])
            if os.path.exists(mask_path):
                # Carica il numpy array piatto (4096,) e converti a tensor
                target_flat = np.load(mask_path)
                depth_targets[img_id_int] = torch.tensor(target_flat).float()
            else:
                missing += 1

    if missing > 0:
        print(f"⚠️ Attenzione: mancano {missing} maschere. Hai lanciato prepare_depth_targets.py?")

    # 3. Creazione tensori Finali
    X_tr, Y_tr, X_va, Y_va = [], [], [], []
    for i, lbl in enumerate(labels):
        lbl_int = int(lbl)
        if lbl_int in depth_targets:
            if lbl_int in train_ids:
                X_tr.append(X_raw[i])
                Y_tr.append(depth_targets[lbl_int].unsqueeze(0))
            elif lbl_int in val_ids:
                X_va.append(X_raw[i])
                Y_va.append(depth_targets[lbl_int].unsqueeze(0))

    X_train = torch.tensor(np.array(X_tr), dtype=torch.float32).to(device)
    Y_train = torch.cat(Y_tr).to(device)
    X_val = torch.tensor(np.array(X_va), dtype=torch.float32).to(device)
    Y_val = torch.cat(Y_va).to(device)

    print(f"✅ Split completato: Train {X_train.shape[0]} trial | Val {X_val.shape[0]} trial")

    # 4. Training
    target_dim = 64 * 64 
    aligner = SimpleLinearAligner(X_train.shape[1], target_dim).to(device)
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=60, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss()

    l2_lambda, l1_lambda = 1e-4, 1e-5
    best_val_loss = float('inf')

    print(f"🔥 Training Segmentation Aligner (L1: {l1_lambda}, L2: {l2_lambda})...")
    for ep in range(args.epochs):
        aligner.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(aligner(X_train), Y_train)
            reg_loss = 0
            for param in aligner.parameters():
                reg_loss += (0.5 * l2_lambda * torch.sum(param ** 2)) + (l1_lambda * torch.sum(torch.abs(param)))
            loss += reg_loss
            loss.backward()
            return loss
        
        train_loss = optimizer.step(closure)
        
        aligner.eval()
        with torch.no_grad():
            val_loss = criterion(aligner(X_val), Y_val)
            sparsity = (aligner.linear.weight.abs() < 1e-4).sum().item() / aligner.linear.weight.numel() * 100

        print(f"Ep {ep+1:02d} | Train: {train_loss.item():.4f} | Val: {val_loss.item():.4f} | Sparsity: {sparsity:.2f}%", end="")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_dir, f"S{args.subject}_depth_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

if __name__ == "__main__":
    main()