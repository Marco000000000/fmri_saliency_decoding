import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import bdpy
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from tqdm import tqdm
from PIL import Image

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, clip_dim) 
    def forward(self, x): return self.linear(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--epochs", type=int, default=50)
    
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    args = parser.parse_args()

    h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTraining_fmriprep_volume_native.h5"
    tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTraining.tsv"
    img_dir = f"{base_dir}/data/GOD_Dataset/images/training"
    save_dir = f"{base_dir}/trained_fmri_decoders/shen_clip_sparse"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Caricamento e TSV
    print(f"🧠 Caricamento fMRI S{args.subject} e TSV...")
    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    id_to_filename = {int(row[3]): (str(row[0]) + ".JPEG" if not str(row[0]).endswith('.JPEG') else str(row[0])) for _, row in df_tsv.iterrows()}

    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    stim_key = 'image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id'
    labels = bdata.select(stim_key)[:, 0] # Evita flatten() errato

    valid_ids = np.unique(labels[labels > 0])
    
    # --- TRAIN / VAL SPLIT RIGOROSO (Per ID Immagine) ---
    np.random.seed(42)
    shuffled_ids = np.random.permutation(valid_ids)
    split_idx = int(0.9 * len(shuffled_ids))
    train_ids = set(shuffled_ids[:split_idx])
    val_ids = set(shuffled_ids[split_idx:])

    # 2. Estrazione CLIP
    print("🎨 Estrazione CLIP...")
    clip_model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    clip_model.eval()
    
    clip_embeddings = {}
    for img_id in tqdm(valid_ids):
        img_id_int = int(img_id)
        if img_id_int in id_to_filename:
            img_path = os.path.join(img_dir, id_to_filename[img_id_int])
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                inputs = clip_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    clip_embeddings[img_id_int] = clip_model(**inputs).image_embeds.cpu()

    del clip_model
    torch.cuda.empty_cache()

    # 3. Creazione tensori Finali
    X_tr, Y_tr, X_va, Y_va = [], [], [], []
    for i, lbl in enumerate(labels):
        lbl_int = int(lbl)
        if lbl_int in clip_embeddings:
            if lbl_int in train_ids:
                X_tr.append(X_raw[i])
                Y_tr.append(clip_embeddings[lbl_int])
            elif lbl_int in val_ids:
                X_va.append(X_raw[i])
                Y_va.append(clip_embeddings[lbl_int])

    X_train = torch.tensor(np.array(X_tr), dtype=torch.float32).to(device)
    Y_train = torch.cat(Y_tr).to(device)
    X_val = torch.tensor(np.array(X_va), dtype=torch.float32).to(device)
    Y_val = torch.cat(Y_va).to(device)

    print(f"✅ Split completato: Train {X_train.shape[0]} trial | Val {X_val.shape[0]} trial")

    # 4. Training (ElasticNet + L-BFGS)
    aligner = SimpleLinearAligner(X_train.shape[1], 1280).to(device)
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss()

    l2_lambda, l1_lambda = 1e-2, 1e-3
    best_val_loss = float('inf')

    print(f"🔥 Training CLIP Aligner (L1: {l1_lambda}, L2: {l2_lambda})...")
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

        print(f"Ep {ep+1:02d} | Train: {train_loss.item():.4f} | Val: {val_loss.item():.4f} | Sparsity: {sparsity:.2f}%", end="")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_dir, f"S{args.subject}_clip_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

if __name__ == "__main__":
    main()