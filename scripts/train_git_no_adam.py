import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset 

# --- MODELLO LINEARE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.linear(x)

# --- ESTRAZIONE TARGET (ViT Encoder) ---
def extract_vit_targets(loader, caption_model, processor, device):
    all_fmri = []
    all_targets = []
    
    encoder = caption_model.encoder 
    encoder.eval()
    
    print("--- Estrazione Target (Spazio ViT-GPT2) ---")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            all_fmri.append(batch['fmri'])
            
            img_paths = batch['image_path']
            imgs = []
            for p in img_paths:
                try:
                    i = Image.open(p)
                    if i.mode != "RGB": i = i.convert("RGB")
                    imgs.append(i)
                except:
                    imgs.append(Image.new('RGB', (224, 224)))
            
            pixel_values = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
            outputs = encoder(pixel_values)
            
            # Target = [CLS] token (indice 0)
            cls_embedding = outputs.last_hidden_state[:, 0, :] 
            all_targets.append(cls_embedding.cpu())
            
    return torch.cat(all_fmri, dim=0).to(device), torch.cat(all_targets, dim=0).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/caption_aligner_cosine")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training fMRI -> ViT (Cosine Loss) S{args.subject} ---")

    # 1. Setup Teacher
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    processor = ViTImageProcessor.from_pretrained(model_name)
    caption_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    caption_model.eval()

    target_dim = caption_model.config.encoder.hidden_size

    # 2. Dati
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    
    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    
    if raw_data is None: return

    train_dict, val_dict = split_data_for_class(raw_data, 0.9)
    train_loader = DataLoader(MultimodalFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=64, shuffle=False)
    val_loader = DataLoader(MultimodalFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=64, shuffle=False)

    # 3. Estrazione
    X_train, Y_train = extract_vit_targets(train_loader, caption_model, processor, device)
    X_val, Y_val = extract_vit_targets(val_loader, caption_model, processor, device)
    
    del caption_model
    torch.cuda.empty_cache()

    # 4. Optimizer & Loss
    aligner = SimpleLinearAligner(input_dim, output_dim=target_dim).to(device)
    
    # L-BFGS
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")
    
    # LOSS: Cosine Embedding
    criterion_cos = nn.CosineEmbeddingLoss()
    target_ones = torch.ones(X_train.shape[0]).to(device) # Target y=1 (Similarity)
    val_ones = torch.ones(X_val.shape[0]).to(device)

    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)
    
    # Pesi Regolarizzazione
    l2_lambda = 1e-2  # Weight Decay
    l1_lambda = 1e-3  # Sparsità
    norm_lambda = 0.1 # Forza la lunghezza del vettore a essere corretta

    print("Inizio Training Cosine + Norm Constraint...")
    best_val_loss = float('inf')

    for ep in range(args.epochs):
        aligner.train()
        
        def closure():
            optimizer.zero_grad()
            preds = aligner(X_train)
            
            # A. Cosine Loss (Semantica)
            loss_semantic = criterion_cos(preds, Y_train, target_ones)
            
            # B. Norm Loss (Magnitudo)
            # Vogliamo che ||pred|| ≈ ||target||
            # Senza questo, L-BFGS potrebbe trovare vettori piccolissimi o enormi
            pred_norm = preds.norm(p=2, dim=1)
            target_norm = Y_train.norm(p=2, dim=1)
            loss_norm = torch.mean(torch.abs(pred_norm - target_norm))
            
            # C. ElasticNet (Sparsità)
            reg_loss = 0
            epsilon = 1e-6
            for param in aligner.parameters():
                reg_loss += 0.5 * l2_lambda * torch.sum(param ** 2)
                reg_loss += l1_lambda * torch.sum(torch.sqrt(param ** 2 + epsilon))
            
            # Totale
            total_loss = loss_semantic + (norm_lambda * loss_norm) + reg_loss
            
            total_loss.backward()
            return total_loss

        loss = optimizer.step(closure)
        
        # Validation
        aligner.eval()
        with torch.no_grad():
            val_preds = aligner(X_val)
            val_loss = criterion_cos(val_preds, Y_val, val_ones).item()
            
            # Check della norma media (Sanity Check)
            avg_pred_norm = val_preds.norm(dim=1).mean().item()
            avg_tgt_norm = Y_val.norm(dim=1).mean().item()
        
        print(f"Ep {ep+1} | Loss: {loss.item():.4f} | Val Cosine: {val_loss:.4f} | Norm: {avg_pred_norm:.1f}/{avg_tgt_norm:.1f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_caption_aligner.pth"))

if __name__ == "__main__":
    main()