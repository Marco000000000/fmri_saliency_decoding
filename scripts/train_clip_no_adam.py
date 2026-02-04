import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset 

# --- 1. DEFINIZIONE MODELLO LINEARE (Sovrascriviamo tutto qui) ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        # Regressione Lineare Pura: y = Wx + b
        # Nessuna ReLU, nessuna Hidden Layer.
        self.linear = nn.Linear(input_dim, clip_dim) 

    def forward(self, x):
        return self.linear(x)

# --- Funzione helper per caricare tutti i dati in GPU ---
def precompute_embeddings(loader, clip_model, clip_processor, device, desc="Extracting Features"):
    all_fmri = []
    all_targets = []
    clip_model.eval()
    print(f"--- {desc} ---")
    with torch.no_grad():
        for batch in tqdm(loader):
            all_fmri.append(batch['fmri'])
            img_paths = batch['image_path']
            imgs = []
            for p in img_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except:
                    imgs.append(Image.new('RGB', (224, 224)))
            inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
            targets = clip_model(**inputs).image_embeds
            all_targets.append(targets.cpu()) 
            
    return torch.cat(all_fmri, dim=0).to(device), torch.cat(all_targets, dim=0).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--epochs", type=int, default=20) 
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/linear_sparse_lbfgs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training LINEAR SPARSE Aligner (S{args.subject} {args.rois}) [L-BFGS] ---")

    # 1. Setup CLIP (Solo per estrarre target)
    clip_model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    print(f"Caricamento CLIP Vision Model: {clip_model_name}...")
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device)
    clip_model.eval()
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    # 2. Data Preparation
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
    # Batch size alto solo per velocità di caricamento, tanto poi uniamo tutto
    train_loader = DataLoader(MultimodalFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(MultimodalFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=128, shuffle=False, num_workers=4)

    # PRE-COMPUTATION
    X_train, Y_train = precompute_embeddings(train_loader, clip_model, clip_processor, device, "Training Set")
    X_val, Y_val = precompute_embeddings(val_loader, clip_model, clip_processor, device, "Validation Set")
    
    del clip_model
    torch.cuda.empty_cache()

    # 3. Model Setup (USIAMO LA CLASSE LOCALE)
    target_dim = 1280 
    print(f"Creazione Modello Lineare: Input {input_dim} -> Output {target_dim}")
    aligner = SimpleLinearAligner(input_dim, clip_dim=target_dim).to(device)
    
    # --- CONFIGURAZIONE ELASTIC NET ---
    # L2 (Ridge): Stabilità generale
    l2_lambda = 1e-2 
    # L1 (Lasso): Forza i voxel inutili a zero. 
    # Aumenta questo valore se vuoi più sparsità (es. 1e-2), diminuisci se il modello non impara (es. 1e-4).
    l1_lambda = 1e-3 

    # L-BFGS Optimizer (SENZA weight_decay nel costruttore)
    optimizer = optim.LBFGS(aligner.parameters(), 
                            lr=1.0, 
                            max_iter=20, 
                            history_size=100, 
                            line_search_fn="strong_wolfe")

    criterion = nn.CosineEmbeddingLoss() 
    y_target_train = torch.ones(X_train.shape[0]).to(device)
    y_target_val = torch.ones(X_val.shape[0]).to(device)

    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)

    print(f"\nInizio Training... (L1: {l1_lambda} | L2: {l2_lambda})")
    best_val_loss = float('inf')

    for ep in range(args.epochs):
        aligner.train()
        
        # --- CLOSURE PER L-BFGS ---
        def closure():
            optimizer.zero_grad()
            preds = aligner(X_train)
            
            # 1. Loss Principale
            loss = criterion(preds, Y_train, y_target_train)
            
            # 2. ElasticNet Manuale (Smooth L1 + L2)
            reg_loss = 0
            epsilon = 1e-6 # Per evitare divisioni per zero nella derivata
            
            for param in aligner.parameters():
                # L2 Term (Sum of squares)
                l2 = torch.sum(param ** 2)
                # Smooth L1 Term (Approssimazione differenziabile di |w|)
                l1 = torch.sum(torch.sqrt(param ** 2 + epsilon))
                
                reg_loss += (0.5 * l2_lambda * l2) + (l1_lambda * l1)
            
            loss += reg_loss
            loss.backward()
            return loss
        
        # Step optimizer
        train_loss = optimizer.step(closure)
        
        # Validation & Check Sparsità
        aligner.eval()
        with torch.no_grad():
            val_preds = aligner(X_val)
            val_loss = criterion(val_preds, Y_val, y_target_val)
            
            # Calcolo quanti pesi sono effettivamente "spenti" (< 1e-4)
            # Guardiamo solo i pesi (weight), non il bias
            weights = aligner.linear.weight
            zero_count = (weights.abs() < 1e-4).sum().item()
            total_count = weights.numel()
            sparsity = (zero_count / total_count) * 100
        
        print(f"Ep {ep+1:03d} | Train: {train_loss.item():.5f} | Val: {val_loss.item():.5f} | Sparsity: {sparsity:.2f}%", end="")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            # Salviamo
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_linear_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

    print(f"Training finito.")

if __name__ == "__main__":
    main()