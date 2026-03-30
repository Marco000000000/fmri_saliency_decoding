import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset 

# --- 1. DEFINIZIONE MODELLO LINEARE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        # Regressione Lineare Pura: y = Wx + b
        self.linear = nn.Linear(input_dim, clip_dim) 

    def forward(self, x):
        return self.linear(x)

# --- Funzione helper per pre-calcolare embeddings con IP-Adapter ---
def precompute_embeddings(loader, pipe, device, desc="Extracting Features"):
    all_fmri = []
    all_targets = []
    print(f"--- {desc} ---")
    with torch.no_grad():
        for batch in tqdm(loader):
            all_fmri.append(batch['fmri'])
            img_paths = batch['image_path']
            
            batch_targets = []
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except:
                    img = Image.new('RGB', (224, 224))
                
                # Usa lo stesso estrattore dello script video
                emb = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=img, 
                    ip_adapter_image_embeds=None, 
                    device=device, 
                    num_images_per_prompt=1, 
                    do_classifier_free_guidance=False
                )
                # emb è [1, 1, 1280]. Riduciamo a [1280] per il target del regressore lineare
                batch_targets.append(emb[0].squeeze(0).squeeze(0).cpu()) 
            
            all_targets.append(torch.stack(batch_targets))
            
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
    print(f"--- Training LINEAR SPARSE Aligner (S{args.subject} {args.rois}) [L-BFGS / SDXL IP-Adapter] ---")

    # 1. Setup SDXL & IP-Adapter (Solo per estrarre target)
    print("Caricamento SDXL e IP-Adapter per estrazione ground truth...")
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

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
    train_loader = DataLoader(MultimodalFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=128, shuffle=False, num_workers=4)
    val_loader = DataLoader(MultimodalFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=128, shuffle=False, num_workers=4)

    # PRE-COMPUTATION
    X_train, Y_train = precompute_embeddings(train_loader, pipe, device, "Training Set")
    X_val, Y_val = precompute_embeddings(val_loader, pipe, device, "Validation Set")
    
    # Libera VRAM in modo massiccio prima del training
    del pipe
    del controlnet
    torch.cuda.empty_cache()

    # Cast a float32 necessario per la stabilità numerica del L-BFGS
    X_train, Y_train = X_train.to(torch.float32), Y_train.to(torch.float32)
    X_val, Y_val = X_val.to(torch.float32), Y_val.to(torch.float32)

    # 3. Model Setup 
    target_dim = 1280 
    print(f"Creazione Modello Lineare: Input {input_dim} -> Output {target_dim}")
    aligner = SimpleLinearAligner(input_dim, clip_dim=target_dim).to(device)
    
    # --- CONFIGURAZIONE ELASTIC NET ---
    l2_lambda = 1e-2 
    l1_lambda = 1e-3 

    # L-BFGS Optimizer 
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
        
        def closure():
            optimizer.zero_grad()
            preds = aligner(X_train)
            loss = criterion(preds, Y_train, y_target_train)
            
            reg_loss = 0
            epsilon = 1e-6 
            for param in aligner.parameters():
                l2 = torch.sum(param ** 2)
                l1 = torch.sum(torch.sqrt(param ** 2 + epsilon))
                reg_loss += (0.5 * l2_lambda * l2) + (l1_lambda * l1)
            
            loss += reg_loss
            loss.backward()
            return loss
        
        train_loss = optimizer.step(closure)
        
        aligner.eval()
        with torch.no_grad():
            val_preds = aligner(X_val)
            val_loss = criterion(val_preds, Y_val, y_target_val)
            
            weights = aligner.linear.weight
            zero_count = (weights.abs() < 1e-4).sum().item()
            total_count = weights.numel()
            sparsity = (zero_count / total_count) * 100
        
        print(f"Ep {ep+1:03d} | Train: {train_loss.item():.5f} | Val: {val_loss.item():.5f} | Sparsity: {sparsity:.2f}%", end="")

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_linear_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

    print(f"Training finito.")

if __name__ == "__main__":
    main()