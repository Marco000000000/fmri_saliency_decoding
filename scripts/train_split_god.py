import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset 

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=1280):
        super().__init__()
        self.linear = nn.Linear(input_dim, clip_dim) 
    def forward(self, x): return self.linear(x)

def apply_smart_mask(img_pil, mask_np, bbox, mode):
    """Applica Blur simmetrico e BBox Crop per FG"""
    if mode == "full" or mask_np is None: return img_pil
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(img_pil.size, Image.NEAREST)
    blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=30))
    
    if mode == "foreground":
        fg_isolated = Image.composite(img_pil, blurred_img, mask_pil.convert("L"))
        if bbox: return fg_isolated.crop(bbox).resize(img_pil.size, Image.BICUBIC)
        return fg_isolated
    elif mode == "background":
        inv_mask = ImageOps.invert(mask_pil.convert("L"))
        return Image.composite(img_pil, blurred_img, inv_mask)

def precompute_embeddings_masked(loader, clip_model, clip_processor, device, masks_dir, mask_mode, desc="Extracting Features"):
    all_fmri, all_targets = [], []
    clip_model.eval()
    print(f"--- {desc} ({mask_mode.upper()}) ---")
    
    missing_masks = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            all_fmri.append(batch['fmri'])
            img_paths = batch['image_path']
            imgs = []
            
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    basename = os.path.basename(p)
                    mask_name = os.path.splitext(basename)[0] + ".npy"
                    mask_path = os.path.join(masks_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        mask_flat = np.load(mask_path).reshape(64, 64)
                        mask_pil_64 = Image.fromarray((mask_flat * 255).astype(np.uint8))
                        bbox_64 = mask_pil_64.getbbox()
                        bbox = None
                        if bbox_64:
                            scale_x, scale_y = img.size[0] / 64.0, img.size[1] / 64.0
                            bbox = (int(bbox_64[0]*scale_x), int(bbox_64[1]*scale_y), int(bbox_64[2]*scale_x), int(bbox_64[3]*scale_y))
                        
                        imgs.append(apply_smart_mask(img, mask_flat, bbox, mask_mode))
                    else:
                        missing_masks += 1
                        imgs.append(img) 
                except Exception as e:
                    imgs.append(Image.new('RGB', (224, 224)))
                    
            inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
            targets = clip_model(**inputs).image_embeds
            all_targets.append(targets.cpu()) 
            
    if missing_masks > 0: print(f"⚠️ Attenzione: mancano {missing_masks} maschere in {masks_dir}.")
    return torch.cat(all_fmri, dim=0).to(device), torch.cat(all_targets, dim=0).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--masks_dir", type=str, default="data/precomputed_binary_masks")
    parser.add_argument("--mask_mode", type=str, required=True, choices=["foreground", "background"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training LINEAR SPARSE S{args.subject} {args.rois} | {args.mask_mode.upper()} ---")

    clip_model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

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

    X_train, Y_train = precompute_embeddings_masked(train_loader, clip_model, clip_processor, device, args.masks_dir, args.mask_mode, "Training Set")
    X_val, Y_val = precompute_embeddings_masked(val_loader, clip_model, clip_processor, device, args.masks_dir, args.mask_mode, "Validation Set")
    
    del clip_model
    torch.cuda.empty_cache()

    aligner = SimpleLinearAligner(input_dim, clip_dim=1280).to(device)
    l2_lambda, l1_lambda = 1e-2, 1e-3 
    optimizer = optim.LBFGS(aligner.parameters(), lr=1.0, max_iter=20, history_size=100, line_search_fn="strong_wolfe")
    criterion = nn.MSELoss()

    save_path = f"trained_fmri_decoders/linear_sparse_lbfgs_{args.mask_mode}/S{args.subject}_{args.rois}"
    os.makedirs(save_path, exist_ok=True)

    print(f"\nInizio Training... (L1: {l1_lambda} | L2: {l2_lambda})")
    best_val_loss = float('inf')

    for ep in range(args.epochs):
        aligner.train()
        def closure():
            optimizer.zero_grad()
            loss = criterion(aligner(X_train), Y_train)
            reg_loss = 0
            for param in aligner.parameters(): reg_loss += (0.5 * l2_lambda * torch.sum(param ** 2)) + (l1_lambda * torch.sum(torch.sqrt(param ** 2 + 1e-6)))
            loss += reg_loss
            loss.backward()
            return loss
        
        train_loss = optimizer.step(closure)
        aligner.eval()
        with torch.no_grad():
            val_preds = aligner(X_val)
            val_loss = criterion(val_preds, Y_val)
            sparsity = ((aligner.linear.weight.abs() < 1e-4).sum().item() / aligner.linear.weight.numel()) * 100
        
        print(f"Ep {ep+1:03d} | Train: {train_loss.item():.5f} | Val: {val_loss.item():.5f} | Sparsity: {sparsity:.2f}%", end="")
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_linear_aligner.pth"))
            print(" [SAVED *]")
        else: print("")

if __name__ == "__main__": main()