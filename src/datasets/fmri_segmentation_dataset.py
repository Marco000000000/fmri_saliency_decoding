import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
from torchvision import transforms

class FmriSegmentationDataset(Dataset):
    def __init__(self, fmri_data, image_paths, dataset_root, split="training"):
        self.fmri_data = fmri_data
        self.image_paths = image_paths
        self.dataset_root = dataset_root
        self.split = split
        
    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri_data[idx], dtype=torch.float32)
        
        # Path Depth GT
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path).replace(".JPEG", ".png").replace(".jpg", ".png")
        depth_path = os.path.join(self.dataset_root, "derived_maps", "depth_quantized", self.split, fname)
        
        if os.path.exists(depth_path):
            # Carica e Resize
            d_img = Image.open(depth_path).convert("L").resize((224, 224), Image.NEAREST)
            d_arr = np.array(d_img) / 255.0 # Normalizza 0-1
        else:
            # Fallback: Tutto Background
            d_arr = np.zeros((224, 224), dtype=np.float32)

        # --- CREAZIONE TARGET CLASSI (0, 1, 2) ---
        target_mask = np.zeros((224, 224), dtype=np.longlong) # Long per CrossEntropy
        
        # Thresholds
        # 0: BG (< 0.35) -> Già 0 di default
        # 1: MID (>= 0.35 e < 0.65)
        target_mask[(d_arr >= 0.35) & (d_arr < 0.65)] = 1
        # 2: FG (>= 0.65)
        target_mask[d_arr >= 0.65] = 2
        
        return {
            'fmri': fmri,
            'mask_target': torch.from_numpy(target_mask) # [224, 224] con valori 0,1,2
        }