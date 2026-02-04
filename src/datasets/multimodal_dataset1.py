import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MultimodalFmriDataset(Dataset):
    def __init__(self, fmri_data, image_paths, dataset_root, split="training"):
        """
        Args:
            fmri_data: List/Array of fMRI vectors
            image_paths: List of paths to original images
            dataset_root: Root of GOD dataset
            split: 'training' or 'test'
        """
        self.fmri_data = fmri_data
        self.image_paths = image_paths
        self.dataset_root = dataset_root
        self.split = split
        
        # Percorsi per le mappe derivate
        self.depth_root = os.path.join(dataset_root, "derived_maps", "depth", split)
        self.seg_root = os.path.join(dataset_root, "derived_maps", "segmentation", split)

        # --- TRASFORMAZIONE IMMAGINE RGB (ORACLE TARGET) ---
        # SegFormer e DPT sono pre-addestrati su ImageNet.
        # Richiedono questa specifica normalizzazione.
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)), # Standardizziamo a 512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # --- TRASFORMAZIONE MAPPE (Depth/Seg) ---
        # Depth: Tensor 1ch
        self.depth_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor() # [0, 1]
        ])
        
        # Seg: Tensor (Non normalizziamo qui perché potrebbero servire indici o colori raw)
        # Resize nearest per non interpolare le classi se usiamo indici
        self.seg_transform = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        # 1. Carica fMRI
        fmri_vec = torch.tensor(self.fmri_data[idx]).float()
        
        # 2. Carica Immagine Originale (FIX KEYERROR)
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.image_transform(image)
        except Exception as e:
            print(f"Errore caricamento immagine {img_path}: {e}")
            # Fallback nero in caso di errore
            image_tensor = torch.zeros((3, 512, 512))

        # 3. Carica Depth Map
        # Costruiamo il path presunto
        fname = os.path.basename(img_path)
        name_no_ext = os.path.splitext(fname)[0]
        
        # Cerca file depth (png)
        depth_path = os.path.join(self.depth_root, name_no_ext + ".png")
        if os.path.exists(depth_path):
            depth_img = Image.open(depth_path).convert("L") # Grayscale
            depth_tensor = self.depth_transform(depth_img)
        else:
            # Fallback se manca la mappa
            depth_tensor = torch.zeros((1, 512, 512))

        # 4. Carica Segmentation Map
        seg_path = os.path.join(self.seg_root, name_no_ext + ".png")
        if os.path.exists(seg_path):
            seg_img = Image.open(seg_path).convert("RGB") # RGB (Colori ADE20K)
            # Per ora ritorniamo il tensore RGB, la conversione in indici si può fare se serve
            seg_tensor = transforms.ToTensor()(self.seg_transform(seg_img))
        else:
            seg_tensor = torch.zeros((3, 512, 512))

        return {
            'fmri': fmri_vec,
            'image': image_tensor,  # <--- Questa è la chiave che mancava!
            'depth': depth_tensor,
            'seg': seg_tensor,
            'image_path': img_path
        }