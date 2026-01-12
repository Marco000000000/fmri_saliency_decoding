import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MultimodalFmriDataset(Dataset):
    def __init__(self, fmri_data, image_paths, dataset_root, split="training", load_maps=True, alexnet_features=None):
        """
        Args:
            fmri_data: Lista/Array di vettori fMRI.
            image_paths: Lista dei percorsi alle immagini originali (JPEG).
            dataset_root: Root del GOD dataset (es. data/GOD_Dataset).
            split: 'training' o 'test' (per trovare le sottocartelle giuste).
            load_maps: Se True, carica anche Depth e Segmentation png.
            alexnet_features: Dizionario opzionale {image_name_no_ext: numpy_vector}.
        """
        self.fmri = fmri_data
        self.image_paths = image_paths
        self.root = dataset_root
        self.split = split
        self.load_maps = load_maps
        self.alexnet_features = alexnet_features
        
        # Trasformazioni base: Resize a 224x224 e conversione in Tensor [0,1]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        # 1. fMRI Data
        fmri_tensor = torch.from_numpy(self.fmri[idx]).float()
        if fmri_tensor.dim() > 1: # Assicuriamoci che sia piatto
             fmri_tensor = fmri_tensor.flatten()

        # 2. Identificazione Immagine
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        key = os.path.splitext(filename)[0] # es: n01518878_5958
        
        sample = {
            "fmri": fmri_tensor,
            "image_key": key,
            "image_path": img_path
        }

        # 3. Caricamento Mappe (Depth & Segmentation)
        if self.load_maps:
            # Percorsi generati dallo script di preprocessing
            depth_path = os.path.join(self.root, "derived_maps", "depth", self.split, key + ".png")#_quantized_visible
            seg_path = os.path.join(self.root, "derived_maps", "segmentation", self.split, key + ".png")
            
            # Depth (Grayscale -> 1 canale)
            if os.path.exists(depth_path):
                depth_img = Image.open(depth_path).convert("L")
                sample["depth"] = self.transform(depth_img) # [1, 224, 224]
            else:
                # Fallback nero se manca
                sample["depth"] = torch.zeros(1, 224, 224)

            # Segmentation (RGB -> 3 canali)
            if os.path.exists(seg_path):
                seg_img = Image.open(seg_path).convert("RGB")
                sample["seg"] = self.transform(seg_img) # [3, 224, 224]
            else:
                sample["seg"] = torch.zeros(3, 224, 224)

        # 4. Caricamento Feature AlexNet (Target per l'allineamento)
        if self.alexnet_features is not None and key in self.alexnet_features:
            target_feat = self.alexnet_features[key]
            sample["alexnet"] = torch.from_numpy(target_feat).float()

        return sample