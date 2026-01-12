import os
import argparse
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    return parser.parse_args()

class AlexNetSpatial(nn.Module):
    def __init__(self):
        super().__init__()
        # Carica AlexNet originale
        original_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        # Prende tutto tranne gli ultimi 2 blocchi (AvgPool e Classifier)
        # Output atteso: [Batch, 256, 6, 6] -> Flatten -> 9216
        self.features = nn.Sequential(*list(original_model.children())[:-2])
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1) # Flatten (9216)

def main(args):
    model = AlexNetSpatial().to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_root = os.path.join(args.dataset_root, "images")
    out_dir = os.path.join(args.dataset_root, "alexnet_features_spatial")
    os.makedirs(out_dir, exist_ok=True)

    features_dict = {}
    
    # Raccogli tutte le immagini (train e test)
    all_files = []
    for split in ["training", "test"]:
        d = os.path.join(img_root, split)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith("JPEG") or f.endswith("png") or f.endswith("jpg"): 
                    all_files.append(os.path.join(d, f))

    print(f"Estrazione feature spaziali (9216 dim) da {len(all_files)} immagini...")

    with torch.no_grad():
        for img_path in tqdm(all_files):
            try:
                fname = os.path.basename(img_path)
                key = os.path.splitext(fname)[0]
                
                img = Image.open(img_path).convert("RGB")
                tens = transform(img).unsqueeze(0).to(device)
                
                # Estrai feature (9216)
                feat = model(tens).cpu().numpy().flatten()
                features_dict[key] = feat
                
            except Exception as e:
                print(f"Errore {fname}: {e}")

    out_path = os.path.join(out_dir, "features.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(features_dict, f)
    print(f"Salvataggio completato in {out_path}")

if __name__ == "__main__":
    main(get_args())