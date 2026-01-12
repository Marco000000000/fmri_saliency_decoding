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

class AlexNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Usiamo AlexNet standard
        original_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        # Prendiamo fino al primo Linear+ReLU+Dropout (4096 dim)
        # Classifier structure:
        # 0: Dropout, 1: Linear(9216, 4096), 2: ReLU, 3: Dropout, 4: Linear(4096, 4096)
        self.classifier = nn.Sequential(*list(original_model.classifier.children())[:5]) 
    
    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x) # Output: 4096
        print(x.shape)
        return x

def main(args):
    model = AlexNetFeatureExtractor().to(device).eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_root = os.path.join(args.dataset_root, "images")
    out_dir = os.path.join(args.dataset_root, "alexnet_features")
    os.makedirs(out_dir, exist_ok=True)

    # Dizionari per salvare tutto
    features_dict = {} # { "n0123_123": vec }
    class_prototypes = {} # { "n0123": [sum, count] }

    # Processiamo Training e Test insieme per avere prototipi robusti
    all_files = []
    for split in ["training", "test"]:
        d = os.path.join(img_root, split)
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith("JPEG") or f.endswith("png"): 
                    all_files.append(os.path.join(d, f))

    print(f"Estrazione feature da {len(all_files)} immagini...")

    with torch.no_grad():
        for img_path in tqdm(all_files):
            try:
                fname = os.path.basename(img_path)
                key = os.path.splitext(fname)[0] # ID immagine
                class_id = key.split('_')[0]     # ID classe ImageNet
                
                img = Image.open(img_path).convert("RGB")
                tens = transform(img).unsqueeze(0).to(device)
                feat = model(tens).cpu().numpy().flatten()
                
                features_dict[key] = feat
                
                if class_id not in class_prototypes:
                    class_prototypes[class_id] = [feat, 1]
                else:
                    class_prototypes[class_id][0] += feat
                    class_prototypes[class_id][1] += 1
            except Exception as e:
                print(e)

    # Calcolo medie prototipi
    final_prototypes = {k: v[0]/v[1] for k, v in class_prototypes.items()}

    with open(os.path.join(out_dir, "features.pkl"), "wb") as f:
        pickle.dump(features_dict, f)
    with open(os.path.join(out_dir, "prototypes.pkl"), "wb") as f:
        pickle.dump(final_prototypes, f)
    print("Salvataggio completato.")

if __name__ == "__main__":
    main(get_args())