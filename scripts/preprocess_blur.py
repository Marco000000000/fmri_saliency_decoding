import os
import argparse
from PIL import Image, ImageFilter
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--blur_radius", type=float, default=10.0, help="Raggio sfocatura")
    args = parser.parse_args()

    for split in ["training", "test"]:
        in_dir = os.path.join(args.dataset_root, "images", split)
        out_dir = os.path.join(args.dataset_root, "derived_maps", "blurred", split)
        os.makedirs(out_dir, exist_ok=True)
        
        files = [f for f in os.listdir(in_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Sfocatura {split} ({len(files)} files)...")
        
        for f in tqdm(files):
            try:
                img = Image.open(os.path.join(in_dir, f)).convert("RGB")
                # Applica sfocatura
                blurred = img.filter(ImageFilter.GaussianBlur(args.blur_radius))
                blurred.save(os.path.join(out_dir, os.path.splitext(f)[0] + ".png"))
            except Exception as e:
                print(f"Err {f}: {e}")

if __name__ == "__main__":
    main()