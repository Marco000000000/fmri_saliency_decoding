import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

def apply_smart_mask(img_pil, mask_np, bbox, mode):
    """Applica Blur simmetrico e BBox Crop per FG"""
    if mode == "full" or mask_np is None:
        return img_pil
        
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(img_pil.size, Image.NEAREST)
    blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=30))
    
    if mode == "foreground":
        # Incolla il soggetto nitido sullo sfondo sfocato
        fg_isolated = Image.composite(img_pil, blurred_img, mask_pil.convert("L"))
        # Ritaglia la Bounding Box per zoomare
        if bbox:
            return fg_isolated.crop(bbox).resize(img_pil.size, Image.BICUBIC)
        return fg_isolated
        
    elif mode == "background":
        # Incolla lo sfondo nitido sul soggetto sfocato
        inv_mask = ImageOps.invert(mask_pil.convert("L"))
        return Image.composite(img_pil, blurred_img, inv_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["god", "shen"], default="god")
    parser.add_argument("--test_type", choices=["natural", "shapes", "letters"], default="natural")
    parser.add_argument("--num_images", type=int, default=20, help="Quante immagini visualizzare")
    args = parser.parse_args()

    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    masks_dir = f"{base_dir}/data/precomputed_binary_masks"
    out_dir = f"mask_visualizations_{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    
    if args.dataset == "shen":
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
        gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"
    else: 
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv" 
        gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"

    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    id_to_filename, id_to_maskname = {}, {}
    for _, row in df_tsv.iterrows():
        img_id, fname = int(row[3]), str(row[0]).strip()
        if args.dataset == "god":
            key = f"{fname}{ext}" if not fname.endswith(ext) else fname
            id_to_filename[key] = key
            id_to_maskname[key] = f"{fname}.npy"
        else:
            key = f"id_{img_id}.png"
            id_to_filename[key] = f"{fname}{ext}" if not fname.endswith(ext) else fname
            id_to_maskname[key] = f"{fname}.npy"

    files_to_process = list(id_to_filename.keys())[:args.num_images]
    print(f"🎨 Generazione Plot per {len(files_to_process)} immagini di Test ({args.dataset.upper()})...")

    for f in tqdm(files_to_process):
        gt_path = os.path.join(gt_dir, id_to_filename[f])
        if not os.path.exists(gt_path): continue
            
        orig_pil = Image.open(gt_path).convert("RGB").resize((512, 512))
        mask_path = os.path.join(masks_dir, id_to_maskname[f])
        mask_np, bbox = None, None
        
        if os.path.exists(mask_path): 
            mask_np = np.load(mask_path).reshape(64, 64)
            mask_pil_64 = Image.fromarray((mask_np * 255).astype(np.uint8))
            bbox_64 = mask_pil_64.getbbox()
            if bbox_64: 
                bbox = (bbox_64[0]*8, bbox_64[1]*8, bbox_64[2]*8, bbox_64[3]*8)

        if mask_np is None: continue

        fg_pil = apply_smart_mask(orig_pil, mask_np, bbox, "foreground")
        bg_pil = apply_smart_mask(orig_pil, mask_np, bbox, "background")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_pil)
        axes[0].set_title(f"Originale\n({id_to_filename[f]})", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(fg_pil)
        axes[1].set_title(f"Foreground\n(Sfondo sfocato + BBox Crop)", fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(bg_pil)
        axes[2].set_title(f"Background\n(Soggetto sfocato)", fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"plot_{os.path.splitext(id_to_filename[f])[0]}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"✅ Finito! Controlla i plot nella cartella: {out_dir}")

if __name__ == "__main__":
    main()