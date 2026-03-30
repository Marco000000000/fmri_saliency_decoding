import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bdpy
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from diffusers import KandinskyV22Pipeline, KandinskyV22InpaintPipeline
from tqdm import tqdm

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def get_spatial_mask_pil(depth_flat_tensor, spatial_shape, threshold=0.5, target_size=(768, 768)):
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)

    if spatial_shape == "mask":
        return Image.fromarray(bin_64_np * 255).resize(target_size, Image.NEAREST)
    elif spatial_shape == "box":
        coords = np.argwhere(bin_64_np > 0)
        mask_np = np.zeros(target_size[::-1], dtype=np.uint8)
        if len(coords) > 5:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
            x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            mask_np[y1:y2, x1:x2] = 255
        else:
            mask_np[:, :] = 255
        return Image.fromarray(mask_np)

def create_overlay(original_img, mask_pil, spatial_type):
    if "mask" in spatial_type:
        orig_rgba = original_img.resize(mask_pil.size).convert("RGBA")
        green_layer = Image.new("RGBA", orig_rgba.size, (0, 255, 0, 130))
        return Image.composite(green_layer, orig_rgba, mask_pil.convert("L")).convert("RGB")
    else:
        orig_img_resized = original_img.resize(mask_pil.size).convert("RGB")
        bbox = mask_pil.getbbox()
        if bbox:
            ImageDraw.Draw(orig_img_resized).rectangle(bbox, outline="cyan", width=6)
        return orig_img_resized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", choices=["natural", "shapes", "letters"], default="natural")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--spatial", type=str, default="none", choices=["none", "inpaint_mask", "inpaint_box"])
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=1.5) # Kandinsky necessita scale più basse
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    
    if args.test_type == "natural":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTest_fmriprep_volume_native.h5"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
        img_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"
    elif args.test_type == "shapes":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ArtificialShapes_fmriprep_volume_native.h5"
        img_dir = f"{base_dir}/data/Shen2019/stimuli/ArtificialShapes/ArtificialImage"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_ArtificialImage.tsv"
        ext = ".tiff"
    else: # letters
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_perceptionLetterImage_original_VC.h5"
        img_dir = f"{base_dir}/data/Shen2019/stimuli/Letters/LetterImage"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_LetterImage.tsv"
        ext = ".tif"

    print(f"🧠 Caricamento Dati TEST Kandinsky S{args.subject} | {args.spatial}")
    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    labels = bdata.select('image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id')[:, 0] 
    
    averaged_fmri, output_ids = [], []
    for lbl in np.unique(labels[labels > 0]):
        idx = [i for i in np.where(labels == lbl)[0] if i < X_raw.shape[0]]
        if len(idx) > 0:
            averaged_fmri.append(np.mean(X_raw[idx], axis=0))
            output_ids.append(int(lbl))
    X_test = torch.tensor(np.array(averaged_fmri), dtype=torch.float32).to(device)

    train_dim = X_test.shape[1]
    aligner_sem = SimpleLinearAligner(train_dim, output_dim=1280).to(device) # Usa l'aligner CLIP
    aligner_sem.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse/S{args.subject}_clip_aligner.pth", map_location=device))
    aligner_sem.eval()

    if args.spatial != "none":
        aligner_struct = SimpleLinearAligner(train_dim, output_dim=4096).to(device)
        aligner_struct.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_depth_sparse/S{args.subject}_depth_aligner.pth", map_location=device))
        aligner_struct.eval()

    print("🎨 Inizializzazione Kandinsky Pipelines...")
    pipe_base = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", torch_dtype=dtype_inf).to(device)
    
    pipe_inpaint = None
    if args.spatial != "none":
        pipe_inpaint = KandinskyV22InpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=dtype_inf).to(device)

    out_dir = f"reconstructions/kandinsky_S{args.subject}_{args.test_type}_{args.spatial}"
    single_dir = os.path.join(out_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    id_to_filename = {}
    if os.path.exists(tsv_path):
        df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
        id_to_filename = {int(row[3]): (str(row[0]).strip() + ext if not str(row[0]).strip().endswith(ext) else str(row[0]).strip()) for _, row in df_tsv.iterrows()}

    for fmri_vec, img_id in tqdm(zip(X_test, output_ids), total=len(output_ids)):
        with torch.no_grad():
            emb_sem = aligner_sem(fmri_vec.unsqueeze(0))
            
            # 1. NORMALIZZAZIONE ESATTA (Come nel vecchio script, SENZA * 0.95)
            norm = emb_sem.norm(p=2, dim=-1, keepdim=True)
            if norm.item() > 1e-6: 
                emb_sem = emb_sem / norm
            emb_sem = emb_sem.to(dtype_inf)
            
            neg_emb = torch.zeros_like(emb_sem)
            depth_flat = aligner_struct(fmri_vec.unsqueeze(0)) if args.spatial != "none" else None

        generator = torch.Generator(device=device).manual_seed(42)
        
        if args.spatial == "none":
            final_image = pipe_base(
                image_embeds=emb_sem, negative_image_embeds=neg_emb,
                height=768, width=768, num_inference_steps=50, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
        else:
            spatial_shape = args.spatial.split("_")[1]
            mask_pil = get_spatial_mask_pil(depth_flat, spatial_shape, threshold=args.mask_threshold, target_size=(768, 768))
            
            coarse_image = pipe_base(
                image_embeds=emb_sem, negative_image_embeds=neg_emb,
                height=768, width=768, num_inference_steps=50, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            
            mask_pil_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=10))
            final_image = pipe_inpaint(
                image=coarse_image, mask_image=mask_pil_blurred,
                image_embeds=emb_sem, negative_image_embeds=neg_emb,
                height=768, width=768, num_inference_steps=50, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]

        final_image.save(os.path.join(single_dir, f"id_{img_id}.png"))

        # Plotting
        try:
            orig_fname = id_to_filename.get(img_id)
            if orig_fname and os.path.exists(os.path.join(img_dir, orig_fname)):
                orig_img = Image.open(os.path.join(img_dir, orig_fname)).convert("RGB")
                
                if args.spatial == "none":
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    ax1.imshow(orig_img); ax1.set_title("Originale"); ax1.axis("off")
                    ax2.imshow(final_image); ax2.set_title("Kandinsky (None)"); ax2.axis("off")
                else:
                    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                    overlay_img = create_overlay(orig_img, mask_pil, args.spatial)
                    ax[0].imshow(orig_img); ax[0].set_title("Originale"); ax[0].axis("off")
                    ax[1].imshow(overlay_img); ax[1].set_title(f"Overlay ({args.spatial})"); ax[1].axis("off")
                    ax[2].imshow(coarse_image); ax[2].set_title("Coarse"); ax[2].axis("off")
                    ax[3].imshow(final_image); ax[3].set_title("Fine (Inpaint)"); ax[3].axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"comparison_{img_id}.png"))
                plt.close()
        except: pass

if __name__ == "__main__":
    main()