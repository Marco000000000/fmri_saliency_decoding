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
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
from diffusers.image_processor import IPAdapterMaskProcessor
from tqdm import tqdm

# --- MODELLI ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- HELPER FUNCTIONS ---
def get_spatial_mask_pil(depth_flat_tensor, spatial_shape, threshold=0.5, target_size=(1024, 1024)):
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
            y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
            x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
            x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            mask_np[y1:y2, x1:x2] = 255
        else: mask_np[:, :] = 255
        return Image.fromarray(mask_np)

def create_overlay(blank_canvas, mask_pil, spatial_type):
    canvas_rgb = blank_canvas.convert("RGB")
    if "mask" in spatial_type:
        green_layer = Image.new("RGBA", canvas_rgb.size, (0, 255, 0, 130))
        return Image.composite(green_layer, canvas_rgb.convert("RGBA"), mask_pil.convert("L")).convert("RGB")
    else:
        draw = ImageDraw.Draw(canvas_rgb)
        bbox = mask_pil.getbbox()
        if bbox: draw.rectangle(bbox, outline="cyan", width=8)
        return canvas_rgb

def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6: emb = emb / norm
    return emb * 0.95 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--spatial", type=str, default="attn_box", 
                        choices=["none", "inpaint_mask", "inpaint_box", "attn_mask", "attn_box"])
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    
    # 1. Caricamento IMAGERY
    h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_Imagery_fmriprep_volume_native.h5"
    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    
    # Dizionario Categorie (Nomi puliti)
    ids_raw = bdata.select('imagery_image_index')[:, 0]
    names_raw = bdata.get_label('imagery_name')
    unique_ids = np.unique(ids_raw[ids_raw > 0])
    category_dict = {int(uid): names_raw[np.where(ids_raw == uid)[0][0]].split('_', 1)[-1] for uid in unique_ids}

    # Media trial ( SNR boost )
    averaged_fmri, output_ids = [], []
    for uid in unique_ids:
        idx = [i for i in np.where(ids_raw == uid)[0] if i < X_raw.shape[0]]
        if len(idx) > 0:
            averaged_fmri.append(np.mean(X_raw[idx], axis=0))
            output_ids.append(int(uid))
    X_test = torch.tensor(np.array(averaged_fmri), dtype=torch.float32).to(device)

    # 2. Caricamento Decoders
    aligner_sem = SimpleLinearAligner(X_test.shape[1], 1280).to(device)
    aligner_sem.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse/S{args.subject}_clip_aligner.pth", map_location=device))
    aligner_sem.eval()

    aligner_struct = None
    if args.spatial != "none":
        aligner_struct = SimpleLinearAligner(X_test.shape[1], 4096).to(device)
        aligner_struct.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_depth_sparse/S{args.subject}_depth_aligner.pth", map_location=device))
        aligner_struct.eval()

    # 3. Pipelines
    # Step 1: Base Pipeline (Sempre caricata)
    pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf).to(device)
    pipe_base.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_base.set_ip_adapter_scale(1.0)
    
    # Step 2: Img2Img (per attn) o Inpaint (per inpaint)
    pipe_step2 = None
    mask_processor = None
    if args.spatial.startswith("attn"):
        pipe_step2 = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf).to(device)
        mask_processor = IPAdapterMaskProcessor()
    elif args.spatial.startswith("inpaint"):
        pipe_step2 = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtype_inf).to(device)

    if pipe_step2 is not None:
        pipe_step2.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe_step2.set_ip_adapter_scale(1.0)

    out_dir = f"reconstructions/S{args.subject}_imagery_{args.spatial}"
    single_dir = os.path.join(out_dir, "single")
    os.makedirs(single_dir, exist_ok=True)
    
    blank_canvas = Image.new("RGB", (1024, 1024), (128, 128, 128))
    zero_emb = torch.zeros((1, 1, 1280), device=device, dtype=dtype_inf)

    # 4. Generazione
    for fmri_vec, cat_id in tqdm(zip(X_test, output_ids), total=len(output_ids)):
        cat_name = category_dict[cat_id]
        with torch.no_grad():
            emb_sem = normalize_and_cast(aligner_sem(fmri_vec.unsqueeze(0))).to(dtype_inf)
            ip_adapter_embeds = torch.cat([zero_emb, emb_sem.unsqueeze(1)], dim=0)
            depth_flat = aligner_struct(fmri_vec.unsqueeze(0)) if aligner_struct else None

        generator = torch.Generator(device=device).manual_seed(42)
        spatial_type = args.spatial.split("_")[1] if "_" in args.spatial else "none"
        
        # --- STEP 1: COARSE ---
        # Generazione globale (senza maschere) per creare il contesto visivo
        coarse_image = pipe_base(
            prompt="", ip_adapter_image_embeds=[ip_adapter_embeds],
            num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
        ).images[0]
        
        # --- STEP 2: REFINEMENT (INPAINT O ATTN) ---
        if args.spatial == "none":
            final_image = coarse_image
            mask_pil = blank_canvas
        else:
            mask_pil = get_spatial_mask_pil(depth_flat, spatial_type, threshold=args.mask_threshold)
            
            if args.spatial.startswith("attn"):
                # Applichiamo l'attenzione selettiva sull'immagine coarse usando Img2Img
                ip_mask = mask_processor.preprocess([mask_pil], height=1024, width=1024)
                final_image = pipe_step2(
                    prompt="", image=coarse_image,
                    ip_adapter_image_embeds=[ip_adapter_embeds],
                    cross_attention_kwargs={"ip_adapter_masks": ip_mask},
                    strength=0.7, # 0.7 permette di modificare significativamente l'area
                    num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]
            
            elif args.spatial.startswith("inpaint"):
                mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
                final_image = pipe_step2(
                    prompt="", image=coarse_image, mask_image=mask_blurred,
                    ip_adapter_image_embeds=[ip_adapter_embeds],
                    num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]

        # Salvataggio Singolo
        final_image.save(os.path.join(single_dir, f"imagined_{cat_id:02d}_{cat_name}.png"))

        # Plot 4 immagini (Stile Shen originale)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        overlay_img = create_overlay(blank_canvas, mask_pil, args.spatial)
        
        ax[0].imshow(blank_canvas); ax[0].set_title(f"Target (ID {cat_id})"); ax[0].axis("off")
        ax[1].imshow(overlay_img); ax[1].set_title(f"Overlay {args.spatial}"); ax[1].axis("off")
        ax[2].imshow(coarse_image); ax[2].set_title("Step 1: Coarse"); ax[2].axis("off")
        ax[3].imshow(final_image); ax[3].set_title(f"Step 2: {cat_name}"); ax[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comparison_{cat_id:02d}.png"))
        plt.close()

if __name__ == "__main__":
    main()