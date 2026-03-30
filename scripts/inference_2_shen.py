import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bdpy
import pandas as pd
from PIL import Image, ImageFilter
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.image_processor import IPAdapterMaskProcessor
from tqdm import tqdm

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def get_spatial_masks(depth_flat_tensor, threshold=0.5, target_size=(1024, 1024)):
    """Restituisce sia la Maschera intera che il Bounding Box."""
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)

    # 1. Full Mask
    mask_pil = Image.fromarray(bin_64_np * 255).resize(target_size, Image.NEAREST)
    
    # 2. Box Mask
    coords = np.argwhere(bin_64_np > 0)
    box_np = np.zeros(target_size[::-1], dtype=np.uint8) 
    if len(coords) > 5:
        y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
        x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
        x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
        if x2 <= x1: x2 = x1 + 10
        if y2 <= y1: y2 = y1 + 10
        box_np[y1:y2, x1:x2] = 255
    else: box_np[:, :] = 255
    box_pil = Image.fromarray(box_np)

    return mask_pil, box_pil

def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6: emb = emb / norm
    return emb * 0.95 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", choices=["natural", "shapes", "letters", "imagery"], default="natural")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    
    # 1. Configurazione Data
    if args.test_type == "natural":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTest_fmriprep_volume_native.h5"
    elif args.test_type == "shapes":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ArtificialShapes_fmriprep_volume_native.h5"
    elif args.test_type == "letters":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_perceptionLetterImage_original_VC.h5"
    else:
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_Imagery_fmriprep_volume_native.h5"

    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    
    # Gestione label
    if args.test_type == "imagery":
        stim_key = 'imagery_image_index'
        labels = bdata.select(stim_key)[:, 0]
        names_raw = bdata.get_label('imagery_name')
        unique_ids = np.unique(labels[labels > 0])
        dict_names = {int(uid): names_raw[np.where(labels == uid)[0][0]].split('_', 1)[-1] for uid in unique_ids}
    else:
        stim_key = 'image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id'
        labels = bdata.select(stim_key)[:, 0] 
        unique_ids = np.unique(labels[labels > 0])
        dict_names = {int(uid): f"id_{int(uid)}" for uid in unique_ids}

    averaged_fmri, output_ids = [], []
    for lbl in unique_ids:
        idx = [i for i in np.where(labels == lbl)[0] if i < X_raw.shape[0]]
        if len(idx) > 0:
            averaged_fmri.append(np.mean(X_raw[idx], axis=0))
            output_ids.append(int(lbl))
            
    X_test = torch.tensor(np.array(averaged_fmri), dtype=torch.float32).to(device)
    train_dim = X_test.shape[1]

    # 2. Caricamento dei 3 Decoders
    aligner_bg = SimpleLinearAligner(train_dim, 1280).to(device)
    aligner_bg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse_background/S{args.subject}_clip_aligner_ba.pth", map_location=device))
    aligner_bg.eval()

    aligner_fg = SimpleLinearAligner(train_dim, 1280).to(device)
    aligner_fg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse_foreground/S{args.subject}_clip_aligner_fo.pth", map_location=device))
    aligner_fg.eval()

    aligner_struct = SimpleLinearAligner(train_dim, 4096).to(device)
    aligner_struct.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_depth_sparse/S{args.subject}_depth_aligner.pth", map_location=device))
    aligner_struct.eval()

    # 3. Caricamento Pipelines SDXL
    print("🎨 Caricamento Pipelines Diffusers (Base, Inpaint, Img2Img)...")
    pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf).to(device)
    pipe_base.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_base.set_ip_adapter_scale(1.0)
    
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtype_inf).to(device)
    pipe_inpaint.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_inpaint.set_ip_adapter_scale(1.0)

    pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe_base)
    mask_processor = IPAdapterMaskProcessor()

    # Creazione delle 6 cartelle di output
    out_dir = f"reconstructions/S{args.subject}_{args.test_type}_ALL_VERSIONS"
    subdirs = ["bg_only", "fg_only", "inpaint_mask", "inpaint_box", "attn_mask", "attn_box"]
    for d in subdirs: os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    
    zero_emb = torch.zeros((1, 1, 1280), device=device, dtype=dtype_inf)

    # 4. Generazione Massiva
    print(f"🚀 Inizio inferenza su {len(output_ids)} immagini. Genero 6 varianti per ognuna...")

    for fmri_vec, cat_id in tqdm(zip(X_test, output_ids), total=len(output_ids)):
        cat_name = dict_names[cat_id]
        file_name = f"{cat_name}.png"
        
        with torch.no_grad():
            emb_bg = normalize_and_cast(aligner_bg(fmri_vec.unsqueeze(0))).to(dtype_inf)
            emb_fg = normalize_and_cast(aligner_fg(fmri_vec.unsqueeze(0))).to(dtype_inf)
            ip_embeds_bg = torch.cat([zero_emb, emb_bg.unsqueeze(1)], dim=0) 
            ip_embeds_fg = torch.cat([zero_emb, emb_fg.unsqueeze(1)], dim=0) 
            
            depth_flat = aligner_struct(fmri_vec.unsqueeze(0))

        generator = torch.Generator(device=device).manual_seed(42)
        mask_pil, box_pil = get_spatial_masks(depth_flat, threshold=args.mask_threshold)
        
        mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
        box_blurred = box_pil.filter(ImageFilter.GaussianBlur(radius=8))
        ip_mask_full = mask_processor.preprocess([mask_pil], height=1024, width=1024)
        ip_mask_box = mask_processor.preprocess([box_pil], height=1024, width=1024)

        # 1. Background Only (Funge anche da Coarse Image per tutti)
        img_bg = pipe_base(prompt="", ip_adapter_image_embeds=[ip_embeds_bg], num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_bg.save(os.path.join(out_dir, "bg_only", file_name))

        # 2. Foreground Only (Per pura comparazione metrica)
        img_fg = pipe_base(prompt="", ip_adapter_image_embeds=[ip_embeds_fg], num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_fg.save(os.path.join(out_dir, "fg_only", file_name))

        # 3. Inpaint Mask
        img_inp_mask = pipe_inpaint(prompt="", image=img_bg, mask_image=mask_blurred, ip_adapter_image_embeds=[ip_embeds_fg], num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_inp_mask.save(os.path.join(out_dir, "inpaint_mask", file_name))

        # 4. Inpaint Box
        img_inp_box = pipe_inpaint(prompt="", image=img_bg, mask_image=box_blurred, ip_adapter_image_embeds=[ip_embeds_fg], num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_inp_box.save(os.path.join(out_dir, "inpaint_box", file_name))

        # 5. Attn Mask
        img_attn_mask = pipe_img2img(prompt="", image=img_bg, ip_adapter_image_embeds=[ip_embeds_fg], cross_attention_kwargs={"ip_adapter_masks": ip_mask_full}, strength=1.0, num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_attn_mask.save(os.path.join(out_dir, "attn_mask", file_name))

        # 6. Attn Box
        img_attn_box = pipe_img2img(prompt="", image=img_bg, ip_adapter_image_embeds=[ip_embeds_fg], cross_attention_kwargs={"ip_adapter_masks": ip_mask_box}, strength=1.0, num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_attn_box.save(os.path.join(out_dir, "attn_box", file_name))

    print(f"✅ Completato! Tutte le 6 varianti salvate in: {out_dir}")

if __name__ == "__main__": main()