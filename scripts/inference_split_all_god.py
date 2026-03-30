import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.image_processor import IPAdapterMaskProcessor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6: emb = emb / norm
    return emb * 0.95 

def get_spatial_masks(depth_flat_tensor, threshold=0.5, target_size=(1024, 1024)):
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)

    mask_pil = Image.fromarray(bin_64_np * 255).resize(target_size, Image.NEAREST)
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_sdxl_split_all")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16 
    print(f"--- Inference Split-Semantics ALL (GOD S{args.subject}) ---")

    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # 1. Caricamento Decoders
    print("🧠 Caricamento Modelli fMRI (BG, FG, Spaziale)...")
    aligner_bg = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    aligner_bg.load_state_dict(torch.load(f"trained_fmri_decoders/linear_sparse_lbfgs_background/S{args.subject}_{args.rois}/best_linear_aligner.pth", map_location=device))
    aligner_bg.eval()

    aligner_fg = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    aligner_fg.load_state_dict(torch.load(f"trained_fmri_decoders/linear_sparse_lbfgs_foreground/S{args.subject}_{args.rois}/best_linear_aligner.pth", map_location=device))
    aligner_fg.eval()
    
    aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
    aligner_struct.load_state_dict(torch.load(f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth", map_location=device))
    aligner_struct.eval()

    # 2. Caricamento SDXL Pipelines
    print("🎨 Caricamento Pipelines Diffusers...")
    pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf).to(device)
    pipe_base.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_base.set_ip_adapter_scale(1.0) 
    
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtype_inf).to(device)
    pipe_inpaint.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_inpaint.set_ip_adapter_scale(1.0)

    pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe_base)
    mask_processor = IPAdapterMaskProcessor()

    # 3. Data Load
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"), test_h5, rois_dict
    )
    
    grouped = defaultdict(list)
    for i, vec in enumerate(raw_data['fmri']):
        p = raw_data['image_paths'][i]
        fname = os.path.basename(p)
        grouped[fname].append(vec)
    
    unique_files = sorted(list(grouped.keys()))
    
    # Setup Cartelle di output
    base_save_dir = os.path.join(args.output_dir, f"S{args.subject}_{args.rois}")
    subdirs = ["bg_only", "fg_only", "inpaint_mask", "inpaint_box", "attn_mask", "attn_box"]
    for d in subdirs: os.makedirs(os.path.join(base_save_dir, d), exist_ok=True)

    zero_emb = torch.zeros((1, 1, 1280), device=device, dtype=dtype_inf)

    print(f"Generazione in blocco (6 varianti) su {len(unique_files)} immagini...")

    for fname in tqdm(unique_files):
        fmri_input = torch.tensor(np.array(grouped[fname]), dtype=torch.float32).to(device).mean(dim=0, keepdim=True)

        with torch.no_grad():
            emb_bg = normalize_and_cast(aligner_bg(fmri_input)).to(dtype_inf)
            emb_fg = normalize_and_cast(aligner_fg(fmri_input)).to(dtype_inf)
            ip_embeds_bg = torch.cat([zero_emb, emb_bg.unsqueeze(1)], dim=0) 
            ip_embeds_fg = torch.cat([zero_emb, emb_fg.unsqueeze(1)], dim=0) 
            
            depth_flat = aligner_struct(fmri_input)

        generator = torch.Generator(device=device).manual_seed(42)
        mask_pil, box_pil = get_spatial_masks(depth_flat, threshold=args.mask_threshold)
        
        mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
        box_blurred = box_pil.filter(ImageFilter.GaussianBlur(radius=8))
        ip_mask_full = mask_processor.preprocess([mask_pil], height=1024, width=1024)
        ip_mask_box = mask_processor.preprocess([box_pil], height=1024, width=1024)

        # 1. Background Only
        img_bg = pipe_base(prompt="", ip_adapter_image_embeds=[ip_embeds_bg], num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_bg.resize((512, 512)).save(os.path.join(base_save_dir, "bg_only", fname))

        # 2. Foreground Only (Base Model)
        img_fg = pipe_base(prompt="", ip_adapter_image_embeds=[ip_embeds_fg], num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_fg.resize((512, 512)).save(os.path.join(base_save_dir, "fg_only", fname))

        # 3. Inpaint Mask (Sfondo + Soggetto su maschera)
        img_inp_mask = pipe_inpaint(prompt="", image=img_bg, mask_image=mask_blurred, ip_adapter_image_embeds=[ip_embeds_fg], num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_inp_mask.resize((512, 512)).save(os.path.join(base_save_dir, "inpaint_mask", fname))

        # 4. Inpaint Box
        img_inp_box = pipe_inpaint(prompt="", image=img_bg, mask_image=box_blurred, ip_adapter_image_embeds=[ip_embeds_fg], num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_inp_box.resize((512, 512)).save(os.path.join(base_save_dir, "inpaint_box", fname))

        # 5. Attn Mask
        img_attn_mask = pipe_img2img(prompt="", image=img_bg, ip_adapter_image_embeds=[ip_embeds_fg], cross_attention_kwargs={"ip_adapter_masks": ip_mask_full}, strength=1.0, num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_attn_mask.resize((512, 512)).save(os.path.join(base_save_dir, "attn_mask", fname))

        # 6. Attn Box
        img_attn_box = pipe_img2img(prompt="", image=img_bg, ip_adapter_image_embeds=[ip_embeds_fg], cross_attention_kwargs={"ip_adapter_masks": ip_mask_box}, strength=1.0, num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
        img_attn_box.resize((512, 512)).save(os.path.join(base_save_dir, "attn_box", fname))

    print(f"✅ Finito! Tutte le varianti per GOD sono salvate in {base_save_dir}")

if __name__ == "__main__":
    main()