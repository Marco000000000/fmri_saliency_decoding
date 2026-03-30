import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import IPAdapterMaskProcessor

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def get_image_id(image_path):
    basename = os.path.basename(image_path)
    cat_id = int(basename.split('_')[0][1:])
    img_id = int(basename.split('_')[1].split('.')[0])
    return float('%d.%06d' % (cat_id, img_id))

def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6: emb = emb / norm
    emb = emb * 0.95
    return emb

def extract_spatial_pil(depth_flat_tensor, spatial_type, threshold=0.6):
    mask_64 = depth_flat_tensor.view(64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    if spatial_type == "box":
        coords = torch.nonzero(mask_norm > threshold)
        box_mask = np.zeros((1024, 1024), dtype=np.uint8)
        if len(coords) > 5:
            y_min, x_min = torch.min(coords, dim=0)[0]
            y_max, x_max = torch.max(coords, dim=0)[0]
            x1, y1 = int((x_min.item() / 64.0) * 1024), int((y_min.item() / 64.0) * 1024)
            x2, y2 = int((x_max.item() / 64.0) * 1024), int((y_max.item() / 64.0) * 1024)
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            box_mask[y1:y2, x1:x2] = 255
        else:
            box_mask[:, :] = 255
        return Image.fromarray(box_mask)
        
    elif spatial_type == "mask":
        clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm)).unsqueeze(0).unsqueeze(0)
        eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
        binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
        if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
        bin_64_np = (binary_64[0, 0].cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(bin_64_np).resize((1024, 1024), Image.NEAREST)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_final/kamitani_baseline")
    parser.add_argument("--guidance_scale", type=float, default=4.5) 
    parser.add_argument("--spatial", type=str, default="none", choices=["none", "mask", "box"])
    parser.add_argument("--results_file", type=str, default="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/kamitani_method/results/GenericObjectDecoding.pkl")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16 
    print(f"--- Inference KAMITANI Baseline (1-Step Attention) S{args.subject} | {args.rois} | {args.spatial} ---")

    # 1. Carica i risultati Semantici di Kamitani
    with open(args.results_file, 'rb') as f:
        results_df = pickle.load(f)
    sbj_str = f"Subject{args.subject}"
    filtered_df = results_df[(results_df['subject'] == sbj_str) & (results_df['roi'] == args.rois)]
    if filtered_df.empty: return print(f"❌ ERRORE: Nessun risultato Kamitani per {sbj_str} {args.rois}.")
        
    row = filtered_df.iloc[0]
    feature_dict = {lbl: feat for lbl, feat in zip(row['test_label_set_percept'], row['predicted_feature_averaged_percept'])}
    
    # 2. Carica Decoder Strutturale (TUA RETE) se non è 'none'
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    if args.spatial != "none":
        aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
        aligner_struct.load_state_dict(torch.load(f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth", map_location=device))
        aligner_struct.eval()

    # 3. Pipeline SDXL
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0) 
    mask_processor = IPAdapterMaskProcessor()

    # 4. Data Load per mappare i file reali
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(os.path.join(args.dataset_root, "fmri_files", "test"), os.path.join(args.dataset_root, "images", "test"), f"Subject{args.subject}_ImageNetTest.h5", rois_dict)
    
    grouped_fmri = defaultdict(list)
    for i, vec in enumerate(raw_data['fmri']):
        grouped_fmri[os.path.basename(raw_data['image_paths'][i])].append(vec)

    save_dir = os.path.join(args.output_dir, f"S{args.subject}_{args.rois}_{args.spatial}")
    os.makedirs(os.path.join(save_dir, "single"), exist_ok=True)

    # --- INFERENCE LOOP ---
    for fname in tqdm(sorted(list(grouped_fmri.keys()))):
        real_key = os.path.splitext(fname)[0]
        img_id = get_image_id(fname)
        
        if img_id not in feature_dict: continue

        fmri_input = torch.tensor(np.array(grouped_fmri[fname]), dtype=torch.float32).to(device).mean(dim=0, keepdim=True)

        with torch.no_grad():
            # A. Semantica di Kamitani
            emb_sem = torch.tensor(feature_dict[img_id], dtype=torch.float32).unsqueeze(0).to(device)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf).unsqueeze(1)
            ip_adapter_embeds = torch.cat([torch.zeros_like(emb_sem), emb_sem], dim=0)
            
            # B. Spazio (Tua rete)
            ip_mask = None
            if args.spatial != "none":
                depth_flat = aligner_struct(fmri_input)
                sp_pil = extract_spatial_pil(depth_flat, args.spatial)
                ip_mask = mask_processor.preprocess([sp_pil])

        # C. Generazione 1-Step Attention
        kwargs = {"cross_attention_kwargs": {"ip_adapter_masks": ip_mask}} if ip_mask is not None else {}
        img = pipe(prompt="", ip_adapter_image_embeds=[ip_adapter_embeds], num_inference_steps=20, guidance_scale=args.guidance_scale, generator=torch.Generator(device=device).manual_seed(42), **kwargs).images[0]
        
        img.resize((512, 512)).save(os.path.join(save_dir, "single", f"{real_key}.png"))

if __name__ == "__main__":
    main()