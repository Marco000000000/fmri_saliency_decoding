import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UniPCMultistepScheduler

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.latent_projector import LinearLatentProjector
from src.models.multimodal_decoders import ParallelAlexNetAligner 

# --- CLASSI TEST IMAGENET (GOD Dataset) ---
TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]

def draw_text(img, text, pos):
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except: font = ImageFont.load_default()
    bbox = draw.textbbox(pos, text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill="black")
    draw.text(pos, text, fill="white", font=font)

def main():
    parser = argparse.ArgumentParser(description="Inference: Linear Latent Injection -> VAE -> ControlNet")
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--models_dir", type=str, default="trained_latent_models", help="Dove sono i proiettori lineari")
    parser.add_argument("--aligner_dir", type=str, default="trained_fmri_decoders/multimodal", help="Dove è l'aligner AlexNet")
    parser.add_argument("--output_dir", type=str, default="results_latent_injection")
    parser.add_argument("--scale_depth", type=float, default=0.6)
    parser.add_argument("--scale_seg", type=float, default=0.6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference Latent Injection S{args.subject} ---")

    # 1. SETUP DATI E MODELLI
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # A. VAE (Oracle Decoder) - FIX: Caricato in Float16
    print("Loading VAE (Float16)...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)
    
    # B. Proiettori Lineari (fMRI -> Latent)
    print("Loading Linear Projectors...")
    proj_depth = LinearLatentProjector(input_dim).to(device)
    proj_depth.load_state_dict(torch.load(os.path.join(args.models_dir, "best_latent_depth.pth"), map_location=device))
    proj_depth.eval()
    
    proj_seg = LinearLatentProjector(input_dim).to(device)
    proj_seg.load_state_dict(torch.load(os.path.join(args.models_dir, "best_latent_seg.pth"), map_location=device))
    proj_seg.eval()

    # C. Aligner (fMRI -> Class)
    print("Loading Aligner (for retrieval)...")
    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aln_path = os.path.join(args.aligner_dir, f"S{args.subject}", args.rois, "best_aligner.pth")
    
    if not os.path.exists(aln_path):
        # Fallback path
        aln_path = os.path.join(args.models_dir, "best_aligner.pth")
    
    if os.path.exists(aln_path):
        aligner.load_state_dict(torch.load(aln_path, map_location=device))
        print(f"Aligner loaded from {aln_path}")
    else:
        print(f"WARNING: Aligner not found at {aln_path}! Classification will be random.")
    aligner.eval()

    # D. Feature Bank per Retrieval
    feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f: all_features = pickle.load(f)
    else:
        print("WARN: Feature bank not found. Retrieval random.")
        all_features = {}

    # E. ControlNet Pipeline
    print("Loading ControlNet Pipeline...")
    cn_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    cn_seg = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=[cn_depth, cn_seg], 
        torch_dtype=torch.float16,
        vae=vae # Passiamo il VAE in float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 2. CARICAMENTO DATASET TEST
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )
    
    grouped = defaultdict(list)
    paths = {}
    for i, vec in enumerate(raw_data['fmri']):
        path = raw_data['image_paths'][i]
        fname = os.path.basename(path)
        grouped[fname].append(vec)
        paths[fname] = path
    
    # Prepare Retrieval Tensor
    if all_features:
        ret_vals = []
        for k in sorted(all_features.keys()):
            ret_vals.append(all_features[k].flatten())
        if len(ret_vals) > 0:
            ret_tensor = torch.tensor(np.array(ret_vals)).float().to(device)
            ret_tensor = F.normalize(ret_tensor, p=2, dim=1)
        else:
            ret_tensor = None
    else:
        ret_tensor = None

    os.makedirs(args.output_dir, exist_ok=True)
    scaler = 0.18215 

    # 3. LOOP INFERENZA
    sorted_imgs = sorted(list(grouped.keys()))
    
    for i, fname in enumerate(tqdm(sorted_imgs, desc="Generating")):
        trials = torch.tensor(np.array(grouped[fname])).float().to(device)
        fmri = torch.mean(trials, dim=0, keepdim=True)
        
        real_key = os.path.splitext(fname)[0]
        gt_class_idx = i if i < 50 else -1 
        gt_class_name = TEST_CLASSES[gt_class_idx] if gt_class_idx >= 0 and gt_class_idx < len(TEST_CLASSES) else "??"

        # B. RETRIEVAL
        pred_class_name = "??"
        if ret_tensor is not None:
            with torch.no_grad():
                pf, _ = aligner(fmri)
                pf = F.normalize(pf.view(1,-1), p=2, dim=1)
                sims = torch.mm(pf, ret_tensor.T)
                best_idx = torch.argmax(sims).item()
                pred_class_name = TEST_CLASSES[best_idx % 50] 

        # C. RICOSTRUZIONE MAPPE (Latent Injection)
        with torch.no_grad():
            # 1. Proiezione (Output è Float32)
            lat_depth = proj_depth(fmri)
            lat_seg = proj_seg(fmri)
            
            # FIX: Convertiamo in Float16 per il VAE
            lat_depth = lat_depth.to(dtype=vae.dtype)
            lat_seg = lat_seg.to(dtype=vae.dtype)

            # 2. VAE Decoding
            rec_depth = vae.decode(lat_depth / scaler).sample 
            rec_seg = vae.decode(lat_seg / scaler).sample
            
            # 3. Post-Process
            rec_depth = (rec_depth / 2 + 0.5).clamp(0, 1)
            rec_seg = (rec_seg / 2 + 0.5).clamp(0, 1)
            
            d_pil = Image.fromarray((rec_depth.squeeze().permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))
            s_pil = Image.fromarray((rec_seg.squeeze().permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8))
            
            d_pil = d_pil.resize((512, 512), Image.BICUBIC)
            s_pil = s_pil.resize((512, 512), Image.NEAREST)

        # D. GENERAZIONE
        prompt = f"a photo of {pred_class_name}, high quality, realistic, 8k, trending on artstation"
        neg_prompt = "cartoon, drawing, anime, low quality, blur, bad anatomy, text, watermark"
        
        gen_img = pipe(
            prompt, 
            negative_prompt=neg_prompt,
            image=[d_pil, s_pil], 
            num_inference_steps=25,
            controlnet_conditioning_scale=[args.scale_depth, args.scale_seg]
        ).images[0]

        # E. VISUALIZZAZIONE GRID
        gt_img_path = paths[fname]
        gt_img = Image.open(gt_img_path).convert("RGB").resize((512, 512))
        
        gt_d_path = gt_img_path.replace("images", "derived_maps/depth").replace(".JPEG", ".png").replace(".jpg", ".png")
        gt_s_path = gt_img_path.replace("images", "derived_maps/segmentation").replace(".JPEG", ".png").replace(".jpg", ".png")
        gt_d = Image.open(gt_d_path).resize((512, 512)) if os.path.exists(gt_d_path) else Image.new("RGB", (512, 512))
        gt_s = Image.open(gt_s_path).resize((512, 512)) if os.path.exists(gt_s_path) else Image.new("RGB", (512, 512))

        W, H = 512, 512
        grid = Image.new("RGB", (W*4, H*2), "black")
        
        grid.paste(gt_img, (0, 0))
        grid.paste(gt_d, (W, 0))
        grid.paste(gt_s, (2*W, 0))
        
        grid.paste(gen_img, (0, H))
        grid.paste(d_pil.convert("RGB"), (W, H))
        grid.paste(s_pil, (2*W, H))
        
        draw_text(grid, f"GT: {gt_class_name}", (10, 10))
        draw_text(grid, "GT Depth", (W+10, 10))
        draw_text(grid, "GT Seg", (2*W+10, 10))
        
        draw_text(grid, f"Pred: {pred_class_name}", (10, H+10))
        draw_text(grid, "Rec Latent Depth", (W+10, H+10))
        draw_text(grid, "Rec Latent Seg", (2*W+10, H+10))
        
        info_img = Image.new("RGB", (W, H), (30, 30, 30))
        draw = ImageDraw.Draw(info_img)
        draw.text((20, 50), f"Subject: {args.subject}", fill="white")
        draw.text((20, 100), f"Pipeline: Latent Injection", fill="cyan")
        draw.text((20, 150), f"Retrieval: {pred_class_name}", fill="yellow")
        grid.paste(info_img, (3*W, H))

        grid.save(os.path.join(args.output_dir, f"{real_key}_latent.png"))

    print("Inference Completed.")

if __name__ == "__main__":
    main()