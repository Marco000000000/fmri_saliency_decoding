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
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# Setup path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.sota_wrappers import FmriToSota
from src.models.multimodal_decoders import ParallelAlexNetAligner 

# --- PALETTE ADE20K (Per convertire indici -> colori per ControlNet) ---
def get_ade20k_palette():
    return [
        0,0,0, 120,120,120, 180,120,120, 6,230,230, 80,50,50, 4,200,3, 120,120,80, 140,140,140, 204,5,255, 230,230,230, 4,250,7, 224,5,255, 235,255,7, 150,5,61, 120,120,70, 8,255,51, 255,6,82, 143,255,140, 204,255,4, 255,51,7, 204,70,3, 0,102,200, 61,230,250, 255,6,51, 11,102,255, 255,7,71, 255,9,224, 9,7,230, 220,220,220, 255,9,92, 112,9,255, 8,255,214, 7,255,224, 255,184,6, 10,255,71, 255,41,10, 7,255,255, 224,255,8, 102,8, 255,255,61, 6,255,194, 7,255,122, 8,0,255, 20,255,8, 41,255,5, 153,6,51, 255,235,12, 255,160,150, 20,0,163, 255,140,140, 140,250,10, 15,20,255, 0,31,255, 0,255,31, 0,255,224, 0,153,255, 0,0,0, 255,255,71, 0,0,235, 255,0,173, 255,31,0, 255,11,200, 200,255,82, 0,0,255, 245,0,61, 255,0,255, 112,0,255, 133,255,0, 0,255,163, 0,255,102, 0,194,255, 0,0,143, 255,51,255, 0,0,82, 255,0,255, 41,0,255, 173,10,0, 255,173,255, 0,0,255, 153,255,92, 0,255,0, 255,255,0, 245,255,0, 102,255,173, 0,255,0, 20,255,184, 184,0,31, 255,0,255, 61,0,71, 255,255,0, 204,0,255, 194,0,255, 82,0,10, 255,0,112, 255,51,0, 255,0,194, 255,0,122, 255,0,255, 163,255,150, 20,0,255, 10,0,255, 122,0,255, 10,255,0, 0,224,255, 143,255,0, 82,0,255, 163,255,0, 255,235,0, 8,184,170, 133,0,255, 0,255,92, 184,0,255, 255,0,31, 0,184,255, 0,214,255, 255,0,112, 92,255,0, 0,224,255, 112,224,255, 70,184,160, 163,0,255, 153,0,255, 71,255,0, 255,0,163, 255,204,0, 255,0,143, 0,255,235, 133,255,0, 255,0,235, 245,0,255, 255,0,122, 255,245,0, 10,190,212, 214,255,0, 0,204,255, 20,0,255, 255,255,0, 0,153,255, 0,41,255, 0,255,204, 41,0,255, 41,255,0, 173,0,255, 0,245,255, 71,0,255, 122,0,255, 0,255,184, 0,92,255, 184,255,0, 0,133,255, 255,214,0, 25,194,194, 102,255,0, 92,0,255
    ]

# Classi Test GOD (Prime 50)
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
    parser = argparse.ArgumentParser(description="Inference SOTA: Fmri -> Adapter -> SegFormer/DPT -> ControlNet")
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--models_dir", type=str, default="trained_sota_models")
    parser.add_argument("--aligner_dir", type=str, default="trained_fmri_decoders/multimodal")
    parser.add_argument("--output_dir", type=str, default="results_sota_reconstruction")
    parser.add_argument("--scale_depth", type=float, default=0.7)
    parser.add_argument("--scale_seg", type=float, default=0.6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference SOTA Reconstruction S{args.subject} ---")

    # 1. SETUP MODELS
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # A. Reconstruction Models (SOTA Wrappers)
    print("Loading FmriToSota Models...")
    
    # SegFormer (Segmentation)
    model_seg = FmriToSota(input_dim, task='seg').to(device)
    # Carichiamo usando strict=False perché nel checkpoint salvato dal train
    # c'erano SOLO i pesi allenabili (linear + adapter), mentre qui istanziamo tutto il modello SOTA.
    # I pesi SOTA congelati vengono ricaricati da HuggingFace all'init.
    model_seg.load_state_dict(torch.load(os.path.join(args.models_dir, "model_seg_best.pth"), map_location=device), strict=False)
    model_seg.eval()
    
    # DPT (Depth)
    model_depth = FmriToSota(input_dim, task='depth').to(device)
    model_depth.load_state_dict(torch.load(os.path.join(args.models_dir, "model_depth_best.pth"), map_location=device), strict=False)
    model_depth.eval()

    # B. Aligner (Retrieval)
    print("Loading Aligner...")
    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aln_path = os.path.join(args.aligner_dir, f"S{args.subject}", args.rois, "best_aligner.pth")
    if os.path.exists(aln_path):
        aligner.load_state_dict(torch.load(aln_path, map_location=device))
    else:
        print("WARN: Aligner not found, using random retrieval.")
    aligner.eval()

    # C. Feature Bank
    feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f: all_features = pickle.load(f)
        # Prepare Retrieval Tensor
        ret_vals = [all_features[k].flatten() for k in sorted(all_features.keys())]
        ret_tensor = torch.tensor(np.array(ret_vals)).float().to(device)
        ret_tensor = F.normalize(ret_tensor, p=2, dim=1)
    else:
        ret_tensor = None

    # D. ControlNet
    print("Loading ControlNet...")
    cn_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    cn_seg = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=[cn_depth, cn_seg], torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 2. DATASET
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )
    
    # Raggruppa trial
    grouped = defaultdict(list)
    paths = {}
    for i, vec in enumerate(raw_data['fmri']):
        path = raw_data['image_paths'][i]
        fname = os.path.basename(path)
        grouped[fname].append(vec)
        paths[fname] = path

    os.makedirs(args.output_dir, exist_ok=True)
    palette = get_ade20k_palette()

    # 3. LOOP
    sorted_imgs = sorted(list(grouped.keys()))
    
    for i, fname in enumerate(tqdm(sorted_imgs, desc="Generating")):
        # Average fMRI
        trials = torch.tensor(np.array(grouped[fname])).float().to(device)
        fmri = torch.mean(trials, dim=0, keepdim=True)
        
        real_key = os.path.splitext(fname)[0]
        gt_class_idx = i if i < 50 else -1
        gt_class_name = TEST_CLASSES[gt_class_idx] if 0 <= gt_class_idx < 50 else "??"

        # A. RETRIEVAL
        pred_class_name = "??"
        if ret_tensor is not None:
            with torch.no_grad():
                pf, _ = aligner(fmri)
                pf = F.normalize(pf.view(1,-1), p=2, dim=1)
                sims = torch.mm(pf, ret_tensor.T)
                best_idx = torch.argmax(sims).item()
                pred_class_name = TEST_CLASSES[best_idx % 50]

        # B. RECONSTRUCTION (fMRI -> SOTA Decoder)
        with torch.no_grad():
            # 1. DEPTH (DPT)
            d_feats = model_depth.forward_fmri(fmri)
            d_map = model_depth.decode_from_features(d_feats) # [1, H, W]
            
            # Normalize Depth for ControlNet [0, 255]
            d_min, d_max = d_map.min(), d_map.max()
            d_norm = (d_map - d_min) / (d_max - d_min + 1e-8)
            d_pil = Image.fromarray((d_norm.squeeze().cpu().numpy() * 255).astype(np.uint8)).resize((512,512))

            # 2. SEGMENTATION (SegFormer)
            s_feats = model_seg.forward_fmri(fmri) # List of tensors
            s_logits = model_seg.decode_from_features(s_feats) # [1, 150, 512, 512]
            
            # Argmax to get class indices
            s_idx = torch.argmax(s_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8) # [512, 512]
            
            # Apply Palette
            s_pil = Image.fromarray(s_idx)
            s_pil.putpalette(palette)
            s_pil = s_pil.convert("RGB") # ControlNet vuole RGB

        # C. GENERATION
        prompt = f"a photo of {pred_class_name}, high quality, realistic, 8k"
        neg = "cartoon, blur, low quality, bad anatomy, text"
        
        gen_img = pipe(
            prompt, negative_prompt=neg,
            image=[d_pil, s_pil],
            num_inference_steps=25,
            controlnet_conditioning_scale=[args.scale_depth, args.scale_seg]
        ).images[0]

        # D. SAVE GRID
        gt_path = paths[fname]
        gt_img = Image.open(gt_path).convert("RGB").resize((512, 512))
        
        W, H = 512, 512
        grid = Image.new("RGB", (W*3, H*2), "black")
        
        # Top: GT, Rec Depth, Rec Seg
        grid.paste(gt_img, (0, 0))
        grid.paste(d_pil.convert("RGB"), (W, 0))
        grid.paste(s_pil, (2*W, 0))
        
        # Bottom: Gen, Info
        grid.paste(gen_img, (0, H))
        
        info = Image.new("RGB", (W*2, H), (20,20,20))
        draw = ImageDraw.Draw(info)
        draw.text((20, 50), f"GT Class: {gt_class_name}", fill="white")
        draw.text((20, 100), f"Pred Class: {pred_class_name}", fill="yellow")
        draw.text((20, 150), f"Pipeline: SOTA Alignment (Adapter)", fill="cyan")
        grid.paste(info, (W, H))

        grid.save(os.path.join(args.output_dir, f"{real_key}_sota.png"))

    print("Done.")

if __name__ == "__main__":
    main()