import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from skimage.segmentation import slic
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner 

# --- CONFIGURAZIONE E UTILS ---

TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]

# ADE20K Palette
ADE20K_URL = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/data/objectInfo150.csv"
def get_palette():
    csv_name = "objectInfo150.csv"
    if not os.path.exists(csv_name):
        try: urllib.request.urlretrieve(ADE20K_URL, csv_name)
        except: pass
    try:
        return pd.read_csv(csv_name)[['R','G','B']].values.astype(np.uint8)
    except:
        return np.random.randint(0, 255, (150, 3), dtype=np.uint8)

PALETTE = get_palette()
TREE = cKDTree(PALETTE)

def draw_label_bg(draw, text, x, y, color="white", bg="black", font_size=20):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0]-4, bbox[1]-4, bbox[2]+4, bbox[3]+4], fill=bg)
    draw.text((x, y), text, fill=color, font=font)

# --- QUANTIZZAZIONE ---

def quantize_depth(depth_pil, n_clusters=3):
    arr = np.array(depth_pil).astype(np.float32) / 255.0
    h, w = arr.shape
    pixels = arr.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.flatten()
    map_label = {old: new for new, old in enumerate(np.argsort(centers))}
    mapped = np.array([map_label[l] for l in labels]).reshape(h, w)
    
    vals = [30, 128, 255]
    out = np.zeros_like(mapped, dtype=np.uint8)
    for i in range(n_clusters): out[mapped==i] = vals[i]
    return Image.fromarray(out)

def quantize_seg(seg_pil, n_segments=30):
    img = np.array(seg_pil)
    if len(img.shape)==2: img = np.stack([img]*3, axis=-1)
    segments = slic(img, n_segments=n_segments, compactness=10, start_label=1)
    out = np.zeros_like(img)
    for idx in np.unique(segments):
        mask = (segments == idx)
        if not np.any(mask): continue
        mean_c = img[mask].mean(axis=0)
        _, pal_idx = TREE.query(mean_c)
        out[mask] = PALETTE[pal_idx]
    return Image.fromarray(out.astype(np.uint8))

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--scale_depth", type=float, default=0.5)
    parser.add_argument("--scale_seg", type=float, default=0.5)
    parser.add_argument("--segments", type=int, default=30)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    parser.add_argument("--output_dir", type=str, default="results_multicontrolnet_full")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Multi-ControlNet FULL Inference (S{args.subject}) ---")

    # 1. SETUP MODELLI
    roi_keys = ["VC"]
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    model_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{args.subject}", "VC")
    
    decoder = MultiTaskDecoder(input_dim).to(device)
    decoder.load_state_dict(torch.load(os.path.join(model_dir, "best_map_decoder.pth"), map_location=device))
    decoder.eval()

    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aligner.load_state_dict(torch.load(os.path.join(model_dir, "best_aligner.pth"), map_location=device))
    aligner.eval()
    
    feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    with open(feat_path, "rb") as f: all_features = pickle.load(f)

    # 2. CONTROLNET SETUP
    cn_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    cn_seg = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        controlnet=[cn_depth, cn_seg], # [0]=Depth, [1]=Seg
        torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 3. DATA LOAD
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )
    
    grouped = defaultdict(list)
    path_map = {}
    for i, vec in enumerate(raw_data['fmri']):
        path = raw_data['image_paths'][i]
        fname = os.path.basename(path)
        grouped[fname].append(vec)
        path_map[fname] = path
    unique_imgs = sorted(list(grouped.keys()))

    ret_k, ret_v = [], []
    for f in unique_imgs:
        k = os.path.splitext(f)[0]
        if k in all_features: ret_k.append(k); ret_v.append(all_features[k].flatten())
    ret_tensor = F.normalize(torch.tensor(np.array(ret_v)).float().to(device), p=2, dim=1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 4. LOOP
    for i, fname in enumerate(tqdm(unique_imgs)):
        real_key = os.path.splitext(fname)[0]
        gt_class = TEST_CLASSES[i] if i < len(TEST_CLASSES) else "??"
        
        trials = torch.tensor(np.array(grouped[fname])).float().to(device)
        fmri = torch.mean(trials, dim=0, keepdim=True)

        # A. RETRIEVAL
        with torch.no_grad():
            pf, _ = aligner(fmri)
            pf = F.normalize(pf.view(1,-1), p=2, dim=1)
            sims = torch.mm(pf, ret_tensor.T)
            best_idx = torch.argmax(sims).item()
            pred_class = TEST_CLASSES[best_idx] if best_idx < len(TEST_CLASSES) else "??"

        # B. PREDICTION & QUANTIZATION
        with torch.no_grad():
            d_raw, s_raw = decoder(fmri)
            
            d_npy = d_raw.squeeze().cpu().numpy()
            d_npy = (d_npy - d_npy.min()) / (d_npy.max() - d_npy.min() + 1e-6) * 255
            d_pil = Image.fromarray(d_npy.astype(np.uint8)).resize((512,512), Image.BICUBIC)
            
            s_npy = s_raw.squeeze().cpu().permute(1,2,0).numpy()
            s_npy = (s_npy - s_npy.min()) / (s_npy.max() - s_npy.min() + 1e-6) * 255
            s_pil = Image.fromarray(s_npy.astype(np.uint8)).resize((512,512), Image.BICUBIC)

        d_quant = quantize_depth(d_pil, n_clusters=3)
        s_quant = quantize_seg(s_pil, n_segments=args.segments)

        # C. GT DATA
        gt_img = Image.open(path_map[fname]).convert("RGB").resize((512,512))
        
        # GT Depth (path replacement)
        gt_depth_path = path_map[fname].replace("images", "derived_maps/depth").replace(".JPEG", ".png").replace(".jpg", ".png")
        if os.path.exists(gt_depth_path):
            gt_depth = Image.open(gt_depth_path).convert("RGB").resize((512,512))
        else:
            gt_depth = Image.new("RGB", (512,512), "black")
            
        # GT Seg (Simulated via SLIC on GT Image)
        gt_seg = quantize_seg(gt_img, n_segments=args.segments)

        # D. GENERATION (3 PASSAGGI)
        prompt = f"a photo of {pred_class}, high quality, realistic, 8k"
        neg = "low quality, blur, bad anatomy, deformed"
        
        # 1. DEPTH ONLY (Seg Scale = 0.0)
        img_depth_only = pipe(
            prompt, image=[d_quant, s_quant], negative_prompt=neg, num_inference_steps=20,
            controlnet_conditioning_scale=[args.scale_depth, 0.0]
        ).images[0]
        
        # 2. SEG ONLY (Depth Scale = 0.0)
        img_seg_only = pipe(
            prompt, image=[d_quant, s_quant], negative_prompt=neg, num_inference_steps=20,
            controlnet_conditioning_scale=[0.0, args.scale_seg]
        ).images[0]
        
        # 3. COMBINED (Both active)
        img_combined = pipe(
            prompt, image=[d_quant, s_quant], negative_prompt=neg, num_inference_steps=20,
            controlnet_conditioning_scale=[args.scale_depth, args.scale_seg]
        ).images[0]

        # E. VISUALIZATION GRID (3x3)
        W, H = 512, 512
        grid = Image.new("RGB", (W*3, H*3), "black")
        draw = ImageDraw.Draw(grid)
        
        # Row 1: GTs
        grid.paste(gt_img, (0, 0)); grid.paste(gt_depth, (W, 0)); grid.paste(gt_seg, (2*W, 0))
        
        # Row 2: Generations
        grid.paste(img_combined, (0, H)); grid.paste(img_depth_only, (W, H)); grid.paste(img_seg_only, (2*W, H))
        
        # Row 3: Inputs fMRI
        # Mettiamo un'info box nella prima colonna
        info_img = Image.new("RGB", (W, H), "black")
        d_info = ImageDraw.Draw(info_img)
        d_info.text((50, 100), f"Subject: {args.subject}", fill="white", font_size=30)
        d_info.text((50, 150), f"Pred Class: {pred_class}", fill="green", font_size=30)
        d_info.text((50, 200), f"Depth Scale: {args.scale_depth}", fill="yellow", font_size=30)
        d_info.text((50, 250), f"Seg Scale: {args.scale_seg}", fill="cyan", font_size=30)
        
        grid.paste(info_img, (0, 2*H)); grid.paste(d_quant.convert("RGB"), (W, 2*H)); grid.paste(s_quant, (2*W, 2*H))

        # Labels
        # Row 1
        draw_label_bg(draw, f"GT Photo: {gt_class}", 10, 10)
        draw_label_bg(draw, "GT Depth (Ref)", W+10, 10)
        draw_label_bg(draw, "GT Seg (Simulated)", 2*W+10, 10)
        
        # Row 2
        draw_label_bg(draw, "GEN COMBINED", 10, H+10, color="green")
        draw_label_bg(draw, "GEN DEPTH ONLY", W+10, H+10, color="yellow")
        draw_label_bg(draw, "GEN SEG ONLY", 2*W+10, H+10, color="cyan")
        
        # Row 3
        draw_label_bg(draw, "fMRI Depth (Input)", W+10, 2*H+10)
        draw_label_bg(draw, "fMRI Seg (Input)", 2*W+10, 2*H+10)

        grid.save(os.path.join(args.output_dir, f"{real_key}_full.png"))

    print("Inference Finished.")

if __name__ == "__main__":
    main()