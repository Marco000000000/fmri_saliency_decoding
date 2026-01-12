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
from scipy.spatial import cKDTree
from skimage.segmentation import slic
from skimage.util import img_as_float
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner 

# --- ADE20K PALETTE SETUP ---
ADE20K_OFFICIAL_CSV_URL = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/data/objectInfo150.csv"

def get_ade20k_palette():
    csv_path = "objectInfo150.csv"
    if not os.path.exists(csv_path):
        try:
            urllib.request.urlretrieve(ADE20K_OFFICIAL_CSV_URL, csv_path)
        except:
            return np.random.randint(0, 255, (150, 3), dtype=np.uint8)
    try:
        df = pd.read_csv(csv_path)
        return df[['R', 'G', 'B']].values.astype(np.uint8)
    except:
        return np.random.randint(0, 255, (150, 3), dtype=np.uint8)

PALETTE = get_ade20k_palette()
TREE = cKDTree(PALETTE)

def quantize_segmentation_slic(seg_map_pil, n_segments=100, compactness=10):
    """
    Applica Superpixel Segmentation (SLIC) per pulire la mappa semantica.
    """
    img = np.array(seg_map_pil)
    
    # Se l'immagine è in scala di grigi o ha dimensione strana, forza RGB
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    
    # SLIC
    segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=1)
    out_img = np.zeros_like(img)
    
    # Loop sui segmenti
    for seg_idx in np.unique(segments):
        mask = (segments == seg_idx)
        if np.sum(mask) == 0: continue
        
        # Colore medio -> Snap to Palette
        mean_color = img[mask].mean(axis=0)
        dist, idx = TREE.query(mean_color)
        best_color = PALETTE[idx]
        
        out_img[mask] = best_color
        
    return Image.fromarray(out_img.astype(np.uint8))

TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--scale", type=float, default=0.5, help="ControlNet Strength")
    parser.add_argument("--segments", type=int, default=50, help="Numero di superpixels")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    parser.add_argument("--output_dir", type=str, default="results_seg_quantized")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference Seg Quantization (SLIC) S{args.subject} ---")

    # 1. Load Data & Models
    # ... (Caricamento feature solito) ...
    feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    with open(feat_path, "rb") as f: all_features = pickle.load(f)

    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # Decoder Multimodale (Continuo)
    model_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{args.subject}", f"{args.rois}")
    map_decoder = MultiTaskDecoder(input_dim).to(device)
    map_decoder.load_state_dict(torch.load(os.path.join(model_dir, "best_map_decoder.pth"), map_location=device))
    map_decoder.eval()

    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aligner.load_state_dict(torch.load(os.path.join(model_dir, "best_aligner.pth"), map_location=device))
    aligner.eval()

    # ControlNet
    cn_seg = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=cn_seg, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Load Data
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )
    
    grouped_data = defaultdict(list)
    image_path_map = {}
    for i, fmri_vec in enumerate(raw_data['fmri']):
        path = raw_data['image_paths'][i]
        fname = os.path.basename(path)
        grouped_data[fname].append(fmri_vec)
        image_path_map[fname] = path
    unique_images = sorted(list(grouped_data.keys()))
    
    # Retrieval Setup
    ret_keys = []
    ret_vals = []
    for f in unique_images:
        k = os.path.splitext(f)[0]
        if k in all_features:
            ret_keys.append(k)
            ret_vals.append(all_features[k].flatten())
    ret_tensor = F.normalize(torch.tensor(np.array(ret_vals)).float().to(device), p=2, dim=1)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inference Loop
    for i, fname in enumerate(tqdm(unique_images)):
        real_key = os.path.splitext(fname)[0]
        gt_class = TEST_CLASSES[i] if i < len(TEST_CLASSES) else "Unknown"
        
        trials = torch.tensor(np.array(grouped_data[fname])).float().to(device)
        fmri_mean = torch.mean(trials, dim=0, keepdim=True)

        # A. Retrieval
        with torch.no_grad():
            pred_feat, _ = aligner(fmri_mean)
            pred_feat = F.normalize(pred_feat.view(1,-1), p=2, dim=1)
            sims = torch.mm(pred_feat, ret_tensor.T)
            best_idx = torch.argmax(sims).item()
            pred_class = TEST_CLASSES[best_idx] if best_idx < len(TEST_CLASSES) else "Unknown"

        # B. Raw fMRI Prediction
        with torch.no_grad():
            _, s_out = map_decoder(fmri_mean) # [1, 3, 224, 224] (Output del modello continuo)
            s_npy = s_out.squeeze().cpu().permute(1, 2, 0).numpy()
            s_npy = (s_npy - s_npy.min()) / (s_npy.max() - s_npy.min() + 1e-6) * 255
            s_pil_pred = Image.fromarray(s_npy.astype(np.uint8)).resize((512,512), Image.BICUBIC)

        # C. SLIC Quantization (PREDICTED)
        s_quant_pred = quantize_segmentation_slic(s_pil_pred, n_segments=args.segments)

        # D. Pseudo-GT Segmentation
        # Carichiamo la foto vera e applichiamo lo stesso SLIC per vedere "come dovrebbe essere"
        gt_img = Image.open(image_path_map[fname]).convert("RGB").resize((512,512))
        s_quant_gt = quantize_segmentation_slic(gt_img, n_segments=args.segments)

        # E. Generation (Usa la predizione quantizzata)
        prompt = f"a photo of {pred_class}, high quality, realistic, 8k"
        img_gen = pipe(prompt, image=s_quant_pred, num_inference_steps=20, controlnet_conditioning_scale=args.scale).images[0]

        # F. Visualization Grid
        # Cols: GT Image | Pseudo-GT Seg | Pred Raw Seg | Pred Quant Seg | Generated
        W, H = 512, 512
        grid = Image.new("RGB", (W*5, H+60), "black")
        draw = ImageDraw.Draw(grid)
        
        items = [
            (gt_img, f"GT Photo: {gt_class}"),
            (s_quant_gt, "GT Seg (Target)"),
            (s_pil_pred, "fMRI Raw Output"),
            (s_quant_pred, f"fMRI Seg (SLIC {args.segments})"),
            (img_gen, f"Gen: {pred_class}")
        ]
        
        for idx, (img, lbl) in enumerate(items):
            grid.paste(img, (idx*W, 60))
            # Header colorato
            color = "cyan" if idx < 2 else ("yellow" if idx < 4 else "green")
            draw.text((idx*W+10, 20), lbl, fill=color, font_size=20)
            
        grid.save(os.path.join(args.output_dir, f"{real_key}_seg_compare.png"))

    print("Finito.")

if __name__ == "__main__":
    main()