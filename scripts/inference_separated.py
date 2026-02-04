import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Imposta la GPU specifica come richiesto
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from skimage.segmentation import slic
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner 

# --- CONFIG & PALETTE (Ridotta) ---
TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]

# Indici Macro-Classi (Wall, Floor, Person, Animal, Car, etc.)
TARGET_INDICES = [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 15, 19, 20, 21, 126, 127]

# ADE20K Palette (Hardcoded per brevità, o caricata da file)
# Qui usiamo un placeholder o la logica di caricamento del file precedente
def get_reduced_palette_tree():
    import urllib.request
    url = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/data/objectInfo150.csv"
    if not os.path.exists("objectInfo150.csv"):
        try: urllib.request.urlretrieve(url, "objectInfo150.csv")
        except: pass
    try:
        df = pd.read_csv("objectInfo150.csv")
        full_pal = df[['R','G','B']].values.astype(np.uint8)
        reduced = full_pal[TARGET_INDICES]
        return cKDTree(reduced), reduced
    except:
        # Fallback random se fallisce download
        rnd = np.random.randint(0,255, (len(TARGET_INDICES), 3))
        return cKDTree(rnd), rnd

REDUCED_TREE, REDUCED_COLORS = get_reduced_palette_tree()

# --- UTILS VISUALIZZAZIONE ---
def draw_label(draw, text, x, y, color="white", bg="black", font_size=20):
    try: font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except: font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=bg)
    draw.text((x, y), text, fill=color, font=font)

# --- POST-PROCESSING ---
def process_depth(tensor):
    # Tensor [1, 1, 224, 224] -> PIL Normalized
    d = tensor.squeeze().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6) * 255
    return Image.fromarray(d.astype(np.uint8)).resize((512,512), Image.BICUBIC)

def process_seg(tensor):
    # Tensor [1, 3, 224, 224] -> PIL Normalized
    s = tensor.squeeze().cpu().permute(1,2,0).numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-6) * 255
    return Image.fromarray(s.astype(np.uint8)).resize((512,512), Image.BICUBIC)

def quantize_depth(pil, k=3):
    arr = np.array(pil).astype(np.float32)/255.0
    h,w = arr.shape
    pixels = arr.reshape(-1,1)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3).fit(pixels)
    centers = kmeans.cluster_centers_.flatten()
    map_l = {old: new for new, old in enumerate(np.argsort(centers))} # 0=Dark
    mapped = np.array([map_l[l] for l in kmeans.labels_]).reshape(h,w)
    out = np.zeros_like(mapped, dtype=np.uint8)
    vals = [30, 128, 255]
    for i in range(k): out[mapped==i] = vals[i]
    return Image.fromarray(out)

def quantize_seg(pil, n_seg=30):
    img = np.array(pil)
    if len(img.shape)==2: img=np.stack([img]*3, axis=-1)
    segments = slic(img, n_segments=n_seg, compactness=10, start_label=1)
    out = np.zeros_like(img)
    for idx in np.unique(segments):
        mask = segments==idx
        if not np.any(mask): continue
        mean = img[mask].mean(axis=0)
        _, pi = REDUCED_TREE.query(mean)
        out[mask] = REDUCED_COLORS[pi]
    return Image.fromarray(out.astype(np.uint8))

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--scale_depth", type=float, default=0.6)
    parser.add_argument("--scale_seg", type=float, default=0.6)
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    parser.add_argument("--output_dir", type=str, default="results_comparison_modes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Comparison: AvgInput vs AvgOutput vs Single ---")

    # 1. LOAD MODELS
    input_dim = sum([subject_dims[k][args.subject-1] for k in parse_roi_keys(args.rois)])
    model_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{args.subject}", args.rois)
    
    # Decoder Depth
    dec_d = MultiTaskDecoder(input_dim).to(device)
    dec_d.load_state_dict(torch.load(os.path.join(model_dir, "best_depth_decoder.pth")))
    dec_d.eval()
    
    # Decoder Seg
    dec_s = MultiTaskDecoder(input_dim).to(device)
    dec_s.load_state_dict(torch.load(os.path.join(model_dir, "best_seg_decoder.pth")))
    dec_s.eval()
    
    # Aligner
    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aligner.load_state_dict(torch.load(os.path.join(model_dir, "best_aligner.pth")))
    aligner.eval()

    # Features for Retrieval
    with open("data/GOD_Dataset/alexnet_features_spatial/features.pkl", "rb") as f:
        all_feats = pickle.load(f)

    # 2. CONTROLNET
    cn_d = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    cn_s = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=[cn_d, cn_s], torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 3. DATA
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in parse_roi_keys(args.rois)}
    raw = load_fmri_image_pairs("data/GOD_Dataset/fmri_files/test", "data/GOD_Dataset/images/test", test_h5, rois_dict)
    
    grouped = defaultdict(list)
    paths = {}
    for i, vec in enumerate(raw['fmri']):
        fname = os.path.basename(raw['image_paths'][i])
        grouped[fname].append(vec)
        paths[fname] = raw['image_paths'][i]
    
    # Retrieval Tensor
    ret_vals = []
    for f in sorted(list(grouped.keys())):
        k = os.path.splitext(f)[0]
        if k in all_feats: ret_vals.append(all_feats[k].flatten())
    ret_t = F.normalize(torch.tensor(np.array(ret_vals)).float().to(device), p=2, dim=1)

    os.makedirs(args.output_dir, exist_ok=True)

    # 4. LOOP IMMAGINI
    unique_imgs = sorted(list(grouped.keys()))
    
    # Processiamo solo le prime 5 per brevità del test
    for i, fname in enumerate(tqdm(unique_imgs)):
        
        # A. Preparazione Dati
        trials_np = np.array(grouped[fname]) # [N_trials, InputDim]
        trials = torch.tensor(trials_np).float().to(device)
        gt_img = Image.open(paths[fname]).convert("RGB").resize((512,512))
        real_key = os.path.splitext(fname)[0]
        
        # Helper Retrieval
        def get_class(fmri_vec):
            with torch.no_grad():
                pf, _ = aligner(fmri_vec)
                pf = F.normalize(pf.view(1,-1), p=2, dim=1)
                sims = torch.mm(pf, ret_t.T)
                idx = torch.argmax(sims).item()
                return TEST_CLASSES[idx] if idx < len(TEST_CLASSES) else "??"
        
        # --- MODE 1: AVG INPUT ---
        # Media prima del modello
        fmri_avg_in = torch.mean(trials, dim=0, keepdim=True)
        class_1 = get_class(fmri_avg_in)
        
        with torch.no_grad():
            d1_raw, _ = dec_d(fmri_avg_in)
            _, s1_raw = dec_s(fmri_avg_in)
        
        # --- MODE 2: AVG OUTPUT ---
        # Passa tutti i trial, media dopo
        class_2 = class_1 # Per retrieval usiamo avg input solitamente, o media feature. Usiamo same per consistenza visiva
        all_d, all_s = [], []
        # Batch processing per velocità
        # Se troppi trial, fai a chunk. Qui GOD ha ~35 trial.
        with torch.no_grad():
            d_batch, _ = dec_d(trials) # [N, 1, 224, 224]
            _, s_batch = dec_s(trials) # [N, 3, 224, 224]
            
            d2_raw = torch.mean(d_batch, dim=0, keepdim=True)
            s2_raw = torch.mean(s_batch, dim=0, keepdim=True)

        # --- MODE 3: SINGLE TRIAL ---
        # Prendi il primo (indice 0)
        fmri_single = trials[0].unsqueeze(0)
        class_3 = get_class(fmri_single) # Vediamo se retrieval funziona su singolo
        
        with torch.no_grad():
            d3_raw, _ = dec_d(fmri_single)
            _, s3_raw = dec_s(fmri_single)

        # --- GENERATION LOOP ---
        results = []
        # Lista tuples: (Label, RawDepth, RawSeg, PredClass)
        scenarios = [
            ("Avg Input (Standard)", d1_raw, s1_raw, class_1),
            ("Avg Output (Ensemble)", d2_raw, s2_raw, class_2),
            ("Single Trial (Noisy)", d3_raw, s3_raw, class_3)
        ]
        
        for lbl, d_tens, s_tens, cls_name in scenarios:
            # Proc
            d_pil = process_depth(d_tens)
            s_pil = process_seg(s_tens)
            
            # Quant
            d_q = quantize_depth(d_pil)
            s_q = quantize_seg(s_pil)
            
            # Gen
            prompt = f"a photo of {cls_name}, high quality, realistic, 8k"
            img = pipe(prompt, image=[d_q, s_q], num_inference_steps=20, 
                       controlnet_conditioning_scale=[args.scale_depth, args.scale_seg]).images[0]
            
            results.append({
                'label': lbl, 'class': cls_name,
                'depth': d_q, 'seg': s_q, 'gen': img
            })

        # --- VISUALIZATION TABLE ---
        # Rows: GT, AvgIn, AvgOut, Single
        # Cols: Label/Class | Depth Map | Seg Map | Generated
        W, H = 512, 512
        grid = Image.new("RGB", (W*4, H*4), "black")
        draw = ImageDraw.Draw(grid)
        
        # Row 0: GT (Carichiamo derived se ci sono)
        gt_d_path = paths[fname].replace("images", "derived_maps/depth").replace(".JPEG", ".png").replace(".jpg", ".png")
        gt_s_path = paths[fname].replace("images", "derived_maps/segmentation").replace(".JPEG", ".png").replace(".jpg", ".png")
        
        gt_d = Image.open(gt_d_path).convert("RGB").resize((W,H)) if os.path.exists(gt_d_path) else Image.new("RGB",(W,H))
        gt_s = Image.open(gt_s_path).convert("RGB").resize((W,H)) if os.path.exists(gt_s_path) else Image.new("RGB",(W,H))
        
        # Paste Row 0
        grid.paste(gt_img, (3*W, 0)) # Metto GT image alla fine per confronto con gen
        grid.paste(gt_d, (W, 0))
        grid.paste(gt_s, (2*W, 0))
        draw_label(draw, "GROUND TRUTH", 20, 200, font_size=40)
        draw_label(draw, "GT Depth", W+20, 20)
        draw_label(draw, "GT Seg", 2*W+20, 20)
        draw_label(draw, "Original Photo", 3*W+20, 20)
        
        # Paste Scenario Rows
        for idx, res in enumerate(results):
            y = (idx + 1) * H
            # Col 0: Label Box
            draw.text((20, y+150), res['label'], fill="cyan", font_size=40)
            draw.text((20, y+220), f"Class: {res['class']}", fill="yellow", font_size=30)
            
            # Col 1: Depth
            grid.paste(res['depth'].convert("RGB"), (W, y))
            # Col 2: Seg
            grid.paste(res['seg'], (2*W, y))
            # Col 3: Gen
            grid.paste(res['gen'], (3*W, y))

        grid.save(os.path.join(args.output_dir, f"{real_key}_comparison.png"))

if __name__ == "__main__":
    main()