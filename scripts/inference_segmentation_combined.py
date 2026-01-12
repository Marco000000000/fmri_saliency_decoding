import argparse
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner 

TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]

def draw_text_centered(draw, text, center_x, top_y, color="white", font_size=20):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
    except AttributeError:
        w, h = draw.textsize(text, font=font)
    x = center_x - w // 2
    y = top_y
    draw.text((x, y), text, fill=color, font=font)

def quantize_and_split(depth_map_pil, n_clusters=3):
    """
    K-Means + Splitting dei piani.
    Ritorna:
      - quant_img: Immagine completa a 3 colori (30, 128, 255)
      - planes: Lista di 3 immagini dove SOLO quel piano è attivo (255) e il resto è nero (0).
    """
    arr = np.array(depth_map_pil).astype(np.float32) / 255.0
    h, w = arr.shape
    pixels = arr.reshape(-1, 1)
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.flatten()
    
    # Ordina centri (0=Lontano/Scuro, 2=Vicino/Chiaro)
    sorted_indices = np.argsort(centers)
    
    # Mappa label
    map_label = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    mapped_labels = np.array([map_label[l] for l in labels]).reshape(h, w)
    
    # 1. Immagine Combinata (Grigi relativi corretti)
    quant_values = [30, 128, 255] # BG, MID, FG
    quant_img_arr = np.zeros_like(mapped_labels, dtype=np.uint8)
    for i in range(n_clusters):
        quant_img_arr[mapped_labels == i] = quant_values[i]
    
    # 2. Piani Singoli (Isolati)
    # Per vedere l'effetto di un piano, lo mettiamo Bianco (255) su Nero (0)
    # Così ControlNet vede "Qualcosa qui" vs "Nulla altrove".
    planes = []
    for i in range(n_clusters):
        plane_arr = np.zeros_like(mapped_labels, dtype=np.uint8)
        plane_arr[mapped_labels == i] = 255 # Accendiamo solo questo piano
        planes.append(Image.fromarray(plane_arr))
        
    return Image.fromarray(quant_img_arr), planes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--scale", type=float, default=0.5, help="ControlNet Strength")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    parser.add_argument("--output_dir", type=str, default="results_kmeans_planes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference K-Means Planes (S{args.subject}) Scale={args.scale} ---")

    # 1. Load Data & Models
    # (Identico a prima, omesso per brevità)
    feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    with open(feat_path, "rb") as f: all_features = pickle.load(f)

    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    # Load Models
    model_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{args.subject}", f"{args.rois}")
    
    # Map Decoder (Continuo)
    map_decoder = MultiTaskDecoder(input_dim).to(device)
    map_path = os.path.join(model_dir, "best_map_decoder.pth")
    if not os.path.exists(map_path): print("No map decoder"); return
    map_decoder.load_state_dict(torch.load(map_path, map_location=device))
    map_decoder.eval()

    # Aligner
    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aligner.load_state_dict(torch.load(os.path.join(model_dir, "best_aligner.pth"), map_location=device))
    aligner.eval()

    # ControlNet
    cn_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=cn_depth, torch_dtype=torch.float16
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
    
    # Organize Data
    grouped_data = defaultdict(list)
    image_path_map = {}
    for i, fmri_vec in enumerate(raw_data['fmri']):
        path = raw_data['image_paths'][i]
        fname = os.path.basename(path)
        grouped_data[fname].append(fmri_vec)
        image_path_map[fname] = path
    unique_images = sorted(list(grouped_data.keys()))
    
    # Retrieval
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

        # B. Depth Generation (Continuous)
        with torch.no_grad():
            d_out, _ = map_decoder(fmri_mean)
            d_npy = d_out.squeeze().cpu().numpy()
            d_npy = (d_npy - d_npy.min()) / (d_npy.max() - d_npy.min() + 1e-6) * 255
            d_pil = Image.fromarray(d_npy.astype(np.uint8)).resize((512,512), Image.BICUBIC)

        # C. K-Means (Combined + Individual Planes)
        d_comb, planes = quantize_and_split(d_pil, n_clusters=3)

        # D. GT Load
        gt_img_path = image_path_map[fname]
        gt_img = Image.open(gt_img_path).convert("RGB").resize((512,512))
        
        # GT Depth (se esiste)
        gt_depth_path = gt_img_path.replace("images", "derived_maps/depth_quantized_visible").replace(".JPEG", ".png").replace(".jpg", ".png")
        if os.path.exists(gt_depth_path):
            gt_depth = Image.open(gt_depth_path).convert("RGB").resize((512,512))
        else:
            gt_depth = Image.new("RGB", (512,512), "black")

        # E. Multi-Generation Loop
        # Generiamo un'immagine per ogni condizione
        prompt = f"a photo of {pred_class}, high quality, realistic, 8k"
        neg_prompt = "low quality, bad anatomy, blur"
        
        # Helper Gen
        def generate(ctrl_image):
            return pipe(
                prompt, image=ctrl_image, negative_prompt=neg_prompt,
                num_inference_steps=20, controlnet_conditioning_scale=args.scale
            ).images[0]

        img_comb = generate(d_comb)
        img_p1 = generate(planes[0]) # BG
        img_p2 = generate(planes[1]) # MID
        img_p3 = generate(planes[2]) # FG

        # F. Grid Visualization (5 Colonne)
        W, H = 512, 512
        header_h = 80
        grid = Image.new("RGB", (W*5, H*2 + header_h), "black")
        draw = ImageDraw.Draw(grid)
        
        # Row 1: Images (GT + 4 Gen)
        imgs = [gt_img, img_comb, img_p1, img_p2, img_p3]
        for idx, img in enumerate(imgs):
            grid.paste(img, (idx*W, header_h))
            
        # Row 2: Maps (GT Depth + 4 Maps Used)
        maps = [gt_depth, d_comb.convert("RGB"), planes[0].convert("RGB"), planes[1].convert("RGB"), planes[2].convert("RGB")]
        for idx, mp in enumerate(maps):
            grid.paste(mp, (idx*W, header_h + H))
            
        # Headers
        headers = [
            (f"GT: {gt_class}", "Original Data"),
            ("Combined Planes", f"Scale {args.scale}"),
            ("Plane 1 (BG)", "Focus Background"),
            ("Plane 2 (MID)", "Focus Midground"),
            ("Plane 3 (FG)", "Focus Foreground")
        ]
        
        for idx, (top, bot) in enumerate(headers):
            color = "cyan" if idx == 0 else "green"
            draw_text_centered(draw, top, idx*W + W//2, 10, color, 24)
            draw_text_centered(draw, bot, idx*W + W//2, 45, "gray", 16)
            
        grid.save(os.path.join(args.output_dir, f"{real_key}_planes_S{args.scale}.png"))

    print("Finito.")

if __name__ == "__main__":
    main()