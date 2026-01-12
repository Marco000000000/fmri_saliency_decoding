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
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.multimodal_decoders import MultiTaskDecoder, ParallelAlexNetAligner
# Assicurati che src/postprocessing.py sia l'ultima versione corretta (con i fix per float32)
from src.postprocessing import refine_depth, refine_segmentation, refine_jointly, normalize_min_max

# --- LISTA CLASSI TEST ---
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--checkpoints_dir", type=str, default="trained_fmri_decoders")
    parser.add_argument("--output_dir", type=str, default="results_complete_analysis_v2") # Nuova cartella output
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Complete Inference Analysis V2 (S{args.subject}) ---")

    # 1. Load Features & Data (Identico a prima)
    feat_path = os.path.join(args.dataset_root, "alexnet_features_spatial", "features.pkl")
    if not os.path.exists(feat_path): return
    with open(feat_path, "rb") as f: all_features = pickle.load(f)

    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    roi_keys = parse_roi_keys(args.rois)
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
    
    unique_images_sorted = sorted(list(grouped_data.keys())) 
    
    # 2. Retrieval Bank (Identico a prima)
    retrieval_keys = []
    retrieval_matrix = []
    for fname in unique_images_sorted:
        key = os.path.splitext(fname)[0]
        if key in all_features:
            retrieval_keys.append(key)
            retrieval_matrix.append(all_features[key].flatten())
    retrieval_tensor = torch.tensor(np.array(retrieval_matrix)).float().to(device)
    retrieval_tensor = F.normalize(retrieval_tensor, p=2, dim=1)

    # 3. Load Models (Identico a prima)
    roi_str = "_".join(roi_keys)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    model_dir = os.path.join(args.checkpoints_dir, "multimodal", f"S{args.subject}", roi_str)

    map_decoder = MultiTaskDecoder(input_dim).to(device)
    map_decoder.load_state_dict(torch.load(os.path.join(model_dir, "best_map_decoder.pth"), map_location=device))
    map_decoder.eval()

    aligner = ParallelAlexNetAligner(input_dim).to(device)
    aligner.load_state_dict(torch.load(os.path.join(model_dir, "best_aligner.pth"), map_location=device))
    aligner.eval()

    # 4. SD Pipeline (Identico a prima)
    cn_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    cn_seg = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=[cn_depth, cn_seg], torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    out_dir = os.path.join(args.output_dir, f"S{args.subject}_{roi_str}")
    os.makedirs(out_dir, exist_ok=True)
    limit = args.num_samples if args.num_samples > 0 else len(unique_images_sorted)

    def process_tensors(d_tensor, s_tensor, refine_mode=False):
        """
        Converte tensori in mappe PIL gestendo correttamente le dimensioni.
        d_tensor: [1, 1, 224, 224] -> deve diventare [224, 224]
        s_tensor: [1, 3, 224, 224] -> deve diventare [224, 224, 3]
        """
        # 1. Depth: squeeze() senza argomenti rimuove TUTTE le dimensioni unitarie (Batch e Canale)
        #    Risultato: [224, 224] (Perfetto per PIL 'L')
        d = d_tensor.detach().cpu().squeeze() 
        
        # 2. Seg: squeeze(0) rimuove Batch, poi permute sposta i canali alla fine
        #    Input: [1, 3, 224, 224] -> squeeze(0) -> [3, 224, 224] -> permute -> [224, 224, 3]
        s = s_tensor.detach().cpu().squeeze(0).permute(1, 2, 0)
        
        # 3. Normalize & Numpy
        d_np = normalize_min_max(d.numpy())
        s_np = normalize_min_max(s.numpy())
        
        # 4. Converti in PIL
        #    Ora d_np è (224, 224) -> OK
        #    s_np è (224, 224, 3) -> OK
        dp = Image.fromarray(d_np).resize((512,512), Image.BICUBIC)
        sp = Image.fromarray(s_np).resize((512,512), Image.BICUBIC)
        
        if refine_mode:
            dp, sp = refine_jointly(dp, sp)
            
        return dp, sp
    # --- LOOP ---
    for i, fname in enumerate(tqdm(unique_images_sorted[:limit])):
        real_key = os.path.splitext(fname)[0]
        gt_class_name = TEST_CLASSES[i] if i < len(TEST_CLASSES) else "Unknown"

        trials = np.array(grouped_data[fname]) 
        trials_tensor = torch.from_numpy(trials).float().to(device)

        # A. Retrieval
        fmri_mean = torch.mean(trials_tensor, dim=0, keepdim=True)
        with torch.no_grad():
            pred_feat, _ = aligner(fmri_mean) 
            pred_feat = pred_feat.view(1, -1) 
            pred_feat = F.normalize(pred_feat, p=2, dim=1)
            sims = torch.mm(pred_feat, retrieval_tensor.T) 
            best_idx = torch.argmax(sims).item()
            pred_class_name = TEST_CLASSES[best_idx] if best_idx < len(TEST_CLASSES) else "Unknown"

        # B. Map Generation
        with torch.no_grad():
            # 1. Avg Input
            d_in, s_in = map_decoder(fmri_mean)
            maps_raw = process_tensors(d_in, s_in, refine_mode=False)
            maps_in_ref = process_tensors(d_in, s_in, refine_mode=True)
            
            # 2. Avg Output (Best maps)
            d_batch, s_batch = map_decoder(trials_tensor)
            d_out = torch.mean(d_batch, dim=0, keepdim=True)
            s_out = torch.mean(s_batch, dim=0, keepdim=True)
            maps_out_ref = process_tensors(d_out, s_out, refine_mode=True)

        # C. GT Maps
        gt_d_path = os.path.join(args.dataset_root, "derived_maps", "depth", "test", fname.replace(".JPEG", ".png").replace(".jpg", ".png"))
        gt_s_path = os.path.join(args.dataset_root, "derived_maps", "segmentation", "test", fname.replace(".JPEG", ".png").replace(".jpg", ".png"))
        maps_gt = (
            Image.open(gt_d_path).convert("L").resize((512,512)) if os.path.exists(gt_d_path) else Image.new("L", (512,512)),
            Image.open(gt_s_path).convert("RGB").resize((512,512)) if os.path.exists(gt_s_path) else Image.new("RGB", (512,512))
        )

        # D. Image Generation Helper (Aggiornato per scale flessibili)
        def generate(cls, d_map, s_map, scales_list):
            """
            scales_list: lista di 2 float [scale_depth, scale_seg].
            Passa 0.0 per disabilitare una modalità.
            """
            prompt = f"a photo of {cls}, high quality, realistic, 8k"
            neg = "low quality, bad anatomy, blur"
            return pipe(
                prompt, 
                image=[d_map, s_map], # Passa sempre entrambe le mappe
                negative_prompt=neg, 
                num_inference_steps=20, 
                guidance_scale=args.guidance_scale, 
                controlnet_conditioning_scale=scales_list # Le scale decidono cosa usare
            ).images[0].resize((512,512))

        scale = args.controlnet_conditioning_scale
        
        # 1. Generazioni Congiunte (Joint)
        img_raw     = generate(pred_class_name, maps_raw[0], maps_raw[1], [scale, scale])
        img_in_ref  = generate(pred_class_name, maps_in_ref[0], maps_in_ref[1], [scale, scale])
        img_out_ref = generate(pred_class_name, maps_out_ref[0], maps_out_ref[1], [scale, scale])
        img_oracle  = generate(gt_class_name, maps_gt[0], maps_gt[1], [scale, scale])

        # 2. Generazioni Separate (Solo per Avg Output Refined)
        # Depth Only: Scala Seg a 0.0
        img_out_depth_only = generate(pred_class_name, maps_out_ref[0], maps_out_ref[1], [scale, 0.0])
        # Seg Only: Scala Depth a 0.0
        img_out_seg_only   = generate(pred_class_name, maps_out_ref[0], maps_out_ref[1], [0.0, scale])

        # E. Grid Construction (7 Colonne)
        W, H = 512, 512
        N_COLS = 7
        header_h = 80
        grid = Image.new("RGB", (W * N_COLS, H * 3 + header_h), "black")
        draw = ImageDraw.Draw(grid)
        
        gt_img = Image.open(image_path_map[fname]).convert("RGB").resize((W,H))
        
        # --- Define Columns Content ---
        # Row 1: Final Images
        imgs_row1 = [gt_img, img_raw, img_in_ref, img_out_ref, img_out_depth_only, img_out_seg_only, img_oracle]
        
        # Row 2 & 3: Maps used (Visualizziamo le mappe di input, anche se una è stata ignorata dalla scala 0)
        # Per le colonne separate, mostriamo le mappe Out Refined che sono state passate al pipe.
        depths_row2 = [maps_gt[0], maps_raw[0], maps_in_ref[0], maps_out_ref[0], maps_out_ref[0], maps_out_ref[0], maps_gt[0]]
        segs_row3   = [maps_gt[1], maps_raw[1], maps_in_ref[1], maps_out_ref[1], maps_out_ref[1], maps_out_ref[1], maps_gt[1]]

        # --- Paste Images ---
        for idx, img in enumerate(imgs_row1):   grid.paste(img, (idx*W, header_h))
        for idx, img in enumerate(depths_row2): grid.paste(img.convert("RGB"), (idx*W, header_h + H))
        for idx, img in enumerate(segs_row3):   grid.paste(img, (idx*W, header_h + 2*H))

        # --- Headers & Labels ---
        col_pred = "green" if pred_class_name == gt_class_name else "red"
        
        labels = [
            (f"GT: {gt_class_name}", "Original", "white", "gray"),
            (f"Pred: {pred_class_name}", "Avg Input (RAW)", col_pred, "yellow"),
            (f"Pred: {pred_class_name}", "Avg Input (Refined)", col_pred, "gray"),
            (f"Pred: {pred_class_name}", "Avg Output (Refined)", col_pred, "green"),
            (f"Pred: {pred_class_name}", "Avg Out (Depth Only)", col_pred, "orange"), # Nuovo
            (f"Pred: {pred_class_name}", "Avg Out (Seg Only)", col_pred, "purple"),   # Nuovo
            (f"Prompt: {gt_class_name}", "Oracle (GT Maps)", "cyan", "gray")
        ]

        for idx, (top_txt, bot_txt, top_col, bot_col) in enumerate(labels):
            draw_text_centered(draw, top_txt, idx*W + W//2, 10, top_col, 24)
            draw_text_centered(draw, bot_txt, idx*W + W//2, 45, bot_col, 16)

        grid.save(os.path.join(out_dir, f"{real_key}_analysis_v2.png"))

    print(f"Finito. Risultati in {out_dir}")

if __name__ == "__main__":
    main()