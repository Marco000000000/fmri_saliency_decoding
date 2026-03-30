import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Add root to path (necessario per caricare roi_config e dataset)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import IPAdapterMaskProcessor

# --- CLASSI MODELLO ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- HELPER FUNCTIONS ---
def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6:
        emb = emb / norm
    emb = emb * 0.95
    return emb

def get_image_id(image_path):
    basename = os.path.basename(image_path)
    cat_id = int(basename.split('_')[0][1:])
    img_id = int(basename.split('_')[1].split('.')[0])
    return float('%d.%06d' % (cat_id, img_id))

def extract_custom_shape_mask(depth_flat_tensor, threshold=0.6):
    """Estrae la sagoma organica predetta dalla tua V1 e applica il blur per SDXL."""
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    
    # Sfondo al 6% per evitare il collasso a griglia
    binary_shape = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(15.0/255.0).to(eroded.device))
    
    if binary_shape.sum() < 5:
        binary_shape = torch.ones_like(binary_shape)
    
    shape_1024 = F.interpolate(binary_shape, size=(1024, 1024), mode='bilinear', align_corners=False)
    shape_np = (shape_1024[0, 0].cpu().numpy() * 255).astype(np.uint8)
    
    shape_pil = Image.fromarray(shape_np)
    shape_pil = shape_pil.filter(ImageFilter.GaussianBlur(radius=25))
    return shape_pil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--roi", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="reconstructions_kamitani_masked")
    parser.add_argument("--guidance_scale", type=float, default=4.5) # Abbassato per evitare griglie
    parser.add_argument("--mask_threshold", type=float, default=0.6)
    parser.add_argument("--results_file", type=str, default="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/kamitani_method/results/GenericObjectDecoding.pkl")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    print(f"--- Inference KAMITANI (Semantica) + TUO MODELLO (Spazio) S{args.subject} | {args.roi} ---")

    # 1. Carica i risultati uniti di Kamitani
    if not os.path.exists(args.results_file):
        print(f"ERRORE: Manca il file dei risultati in: {args.results_file}")
        return
        
    with open(args.results_file, 'rb') as f:
        results_df = pickle.load(f)

    sbj_str = f"Subject{args.subject}"
    filtered_df = results_df[(results_df['subject'] == sbj_str) & (results_df['roi'] == args.roi)]
    
    if filtered_df.empty:
        print(f"ERRORE: Nessun risultato trovato per {sbj_str} e ROI {args.roi}.")
        return
        
    row = filtered_df.iloc[0]
    predicted_features = row['predicted_feature_averaged_percept'] # (50, 1280)
    test_labels = row['test_label_set_percept'] # (50,) ID

    # 2. Carica il TUO Decoder Strutturale (per la maschera)
    print("Caricamento Decoder Strutturale (fMRI -> Mask)...")
    roi_keys = parse_roi_keys(args.roi)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
    path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.roi}/best_depth_decoder.pth"
    if not os.path.exists(path_struct):
        print(f"ERRORE: Manca il decoder strutturale in: {path_struct}")
        return
    aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
    aligner_struct.eval()

    # 3. Mappiamo gli ID alle immagini reali e carichiamo i dati fMRI
    print("Caricamento Dati fMRI Test...")
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"), test_h5, rois_dict
    )
    
    grouped_fmri = defaultdict(list)
    img_paths_map = {}
    for i, vec in enumerate(raw_data['fmri']):
        p = raw_data['image_paths'][i]
        fname = os.path.basename(p)
        base_id = fname.split('_')[0] + '_' + fname.split('_')[1].split('.')[0]
        
        grouped_fmri[base_id].append(vec)
        img_id = get_image_id(p)
        img_paths_map[img_id] = p

    # 4. Pipeline SDXL + IP-Adapter
    print("Caricamento SDXL Base e IP-Adapter (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf
    ).to(device)
    
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe.set_ip_adapter_scale(1.0)
    mask_processor = IPAdapterMaskProcessor()

    # Setup cartelle output
    save_dir = os.path.join(args.output_dir, args.roi, str(args.subject))
    singles_dir = os.path.join(save_dir, "singles")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(singles_dir, exist_ok=True)

    print(f"Generazione Ibrida su {len(test_labels)} immagini...")

    # --- INFERENCE LOOP SUI RISULTATI IBRIDI ---
    for i, img_id in enumerate(tqdm(test_labels)):
        # A. SEMANTICA DA KAMITANI
        emb_sem_np = predicted_features[i]
        emb_sem_np = np.expand_dims(emb_sem_np, axis=0)

        real_img_path = img_paths_map.get(img_id)
        if not real_img_path: continue
        
        real_key = os.path.splitext(os.path.basename(real_img_path))[0]
        base_id = real_key.split('_')[0] + '_' + real_key.split('_')[1]

        with torch.no_grad():
            # Prepariamo l'Embedding Semantico di Kamitani
            emb_sem = torch.tensor(emb_sem_np, dtype=torch.float32).to(device)
            emb_sem = normalize_and_cast(emb_sem).to(dtype=dtype_inf)
            emb_sem_formatted = emb_sem.unsqueeze(1) 
            uncond_emb = torch.zeros_like(emb_sem_formatted)
            ip_adapter_embeds = torch.cat([uncond_emb, emb_sem_formatted], dim=0)
            
            # B. MASCHERA DAL TUO MODELLO
            trials_np = np.array(grouped_fmri[base_id])
            fmri_input = torch.tensor(trials_np, dtype=torch.float32).to(device).mean(dim=0, keepdim=True)
            
            depth_flat = aligner_struct(fmri_input)
            mask_pil = extract_custom_shape_mask(depth_flat, threshold=args.mask_threshold)
            ip_mask = mask_processor.preprocess([mask_pil])
            
        generator = torch.Generator(device=device).manual_seed(42)
        
        # C. GENERAZIONE
        generated_image = pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image_embeds=[ip_adapter_embeds],
            cross_attention_kwargs={"ip_adapter_masks": ip_mask}, 
            num_inference_steps=20,
            guidance_scale=args.guidance_scale, 
            height=1024, width=1024, 
            generator=generator
        ).images[0]

        # D. SALVATAGGIO SINGOLO (Per le Metriche)
        gen_thumb = generated_image.resize((512, 512))
        gen_thumb.save(os.path.join(singles_dir, f"{real_key}_hybrid.png"))

        # E. GRIGLIA DI CONFRONTO
        if os.path.exists(real_img_path):
            gt_img = Image.open(real_img_path).convert("RGBA").resize((512, 512))
        else:
            gt_img = Image.new("RGBA", (512, 512))

        # Sovrapposizione maschera su GT (Overlay Ciano)
        mask_resized = mask_pil.resize((512, 512)).convert("L")
        cyan_layer = Image.new("RGBA", (512, 512), (0, 255, 255, 128))
        gt_img.paste(cyan_layer, (0, 0), mask_resized)
        gt_img = gt_img.convert("RGB")

        grid = Image.new("RGB", (512 * 2, 512))
        grid.paste(gt_img, (0, 0))
        grid.paste(gen_thumb, (512, 0))
        
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "Ground Truth + TUO Bbox", fill="white")
        d.text((522, 10), "Kamitani Semantic + Tua Maschera", fill="lime")
        
        grid.save(os.path.join(save_dir, f"{real_key}_hybrid.png"))

    print(f"Finito. Singole per le metriche salvate in: {singles_dir}")

if __name__ == "__main__":
    main()