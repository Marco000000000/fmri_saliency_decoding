import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bdpy
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from diffusers.image_processor import IPAdapterMaskProcessor
from tqdm import tqdm

# --- MODELLI ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- HELPER FUNCTIONS ---
def get_spatial_mask_pil(depth_flat_tensor, spatial_shape, threshold=0.5, target_size=(1024, 1024)):
    """Estrae la maschera spaziale o il bounding box dal segnale fMRI predetto."""
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)

    if spatial_shape == "mask":
        return Image.fromarray(bin_64_np * 255).resize(target_size, Image.NEAREST)
        
    elif spatial_shape == "box":
        coords = np.argwhere(bin_64_np > 0)
        mask_np = np.zeros(target_size[::-1], dtype=np.uint8) # (height, width)
        if len(coords) > 5:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
            x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            mask_np[y1:y2, x1:x2] = 255
        else:
            mask_np[:, :] = 255
        return Image.fromarray(mask_np)

def create_overlay(original_img, mask_pil, spatial_type):
    """Crea l'overlay verde per la maschera o il rettangolo ciano per i box."""
    if "mask" in spatial_type:
        orig_rgba = original_img.resize(mask_pil.size).convert("RGBA")
        green_layer = Image.new("RGBA", orig_rgba.size, (0, 255, 0, 130)) # Verde alpha=130
        mask_l = mask_pil.convert("L")
        return Image.composite(green_layer, orig_rgba, mask_l).convert("RGB")
    else:
        orig_img_resized = original_img.resize(mask_pil.size).convert("RGB")
        bbox = mask_pil.getbbox()
        if bbox:
            draw = ImageDraw.Draw(orig_img_resized)
            draw.rectangle(bbox, outline="cyan", width=8)
        return orig_img_resized

def normalize_and_cast(emb):
    emb = torch.clamp(emb, -1.0, 1.0)
    norm = emb.norm(p=2, dim=-1, keepdim=True)
    if norm.item() > 1e-6: emb = emb / norm
    return emb * 0.95 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", choices=["natural", "shapes", "letters"], default="natural")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--spatial", type=str, default="none", 
                        choices=["none", "inpaint_mask", "inpaint_box", "attn_mask", "attn_box"])
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    
    # 1. Configurazione Percorsi
    if args.test_type == "natural":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTest_fmriprep_volume_native.h5"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
        img_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"
    elif args.test_type == "shapes":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ArtificialShapes_fmriprep_volume_native.h5"
        img_dir = f"{base_dir}/data/Shen2019/stimuli/ArtificialShapes/ArtificialImage"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_ArtificialImage.tsv"
        ext = ".tiff"
    else: # letters
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_perceptionLetterImage_original_VC.h5"
        img_dir = f"{base_dir}/data/Shen2019/stimuli/Letters/LetterImage"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_LetterImage.tsv"
        ext = ".tif"

    print(f"🧠 Caricamento dati TEST ({args.test_type}) S{args.subject} | Metodo Spaziale: {args.spatial}")
    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    
    stim_key = 'image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id'
    labels = bdata.select(stim_key)[:, 0] 
    unique_labels = np.unique(labels[labels > 0])
    
    averaged_fmri, output_ids = [], []
    for lbl in unique_labels:
        idx = [i for i in np.where(labels == lbl)[0] if i < X_raw.shape[0]]
        if len(idx) > 0:
            averaged_fmri.append(np.mean(X_raw[idx], axis=0))
            output_ids.append(int(lbl))
    X_test = torch.tensor(np.array(averaged_fmri), dtype=torch.float32).to(device)

    # 2. Caricamento Decoders (CLIP + Depth)
    train_dim = X_test.shape[1]
    
    aligner_sem = SimpleLinearAligner(train_dim, output_dim=1280).to(device)
    sem_path = f"{base_dir}/trained_fmri_decoders/shen_clip_sparse/S{args.subject}_clip_aligner.pth"
    aligner_sem.load_state_dict(torch.load(sem_path, map_location=device))
    aligner_sem.eval()

    if args.spatial != "none":
        aligner_struct = SimpleLinearAligner(train_dim, output_dim=4096).to(device)
        struct_path = f"{base_dir}/trained_fmri_decoders/shen_depth_sparse/S{args.subject}_depth_aligner.pth"
        aligner_struct.load_state_dict(torch.load(struct_path, map_location=device))
        aligner_struct.eval()

    # 3. SDXL Pipelines & Processors
    print("🎨 Inizializzazione SDXL Pipeline...")
    pipe_base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf, use_safetensors=True
    ).to(device)
    pipe_base.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_base.set_ip_adapter_scale(1.0)
    
    pipe_inpaint = None
    mask_processor = None

    if args.spatial.startswith("inpaint"):
        pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtype_inf, use_safetensors=True
        ).to(device)
        pipe_inpaint.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe_inpaint.set_ip_adapter_scale(1.0)
    elif args.spatial.startswith("attn"):
        mask_processor = IPAdapterMaskProcessor()

    out_dir = f"reconstructions/S{args.subject}_{args.test_type}_{args.spatial}"
    single_dir = os.path.join(out_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    # Mapping Immagini Originali per il Plot
    id_to_filename = {}
    if os.path.exists(tsv_path):
        df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
        for _, row in df_tsv.iterrows():
            f_name = str(row[0]).strip()
            if not f_name.lower().endswith(ext.lower()): f_name += ext
            id_to_filename[int(row[3])] = f_name

    # 4. Generazione
    print(f"🚀 Inizio inferenza su {len(output_ids)} immagini...")
    zero_emb = torch.zeros((1, 1, 1280), device=device, dtype=dtype_inf)

    for fmri_vec, img_id in tqdm(zip(X_test, output_ids), total=len(output_ids)):
        with torch.no_grad():
            emb_sem = normalize_and_cast(aligner_sem(fmri_vec.unsqueeze(0))).to(dtype_inf)
            ip_adapter_embeds = torch.cat([zero_emb, emb_sem.unsqueeze(1)], dim=0)
            
            depth_flat = aligner_struct(fmri_vec.unsqueeze(0)) if args.spatial != "none" else None

        seed = 42
        generator = torch.Generator(device=device).manual_seed(seed)
        prompt = "" if args.test_type == "natural" else ""
        
        coarse_image = None
        final_image = None

        if args.spatial == "none":
            final_image = pipe_base(
                prompt=prompt, ip_adapter_image_embeds=[ip_adapter_embeds],
                num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            mask_pil = None
            
        elif args.spatial.startswith("attn"):
            spatial_shape = args.spatial.split("_")[1] # "mask" o "box"
            mask_pil = get_spatial_mask_pil(depth_flat, spatial_shape, threshold=args.mask_threshold)
            ip_mask = mask_processor.preprocess([mask_pil])
            
            final_image = pipe_base(
                prompt=prompt, ip_adapter_image_embeds=[ip_adapter_embeds],
                cross_attention_kwargs={"ip_adapter_masks": ip_mask},
                num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            
        elif args.spatial.startswith("inpaint"):
            spatial_shape = args.spatial.split("_")[1]
            mask_pil = get_spatial_mask_pil(depth_flat, spatial_shape, threshold=args.mask_threshold)
            
            coarse_image = pipe_base(
                prompt=prompt, ip_adapter_image_embeds=[ip_adapter_embeds],
                num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            
            mask_pil_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))

            final_image = pipe_inpaint(
                prompt=prompt, image=coarse_image, mask_image=mask_pil_blurred,
                ip_adapter_image_embeds=[ip_adapter_embeds],
                num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]

        # Salva output puro
        final_image.save(os.path.join(single_dir, f"id_{img_id}.png"))

        # --- PLOTTING INTELLIGENTE ---
        try:
            orig_fname = id_to_filename.get(img_id)
            if orig_fname and os.path.exists(os.path.join(img_dir, orig_fname)):
                orig_img = Image.open(os.path.join(img_dir, orig_fname)).convert("RGB")
                
                if args.spatial == "none":
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    ax1.imshow(orig_img); ax1.set_title(f"Originale (ID {img_id})"); ax1.axis("off")
                    ax2.imshow(final_image); ax2.set_title("SDXL Recon (None)"); ax2.axis("off")
                
                elif args.spatial.startswith("attn"):
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                    overlay_img = create_overlay(orig_img, mask_pil, args.spatial)
                    
                    ax1.imshow(orig_img); ax1.set_title(f"Originale (ID {img_id})"); ax1.axis("off")
                    ax2.imshow(overlay_img); ax2.set_title(f"Overlay ({args.spatial})"); ax2.axis("off")
                    ax3.imshow(final_image); ax3.set_title("Ricostruita (IP Attentivo)"); ax3.axis("off")
                
                elif args.spatial.startswith("inpaint"):
                    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                    overlay_img = create_overlay(orig_img, mask_pil, args.spatial)
                    
                    ax[0].imshow(orig_img); ax[0].set_title(f"Originale (ID {img_id})"); ax[0].axis("off")
                    ax[1].imshow(overlay_img); ax[1].set_title(f"Overlay ({args.spatial})"); ax[1].axis("off")
                    ax[2].imshow(coarse_image); ax[2].set_title("Step 1: Globale (SDXL)"); ax[2].axis("off")
                    ax[3].imshow(final_image); ax[3].set_title("Step 2: Locale (Inpaint)"); ax[3].axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"comparison_{img_id}.png"))
                plt.close()
        except Exception as e:
            print(f"⚠️ Errore nel plot per ID {img_id}: {e}")

    print(f"✅ Completato! Analizza i plot in: {out_dir}")

if __name__ == "__main__":
    main()