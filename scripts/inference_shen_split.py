import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import bdpy
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
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
        mask_np = np.zeros(target_size[::-1], dtype=np.uint8) 
        if len(coords) > 5:
            y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
            x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
            x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
            if x2 <= x1: x2 = x1 + 10
            if y2 <= y1: y2 = y1 + 10
            mask_np[y1:y2, x1:x2] = 255
        else: mask_np[:, :] = 255
        return Image.fromarray(mask_np)

def create_overlay(original_img, mask_pil, spatial_type):
    if "mask" in spatial_type:
        orig_rgba = original_img.resize(mask_pil.size).convert("RGBA")
        green_layer = Image.new("RGBA", orig_rgba.size, (0, 255, 0, 130))
        return Image.composite(green_layer, orig_rgba, mask_pil.convert("L")).convert("RGB")
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
    parser.add_argument("--spatial", type=str, default="attn_box", 
                        choices=["inpaint_mask", "inpaint_box", "attn_mask", "attn_box"])
    parser.add_argument("--order", type=str, default="bg_first", 
                        choices=["bg_first", "fg_first"], help="Ordine di generazione: Sfondo prima o Soggetto prima")
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    
    # 1. Configurazione Percorsi
    if args.test_type == "natural":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTest_fmriprep_volume_native.h5"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
        img_dir = f"{base_dir}/data/GOD_Dataset/images/test"; ext = ".JPEG"
    elif args.test_type == "shapes":
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ArtificialShapes_fmriprep_volume_native.h5"
        img_dir = f"{base_dir}/data/Shen2019/stimuli/ArtificialShapes/ArtificialImage"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_ArtificialImage.tsv"; ext = ".tiff"
    else: # letters
        h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_perceptionLetterImage_original_VC.h5"
        img_dir = f"{base_dir}/data/Shen2019/stimuli/Letters/LetterImage"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_LetterImage.tsv"; ext = ".tif"

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
    train_dim = X_test.shape[1]

    # 2. Caricamento Decoders
    print(f"🧠 Caricamento Decoders | Ordine: {args.order.upper()} | Spaziale: {args.spatial}")
    
    aligner_bg = SimpleLinearAligner(train_dim, 1280).to(device)
    aligner_bg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse_background/S{args.subject}_clip_aligner_ba.pth", map_location=device))
    aligner_bg.eval()

    aligner_fg = SimpleLinearAligner(train_dim, 1280).to(device)
    aligner_fg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse_foreground/S{args.subject}_clip_aligner_fo.pth", map_location=device))
    aligner_fg.eval()

    aligner_struct = SimpleLinearAligner(train_dim, 4096).to(device)
    # Assicurati del path corretto per le maschere
    aligner_struct.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_depth_sparse/S{args.subject}_depth_aligner.pth", map_location=device))
    aligner_struct.eval()

    # 3. Caricamento Pipelines
    pipe_base = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype_inf).to(device)
    pipe_base.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipe_base.set_ip_adapter_scale(1.0)
    
    pipe_inpaint = None
    pipe_img2img = None
    mask_processor = None

    if "inpaint" in args.spatial:
        pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=dtype_inf).to(device)
        pipe_inpaint.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        pipe_inpaint.set_ip_adapter_scale(1.0)
        
    if "attn" in args.spatial or args.order == "fg_first":
        # Ricicliamo la pipe_base per l'Img2Img risparmiando VRAM
        pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe_base)
        if "attn" in args.spatial:
            mask_processor = IPAdapterMaskProcessor()

    out_dir = f"reconstructions/S{args.subject}_{args.test_type}_{args.spatial}_{args.order}"
    single_dir = os.path.join(out_dir, "single"); os.makedirs(single_dir, exist_ok=True)
    
    blank_canvas = Image.new("RGB", (1024, 1024), (128, 128, 128))
    zero_emb = torch.zeros((1, 1, 1280), device=device, dtype=dtype_inf)

    id_to_filename = {}
    if os.path.exists(tsv_path):
        df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
        for _, row in df_tsv.iterrows():
            f_name = str(row[0]).strip()
            if not f_name.lower().endswith(ext.lower()): f_name += ext
            id_to_filename[int(row[3])] = f_name

    # 4. Generazione Split Semantics con Ordine Dinamico
    print(f"🚀 Inizio inferenza a Semantica Divisa ({len(output_ids)} immagini)...")

    for fmri_vec, img_id in tqdm(zip(X_test, output_ids), total=len(output_ids)):
        with torch.no_grad():
            emb_bg = normalize_and_cast(aligner_bg(fmri_vec.unsqueeze(0))).to(dtype_inf)
            emb_fg = normalize_and_cast(aligner_fg(fmri_vec.unsqueeze(0))).to(dtype_inf)
            ip_embeds_bg = torch.cat([zero_emb, emb_bg.unsqueeze(1)], dim=0) 
            ip_embeds_fg = torch.cat([zero_emb, emb_fg.unsqueeze(1)], dim=0) 
            depth_flat = aligner_struct(fmri_vec.unsqueeze(0))

        generator = torch.Generator(device=device).manual_seed(42)
        spatial_type = args.spatial.split("_")[1]
        mask_pil = get_spatial_mask_pil(depth_flat, spatial_type, threshold=args.mask_threshold)
        
        coarse_image = None
        final_image = None

        # ==========================================================
        # OPZIONE 1: BACKGROUND PRIMA (Classico)
        # ==========================================================
        if args.order == "bg_first":
            # Step 1: Tela Sfondo (Base)
            coarse_image = pipe_base(
                prompt="", ip_adapter_image_embeds=[ip_embeds_fg],
                num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
            ).images[0]
            
            # Step 2: Soggetto (Local Inpaint/Attn)
            if args.spatial.startswith("attn"):
                ip_mask = mask_processor.preprocess([mask_pil], height=1024, width=1024)
                final_image = pipe_img2img(
                    prompt="", image=coarse_image, ip_adapter_image_embeds=[ip_embeds_fg], 
                    cross_attention_kwargs={"ip_adapter_masks": ip_mask}, strength=0.75,
                    num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]
            elif args.spatial.startswith("inpaint"):
                mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
                final_image = pipe_inpaint(
                    prompt="", image=coarse_image, mask_image=mask_blurred,
                    ip_adapter_image_embeds=[ip_embeds_bg], 
                    num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]

        # ==========================================================
        # OPZIONE 2: FOREGROUND PRIMA (Invertito)
        # ==========================================================
        elif args.order == "fg_first":
            if args.spatial.startswith("attn"):
                # Step 1: Soggetto tramite maschera di attenzione su base pipe
                ip_mask = mask_processor.preprocess([mask_pil], height=1024, width=1024)
                coarse_image = pipe_base(
                    prompt="", ip_adapter_image_embeds=[ip_embeds_fg],
                    cross_attention_kwargs={"ip_adapter_masks": ip_mask},
                    num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]
                
                # Step 2: Sfondo tramite Img2Img globale per armonizzare
                final_image = pipe_img2img(
                    prompt="", image=coarse_image, ip_adapter_image_embeds=[ip_embeds_bg],
                    strength=0.5, # 0.5 per non distruggere l'oggetto creato
                    num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]
                
            elif args.spatial.startswith("inpaint"):
                # Step 1: Inpaint del soggetto su tela grigia
                mask_blurred = mask_pil.filter(ImageFilter.GaussianBlur(radius=8))
                coarse_image = pipe_inpaint(
                    prompt="", image=blank_canvas, mask_image=mask_blurred,
                    ip_adapter_image_embeds=[ip_embeds_fg], strength=1.0,
                    num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]
                
                # Step 2: Invertiamo la maschera per dipingere lo sfondo attorno al soggetto
                inv_mask = ImageOps.invert(mask_pil.convert('L')).filter(ImageFilter.GaussianBlur(radius=8))
                final_image = pipe_inpaint(
                    prompt="", image=coarse_image, mask_image=inv_mask,
                    ip_adapter_image_embeds=[ip_embeds_bg], strength=1.0,
                    num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator
                ).images[0]

        final_image.save(os.path.join(single_dir, f"id_{img_id}.png"))

        # --- PLOTTING INTELLIGENTE ---
        try:
            orig_fname = id_to_filename.get(img_id)
            if orig_fname and os.path.exists(os.path.join(img_dir, orig_fname)):
                orig_img = Image.open(os.path.join(img_dir, orig_fname)).convert("RGB")
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                overlay_img = create_overlay(orig_img, mask_pil, args.spatial)
                
                title_step1 = "Step 1: Background" if args.order == "bg_first" else "Step 1: Foreground"
                title_step2 = "Step 2: + Foreground" if args.order == "bg_first" else "Step 2: + Background"

                ax[0].imshow(orig_img); ax[0].set_title(f"Originale (ID {img_id})"); ax[0].axis("off")
                ax[1].imshow(overlay_img); ax[1].set_title(f"Overlay Mask"); ax[1].axis("off")
                ax[2].imshow(coarse_image); ax[2].set_title(title_step1); ax[2].axis("off")
                ax[3].imshow(final_image); ax[3].set_title(title_step2); ax[3].axis("off")
                
                plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"comparison_{img_id}.png")); plt.close()
        except Exception as e: print(f"⚠️ Errore plot ID {img_id}: {e}")

    print(f"✅ Completato! Analizza i plot in: {out_dir}")

if __name__ == "__main__": main()