import argparse
import os
import glob
import tarfile
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline
from diffusers.image_processor import IPAdapterMaskProcessor
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from tqdm import tqdm

# --- MODELLI ---
# NOTA: Per BOLD5000 usiamo bias=False per sconfiggere il Center Bias
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- HELPER FUNCTIONS ---
def get_spatial_mask_pil(depth_flat_tensor, spatial_shape, threshold=0.5, target_size=(1024, 1024)):
    """Estrae la maschera spaziale o il bounding box dal segnale fMRI predetto."""
    mask_64 = depth_flat_tensor.view(1, 1, 64, 64).to(dtype=torch.float32)
    
    # Sigmoide necessaria per le maschere BCE/Dice addestrate su BOLD5000
    mask_norm = torch.sigmoid(mask_64)
    
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

def create_overlay_transparent(original_img, mask_pil, spatial_type):
    """Crea l'overlay verde TRASPARENTE per la maschera (predetta) o il rettangolo ciano per i box."""
    if "mask" in spatial_type:
        orig_rgba = original_img.resize(mask_pil.size).convert("RGBA")
        # Verde alpha=180 (circa 70% opacità) per la trasparenza richiesta
        green_layer = Image.new("RGBA", orig_rgba.size, (0, 255, 0, 180)) 
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

# --- HELPER BOLD5000 ---
def extract_fmri_from_tars(tar_dir):
    """Estrae i tensori fMRI dal Train Set locale per calcolare lo Z-Score."""
    X_fmri = []
    tar_files = glob.glob(os.path.join(tar_dir, "*.tar"))
    for tf in tar_files:
        with tarfile.open(tf, "r") as tar:
            fmri_member = next((m for m in tar.getmembers() if m.name.endswith('.voxel.pyd')), None)
            if fmri_member:
                fmri_vec = np.load(io.BytesIO(tar.extractfile(fmri_member).read()), allow_pickle=True).flatten()
                X_fmri.append(fmri_vec)
    return torch.tensor(np.array(X_fmri), dtype=torch.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="CSI1")
    parser.add_argument("--spatial", type=str, default="none", 
                        choices=["none", "inpaint_mask", "inpaint_box", "attn_mask", "attn_box"])
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_inf = torch.float16
    base_data_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/WAVE-BOLD5000"
    subj_dir = os.path.join(base_data_dir, args.subject)

    # 1. Calcolo Z-Score Parameter dal Train Set
    print("⚖️ Calcolo parametri Z-Scoring dal Training Set locale...")
    X_train_tensors = extract_fmri_from_tars(os.path.join(subj_dir, "train"))
    x_mean = X_train_tensors.mean(dim=0, keepdim=True).to(device)
    x_std = X_train_tensors.std(dim=0, keepdim=True).to(device)
    train_dim = X_train_tensors.shape[1]

    # 2. Caricamento Modelli Decoders BOLD5000
    print(f"🧠 Caricamento dati TEST BOLD5000 S{args.subject} | Metodo Spaziale: {args.spatial}")
    aligner_sem = SimpleLinearAligner(train_dim, output_dim=1280).to(device)
    sem_path = f"trained_fmri_decoders/bold5000_clip_hf/{args.subject}_clip_aligner_hf.pth"
    aligner_sem.load_state_dict(torch.load(sem_path, map_location=device))
    aligner_sem.eval()

    aligner_struct = None
    if args.spatial != "none":
        aligner_struct = SimpleLinearAligner(train_dim, output_dim=4096).to(device)
        struct_path = f"trained_fmri_decoders/bold5000_mask_tar/{args.subject}_mask_aligner.pth"
        aligner_struct.load_state_dict(torch.load(struct_path, map_location=device))
        aligner_struct.eval()

    # 🚨 INIZIALIZZAZIONE RMBG PER VISUALIZZAZIONE GT 🚨
    # Dato che i tar non hanno la maschera GT pronta, la generiamo al volo sull'immagine originale
    print("\n📏 Inizializzazione modello RMBG-1.4 per visualizzazione Ground Truth...")
    rmbg_model_vis = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True).to(device).eval()
    preprocess_image_rmbg = transforms.Compose([
        transforms.Resize((1024, 1024), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

    # 3. SDXL Pipelines & Processors
    print("\n🎨 Inizializzazione SDXL Pipeline...")
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

    out_dir = f"reconstructions/bold5000_{args.subject}_{args.spatial}_sdxl_comparable"
    single_dir = os.path.join(out_dir, "single")
    os.makedirs(single_dir, exist_ok=True)

    test_tars = sorted(glob.glob(os.path.join(subj_dir, "test", "*.tar")))
    print(f"🚀 Inizio inferenza su {len(test_tars)} immagini di Test...")
    
    zero_emb = torch.zeros((1, 1, 1280), device=device, dtype=dtype_inf)

    for tf in tqdm(test_tars):
        base_name = os.path.basename(tf).replace('.tar', '')
        
        with tarfile.open(tf, "r") as tar:
            members = tar.getmembers()
            img_member = next((m for m in members if m.name.endswith('.png')), None)
            fmri_member = next((m for m in members if m.name.endswith('.voxel.pyd')), None)
            
            if not img_member or not fmri_member: continue
            
            # Estrazione Immagine Originale (Ground Truth)
            orig_img = Image.open(tar.extractfile(img_member)).convert("RGB")
            
            # 🚨 GENERAZIONE MASCHERA GROUND TRUTH PER IL PLOT 🚨
            input_tensor_rmbg = preprocess_image_rmbg(orig_img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds_gt_vis = rmbg_model_vis(input_tensor_rmbg)[0][0]
                preds_gt_vis = (preds_gt_vis - preds_gt_vis.min()) / (preds_gt_vis.max() - preds_gt_vis.min() + 1e-8)
                preds_gt_vis = torch.sigmoid(preds_gt_vis).unsqueeze(0).unsqueeze(0)
                # Resize alla dimensione target per plotting (SDXL preferisce 1024)
                gt_mask_pil_vis = Image.fromarray((torch.clamp(preds_gt_vis.squeeze(), 0, 1).cpu().numpy() * 255).astype(np.uint8)).resize((1024, 1024), Image.BILINEAR)
            
            # Estrazione e Z-Scoring fMRI
            fmri_raw = np.load(io.BytesIO(tar.extractfile(fmri_member).read()), allow_pickle=True).flatten()
            fmri_vec = torch.tensor(fmri_raw, dtype=torch.float32).to(device).unsqueeze(0)
            fmri_vec = (fmri_vec - x_mean) / (x_std + 1e-6)

        # Previsione fMRI
        with torch.no_grad():
            emb_sem = normalize_and_cast(aligner_sem(fmri_vec)).to(dtype_inf)
            ip_adapter_embeds = torch.cat([zero_emb, emb_sem.unsqueeze(1)], dim=0)
            depth_flat = aligner_struct(fmri_vec) if args.spatial != "none" else None

        generator = torch.Generator(device=device).manual_seed(42)
        prompt = ""
        
        coarse_image = None
        final_image = None
        mask_pil_pred = None

        # Kandinsky Generation (IP-Adapter SDXL)
        if args.spatial == "none":
            final_image = pipe_base(prompt=prompt, ip_adapter_image_embeds=[ip_adapter_embeds], num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
            
        elif args.spatial.startswith("attn"):
            spatial_shape = args.spatial.split("_")[1]
            mask_pil_pred = get_spatial_mask_pil(depth_flat, spatial_shape, threshold=args.mask_threshold) # Predicted Mask
            ip_mask = mask_processor.preprocess([mask_pil_pred])
            
            final_image = pipe_base(prompt=prompt, ip_adapter_image_embeds=[ip_adapter_embeds], cross_attention_kwargs={"ip_adapter_masks": ip_mask}, num_inference_steps=30, guidance_scale=args.guidance_scale, generator=generator).images[0]
            
        elif args.spatial.startswith("inpaint"):
            spatial_shape = args.spatial.split("_")[1]
            mask_pil_pred = get_spatial_mask_pil(depth_flat, spatial_shape, threshold=args.mask_threshold) # Predicted Mask
            
            coarse_image = pipe_base(prompt=prompt, ip_adapter_image_embeds=[ip_adapter_embeds], num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator).images[0]
            
            mask_pil_blurred = mask_pil_pred.filter(ImageFilter.GaussianBlur(radius=8))
            final_image = pipe_inpaint(prompt=prompt, image=coarse_image, mask_image=mask_pil_blurred, ip_adapter_image_embeds=[ip_adapter_embeds], num_inference_steps=25, guidance_scale=args.guidance_scale, generator=generator).images[0]

        # Salva immagine singola
        final_image.save(os.path.join(single_dir, f"{base_name}.png"))

        # --- PLOTTING AGGIORNATO (CON TRASPARENZA E MASCHERA GT) ---
        try:
            if args.spatial == "none":
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(orig_img); ax1.set_title(f"Originale ({base_name})"); ax1.axis("off")
                ax2.imshow(final_image); ax2.set_title("SDXL Recon (None)"); ax2.axis("off")
            
            elif args.spatial.startswith("attn"):
                # Griglia 1x4 per Metodi Attentivi
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
                # Overlay TRASPARENTE della PREVISIONE spaziale
                overlay_img_pred = create_overlay_transparent(orig_img, mask_pil_pred, args.spatial)
                
                ax1.imshow(orig_img); ax1.set_title(f"Originale ({base_name})"); ax1.axis("off")
                ax2.imshow(gt_mask_pil_vis, cmap='gray'); ax2.set_title("Maschera GT (Reale)"); ax2.axis("off")
                ax3.imshow(overlay_img_pred); ax3.set_title(f"Overlay TRASPARENTE Previsione ({args.spatial})"); ax3.axis("off")
                ax4.imshow(final_image); ax4.set_title("Ricostruita (IP Attentivo)"); ax4.axis("off")
            
            elif args.spatial.startswith("inpaint"):
                # Griglia 1x5 per Inpainting
                fig, ax = plt.subplots(1, 5, figsize=(25, 5))
                # Overlay TRASPARENTE della PREVISIONE spaziale
                overlay_img_pred = create_overlay_transparent(orig_img, mask_pil_pred, args.spatial)
                
                ax[0].imshow(orig_img); ax[0].set_title(f"Originale ({base_name})"); ax[0].axis("off")
                ax[1].imshow(gt_mask_pil_vis, cmap='gray'); ax[1].set_title("Maschera GT (Reale)"); ax[1].axis("off")
                ax[2].imshow(overlay_img_pred); ax[2].set_title(f"Overlay TRASPARENTE Previsione ({args.spatial})"); ax[2].axis("off")
                ax[3].imshow(coarse_image); ax[3].set_title("Step 1: Globale (SDXL)"); ax[3].axis("off")
                ax[4].imshow(final_image); ax[4].set_title("Step 2: Locale (Inpaint)"); ax[4].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"comparison_{base_name}.png"))
            plt.close()
        except Exception as e:
            print(f"⚠️ Errore nel plot per {base_name}: {e}")

    print(f"✅ Completato! Analizza i plot in: {out_dir}")

if __name__ == "__main__":
    main()