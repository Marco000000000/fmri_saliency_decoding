import argparse
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import torchvision.transforms.functional as F_vision
import torchvision.models as models
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

class AlexNetExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier[:-1] 
    def forward(self, x):
        with torch.no_grad():
            x2 = self.features[:6](x).flatten(1)
            x5 = self.features[:13](x).flatten(1)
            x_full = self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))
        return x2, x5, x_full

def normalize_embed(emb): 
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

def apply_smart_mask(img_pil, mask_np, bbox, mode):
    if mode == "full" or mask_np is None: return img_pil
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(img_pil.size, Image.NEAREST)
    blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=30))
    if mode == "foreground":
        fg_isolated = Image.composite(img_pil, blurred_img, mask_pil.convert("L"))
        if bbox: return fg_isolated.crop(bbox).resize(img_pil.size, Image.BICUBIC)
        return fg_isolated
    elif mode == "background":
        inv_mask = ImageOps.invert(mask_pil.convert("L"))
        return Image.composite(img_pil, blurred_img, inv_mask)

def get_accuracy_from_sim(sim_matrix):
    """Calcola Top-1 e Top-5 N-way a partire dalla matrice di similarità pre-calcolata"""
    if len(sim_matrix) == 0: return 0.0, 0.0
    t1, t5 = 0, 0
    n = len(sim_matrix)
    for i in range(n):
        rank = (torch.argsort(sim_matrix[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        if rank == 1: t1 += 1
        if rank <= 5: t5 += 1
    return (t1 / n) * 100, (t5 / n) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["god", "shen"], required=True)
    parser.add_argument("--gen_full", type=str, required=True, help="Dir img generate FULL")
    parser.add_argument("--gen_fg", type=str, required=True, help="Dir img generate FG")
    parser.add_argument("--gen_bg", type=str, required=True, help="Dir img generate BG")
    args = parser.parse_args()

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    masks_dir = f"{base_dir}/data/precomputed_binary_masks"
    
    # Setup Dataset Paths
    tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
    gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
    ext = ".JPEG"

    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    id_to_filename, id_to_maskname = {}, {}
    for _, row in df_tsv.iterrows():
        img_id, fname = int(row[3]), str(row[0]).strip()
        if args.dataset == "god":
            key = f"{fname}{ext}" if not fname.endswith(ext) else fname
            id_to_filename[key] = key
            id_to_maskname[key] = f"{fname}.npy"
        else:
            key = f"id_{img_id}.png"
            id_to_filename[key] = f"{fname}{ext}" if not fname.endswith(ext) else fname
            id_to_maskname[key] = f"{fname}.npy"

    clip_processor_xl = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    clip_model_xl = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device).eval()
    alexnet = AlexNetExtractor().to(device)
    import torchvision.transforms as T
    alex_transform = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    results = {
        'clip_xl_t_full': [], 'clip_xl_t_fg': [], 'clip_xl_t_bg': [],
        'clip_xl_g_full': [], 'clip_xl_g_fg': [], 'clip_xl_g_bg': [],
        'alex5_t_full': [], 'alex5_t_fg': [], 'alex5_t_bg': [],
        'alex5_g_full': [], 'alex5_g_fg': [], 'alex5_g_bg': [],
        'alex7_t_full': [], 'alex7_t_fg': [], 'alex7_t_bg': [],
        'alex7_g_full': [], 'alex7_g_fg': [], 'alex7_g_bg': [],
    }

    # Helper function per cercare file ignorando le estensioni
    def find_img(directory, b_name):
        for ext_test in [".png", ".JPEG", ".jpg"]:
            p = os.path.join(directory, b_name + ext_test)
            if os.path.exists(p): return p
        return None

    # Costruiamo la lista di accoppiamenti validi
    valid_pairs = []
    for gt_key, gt_file in id_to_filename.items():
        base_name = os.path.splitext(gt_key)[0]
        if args.dataset == "shen":
            base_name = gt_key.replace(".png", "")

        gt_path = os.path.join(gt_dir, gt_file)
        path_full = find_img(args.gen_full, base_name)
        path_fg = find_img(args.gen_fg, base_name)
        path_bg = find_img(args.gen_bg, base_name)

        if os.path.exists(gt_path) and path_full and path_fg and path_bg:
            valid_pairs.append((gt_path, path_full, path_fg, path_bg, gt_key))

    if not valid_pairs:
        print("❌ ERRORE FATALE: Nessun match trovato tra Ground Truth e le 3 cartelle generate!")
        return

    print(f"🔄 Avvio ESTRAZIONE FEATURE su {len(valid_pairs)} immagini x 3 modalità...")
    for gt_path, path_full, path_fg, path_bg, gt_key in tqdm(valid_pairs):
        
        gt_pil = Image.open(gt_path).convert("RGB").resize((512, 512))
        gen_full = Image.open(path_full).convert("RGB").resize((512, 512))
        gen_fg = Image.open(path_fg).convert("RGB").resize((512, 512))
        gen_bg = Image.open(path_bg).convert("RGB").resize((512, 512))
        
        mask_np, bbox = None, None
        mask_path = os.path.join(masks_dir, id_to_maskname[gt_key])
        if os.path.exists(mask_path): 
            mask_np = np.load(mask_path).reshape(64, 64)
            bbox_64 = Image.fromarray((mask_np * 255).astype(np.uint8)).getbbox()
            if bbox_64: bbox = (bbox_64[0]*8, bbox_64[1]*8, bbox_64[2]*8, bbox_64[3]*8)

        # Creazione dei 3 target di Ground Truth mascherati
        gt_full_pil = apply_smart_mask(gt_pil, mask_np, bbox, "full")
        gt_fg_pil = apply_smart_mask(gt_pil, mask_np, bbox, "foreground")
        gt_bg_pil = apply_smart_mask(gt_pil, mask_np, bbox, "background")

        with torch.no_grad():
            # Estrazione CLIP XL
            results['clip_xl_t_full'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_full_pil, return_tensors="pt").to(device)).image_embeds).cpu())
            results['clip_xl_t_fg'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_fg_pil, return_tensors="pt").to(device)).image_embeds).cpu())
            results['clip_xl_t_bg'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_bg_pil, return_tensors="pt").to(device)).image_embeds).cpu())
            
            results['clip_xl_g_full'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gen_full, return_tensors="pt").to(device)).image_embeds).cpu())
            results['clip_xl_g_fg'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gen_fg, return_tensors="pt").to(device)).image_embeds).cpu())
            results['clip_xl_g_bg'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gen_bg, return_tensors="pt").to(device)).image_embeds).cpu())

            # Estrazione AlexNet (5 e 7)
            _, a5_gt_full, a7_gt_full = alexnet(alex_transform(gt_full_pil).unsqueeze(0).to(device))
            _, a5_gt_fg, a7_gt_fg = alexnet(alex_transform(gt_fg_pil).unsqueeze(0).to(device))
            _, a5_gt_bg, a7_gt_bg = alexnet(alex_transform(gt_bg_pil).unsqueeze(0).to(device))
            
            _, a5_g_full, a7_g_full = alexnet(alex_transform(gen_full).unsqueeze(0).to(device))
            _, a5_g_fg, a7_g_fg = alexnet(alex_transform(gen_fg).unsqueeze(0).to(device))
            _, a5_g_bg, a7_g_bg = alexnet(alex_transform(gen_bg).unsqueeze(0).to(device))

            results['alex5_t_full'].append(normalize_embed(a5_gt_full).cpu()); results['alex5_g_full'].append(normalize_embed(a5_g_full).cpu())
            results['alex5_t_fg'].append(normalize_embed(a5_gt_fg).cpu());   results['alex5_g_fg'].append(normalize_embed(a5_g_fg).cpu())
            results['alex5_t_bg'].append(normalize_embed(a5_gt_bg).cpu());   results['alex5_g_bg'].append(normalize_embed(a5_g_bg).cpu())

            results['alex7_t_full'].append(normalize_embed(a7_gt_full).cpu()); results['alex7_g_full'].append(normalize_embed(a7_g_full).cpu())
            results['alex7_t_fg'].append(normalize_embed(a7_gt_fg).cpu());   results['alex7_g_fg'].append(normalize_embed(a7_g_fg).cpu())
            results['alex7_t_bg'].append(normalize_embed(a7_gt_bg).cpu());   results['alex7_g_bg'].append(normalize_embed(a7_g_bg).cpu())

    print("\n🏆 --- RISULTATI JOINT ENSEMBLE (LATE FUSION) ABLATION --- 🏆")
    
    for metric_name, t_full_k, t_fg_k, t_bg_k, g_full_k, g_fg_k, g_bg_k in [
        ("CLIP-XL", 'clip_xl_t_full', 'clip_xl_t_fg', 'clip_xl_t_bg', 'clip_xl_g_full', 'clip_xl_g_fg', 'clip_xl_g_bg'),
        ("Alex5",   'alex5_t_full',   'alex5_t_fg',   'alex5_t_bg',   'alex5_g_full',   'alex5_g_fg',   'alex5_g_bg'),
        ("Alex7",   'alex7_t_full',   'alex7_t_fg',   'alex7_t_bg',   'alex7_g_full',   'alex7_g_fg',   'alex7_g_bg')
    ]:
        # Concatenazione Tensori
        g_full = torch.cat(results[g_full_k], dim=0)
        g_fg = torch.cat(results[g_fg_k], dim=0)
        g_bg = torch.cat(results[g_bg_k], dim=0)
        
        t_full = torch.cat(results[t_full_k], dim=0)
        t_fg = torch.cat(results[t_fg_k], dim=0)
        t_bg = torch.cat(results[t_bg_k], dim=0)

        # Matrici di Similarità di Base
        sim_full = torch.matmul(g_full, t_full.T)
        sim_fg = torch.matmul(g_fg, t_fg.T)
        sim_bg = torch.matmul(g_bg, t_bg.T)

        # Combinazioni Late Fusion (Somma Logits)
        sim_full_fg = sim_full + sim_fg
        sim_full_bg = sim_full + sim_bg
        sim_fg_bg   = sim_fg + sim_bg
        sim_all     = sim_full + sim_fg + sim_bg

        # Calcolo Top-1 e Top-5
        res_full = get_accuracy_from_sim(sim_full)
        res_fg = get_accuracy_from_sim(sim_fg)
        res_bg = get_accuracy_from_sim(sim_bg)
        
        res_full_fg = get_accuracy_from_sim(sim_full_fg)
        res_full_bg = get_accuracy_from_sim(sim_full_bg)
        res_fg_bg   = get_accuracy_from_sim(sim_fg_bg)
        res_all     = get_accuracy_from_sim(sim_all)

        print(f"\n--- {metric_name} ---")
        print(f"Singoli   -> FULL: {res_full[0]:.2f}% | FG: {res_fg[0]:.2f}% | BG: {res_bg[0]:.2f}%")
        print(f"Coppie    -> FULL+FG: {res_full_fg[0]:.2f}% | FULL+BG: {res_full_bg[0]:.2f}% | FG+BG: {res_fg_bg[0]:.2f}%")
        print(f"Grand Mix -> FULL+FG+BG: {res_all[0]:.2f}% (Top1) / {res_all[1]:.2f}% (Top5)")

if __name__ == "__main__": main()