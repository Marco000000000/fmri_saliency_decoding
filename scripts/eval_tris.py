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
    parser.add_argument("--gen_dir", type=str, required=True, help="Dir img generate SOLO OLISTICO (Full)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    masks_dir = f"{base_dir}/data/precomputed_binary_masks"
    
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
        'clip_xl': {'t_full': [], 't_fg': [], 't_bg': [], 'g_holistic': []},
        'alex5':   {'t_full': [], 't_fg': [], 't_bg': [], 'g_holistic': []},
        'alex7':   {'t_full': [], 't_fg': [], 't_bg': [], 'g_holistic': []}
    }

    def find_img(directory, b_name):
        for ext_test in [".png", ".JPEG", ".jpg"]:
            p = os.path.join(directory, b_name + ext_test)
            if os.path.exists(p): return p
        return None

    valid_pairs = []
    for gt_key, gt_file in id_to_filename.items():
        base_name = os.path.splitext(gt_key)[0]
        if args.dataset == "shen":
            base_name = gt_key.replace(".png", "")

        gt_path = os.path.join(gt_dir, gt_file)
        gen_path = find_img(args.gen_dir, base_name)

        if os.path.exists(gt_path) and gen_path:
            valid_pairs.append((gt_path, gen_path, gt_key))

    print(f"🔄 Avvio ESTRAZIONE FEATURE su {len(valid_pairs)} immagini Olistiche (INTATTE)...")
    for gt_path, gen_path, gt_key in tqdm(valid_pairs):
        
        gt_pil = Image.open(gt_path).convert("RGB").resize((512, 512))
        gen_pil = Image.open(gen_path).convert("RGB").resize((512, 512)) # IMMAGINE GENERATA INTATTA
        
        mask_np, bbox = None, None
        mask_path = os.path.join(masks_dir, id_to_maskname[gt_key])
        if os.path.exists(mask_path): 
            mask_np = np.load(mask_path).reshape(64, 64)
            bbox_64 = Image.fromarray((mask_np * 255).astype(np.uint8)).getbbox()
            if bbox_64: bbox = (bbox_64[0]*8, bbox_64[1]*8, bbox_64[2]*8, bbox_64[3]*8)

        # MASCHERIAMO SOLO LE GROUND TRUTH
        t_full_pil = apply_smart_mask(gt_pil, mask_np, bbox, "full")
        t_fg_pil = apply_smart_mask(gt_pil, mask_np, bbox, "foreground")
        t_bg_pil = apply_smart_mask(gt_pil, mask_np, bbox, "background")

        with torch.no_grad():
            def extract_all(pil_img):
                cxl = normalize_embed(clip_model_xl(**clip_processor_xl(images=pil_img, return_tensors="pt").to(device)).image_embeds).cpu()
                _, a5, a7 = alexnet(alex_transform(pil_img).unsqueeze(0).to(device))
                return cxl, normalize_embed(a5).cpu(), normalize_embed(a7).cpu()

            # Estrazione Target Mascherati
            cxl_t_full, a5_t_full, a7_t_full = extract_all(t_full_pil)
            cxl_t_fg, a5_t_fg, a7_t_fg = extract_all(t_fg_pil)
            cxl_t_bg, a5_t_bg, a7_t_bg = extract_all(t_bg_pil)

            # Estrazione UNICA per il modello Olistico Intatto
            cxl_g_h, a5_g_h, a7_g_h = extract_all(gen_pil)

            metrics_data = [
                ('clip_xl', cxl_t_full, cxl_t_fg, cxl_t_bg, cxl_g_h),
                ('alex5',   a5_t_full,   a5_t_fg,   a5_t_bg,   a5_g_h),
                ('alex7',   a7_t_full,   a7_t_fg,   a7_t_bg,   a7_g_h)
            ]

            for m_name, t_ful, t_fg, t_bg, g_h in metrics_data:
                results[m_name]['t_full'].append(t_ful)
                results[m_name]['t_fg'].append(t_fg)
                results[m_name]['t_bg'].append(t_bg)
                results[m_name]['g_holistic'].append(g_h)

    print("\n🏆 --- RISULTATI LATE FUSION (SOLO MODELLO OLISTICO INTATTO) --- 🏆")
    
    for metric_name in ['clip_xl', 'alex5', 'alex7']:
        d = results[metric_name]
        
        t_full = torch.cat(d['t_full'], dim=0)
        t_fg   = torch.cat(d['t_fg'], dim=0)
        t_bg   = torch.cat(d['t_bg'], dim=0)
        
        # L'unica immagine olistica
        g_holistic = torch.cat(d['g_holistic'], dim=0)

        # La singola immagine valutata contro le 3 lenti
        sim_h_full = torch.matmul(g_holistic, t_full.T)
        sim_h_fg   = torch.matmul(g_holistic, t_fg.T)
        sim_h_bg   = torch.matmul(g_holistic, t_bg.T)

        # Fusioni a Coppia (Somma Logits)
        sim_full_fg = sim_h_full + sim_h_fg
        sim_full_bg = sim_h_full + sim_h_bg
        sim_fg_bg   = sim_h_fg + sim_h_bg

        # Grand Mix Ensemble Olistico
        sim_holistic_ensemble = sim_h_full + sim_h_fg + sim_h_bg

        # Calcolo Top-1 e Top-5 per tutte le combinazioni
        res_full = get_accuracy_from_sim(sim_h_full)
        res_fg   = get_accuracy_from_sim(sim_h_fg)
        res_bg   = get_accuracy_from_sim(sim_h_bg)
        
        res_full_fg = get_accuracy_from_sim(sim_full_fg)
        res_full_bg = get_accuracy_from_sim(sim_full_bg)
        res_fg_bg   = get_accuracy_from_sim(sim_fg_bg)
        
        res_all = get_accuracy_from_sim(sim_holistic_ensemble)

        print(f"\n--- {metric_name.upper()} (Top1 / Top5) ---")
        print(f"Valutazione Immagine Intatta contro 3 Target:")
        print(f"Singoli -> vs FULL: {res_full[0]:.2f}%/{res_full[1]:.2f}% | vs FG: {res_fg[0]:.2f}%/{res_fg[1]:.2f}% | vs BG: {res_bg[0]:.2f}%/{res_bg[1]:.2f}%")
        print(f"Coppie  -> FULL+FG: {res_full_fg[0]:.2f}%/{res_full_fg[1]:.2f}% | FULL+BG: {res_full_bg[0]:.2f}%/{res_full_bg[1]:.2f}% | FG+BG: {res_fg_bg[0]:.2f}%/{res_fg_bg[1]:.2f}%")
        print(f"Grand Mix Olistico -> FULL+FG+BG: {res_all[0]:.2f}% / {res_all[1]:.2f}%")

if __name__ == "__main__": main()