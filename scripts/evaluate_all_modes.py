import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import bdpy
from collections import defaultdict
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import lpips
import torchvision.transforms.functional as F_vision
import torchvision.models as models
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModelWithProjection
from pathlib import Path

# Per caricare il dataset GOD
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))
try:
    from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
    from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
except ImportError:
    print("Attenzione: moduli src non trovati. Assicurati di lanciare lo script dalla root del progetto.")

# --- MODELLI ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

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

# --- HELPER FUNCTIONS ---
def normalize_embed(emb): return emb / emb.norm(p=2, dim=-1, keepdim=True)

def predict_masks(fmri_tensor, model, threshold=0.5):
    with torch.no_grad(): depth_flat = model(fmri_tensor)
    mask_64 = depth_flat.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    return mask_norm[0, 0].cpu().numpy(), binary_64[0, 0].cpu().numpy().astype(np.uint8)

def calc_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return intersection / union

def get_n_way(g_list, t_list):
    if len(g_list) == 0: return 0.0, 0.0
    sim = torch.matmul(torch.cat(g_list, dim=0), torch.cat(t_list, dim=0).T)
    t1, t5 = 0, 0
    for i in range(len(g_list)):
        rank = (torch.argsort(sim[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        if rank == 1: t1 += 1
        if rank <= 5: t5 += 1
    return (t1 / len(g_list)) * 100, (t5 / len(g_list)) * 100

def apply_smart_mask(img_pil, mask_np, bbox, mode):
    """Applica BBox Crop per FG e Heavy Blur per BG"""
    if mode == "full" or mask_np is None:
        return img_pil
        
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    
    if mode == "foreground":
        # Composizione col nero
        black_bg = Image.new("RGB", (512, 512), "black")
        isolated = Image.composite(img_pil, black_bg, mask_pil.convert("L"))
        # Crop alla Bounding Box e rescale per massimizzare il soggetto
        if bbox:
            isolated = isolated.crop(bbox).resize((512, 512), Image.BICUBIC)
        return isolated
        
    elif mode == "background":
        # Blur pesante su tutta l'immagine
        blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=30))
        inv_mask = ImageOps.invert(mask_pil.convert("L"))
        # Incolliamo lo sfondo nitido sopra l'immagine sfocata
        return Image.composite(img_pil, blurred_img, inv_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["god", "shen"], required=True)
    parser.add_argument("--test_type", choices=["natural", "shapes", "letters", "imagery"], default="natural")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="VC") 
    parser.add_argument("--eval_mode", choices=["full", "foreground", "background"], default="full")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    masks_dir = f"{base_dir}/data/precomputed_binary_masks"
    
    # 1. Configurazione Percorsi
    if args.dataset == "shen":
        parent_dir = f"reconstructions/S{args.subject}_{args.test_type}_ALL_VERSIONS"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
        gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"
        if args.test_type == "natural": h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTest_fmriprep_volume_native.h5"
        elif args.test_type == "shapes": h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ArtificialShapes_fmriprep_volume_native.h5"
        elif args.test_type == "letters": h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_perceptionLetterImage_original_VC.h5"
        else: h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_Imagery_fmriprep_volume_native.h5"
    else: 
        parent_dir = f"results_sdxl_split_all/S{args.subject}_{args.roi}"
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv" 
        gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"

    # Mappatura Nomi
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

    # 2. Estrazione Dati fMRI 
    print("🧠 Estrazione tensori fMRI per calcolo metriche pure...")
    fmri_dict = {}
    if args.dataset == "shen":
        bdata = bdpy.BData(h5_path)
        X_raw = bdata.select(args.roi)
        input_dim = X_raw.shape[1]
        stim_key = 'imagery_image_index' if args.test_type == 'imagery' else ('image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id')
        labels = bdata.select(stim_key)[:, 0]
        for uid in np.unique(labels[labels > 0]):
            idx = [i for i in np.where(labels == uid)[0] if i < X_raw.shape[0]]
            if len(idx) > 0: fmri_dict[f"id_{int(uid)}.png"] = np.mean(X_raw[idx], axis=0)
    else: # GOD
        roi_keys = parse_roi_keys(args.roi)
        input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
        rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
        raw_data = load_fmri_image_pairs(os.path.join(base_dir, "data/GOD_Dataset/fmri_files/test"), os.path.join(base_dir, "data/GOD_Dataset/images/test"), f"Subject{args.subject}_ImageNetTest.h5", rois_dict)
        grouped = defaultdict(list)
        for i, vec in enumerate(raw_data['fmri']): grouped[os.path.basename(raw_data['image_paths'][i])].append(vec)
        for fname, vecs in grouped.items(): fmri_dict[f"{fname}{ext}" if not fname.endswith(ext) else fname] = np.mean(vecs, axis=0)

    # 3. Caricamento Modelli 
    print(f"Inizializzazione Modelli (Modo: {args.eval_mode.upper()})...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    clip_processor_b = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model_b = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor_xl = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    clip_model_xl = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device).eval()
    alexnet = AlexNetExtractor().to(device)
    import torchvision.transforms as T
    alex_transform = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    aligner_bg = SimpleLinearAligner(input_dim, 1280).to(device)
    aligner_fg = SimpleLinearAligner(input_dim, 1280).to(device)
    aligner_struct = SimpleLinearAligner(input_dim, 4096).to(device)

    if args.dataset == "shen":
        aligner_bg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse_background/S{args.subject}_clip_aligner_ba.pth", map_location=device))
        aligner_fg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_clip_sparse_foreground/S{args.subject}_clip_aligner_fo.pth", map_location=device))
        aligner_struct.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/shen_depth_sparse/S{args.subject}_depth_aligner.pth", map_location=device))
    else:
        aligner_bg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/linear_sparse_lbfgs_background/S{args.subject}_{args.roi}/best_linear_aligner.pth", map_location=device))
        aligner_fg.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/linear_sparse_lbfgs_foreground/S{args.subject}_{args.roi}/best_linear_aligner.pth", map_location=device))
        aligner_struct.load_state_dict(torch.load(f"{base_dir}/trained_fmri_decoders/binary_decoder/S{args.subject}_{args.roi}/best_depth_decoder.pth", map_location=device))

    aligner_bg.eval(); aligner_fg.eval(); aligner_struct.eval()

    methods = ["bg_only", "fg_only", "inpaint_mask", "inpaint_box", "attn_mask", "attn_box"]
    results = {m: {'lpips': [], 'clip_b_g': [], 'clip_b_t': [], 'clip_xl_g': [], 'clip_xl_t': [], 'alex2_g': [], 'alex2_t': [], 'alex5_g': [], 'alex5_t': [], 'alex7_g': [], 'alex7_t': []} for m in methods}
    
    pred_sem_fg_list, pred_sem_bg_list, gt_sem_xl_list = [], [], []
    pred_mask_cont_list, pred_mask_bin_list, gt_mask_bin_list = [], [], []

    valid_files = [f for f in os.listdir(os.path.join(parent_dir, "bg_only")) if f in id_to_filename and f in fmri_dict]
    N = 0

    print("Calcolo Metriche...")
    for f in tqdm(valid_files):
        gt_path = os.path.join(gt_dir, id_to_filename[f])
        if not os.path.exists(gt_path): continue
        gt_pil = Image.open(gt_path).convert("RGB").resize((512, 512))
        
        mask_np, bbox = None, None
        if args.test_type != "imagery":
            mask_path = os.path.join(masks_dir, id_to_maskname[f])
            if os.path.exists(mask_path): 
                mask_np = np.load(mask_path).reshape(64, 64)
                # Estrazione BBox sulla maschera 64x64 poi scalata a 512
                mask_pil_64 = Image.fromarray((mask_np * 255).astype(np.uint8))
                bbox_64 = mask_pil_64.getbbox()
                if bbox_64: bbox = (bbox_64[0]*8, bbox_64[1]*8, bbox_64[2]*8, bbox_64[3]*8)

        # Smart Masking sulla Ground Truth
        gt_pil = apply_smart_mask(gt_pil, mask_np, bbox, args.eval_mode)

        N += 1
        fmri_input = torch.tensor(fmri_dict[f], dtype=torch.float32).to(device).unsqueeze(0)
        
        with torch.no_grad():
            # --- PURE FMRI EVALUATION ---
            pred_emb_bg = normalize_embed(aligner_bg(fmri_input)[0])
            pred_emb_fg = normalize_embed(aligner_fg(fmri_input)[0])
            t_gt_clip_xl = normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_pil, return_tensors="pt").to(device)).image_embeds).cpu()
            
            pred_sem_bg_list.append(pred_emb_bg.cpu())
            pred_sem_fg_list.append(pred_emb_fg.cpu())
            gt_sem_xl_list.append(t_gt_clip_xl)

            if mask_np is not None:
                p_cont, p_bin = predict_masks(fmri_input, aligner_struct)
                pred_mask_cont_list.append(p_cont)
                pred_mask_bin_list.append(p_bin)
                gt_mask_bin_list.append((mask_np > 0.5).astype(np.uint8))

            # --- IMAGE METRICS EVALUATION ---
            t_gt_lpips = (F_vision.to_tensor(gt_pil).to(device) * 2.0 - 1.0).unsqueeze(0)
            t_gt_clip_b = normalize_embed(clip_model_b.get_image_features(**clip_processor_b(images=gt_pil, return_tensors="pt").to(device))).cpu()
            a2_gt, a5_gt, a7_gt = alexnet(alex_transform(gt_pil).unsqueeze(0).to(device))
            a2_gt, a5_gt, a7_gt = normalize_embed(a2_gt).cpu(), normalize_embed(a5_gt).cpu(), normalize_embed(a7_gt).cpu()

            for m in methods:
                gen_path = os.path.join(parent_dir, m, f)
                if not os.path.exists(gen_path): continue
                gen_pil = Image.open(gen_path).convert("RGB").resize((512, 512))
                
                # Smart Masking anche sull'immagine generata
                gen_pil = apply_smart_mask(gen_pil, mask_np, bbox, args.eval_mode)

                t_gen_lpips = (F_vision.to_tensor(gen_pil).to(device) * 2.0 - 1.0).unsqueeze(0)
                results[m]['lpips'].append(loss_fn_vgg(t_gen_lpips, t_gt_lpips).item())
                results[m]['clip_b_g'].append(normalize_embed(clip_model_b.get_image_features(**clip_processor_b(images=gen_pil, return_tensors="pt").to(device))).cpu())
                results[m]['clip_b_t'].append(t_gt_clip_b)
                results[m]['clip_xl_g'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gen_pil, return_tensors="pt").to(device)).image_embeds).cpu())
                results[m]['clip_xl_t'].append(t_gt_clip_xl)
                
                a2_gen, a5_gen, a7_gen = alexnet(alex_transform(gen_pil).unsqueeze(0).to(device))
                results[m]['alex2_g'].append(normalize_embed(a2_gen).cpu())
                results[m]['alex2_t'].append(a2_gt)
                results[m]['alex5_g'].append(normalize_embed(a5_gen).cpu())
                results[m]['alex5_t'].append(a5_gt)
                results[m]['alex7_g'].append(normalize_embed(a7_gen).cpu())
                results[m]['alex7_t'].append(a7_gt)

    if N == 0: return print("❌ Nessuna immagine trovata.")

    # --- Helper Generici per Matrici e Top-K ---
    def get_sim_matrix(pred_list, gt_list):
        if len(pred_list) == 0: return np.zeros((0,0))
        preds = torch.stack([p.view(-1) for p in pred_list])
        gts = torch.stack([g.view(-1) for g in gt_list])
        return torch.matmul(preds, gts.T).cpu().numpy()

    def calc_topk_from_matrix(score_matrix, reverse=True):
        t1, t5 = 0, 0
        N_items = score_matrix.shape[0]
        if N_items == 0: return 0.0, 0.0
        for i in range(N_items):
            rank = sorted(range(N_items), key=lambda x: score_matrix[i, x], reverse=reverse).index(i) + 1
            if rank == 1: t1 += 1
            if rank <= 5: t5 += 1
        return (t1/N_items)*100, (t5/N_items)*100

    # --- Calcolo Metriche Pure fMRI Semantic (Solo Cosine) ---
    sim_matrix_bg = get_sim_matrix(pred_sem_bg_list, gt_sem_xl_list)
    sim_matrix_fg = get_sim_matrix(pred_sem_fg_list, gt_sem_xl_list)

    t1_sem_cos_bg, t5_sem_cos_bg = calc_topk_from_matrix(sim_matrix_bg, reverse=True)
    t1_sem_cos_fg, t5_sem_cos_fg = calc_topk_from_matrix(sim_matrix_fg, reverse=True)

    # --- Calcolo Metriche Pure fMRI Spatial e Joint ---
    ious, mses, cosines = [], [], []
    t1_spa_iou, t5_spa_iou, t1_spa_cos, t5_spa_cos, t1_spa_mse, t5_spa_mse = 0, 0, 0, 0, 0, 0
    N_spa = len(pred_mask_bin_list)
    
    mse_matrix_spa = np.zeros((N_spa, N_spa))

    if N_spa > 0:
        for i in range(N_spa):
            p_cont, p_bin = pred_mask_cont_list[i], pred_mask_bin_list[i]
            p_flat = p_cont.flatten()
            sims_iou, sims_cos = [], []
            
            g_bin_i = gt_mask_bin_list[i]
            g_flat_true = g_bin_i.flatten().astype(np.float32)
            ious.append(calc_iou(p_bin, g_bin_i))
            mses.append(np.mean((p_cont - g_bin_i)**2))
            norm_p, norm_g_true = np.linalg.norm(p_flat), np.linalg.norm(g_flat_true)
            cosines.append(np.dot(p_flat, g_flat_true) / (norm_p * norm_g_true) if norm_p > 0 and norm_g_true > 0 else 0.0)

            for j in range(N_spa):
                g_bin = gt_mask_bin_list[j]
                g_flat = g_bin.flatten().astype(np.float32)
                sims_iou.append(calc_iou(p_bin, g_bin))
                
                # Popoliamo la matrice N-way degli MSE Spaziali
                mse_matrix_spa[i, j] = np.mean((p_cont - g_bin)**2)
                
                norm_g = np.linalg.norm(g_flat)
                sims_cos.append(np.dot(p_flat, g_flat) / (norm_p * norm_g) if norm_p > 0 and norm_g > 0 else 0.0)
                
            rank_iou = sorted(range(N_spa), key=lambda x: sims_iou[x], reverse=True).index(i) + 1
            if rank_iou == 1: t1_spa_iou += 1
            if rank_iou <= 5: t5_spa_iou += 1
            
            rank_cos = sorted(range(N_spa), key=lambda x: sims_cos[x], reverse=True).index(i) + 1
            if rank_cos == 1: t1_spa_cos += 1
            if rank_cos <= 5: t5_spa_cos += 1
            
        t1_spa_mse, t5_spa_mse = calc_topk_from_matrix(mse_matrix_spa, reverse=False)

        # --- JOINT METRICS (Z-Score Normalized) ---
        def z_score_matrix(mat):
            means = mat.mean(axis=1, keepdims=True)
            stds = mat.std(axis=1, keepdims=True) + 1e-8
            return (mat - means) / stds

        # Normalizzazioni
        z_sim_bg = z_score_matrix(sim_matrix_bg)
        z_sim_fg = z_score_matrix(sim_matrix_fg)
        z_mse_spa = -z_score_matrix(mse_matrix_spa) # Moltiplicato per -1 così diventa uno "Score" (Più alto = Meglio)

        # 1. Joint Semantica Pura: Bg + Fg
        t1_j_sem, t5_j_sem = calc_topk_from_matrix(z_sim_bg + z_sim_fg, reverse=True)

        # 2. Joint Classica: Semantica singola + Spazialità
        t1_j_bg_spa, t5_j_bg_spa = calc_topk_from_matrix(z_sim_bg + z_mse_spa, reverse=True)
        t1_j_fg_spa, t5_j_fg_spa = calc_topk_from_matrix(z_sim_fg + z_mse_spa, reverse=True)
        
        # 3. TRIS Supremo: Bg + Fg + Posizione
        t1_tris, t5_tris = calc_topk_from_matrix(z_sim_bg + z_sim_fg + z_mse_spa, reverse=True)

    # --- STAMPE ---
    print("\n" + "="*115)
    print(f"TABELLA 1: METRICHE D'IMMAGINE (Ablazione) - {args.dataset.upper()} S{args.subject} | MODO: {args.eval_mode.upper()} | N={N}")
    print("="*115)
    print(f"{'Metodo':<14} | {'LPIPS ↓':<7} | {'CLIP-B (T1/T5)':<16} | {'CLIP-XL (T1/T5)':<16} | {'Alex2 (T1/T5)':<15} | {'Alex5 (T1/T5)':<15} | {'Alex7 (T1/T5)':<15}")
    print("-" * 115)
    for m in methods:
        avg_lpips = np.mean(results[m]['lpips']) if len(results[m]['lpips']) > 0 else 0
        cb_t1, cb_t5 = get_n_way(results[m]['clip_b_g'], results[m]['clip_b_t'])
        cxl_t1, cxl_t5 = get_n_way(results[m]['clip_xl_g'], results[m]['clip_xl_t'])
        a2_t1, a2_t5 = get_n_way(results[m]['alex2_g'], results[m]['alex2_t'])
        a5_t1, a5_t5 = get_n_way(results[m]['alex5_g'], results[m]['alex5_t'])
        a7_t1, a7_t5 = get_n_way(results[m]['alex7_g'], results[m]['alex7_t'])
        print(f"{m:<14} | {avg_lpips:05.4f} | {cb_t1:05.2f}%/{cb_t5:05.2f}% | {cxl_t1:05.2f}%/{cxl_t5:05.2f}% | {a2_t1:05.2f}%/{a2_t5:05.2f}% | {a5_t1:05.2f}%/{a5_t5:05.2f}% | {a7_t1:05.2f}%/{a7_t5:05.2f}%")
    
    print("\n" + "="*95)
    print(f"TABELLA 2: METRICHE PURE fMRI E JOINT - {args.dataset.upper()} S{args.subject} | MODO: {args.eval_mode.upper()} | N={N}")
    print("="*95)
    print(f"{'Metrica N-way':<65} | {'Top-1':<10} | {'Top-5':<10}")
    print("-" * 95)
    print(f"{'Pure Semantic Background (Cosine Sim)':<65} | {t1_sem_cos_bg:05.2f}%    | {t5_sem_cos_bg:05.2f}%")
    print(f"{'Pure Semantic Foreground (Cosine Sim)':<65} | {t1_sem_cos_fg:05.2f}%    | {t5_sem_cos_fg:05.2f}%")
    if N_spa > 0:
        print("-" * 95)
        print(f"{'Pure Spatial (IoU su Maschera Binaria)':<65} | {(t1_spa_iou/N_spa)*100:05.2f}%    | {(t5_spa_iou/N_spa)*100:05.2f}%")
        print(f"{'Pure Spatial (Cosine Sim su Mappa Continua)':<65} | {(t1_spa_cos/N_spa)*100:05.2f}%    | {(t5_spa_cos/N_spa)*100:05.2f}%")
        print(f"{'Pure Spatial (MSE su Mappa Continua)':<65} | {t1_spa_mse:05.2f}%    | {t5_spa_mse:05.2f}%")
        print("-" * 95)
        print(f"{'Joint Semantic [Bg + Fg]':<65} | {t1_j_sem:05.2f}%    | {t5_j_sem:05.2f}%")
        print(f"{'Joint Mix Bg [Bg + Position]':<65} | {t1_j_bg_spa:05.2f}%    | {t5_j_bg_spa:05.2f}%")
        print(f"{'Joint Mix Fg [Fg + Position]':<65} | {t1_j_fg_spa:05.2f}%    | {t5_j_fg_spa:05.2f}%")
        print(f"{'TRIS [Bg + Fg + Position]':<65} | {t1_tris:05.2f}%    | {t5_tris:05.2f}%")
        print("=" * 95)
        print(f"Medie Globali Spaziali: IoU: {np.mean(ious):.4f} | Cosine: {np.mean(cosines):.4f} | MSE: {np.mean(mses):.4f}")
    print("=" * 95 + "\n")

if __name__ == "__main__": main()