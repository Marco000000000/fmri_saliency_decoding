import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as F_vision
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModelWithProjection
from collections import defaultdict
from pathlib import Path

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# --- CLASSI MODELLO LINEARE ---
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

# --- ESTRATTORE ALEXNET (Livelli 2, 5 e 7) ---
class AlexNetExtractor(nn.Module):
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
            x_full = self.features(x)
            x_full = self.avgpool(x_full)
            x_full = torch.flatten(x_full, 1)
            x7 = self.classifier(x_full)
        return x2, x5, x7

# --- HELPER MASCHERE ---
def load_gt_masks(npy_path):
    if not os.path.exists(npy_path): return None
    mask_64 = np.load(npy_path).reshape(64, 64)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    mask_bin_64 = (mask_norm > 0.5).astype(np.uint8)
    return mask_bin_64

def predict_masks(fmri_tensor, model, threshold=0.6):
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

def normalize_embed(emb):
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

def get_n_way(g_list, t_list):
    sim = torch.matmul(torch.cat(g_list, dim=0), torch.cat(t_list, dim=0).T)
    t1, t5 = 0, 0
    for i in range(len(g_list)):
        rank = (torch.argsort(sim[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        if rank == 1: t1 += 1
        if rank <= 5: t5 += 1
    return (t1 / len(g_list)) * 100, (t5 / len(g_list)) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--gt_dir", type=str, default="data/GOD_Dataset/images/test")
    parser.add_argument("--gt_masks_dir", type=str, default="data/precomputed_binary_masks")
    parser.add_argument("--gen_dir", type=str, required=True, help="Percorso alle immagini generate")
    parser.add_argument("--mask_threshold", type=float, default=0.6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Calcolo Metriche (Pura Gen + Pura fMRI) su GOD Dataset | S{args.subject} | {args.rois} ---")

    # ==========================================
    # 1. INIZIALIZZAZIONE MODELLI DI VALUTAZIONE
    # ==========================================
    print("Inizializzazione modelli (LPIPS, CLIP Base, CLIP XL, AlexNet)...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    
    clip_processor_base = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    clip_processor_xl = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    clip_model_xl = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device).eval()

    alexnet = AlexNetExtractor().to(device)
    alex_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 2. CARICAMENTO MODELLI FMRI E DATI GOD
    # ==========================================
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    aligner_sem = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
    path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs_mse/S{args.subject}_{args.rois}/best_linear_aligner.pth"
    aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
    aligner_sem.eval()

    aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
    path_struct = f"trained_fmri_decoders/binary_decoder/S{args.subject}_{args.rois}/best_depth_decoder.pth"
    aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
    aligner_struct.eval()

    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"), test_h5, rois_dict
    )
    
    grouped_fmri = defaultdict(list)
    for i, vec in enumerate(raw_data['fmri']):
        fname = os.path.basename(raw_data['image_paths'][i])
        base_id = fname.split('_')[0] + '_' + fname.split('_')[1].split('.')[0]
        grouped_fmri[base_id].append(vec)

    # Creazione Coppie (Shared Data)
    gen_files = [f for f in os.listdir(args.gen_dir) if f.endswith(('.png', '.jpg', '.JPEG'))]
    shared_data = []
    for gen_name in gen_files:
        try: base_id = gen_name.split('_')[0] + '_' + gen_name.split('_')[1].split('.')[0]
        except: continue
        gt_name = next((f for f in os.listdir(args.gt_dir) if base_id in f), None)
        if gt_name and base_id in grouped_fmri:
            fmri_mean = torch.tensor(np.array(grouped_fmri[base_id]), dtype=torch.float32).to(device).mean(dim=0, keepdim=True)
            mask_path = os.path.join(args.gt_masks_dir, base_id + ".npy")
            shared_data.append({
                'fmri': fmri_mean, 
                'gen_path': os.path.join(args.gen_dir, gen_name), 
                'gt_path': os.path.join(args.gt_dir, gt_name), 
                'mask_path': mask_path
            })

    N = len(shared_data)
    if N == 0: return print("❌ Nessuna immagine valida trovata.")
    print(f"✅ Trovate {N} coppie per la valutazione.")

    # ==========================================
    # 3. ESTRAZIONE FEATURE
    # ==========================================
    metrics = {
        'lpips': [],
        'clip_base_gen': [], 'clip_base_gt': [],
        'clip_xl_gen': [], 'clip_xl_gt': [],
        'alex2_gen': [], 'alex2_gt': [],
        'alex5_gen': [], 'alex5_gt': [],
        'alex7_gen': [], 'alex7_gt': [],
    }
    
    pred_sem_list, gt_sem_1280_list = [], []
    pred_mask_cont_list, pred_mask_bin_list, gt_mask_bin_list = [], [], []

    print("⚙️ Estrazione feature e calcolo metriche...")
    for data in tqdm(shared_data):
        gen_pil = Image.open(data['gen_path']).convert("RGB").resize((512, 512))
        gt_pil = Image.open(data['gt_path']).convert("RGB").resize((512, 512))
        fmri_input = data['fmri']
        
        with torch.no_grad():
            # --- PURE FMRI SEMANTIC ---
            pred_emb = normalize_embed(aligner_sem(fmri_input)[0])
            gt_1280 = normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_pil, return_tensors="pt").to(device)).image_embeds)[0]
            pred_sem_list.append(pred_emb.cpu())
            gt_sem_1280_list.append(gt_1280.cpu())
            
            # --- PURE FMRI SPATIAL ---
            gt_bin_64 = load_gt_masks(data['mask_path'])
            pred_cont_64, pred_bin_64 = predict_masks(fmri_input, aligner_struct, threshold=args.mask_threshold)
            if gt_bin_64 is not None:
                pred_mask_cont_list.append(pred_cont_64)
                pred_mask_bin_list.append(pred_bin_64)
                gt_mask_bin_list.append(gt_bin_64)

            # --- IMMAGINI ---
            t_gen = (F_vision.to_tensor(gen_pil).to(device) * 2.0 - 1.0).unsqueeze(0)
            t_gt = (F_vision.to_tensor(gt_pil).to(device) * 2.0 - 1.0).unsqueeze(0)
            metrics['lpips'].append(loss_fn_vgg(t_gen, t_gt).item())
            
            metrics['clip_base_gen'].append(normalize_embed(clip_model_base.get_image_features(**clip_processor_base(images=gen_pil, return_tensors="pt").to(device))).cpu())
            metrics['clip_base_gt'].append(normalize_embed(clip_model_base.get_image_features(**clip_processor_base(images=gt_pil, return_tensors="pt").to(device))).cpu())
            metrics['clip_xl_gen'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gen_pil, return_tensors="pt").to(device)).image_embeds).cpu())
            metrics['clip_xl_gt'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_pil, return_tensors="pt").to(device)).image_embeds).cpu())

            a2_gen, a5_gen, a7_gen = alexnet(alex_transform(gen_pil).unsqueeze(0).to(device))
            a2_gt, a5_gt, a7_gt = alexnet(alex_transform(gt_pil).unsqueeze(0).to(device))
            
            metrics['alex2_gen'].append(normalize_embed(a2_gen).cpu())
            metrics['alex2_gt'].append(normalize_embed(a2_gt).cpu())
            metrics['alex5_gen'].append(normalize_embed(a5_gen).cpu())
            metrics['alex5_gt'].append(normalize_embed(a5_gt).cpu())
            metrics['alex7_gen'].append(normalize_embed(a7_gen).cpu())
            metrics['alex7_gt'].append(normalize_embed(a7_gt).cpu())

    # ==========================================
    # 4. CALCOLO E STAMPA TABELLE
    # ==========================================
    avg_lpips = np.mean(metrics['lpips'])
    cb_t1, cb_t5 = get_n_way(metrics['clip_base_gen'], metrics['clip_base_gt'])
    cxl_t1, cxl_t5 = get_n_way(metrics['clip_xl_gen'], metrics['clip_xl_gt'])
    a2_t1, a2_t5 = get_n_way(metrics['alex2_gen'], metrics['alex2_gt'])
    a5_t1, a5_t5 = get_n_way(metrics['alex5_gen'], metrics['alex5_gt'])
    a7_t1, a7_t5 = get_n_way(metrics['alex7_gen'], metrics['alex7_gt'])

    # --- Calcolo Metriche Pure fMRI Semantic ---
    sim_sem = torch.matmul(torch.stack(pred_sem_list), torch.stack(gt_sem_1280_list).T)
    t1_sem, t5_sem = 0, 0
    for i in range(N):
        rank = (torch.argsort(sim_sem[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        if rank == 1: t1_sem += 1
        if rank <= 5: t5_sem += 1

    # --- Calcolo Metriche Pure fMRI Spatial (N-Way per IoU, MSE, Cosine) ---
    ious, mses, cosines = [], [], []
    N_spa = len(pred_mask_bin_list)
    
    t1_spa_iou, t5_spa_iou = 0, 0
    t1_spa_cos, t5_spa_cos = 0, 0
    t1_spa_mse, t5_spa_mse = 0, 0

    for i in range(N_spa):
        p_cont = pred_mask_cont_list[i]
        p_bin = pred_mask_bin_list[i]
        p_flat = p_cont.flatten()
        
        sims_iou, sims_cos, sims_mse = [], [], []
        
        # Medie Globali
        ious.append(calc_iou(p_bin, gt_mask_bin_list[i]))
        mses.append(np.mean((p_cont - gt_mask_bin_list[i])**2))
        
        g_flat_true = gt_mask_bin_list[i].flatten().astype(np.float32)
        norm_p, norm_g_true = np.linalg.norm(p_flat), np.linalg.norm(g_flat_true)
        cosines.append(np.dot(p_flat, g_flat_true) / (norm_p * norm_g_true) if norm_p > 0 and norm_g_true > 0 else 0.0)

        # Loop N-way
        for j in range(N_spa):
            g_bin = gt_mask_bin_list[j]
            g_flat = g_bin.flatten().astype(np.float32)
            
            # IoU
            sims_iou.append(calc_iou(p_bin, g_bin))
            
            # Cosine
            norm_g = np.linalg.norm(g_flat)
            sims_cos.append(np.dot(p_flat, g_flat) / (norm_p * norm_g) if norm_p > 0 and norm_g > 0 else 0.0)
            
            # MSE
            sims_mse.append(np.mean((p_cont - g_bin)**2))
            
        # Classifica IoU (Più alto è meglio -> descending)
        rank_iou = sorted(range(N_spa), key=lambda x: sims_iou[x], reverse=True).index(i) + 1
        if rank_iou == 1: t1_spa_iou += 1
        if rank_iou <= 5: t5_spa_iou += 1
        
        # Classifica Cosine (Più alto è meglio -> descending)
        rank_cos = sorted(range(N_spa), key=lambda x: sims_cos[x], reverse=True).index(i) + 1
        if rank_cos == 1: t1_spa_cos += 1
        if rank_cos <= 5: t5_spa_cos += 1
        
        # Classifica MSE (Più basso è meglio -> ASCENDING!)
        rank_mse = sorted(range(N_spa), key=lambda x: sims_mse[x], reverse=False).index(i) + 1
        if rank_mse == 1: t1_spa_mse += 1
        if rank_mse <= 5: t5_spa_mse += 1

    print("\n" + "="*85)
    print(f"TABELLA 1: METRICHE D'IMMAGINE Pura Gen vs Ground Truth (N={N})")
    print("="*85)
    print(f"{'Metrica':<35} | {'Top-1':<15} | {'Top-5':<15}")
    print("-" * 85)
    print(f"{'LPIPS (VGG) ↓':<35} | {avg_lpips:05.4f}         | -")
    print("-" * 85)
    print(f"{'AlexNet(2) [Struttura Low-Level] ↑':<35} | {a2_t1:05.2f}%         | {a2_t5:05.2f}%")
    print(f"{'AlexNet(5) [Features Mid-Level] ↑':<35} | {a5_t1:05.2f}%         | {a5_t5:05.2f}%")
    print(f"{'AlexNet(7) [Tua Originale] ↑':<35} | {a7_t1:05.2f}%         | {a7_t5:05.2f}%")
    print("-" * 85)
    print(f"{'CLIP Base (512D) [Semantica] ↑':<35} | {cb_t1:05.2f}%         | {cb_t5:05.2f}%")
    print(f"{'CLIP XL (1280D) [Semantica] ↑':<35} | {cxl_t1:05.2f}%         | {cxl_t5:05.2f}%")
    
    print("\n" + "="*85)
    print(f"TABELLA 2: METRICHE PURE fMRI (N={N})")
    print("="*85)
    print(f"{'Metrica N-way':<55} | {'Top-1':<10} | {'Top-5':<10}")
    print("-" * 85)
    print(f"{'Pure Semantic (fMRI 1280D vs Real 1280D)':<55} | {(t1_sem/N)*100:05.2f}%    | {(t5_sem/N)*100:05.2f}%")
    print("-" * 85)
    print(f"{'Pure Spatial (IoU su Maschera Binaria)':<55} | {(t1_spa_iou/N_spa)*100:05.2f}%    | {(t5_spa_iou/N_spa)*100:05.2f}%")
    print(f"{'Pure Spatial (Cosine Sim su Mappa Continua)':<55} | {(t1_spa_cos/N_spa)*100:05.2f}%    | {(t5_spa_cos/N_spa)*100:05.2f}%")
    print(f"{'Pure Spatial (MSE su Mappa Continua)':<55} | {(t1_spa_mse/N_spa)*100:05.2f}%    | {(t5_spa_mse/N_spa)*100:05.2f}%")
    print("=" * 85)
    print(f"{'Medie Globali Spaziali:':<30} IoU: {np.mean(ious):.4f} | Cosine: {np.mean(cosines):.4f} | MSE: {np.mean(mses):.4f}")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()