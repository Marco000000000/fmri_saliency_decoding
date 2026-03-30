import argparse
import os
import sys
import pickle
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

# ==============================================================================
# CONFIGURAZIONE BATCH
# ==============================================================================
SUBJECTS = [1, 2, 3, 4, 5]
ROIS = ["VC"]#, "V1", "V2", "V3", "V4", "LOC", "FFA", "PPA"]

# Il path esatto in cui si trovano le sottocartelle delle ROI (VC, FFA, ecc.)
GEN_BASE_DIR = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/results_final/sdxl_attention" 

# Che tipo di spazio stai valutando? (none, mask, box)
SPATIAL_MODE = "box" 

# Nome che apparirà nel report per essere letto dal tuo script delle tabelle
MODEL_DISPLAY_NAME = "SDXL 1-STEP ATTENTION"
# ==============================================================================

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def get_bbox_mask(mask_2d, target_size=(512, 512)):
    coords = np.argwhere(mask_2d > 0)
    bbox_mask = np.zeros(target_size, dtype=np.uint8)
    if len(coords) > 5:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        x1, y1 = int((x_min / 64.0) * target_size[0]), int((y_min / 64.0) * target_size[1])
        x2, y2 = int((x_max / 64.0) * target_size[0]), int((y_max / 64.0) * target_size[1])
        if x2 <= x1: x2 = x1 + 5
        if y2 <= y1: y2 = y1 + 5
        bbox_mask[y1:y2, x1:x2] = 1
    else:
        bbox_mask[:, :] = 1 
    return np.stack([bbox_mask]*3, axis=-1).astype(np.float32)

def load_gt_masks(npy_path):
    if not os.path.exists(npy_path): return None, None, None
    mask_64 = np.load(npy_path).reshape(64, 64)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    mask_bin_64 = (mask_norm > 0.5).astype(np.uint8)
    mask_pil = Image.fromarray(mask_bin_64 * 255).resize((512, 512), Image.NEAREST)
    return np.stack([np.array(mask_pil) / 255.0]*3, axis=-1).astype(np.float32), get_bbox_mask(mask_bin_64), mask_bin_64

def predict_masks(fmri_tensor, model, threshold=0.6):
    with torch.no_grad(): depth_flat = model(fmri_tensor)
    mask_64 = depth_flat.view(1, 1, 64, 64).to(dtype=torch.float32)
    d_min, d_max = mask_64.min(), mask_64.max()
    mask_norm = (mask_64 - d_min) / (d_max - d_min + 1e-6)
    clean_mask = torch.where(mask_norm > threshold, mask_norm, torch.zeros_like(mask_norm))
    eroded = -F.max_pool2d(-clean_mask, kernel_size=3, stride=1, padding=1)
    binary_64 = torch.where(eroded > 0, torch.tensor(1.0).to(eroded.device), torch.tensor(0.0).to(eroded.device))
    if binary_64.sum() < 5: binary_64 = torch.ones_like(binary_64)
    bin_64_np = binary_64[0, 0].cpu().numpy().astype(np.uint8)
    mask_pil = Image.fromarray(bin_64_np * 255).resize((512, 512), Image.NEAREST)
    return np.stack([np.array(mask_pil) / 255.0]*3, axis=-1).astype(np.float32), get_bbox_mask(bin_64_np), bin_64_np

def calc_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return intersection / union
def get_2d_box_mask(mask_2d):
    """Converte una maschera 2D in una maschera rettangolare (box) 2D."""
    coords = np.argwhere(mask_2d > 0)
    box_mask = np.zeros_like(mask_2d)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Riordina i vertici per creare il rettangolo pieno
        box_mask[y_min:y_max+1, x_min:x_max+1] = 1
    return box_mask

def calc_box_iou(pred_mask_64, gt_mask_64):
    """Calcola la IoU tra le bounding box di due maschere."""
    # Convertiamo entrambe le forme in rettangoli pieni
    pred_box = get_2d_box_mask(pred_mask_64)
    gt_box = get_2d_box_mask(gt_mask_64)
    
    intersection = np.logical_and(pred_box, gt_box).sum()
    union = np.logical_or(pred_box, gt_box).sum()
    
    if union == 0:
        return 0.0
    return intersection / union
def normalize_embed(emb):
    return emb / emb.norm(p=2, dim=-1, keepdim=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--gt_dir", type=str, default="data/GOD_Dataset/images/test")
    parser.add_argument("--gt_masks_dir", type=str, default="data/precomputed_binary_masks")
    parser.add_argument("--mask_threshold", type=float, default=0.6)
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "kandinsky", "kamitani"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("eval_report1", exist_ok=True)

    print(f"🚀 INIZIO BATCH EVALUATION | {len(SUBJECTS)} Soggetti | {len(ROIS)} ROI")
    
    # --- 1. CARICAMENTO MODELLI PESANTI (UNA SOLA VOLTA) ---
    print("⏳ Caricamento VGG, CLIP (512D & 1280D), AlexNet...")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor_1280 = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    clip_model_1280 = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device).eval()

    alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    alexnet.classifier = alexnet.classifier[:-1] 
    alexnet.eval()
    alex_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    kamitani_df = None
    if args.model == "kamitani":
        pkl_path = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/kamitani_method/results/GenericObjectDecoding.pkl"
        with open(pkl_path, 'rb') as f:
            kamitani_df = pickle.load(f)

    def _get_k_id(base_str):
        cat_id = int(base_str.split('_')[0][1:])
        img_id = int(base_str.split('_')[1])
        return float('%d.%06d' % (cat_id, img_id))

    # --- 2. CICLO SU SOGGETTI E ROI ---
    for sub in SUBJECTS:
        print(f"\n" + "="*50)
        print(f"🧠 INIZIO SOGGETTO {sub}")
        print("="*50)
        
        test_h5 = f"Subject{sub}_ImageNetTest.h5"
        
        for roi in ROIS:
            print(f"\n--- Analisi ROI: {roi} ---")
            
            # 2a. Trova la cartella delle immagini generate con la NUOVA STRUTTURA
            # Es: .../results_final/sdxl_attention/FFA/1
            gen_dir = os.path.join(GEN_BASE_DIR, roi, str(sub))
            
            # Fallback di sicurezza: se per caso dentro c'è una cartella 'single', usa quella
            if os.path.exists(os.path.join(gen_dir, "single")):
                gen_dir = os.path.join(gen_dir, "single")
                
            if not os.path.exists(gen_dir):
                print(f"⚠️ Cartella non trovata ({gen_dir}). Salto la ROI {roi} per S{sub}.")
                continue

            # 2b. Carica i Decoders fMRI per questa specifica ROI
            try:
                roi_keys = parse_roi_keys(roi)
                input_dim = sum([subject_dims[k][sub-1] for k in roi_keys])
                rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
                
                raw_data = load_fmri_image_pairs(
                    os.path.join(args.dataset_root, "fmri_files", "test"),
                    os.path.join(args.dataset_root, "images", "test"), test_h5, rois_dict
                )
                
                aligner_sem = SimpleLinearAligner(input_dim, output_dim=1280).to(device)
                path_sem = f"trained_fmri_decoders/linear_sparse_lbfgs_mse/S{sub}_{roi}/best_linear_aligner.pth"
                aligner_sem.load_state_dict(torch.load(path_sem, map_location=device))
                aligner_sem.eval()

                aligner_struct = SimpleLinearAligner(input_dim, output_dim=4096).to(device)
                path_struct = f"trained_fmri_decoders/binary_decoder/S{sub}_{roi}/best_depth_decoder.pth"
                aligner_struct.load_state_dict(torch.load(path_struct, map_location=device))
                aligner_struct.eval()
            except Exception as e:
                print(f"⚠️ Errore caricamento decoder per S{sub} {roi}: {e}. Salto.")
                continue

            grouped_fmri = defaultdict(list)
            for i, vec in enumerate(raw_data['fmri']):
                fname = os.path.basename(raw_data['image_paths'][i])
                base_id = fname.split('_')[0] + '_' + fname.split('_')[1].split('.')[0]
                grouped_fmri[base_id].append(vec)

            gen_files = [f for f in os.listdir(gen_dir) if f.endswith(('.png', '.jpg', '.JPEG'))]
            shared_data = []
            for gen_name in gen_files:
                try: base_id = gen_name.split('_')[0] + '_' + gen_name.split('_')[1].split('.')[0]
                except: continue
                gt_name = next((f for f in os.listdir(args.gt_dir) if base_id in f), None)
                if gt_name and base_id in grouped_fmri:
                    shared_data.append((base_id, gen_name, gt_name))

            shared_data.sort(key=lambda x: x[0])
            N = len(shared_data)
            if N == 0: 
                print(f"⚠️ Nessuna immagine condivisa tra GT e Generate in {gen_dir}. Salto.")
                continue

            # 2c. Preparazione Feature Kamitani
            kamitani_features_dict = {}
            if args.model == "kamitani" and kamitani_df is not None:
                sbj_str = f"Subject{sub}"
                filtered_df = kamitani_df[(kamitani_df['subject'] == sbj_str) & (kamitani_df['roi'] == roi)]
                if not filtered_df.empty:
                    row = filtered_df.iloc[0]
                    k_feat = row['predicted_feature_averaged_percept'] 
                    k_labels = row['test_label_set_percept']           
                    kamitani_features_dict = {_get_k_id(lbl): feat for lbl, feat in zip(k_labels, k_feat)}

            # --- STRUTTURE DATI PER EVAL ---
            modes = ['full', 'pred_mask', 'gt_mask', 'pred_box', 'gt_box']
            metrics = {m: {'lpips': [], 'clip_emb': [], 'alex_emb': [], 'gt_clip_emb': [], 'gt_alex_emb': []} for m in modes}
            pred_sem_list, gt_sem_1280_list = [], []
            pred_mask_64_list, gt_mask_64_list = [], []
            gen_images_np, gt_images_np, gt_box_masks_np = [], [], []

            # 2d. Estrazione Feature (Ciclo sulle immagini)
            for base_id, gen_name, gt_name in tqdm(shared_data, desc=f"Eval S{sub} {roi}", leave=False):
                gt_mask_npy_path = os.path.join(args.gt_masks_dir, base_id + ".npy")
                fmri_input = torch.tensor(np.array(grouped_fmri[base_id]), dtype=torch.float32).to(device).mean(dim=0, keepdim=True)
                gt_pil = Image.open(os.path.join(args.gt_dir, gt_name)).convert("RGB").resize((512, 512))
                
                # Pure Semantic
                if args.model == "kamitani":
                    k_id = _get_k_id(base_id)
                    pred_emb = torch.tensor(kamitani_features_dict.get(k_id, np.zeros(1280)), dtype=torch.float32).unsqueeze(0).to(device)
                    pred_sem_list.append(pred_emb.cpu())
                else:
                    with torch.no_grad(): 
                        pred_sem_list.append(aligner_sem(fmri_input)[0].cpu())
                    
                # GT 1280D
                with torch.no_grad():
                    gt_1280 = clip_model_1280(**clip_processor_1280(images=gt_pil, return_tensors="pt").to(device)).image_embeds
                    gt_sem_1280_list.append(gt_1280[0].cpu())
                
                # Maschere
                gt_shape, gt_box, gt_bin_64 = load_gt_masks(gt_mask_npy_path)
                pred_shape, pred_box, pred_bin_64 = predict_masks(fmri_input, aligner_struct, threshold=args.mask_threshold)
                if gt_shape is None: continue
                
                pred_mask_64_list.append(pred_bin_64)
                gt_mask_64_list.append(gt_bin_64)
                
                gen_pil = Image.open(os.path.join(gen_dir, gen_name)).convert("RGB").resize((512, 512))
                gen_np, gt_np = np.array(gen_pil), np.array(gt_pil)
                gen_images_np.append(gen_np)
                gt_images_np.append(gt_np)
                gt_box_masks_np.append(gt_box)

                # Ablation
                img_dict = {
                    'full': (gen_pil, gt_pil),
                    'pred_mask': (Image.fromarray((gen_np * pred_shape).astype(np.uint8)), Image.fromarray((gt_np * gt_shape).astype(np.uint8))),
                    'gt_mask': (Image.fromarray((gen_np * gt_shape).astype(np.uint8)), Image.fromarray((gt_np * gt_shape).astype(np.uint8))),
                    'pred_box': (Image.fromarray((gen_np * pred_box).astype(np.uint8)), Image.fromarray((gt_np * gt_box).astype(np.uint8))),
                    'gt_box': (Image.fromarray((gen_np * gt_box).astype(np.uint8)), Image.fromarray((gt_np * gt_box).astype(np.uint8)))
                }

                with torch.no_grad():
                    for mode in modes:
                        gen_img, gt_img = img_dict[mode]
                        t_gen = (F_vision.to_tensor(gen_img).to(device) * 2.0 - 1.0).unsqueeze(0)
                        t_gt = (F_vision.to_tensor(gt_img).to(device) * 2.0 - 1.0).unsqueeze(0)
                        metrics[mode]['lpips'].append(loss_fn_vgg(t_gen, t_gt).item())
                        
                        e_g = clip_model.get_image_features(**clip_processor(images=gen_img, return_tensors="pt").to(device))
                        e_t = clip_model.get_image_features(**clip_processor(images=gt_img, return_tensors="pt").to(device))
                        metrics[mode]['clip_emb'].append(normalize_embed(e_g).cpu())
                        metrics[mode]['gt_clip_emb'].append(normalize_embed(e_t).cpu())

                        a_g = alexnet(alex_transform(gen_img).unsqueeze(0).to(device))
                        a_t = alexnet(alex_transform(gt_img).unsqueeze(0).to(device))
                        metrics[mode]['alex_emb'].append(normalize_embed(a_g).cpu())
                        metrics[mode]['gt_alex_emb'].append(normalize_embed(a_t).cpu())

            # 2e. Calcolo Matematico N-Way
            def get_n_way(g_list, t_list):
                sim = torch.matmul(torch.cat(g_list, dim=0), torch.cat(t_list, dim=0).T)
                t1, t5 = 0, 0
                for i in range(len(g_list)):
                    r = (torch.argsort(sim[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
                    if r == 1: t1 += 1
                    if r <= 5: t5 += 1
                return (t1 / len(g_list)) * 100, (t5 / len(g_list)) * 100

            results_ablation = {}
            for mode in modes:
                c1, c5 = get_n_way(metrics[mode]['clip_emb'], metrics[mode]['gt_clip_emb'])
                a1, a5 = get_n_way(metrics[mode]['alex_emb'], metrics[mode]['gt_alex_emb'])
                results_ablation[mode] = {'lpips': np.mean(metrics[mode]['lpips']), 'c1': c1, 'c5': c5, 'a1': a1, 'a5': a5}

            avg_iou = np.mean([calc_iou(pred_mask_64_list[i], gt_mask_64_list[i]) for i in range(N)])

            pred_sem_stack = torch.nn.functional.normalize(torch.stack(pred_sem_list), p=2, dim=1)
            gt_sem_stack = torch.nn.functional.normalize(torch.stack(gt_sem_1280_list), p=2, dim=1)
            sim_sem = torch.matmul(pred_sem_stack, gt_sem_stack.T)
            t1_sem, t5_sem = sum((torch.argsort(sim_sem[i], descending=True) == i).nonzero(as_tuple=True)[0].item() == 0 for i in range(N)), sum((torch.argsort(sim_sem[i], descending=True) == i).nonzero(as_tuple=True)[0].item() < 5 for i in range(N))

            t1_spa, t5_spa = 0, 0
            for i in range(N):
                ious = [calc_iou(pred_mask_64_list[i], gt_mask_64_list[j]) for j in range(N)]
                rank = sorted(range(N), key=lambda x: ious[x], reverse=True).index(i) + 1
                if rank == 1: t1_spa += 1
                if rank <= 5: t5_spa += 1

            

            # --- 2f. SCRITTURA FILE DI REPORT ---
            report_lines = []
            report_lines.append(f"REPORT VALUTAZIONE COMPLETA: SOGGETTO {sub} - ROI {roi}")
            report_lines.append("=======================================================================\n")
            report_lines.append(f"#######################################################################")
            report_lines.append(f"                      VARIANTE SPAZIALE: {SPATIAL_MODE}")
            report_lines.append(f"#######################################################################\n")
            report_lines.append(f"--- MODELLO: {MODEL_DISPLAY_NAME} | SPAZIO: {SPATIAL_MODE} ---")
            
            report_lines.append("==============================================================================================================")
            report_lines.append(f"TABELLA 1: ABLATION MULTI-MODELLO E MULTI-MASCHERA (N={N})")
            report_lines.append("==============================================================================================================")
            report_lines.append(f"{'Metrica':<20} | {'Full Image':<14} | {'Pred MASK':<14} | {'GT MASK':<14} | {'Pred BOX':<14} | {'GT BOX':<14}")
            report_lines.append("-" * 110)
            report_lines.append(f"{'LPIPS (VGG) ↓':<20} | {results_ablation['full']['lpips']:05.4f}          | {results_ablation['pred_mask']['lpips']:05.4f}          | {results_ablation['gt_mask']['lpips']:05.4f}          | {results_ablation['pred_box']['lpips']:05.4f}          | {results_ablation['gt_box']['lpips']:05.4f}")
            report_lines.append("-" * 110)
            report_lines.append(f"{'CLIP Top-1 ↑':<20} | {results_ablation['full']['c1']:05.2f}%         | {results_ablation['pred_mask']['c1']:05.2f}%         | {results_ablation['gt_mask']['c1']:05.2f}%         | {results_ablation['pred_box']['c1']:05.2f}%         | {results_ablation['gt_box']['c1']:05.2f}%")
            report_lines.append(f"{'CLIP Top-5 ↑':<20} | {results_ablation['full']['c5']:05.2f}%         | {results_ablation['pred_mask']['c5']:05.2f}%         | {results_ablation['gt_mask']['c5']:05.2f}%         | {results_ablation['pred_box']['c5']:05.2f}%         | {results_ablation['gt_box']['c5']:05.2f}%")
            report_lines.append("-" * 110)
            report_lines.append(f"{'AlexNet Top-1 ↑':<20} | {results_ablation['full']['a1']:05.2f}%         | {results_ablation['pred_mask']['a1']:05.2f}%         | {results_ablation['gt_mask']['a1']:05.2f}%         | {results_ablation['pred_box']['a1']:05.2f}%         | {results_ablation['gt_box']['a1']:05.2f}%")
            report_lines.append(f"{'AlexNet Top-5 ↑':<20} | {results_ablation['full']['a5']:05.2f}%         | {results_ablation['pred_mask']['a5']:05.2f}%         | {results_ablation['gt_mask']['a5']:05.2f}%         | {results_ablation['pred_box']['a5']:05.2f}%         | {results_ablation['gt_box']['a5']:05.2f}%")
            report_lines.append("==============================================================================================================")
            report_lines.append(f"Spatial Mask IoU (Predicted vs Ground Truth) : {avg_iou:.4f}\n")
            
            report_lines.append("=====================================================================================")
            report_lines.append(f"TABELLA 2: PUNTEGGI PURI E UNBIASED (N={N})")
            report_lines.append("=====================================================================================")
            report_lines.append(f"{'Metrica N-way (Top-1 / Top-5)':<45} | {'Top-1':<10} | {'Top-5':<10}")
            report_lines.append("-" * 85)
            report_lines.append(f"{'1. PURE SEMANTIC (fMRI 1280D vs Real 1280D)':<45} | {(t1_sem/N)*100:05.2f}%    | {(t5_sem/N)*100:05.2f}%")
            report_lines.append(f"{'2. PURE SPATIAL (fMRI Mask vs Real Mask)':<45} | {(t1_spa/N)*100:05.2f}%    | {(t5_spa/N)*100:05.2f}%")
            report_lines.append("-" * 85)
            report_lines.append(f"{'3. PURE IMAGE - CLIP (Full Gen vs Full GT)':<45} | {results_ablation['full']['c1']:05.2f}%    | {results_ablation['full']['c5']:05.2f}%")
            report_lines.append(f"{'4. PURE IMAGE - ALEXNET (Full Gen vs GT)':<45} | {results_ablation['full']['a1']:05.2f}%    | {results_ablation['full']['a5']:05.2f}%")
            report_lines.append("=====================================================================================\n\n")

            report_path = f"eval_report1/S{sub}_{roi}_FULL_REPORT.txt"
            with open(report_path, "a") as f:
                f.write("\n".join(report_lines))
                
            print(f"✅ Report accodato in: {report_path}")

    print("\n🎉 BATCH COMPLETATO! I tuoi file di report sono pronti in 'eval_report/'.")

if __name__ == "__main__":
    main()