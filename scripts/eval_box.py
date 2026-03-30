import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

# --- GESTIONE PATH ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# ================= CONFIGURAZIONE =================
SUBJECT = 3
ROI = "VC"
THRESHOLD = 0.6  
DATASET_ROOT = "data/GOD_Dataset"
GT_DIR = "data/GOD_Dataset/images/test"
GT_MASKS_DIR = "data/precomputed_binary_masks"
OUTPUT_DIR = f"visual_validation_all/S{SUBJECT}_{ROI}"
# ==================================================

class Aligner(nn.Module):
    def __init__(self, in_dim, out_dim=4096):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.linear(x)

def calc_mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return inter / union if union > 0 else 0.0

def get_bin_mask(fmri_tensor, model, threshold=0.6):
    with torch.no_grad():
        d_flat = model(fmri_tensor)
    m64_tensor = d_flat.view(1, 1, 64, 64).to(dtype=torch.float32)
    m_min, m_max = m64_tensor.min(), m64_tensor.max()
    m_norm_tensor = (m64_tensor - m_min) / (m_max - m_min + 1e-6)
    
    m_norm = m_norm_tensor[0,0].cpu().numpy()
    mask_bin = (m_norm > threshold).astype(np.uint8)
    return mask_bin

def create_panel(orig_img, gt_mask_64, pred_mask_64, mask_iou, title_text, font):
    p_w, p_h = 512, 512
    
    # 1. Maschere in scala di grigi per visualizzazione base
    gt_pil = Image.fromarray(gt_mask_64 * 255).resize((p_w, p_h), Image.NEAREST).convert("L")
    pr_pil = Image.fromarray(pred_mask_64 * 255).resize((p_w, p_h), Image.NEAREST).convert("L")

    # --- PANNELLO A: GT Mask (Solo per coerenza, ora ci dipingiamo sopra) ---
    img_a = orig_img.convert("RGB")
    mask_cyan = Image.new("RGB", (p_w, p_h), (0, 255, 255))
    overlaid_a = Image.composite(mask_cyan, img_a, gt_pil)
    img_a = Image.blend(img_a, overlaid_a, alpha=0.4)

    # --- PANNELLO B: OVERLAY MASCHERA PREDETTA ---
    img_b = orig_img.convert("RGB")
    mask_green = Image.new("RGB", (p_w, p_h), (0, 255, 0))
    overlaid_b = Image.composite(mask_green, img_b, pr_pil)
    img_b = Image.blend(img_b, overlaid_b, alpha=0.4)

    # --- PANNELLO C: CONFRONTO BOUNDING BOX ---
    img_c = Image.new("RGB", (p_w, p_h), (0, 0, 0))
    img_c.paste(orig_img.resize((p_w, p_h)).convert("L"), (0,0))
    draw_c = ImageDraw.Draw(img_c)
    
    gt_coords = np.argwhere(gt_mask_64 > 0)
    if len(gt_coords) > 0:
        y1, x1 = gt_coords.min(0); y2, x2 = gt_coords.max(0)
        draw_c.rectangle([int(x1*8), int(y1*8), int(x2*8), int(y2*8)], outline="cyan", width=4)

    pr_coords = np.argwhere(pred_mask_64 > 0)
    if len(pr_coords) > 0:
        y1, x1 = pr_coords.min(0); y2, x2 = pr_coords.max(0)
        draw_c.rectangle([int(x1*8), int(y1*8), int(x2*8), int(y2*8)], outline="lime", width=4)

    # --- MONTAGGIO GRIGLIA ---
    grid = Image.new("RGB", (p_w * 3, p_h + 50), (30, 30, 30))
    grid.paste(img_a, (0, 50))
    grid.paste(img_b, (p_w, 50))
    grid.paste(img_c, (p_w * 2, 50))
    
    draw_g = ImageDraw.Draw(grid)
    draw_g.text((10, 10), f"Target: GT Overlay", fill="cyan", font=font)
    draw_g.text((p_w+10, 10), f"Pred: Brain Overlay (IoU: {mask_iou:.4f})", fill="lime", font=font)
    draw_g.text((p_w*2+10, 10), f"Geometrical Bounding Boxes", fill="white", font=font)
    
    return grid

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try: font = ImageFont.truetype("arial.ttf", 26)
    except: font = ImageFont.load_default()

    print(f"🔬 Generazione pannelli visivi per tutte le 50 immagini (S{SUBJECT} {ROI})...")

    # 1. Carica Modello
    try:
        rk = parse_roi_keys(ROI); idim = sum([subject_dims[k][SUBJECT-1] for k in rk])
        model = Aligner(idim).to(device)
        model.load_state_dict(torch.load(f"trained_fmri_decoders/binary_decoder/S{SUBJECT}_{ROI}/best_depth_decoder.pth", map_location=device))
        model.eval()
    except Exception as e:
        print(f"⚠️ Errore caricamento decoder: {e}")
        return

    # 2. Carica Dati (Trial Mediati)
    raw = load_fmri_image_pairs(os.path.join(DATASET_ROOT, "fmri_files", "test"), os.path.join(DATASET_ROOT, "images", "test"), f"Subject{SUBJECT}_ImageNetTest.h5", {k: ALL_ROI_MAPPINGS[k] for k in rk})
    
    grouped_fmri = defaultdict(list)
    img_map = {}
    for i, vec in enumerate(raw['fmri']):
        fname = os.path.basename(raw['image_paths'][i])
        grouped_fmri[fname].append(vec)
        img_map[fname] = raw['image_paths'][i]

    # 3. Inferenza e Calcolo IoU
    all_scores = []
    print("⏳ Calcolo predizioni...")
    for fname, vecs in tqdm(grouped_fmri.items(), leave=False):
        bid = os.path.splitext(fname)[0]
        f_avg = torch.tensor(np.array(vecs)).mean(0, keepdim=True).to(device).float()
        
        mask_gt_path = os.path.join(GT_MASKS_DIR, f"{bid}.npy")
        if not os.path.exists(mask_gt_path): continue
        gt_m64 = np.load(mask_gt_path).reshape(64,64)
        gt_mask = (gt_m64 > 0.5).astype(np.uint8)

        pred_mask = get_bin_mask(f_avg, model, threshold=THRESHOLD)
        iou = calc_mask_iou(pred_mask, gt_mask)
        
        all_scores.append({
            "id": bid, "iou": iou, "img_path": img_map[fname], 
            "gt_mask": gt_mask, "pred_mask": pred_mask
        })

    # 4. Ordinamento dal migliore al peggiore
    sorted_scores = sorted(all_scores, key=lambda x: x['iou'], reverse=True)

    # 5. Salvataggio Singole Immagini
    print(f"💾 Salvataggio di {len(sorted_scores)} pannelli nella cartella: {OUTPUT_DIR}")
    for rank, sample in enumerate(tqdm(sorted_scores)):
        orig_img = Image.open(sample['img_path']).convert("RGB")
        title = f"Rank {rank+1:02d} | ID: {sample['id']} | Mask IoU: {sample['iou']:.4f}"
        
        panel = create_panel(orig_img, sample['gt_mask'], sample['pred_mask'], sample['iou'], title, font)
        
        # Nome file es: 01_iou0.8523_n01440764_10026.png
        filename = f"{rank+1:02d}_iou{sample['iou']:.4f}_{sample['id']}.png"
        panel.save(os.path.join(OUTPUT_DIR, filename))

    print("✅ Completato! Apri la cartella per scorrere tutte le 50 immagini ordinate per performance.")

if __name__ == "__main__":
    main()