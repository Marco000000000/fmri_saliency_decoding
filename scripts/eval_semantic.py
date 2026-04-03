import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from collections import defaultdict

# ==========================================
# CONFIGURAZIONE PERCORSI E COSTANTI
# ==========================================
BASE_DIR = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
GT_DIR = f"{BASE_DIR}/data/GOD_Dataset/images/test"
MASKS_DIR = f"{BASE_DIR}/data/precomputed_binary_masks"
TSV_PATH = f"{BASE_DIR}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# Aggiunta path per importare i dataset manager
sys.path.append(BASE_DIR)
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.config.roi_config import ALL_ROI_MAPPINGS

# ==========================================
# CLASSI E FUNZIONI MODELLO
# ==========================================
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    def forward(self, x): return self.linear(x)

def normalize_embed(emb):
    return emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)

def min_max_normalize(scores):
    """Normalizza i punteggi nel range [0, 1] per la fusione."""
    s_min = scores.min(dim=1, keepdim=True)[0]
    s_max = scores.max(dim=1, keepdim=True)[0]
    return (scores - s_min) / (s_max - s_min + 1e-8)

def extract_gt_features(targets_info, model, processor, device):
    """Estrae embedding CLIP e Maschere GT per i 50 target del test set."""
    clip_embs = []
    mask_tensors = []
    print(f"📦 Estrazione Feature Ground Truth (50 target)...")
    
    with torch.no_grad():
        for t_info in tqdm(targets_info):
            # 1. CLIP Embedding dall'immagine reale
            img_pil = Image.open(t_info['gt_path']).convert("RGB")
            c_input = processor(images=img_pil, return_tensors="pt").to(device)
            c_emb = normalize_embed(model(**c_input).image_embeds)
            clip_embs.append(c_emb.detach())
            
            # 2. Maschera GT binaria (64x64)
            m_path = t_info['mask_path']
            if os.path.exists(m_path):
                m_np = np.load(m_path).reshape(64, 64)
                m_tensor = torch.from_numpy(m_np).float().to(device)
            else:
                m_tensor = torch.zeros((64, 64)).to(device)
            mask_tensors.append(m_tensor.unsqueeze(0))
            
    return torch.cat(clip_embs, dim=0), torch.cat(mask_tensors, dim=0)

def load_dynamic_model(path, output_dim, device):
    """Carica un modello inferendo la dimensione di input dal checkpoint e ritorna (model, input_dim)."""
    if not os.path.exists(path):
        return None, 0
    state_dict = torch.load(path, map_location=device)
    # Estraiamo la dimensione di input corretta (es. 3444 invece di 4466)
    input_dim = state_dict['linear.weight'].shape[1]
    model = SimpleLinearAligner(input_dim, output_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, input_dim

def evaluate_subject_pure_fmri(sub_idx, t_clip, t_masks, targets_info, device):
    """Classificazione N-way basata puramente sugli embedding decodificati dall'fMRI."""
    
    # 1. Caricamento Dinamico (Risolve il problema del mismatch S1)
    path_sem = f"{BASE_DIR}/trained_fmri_decoders/linear_sparse_lbfgs_mse/S{sub_idx}_VC/best_linear_aligner.pth"
    path_struct = f"{BASE_DIR}/trained_fmri_decoders/binary_decoder/S{sub_idx}_VC/best_depth_decoder.pth"
    
    aligner_sem, in_dim_sem = load_dynamic_model(path_sem, 1280, device)
    aligner_struct, in_dim_struct = load_dynamic_model(path_struct, 4096, device)

    if aligner_sem is None:
        return None

    # 2. Caricamento Dati fMRI reali
    test_h5 = f"Subject{sub_idx}_ImageNetTest.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(BASE_DIR, "data/GOD_Dataset/fmri_files/test"),
        os.path.join(BASE_DIR, "data/GOD_Dataset/images/test"), 
        test_h5, {"VC": ALL_ROI_MAPPINGS["VC"]}
    )
    
    grouped_fmri = defaultdict(list)
    for i, vec in enumerate(raw_data['fmri']):
        fname = os.path.basename(raw_data['image_paths'][i])
        base_id = os.path.splitext(fname)[0]
        grouped_fmri[base_id].append(vec)

    pred_clip_list, pred_mask_list, target_indices = [], [], []

    # 3. Inferenza dai decodificatori
    with torch.no_grad():
        for i, t_info in enumerate(targets_info):
            base_id = t_info['id']
            if base_id in grouped_fmri:
                # Calcoliamo la media delle trial
                fmri_mean = torch.tensor(np.array(grouped_fmri[base_id]), dtype=torch.float32).to(device).mean(dim=0, keepdim=True)
                
                # --- FIX SLICING ---
                # Adattiamo il vettore fMRI alla dimensione attesa dal modello (es. se VC è 4466 ma il modello è 3444)
                fmri_sem = fmri_mean[:, :in_dim_sem]
                fmri_struct = fmri_mean[:, :in_dim_struct] if aligner_struct else None
                
                # Predizione Semantica
                p_clip = normalize_embed(aligner_sem(fmri_sem))
                pred_clip_list.append(p_clip)
                
                # Predizione Spaziale (se disponibile)
                if aligner_struct:
                    p_mask_flat = aligner_struct(fmri_struct)
                    p_mask_64 = p_mask_flat.view(1, 64, 64)
                    d_min, d_max = p_mask_64.min(), p_mask_64.max()
                    p_mask_norm = (p_mask_64 - d_min) / (d_max - d_min + 1e-6)
                    pred_mask_list.append(p_mask_norm)
                
                target_indices.append(i)

    if not pred_clip_list: return None

    p_clip = torch.cat(pred_clip_list, dim=0)
    
    # 4. Calcolo Similarità Semantica (Pure fMRI Embedding vs GT Image Embedding)
    # Cosine Similarity
    sim_cos = torch.matmul(p_clip, t_clip.T) 
    # MSE Similarity (Convertita in similarità: 1 - MSE)
    dist_mse = torch.cdist(p_clip, t_clip, p=2)**2
    sim_mse = 1.0 - (dist_mse / 4.0) # Max dist tra vettori unitari è 2, dist^2 è 4

    # 5. Calcolo Spaziale (se disponibile)
    sim_spatial = None
    if pred_mask_list:
        p_masks = torch.cat(pred_mask_list, dim=0)
        n_trials = p_masks.size(0)
        sim_spatial = torch.zeros((n_trials, 50)).to(device)
        for j in range(n_trials):
            mse_m = torch.mean((p_masks[j] - t_masks)**2, dim=(1, 2))
            sim_spatial[j] = 1.0 - mse_m

    # 6. Joint Score (Normalizzato)
    norm_sem = min_max_normalize(sim_cos)
    sim_joint = norm_sem
    if sim_spatial is not None:
        norm_spat = min_max_normalize(sim_spatial)
        sim_joint = norm_sem + norm_spat
    
    def get_acc(matrix, indices):
        t1, t5 = 0, 0
        for j in range(len(indices)):
            correct_idx = indices[j]
            sorted_idx = torch.argsort(matrix[j], descending=True)
            rank = (sorted_idx == correct_idx).nonzero(as_tuple=True)[0].item() + 1
            if rank == 1: t1 += 1
            if rank <= 5: t5 += 1
        return (t1 / len(indices)) * 100, (t5 / len(indices)) * 100

    return {
        'cos': get_acc(sim_cos, target_indices),
        'mse': get_acc(sim_mse, target_indices),
        'spatial': get_acc(sim_spatial, target_indices) if sim_spatial is not None else (0,0),
        'joint': get_acc(sim_joint, target_indices),
        'count': len(target_indices)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Caricamento modello CLIP Ground Truth...")
    processor = CLIPImageProcessor.from_pretrained(MODEL_ID)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID).to(device).eval()

    df_tsv = pd.read_csv(TSV_PATH, sep='\t', header=None)
    targets_info = []
    for _, row in df_tsv.iterrows():
        fname = str(row[0]).strip()
        targets_info.append({
            'id': fname,
            'gt_path': os.path.join(GT_DIR, f"{fname}.JPEG"),
            'mask_path': os.path.join(MASKS_DIR, f"{fname}.npy")
        })

    # Estraiamo le feature reali (Ground Truth)
    t_clip, t_masks = extract_gt_features(targets_info, clip_model, processor, device)

    print("\n" + "="*95)
    print("🧪 VALUTAZIONE PURE fMRI DECODING (N-WAY CLASSIFICATION)")
    print("🧪 METRICA: DISTANZA TRA EMBEDDING PREDETTI E GROUND TRUTH (NO RECONSTRUCTIONS)")
    print("="*95)

    for s in range(1, 6):
        print(f"\n--- SOGGETTO S{s} ---")
        res = evaluate_subject_pure_fmri(s, t_clip, t_masks, targets_info, device)
        
        if res:
            print(f"✅ Analisi su {res['count']} stimoli (media delle presentazioni):")
            print(f"   [Pure Semantics - COSINE] Top-1: {res['cos'][0]:.2f}% | Top-5: {res['cos'][1]:.2f}%")
            print(f"   [Pure Semantics - MSE]    Top-1: {res['mse'][0]:.2f}% | Top-5: {res['mse'][1]:.2f}%")
            if res['spatial'][0] > 0:
                print(f"   [Pure Spatial Prediction] Top-1: {res['spatial'][0]:.2f}% | Top-5: {res['spatial'][1]:.2f}%")
                print(f"   [Joint fMRI Decoding]    Top-1: {res['joint'][0]:.2f}% | Top-5: {res['joint'][1]:.2f}% ⭐")
        else:
            print(f"⚠️ Dati o modelli non trovati per S{s}. Verifica i percorsi.")
            
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()