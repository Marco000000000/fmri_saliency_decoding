import argparse
import os
import sys
import pickle
from time import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Importiamo le librerie ufficiali del Kamitani Lab
from slir import SparseLinearRegression
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset import MultimodalFmriDataset 

# --- Funzione helper per pre-calcolare embeddings con IP-Adapter (invariata) ---
def precompute_embeddings(loader, pipe, device, desc="Extracting Features"):
    all_fmri = []
    all_targets = []
    print(f"--- {desc} ---")
    with torch.no_grad():
        for batch in tqdm(loader):
            all_fmri.append(batch['fmri'])
            img_paths = batch['image_path']
            
            batch_targets = []
            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except:
                    img = Image.new('RGB', (224, 224))
                
                emb = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=img, 
                    ip_adapter_image_embeds=None, 
                    device=device, 
                    num_images_per_prompt=1, 
                    do_classifier_free_guidance=False
                )
                batch_targets.append(emb[0].squeeze(0).squeeze(0).cpu()) 
            
            all_targets.append(torch.stack(batch_targets))
            
    return torch.cat(all_fmri, dim=0), torch.cat(all_targets, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/kamitani_slir")
    parser.add_argument("--n_voxel", type=int, default=500, help="Numero di voxel da selezionare per feature")
    parser.add_argument("--n_iter", type=int, default=200, help="Iterazioni massime per SLiR")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training KAMITANI SLiR Aligner (S{args.subject} {args.rois}) ---")

    # 1. Setup SDXL & IP-Adapter per estrazione target
    print("Caricamento SDXL e IP-Adapter per estrazione ground truth...")
    controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, torch_dtype=torch.float16
    ).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

    # 2. Data Preparation
    roi_keys = parse_roi_keys(args.rois)
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}

    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    
    if raw_data is None: return

    train_dict, val_dict = split_data_for_class(raw_data, 0.9)
    train_loader = DataLoader(MultimodalFmriDataset(train_dict['fmri'], train_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=64, shuffle=False)
    val_loader = DataLoader(MultimodalFmriDataset(val_dict['fmri'], val_dict['image_paths'], args.dataset_root, load_maps=False), batch_size=64, shuffle=False)

    # PRE-COMPUTATION (PyTorch tensors)
    X_train_pt, Y_train_pt = precompute_embeddings(train_loader, pipe, device, "Training Set")
    X_val_pt, Y_val_pt = precompute_embeddings(val_loader, pipe, device, "Validation Set")
    
    # Libera VRAM per evitare OOM, la SLiR girerà in CPU su array numpy
    del pipe
    del controlnet
    torch.cuda.empty_cache()

    # 3. Conversione in NumPy (Fondamentale per bdpy e slir)
    X_train = X_train_pt.numpy()
    Y_train = Y_train_pt.numpy()
    X_val = X_val_pt.numpy()
    Y_val = Y_val_pt.numpy()

    target_dim = Y_train.shape[1]
    print(f"\nDimensioni Dataset NumPy: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"Esecuzione Regressione Lineare Sparsa su {target_dim} features...")

    # --- PIPELINE KAMITANI & HORIKAWA (2017) ---
    
    # A. Normalizzazione dei dati cerebrali (X)
    # Calcolo su X_train con gradi di libertà = 1 (ddof=1) per stima campionaria
    norm_mean_x = np.mean(X_train, axis=0)
    norm_scale_x = np.std(X_train, axis=0, ddof=1)
    
    # Salvaguardia contro divisioni per zero se un voxel è costante
    norm_scale_x[norm_scale_x == 0] = 1.0

    X_train = (X_train - norm_mean_x) / norm_scale_x
    X_val = (X_val - norm_mean_x) / norm_scale_x

    y_pred_list = []
    models_dict = {}

    # B. Modelli Indipendenti per singola feature (ciclo sulle unità)
    for i in range(target_dim):
        start_time = time()
        
        # Estrazione singola feature d'immagine (1D)
        y_train_unit = Y_train[:, i]
        y_val_unit = Y_val[:, i]

        # Normalizzazione della feature Y (sempre basata solo sul train e con ddof=1)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1.0 if std_y == 0 else std_y  # Gestione di std a zero

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # C. Selezione Voxel (Feature Selection)
        # Calcolo correlazione tra la singola feature Y e l'intera matrice X normalizzata
        corr = corrcoef(y_train_unit, X_train, var='col')
        
        # Selezione dei top n_voxel in base al valore assoluto della correlazione
        x_train_unit, voxel_index = select_top(X_train, np.abs(corr), args.n_voxel, axis=1, verbose=False)
        x_val_unit = X_val[:, voxel_index]

        # D. Aggiunta dell'intercetta (bias term)
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_val_unit = add_bias(x_val_unit, axis=1)

        # E. Regressione Lineare Sparsa
        model = SparseLinearRegression(n_iter=args.n_iter, prune_mode=1)
        
        try:
            model.fit(x_train_unit, y_train_unit)
            y_pred = model.predict(x_val_unit)
        except Exception as e:
            # Fallback matematico (se SLiR converge a vuoto o i dati sono instabili)
            y_pred = np.zeros(y_val_unit.shape)

        # F. Denormalizzazione basata sulla scala del train
        y_pred = y_pred * norm_scale_y + norm_mean_y
        y_pred_list.append(y_pred)

        # G. Salvataggio componenti nel dizionario
        models_dict[i] = {
            'voxel_index': voxel_index,
            'norm_mean_y': norm_mean_y,
            'norm_scale_y': norm_scale_y,
            'model': model
        }

        if (i + 1) % 50 == 0:
            print(f"Elaborate {i + 1}/{target_dim} features... (Ultimo blocco: {time() - start_time:.3f} sec)")

    # 4. Ricostruzione Vettore Predetto Multidimensionale
    Y_predicted = np.vstack(y_pred_list).T

    # 5. Metriche Globali (Cosine Similarity media)
    cos_sim = np.sum(Y_predicted * Y_val, axis=1) / (np.linalg.norm(Y_predicted, axis=1) * np.linalg.norm(Y_val, axis=1) + 1e-8)
    mean_cos_sim = np.mean(cos_sim)
    print(f"\n--- Training Completato ---")
    print(f"Validation Cosine Similarity Media: {mean_cos_sim:.4f}")

    # 6. Esportazione File
    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)
    
    # Il pickle raggruppa le statistiche globali (X) e il dizionario delle unità (modello, y_stats, indici voxel)
    final_save_dict = {
        'norm_mean_x': norm_mean_x,
        'norm_scale_x': norm_scale_x,
        'models': models_dict
    }
    
    # NOTA: se stai facendo un decoder strutturale e uno semantico, 
    # rinomina il file in "kamitani_slir_models_sem.pkl" o "_struct.pkl" a seconda del layer target.
    out_file = os.path.join(save_path, "kamitani_slir_models.pkl")
    with open(out_file, 'wb') as f:
        pickle.dump(final_save_dict, f)
    
    print(f"Modello globale SLiR salvato in: {out_file}")

if __name__ == "__main__":
    main()