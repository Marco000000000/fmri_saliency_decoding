import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from diffusers import FluxPipeline

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
# Usiamo l'Aligner MLP standard (quello veloce da addestrare)
from src.models.flux_aligner import FmriToFluxSequenceAligner

def load_memory_bank(targets_dir, device):
    """
    Carica in RAM tutti gli embedding del Training Set per fare la ricerca.
    """
    print(f"Costruzione Memory Bank da {targets_dir}...")
    clip_bank = []
    t5_bank = []
    bank_paths = []
    
    # Lista dei file .pt generati nel preprocessing
    files = sorted([f for f in os.listdir(targets_dir) if f.endswith(".pt")])
    
    if len(files) == 0:
        raise ValueError("Nessun target trovato! Esegui prima preprocess_flux.py")

    for f in tqdm(files, desc="Loading Bank"):
        path = os.path.join(targets_dir, f)
        data = torch.load(path, map_location="cpu")
        
        # 1. CLIP Embedding (per la ricerca)
        # [1, 768] -> [768]
        clip_vec = data["clip"].squeeze(0).float()
        # Normalizziamo subito per Cosine Similarity veloce
        clip_vec = clip_vec / (clip_vec.norm(dim=-1, keepdim=True) + 1e-6)
        
        # 2. T5 Embedding (per la generazione)
        # Teniamo in bfloat16 per risparmiare RAM e passarlo a Flux
        t5_vec = data["t5"].to(dtype=torch.bfloat16) 
        
        clip_bank.append(clip_vec)
        t5_bank.append(t5_vec)
        bank_paths.append(path)
        
    # Spostiamo CLIP su GPU per ricerca ultra-veloce con matrici
    clip_bank_tensor = torch.stack(clip_bank).to(device) # [N_train, 768]
    
    return clip_bank_tensor, t5_bank, bank_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--targets_dir", type=str, default="data/GOD_Dataset/flux_targets")
    parser.add_argument("--output_dir", type=str, default="results_flux_retrieval_S3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Flux Retrieval Inference S{args.subject} | ROI: {args.rois} ---")

    # 1. SETUP FLUX
    print("Caricamento Flux.1-Schnell...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=torch.bfloat16
    ).to(device)

    # 2. CARICA MEMORY BANK (Il "Cervello Esterno")
    # Qui carichiamo tutte le conoscenze del training set
    bank_clip, bank_t5, bank_paths = load_memory_bank(args.targets_dir, device)
    print(f"Database caricato: {len(bank_paths)} concetti disponibili.")

    # 3. SETUP ALIGNER (Il Decodificatore)
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    aligner = FmriToFluxSequenceAligner(input_dim=input_dim).to(device)
    model_path = f"trained_fmri_decoders/flux_aligner/S{args.subject}_{args.rois}/best_flux_aligner.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello non trovato in {model_path}")
        return

    aligner.load_state_dict(torch.load(model_path, map_location=device))
    aligner.eval()

    # 4. CARICA DATI TEST
    test_h5 = f"Subject{args.subject}_ImageNetTest.h5"
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "test"),
        os.path.join(args.dataset_root, "images", "test"),
        test_h5, rois_dict
    )

    grouped = defaultdict(list)
    img_map = {}
    for i, vec in enumerate(raw_data['fmri']):
        p = raw_data['image_paths'][i]
        fname = os.path.basename(p)
        grouped[fname].append(vec)
        img_map[fname] = p
    
    unique_files = sorted(list(grouped.keys()))
    os.makedirs(args.output_dir, exist_ok=True)
    
    generator = torch.Generator(device).manual_seed(args.seed)

    print(f"Inizio decodifica su {len(unique_files)} immagini...")

    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        
        # Prendi media fMRI
        trials = np.array(grouped[fname]) 
        fmri_mean = torch.tensor(trials.mean(axis=0, keepdims=True)).float().to(device)
        
        with torch.no_grad():
            # A. IL CERVELLO PREDICE
            # L'aligner ci dà un vettore CLIP "sporco" basato sull'attività cerebrale
            pred_clip, _ = aligner(fmri_mean) 
            
            # Normalizziamo per la ricerca
            pred_clip_norm = pred_clip / (pred_clip.norm(dim=-1, keepdim=True) + 1e-6)
            
            # B. RICERCA (RETRIEVAL)
            # Calcoliamo la similarità con TUTTO il training set
            # [1, 768] @ [N_train, 768].T = [1, N_train]
            similarity = torch.matmul(pred_clip_norm, bank_clip.t())
            
            # Troviamo l'indice dell'immagine più simile
            best_idx = similarity.argmax().item()
            score = similarity.max().item()
            
            # Recuperiamo il "Pensiero Pulito" (T5 Embedding) dal database
            # Questo embedding contiene la descrizione perfetta (es. "A photo of a great white shark")
            retrieved_t5 = bank_t5[best_idx].to(device) # Spostiamo su GPU ora
            
            # Recuperiamo il nome del file retrieved per debug
            retrieved_fname = os.path.basename(bank_paths[best_idx])
            
            # C. PREPARAZIONE PER FLUX
            # Usiamo il T5 recuperato per la SEMANTICA (Cosa è?)
            # Usiamo il CLIP predetto dal cervello per lo STILE/COLORE (Com'è?)
            brain_clip_bfloat = pred_clip.to(dtype=torch.bfloat16)

        # D. GENERAZIONE
        # prompt_embeds -> Guida il contenuto (Retrieval)
        # pooled_prompt_embeds -> Guida lo stile (Predizione fMRI)
        gen_image = pipe(
            prompt_embeds=retrieved_t5,      
            pooled_prompt_embeds=brain_clip_bfloat, 
            num_inference_steps=4,
            guidance_scale=0.0,
            height=768, 
            width=768,
            generator=generator
        ).images[0]

        # E. VISUALIZZAZIONE RISULTATI
        W, H = 512, 512
        gen_image = gen_image.resize((W, H))
        grid = Image.new("RGB", (W*2, H), "black")
        
        # 1. Ground Truth (Immagine Reale vista)
        if os.path.exists(img_map[fname]):
            try:
                gt_img = Image.open(img_map[fname]).convert("RGB").resize((W,H))
                grid.paste(gt_img, (0, 0))
            except: pass
        
        # 2. Ricostruzione
        grid.paste(gen_image, (W, 0))
        
        d = ImageDraw.Draw(grid)
        
        # Annotazioni
        d.text((10, 10), "GT (Real)", fill="white")
        d.text((W+10, 10), "Flux Retrieval", fill="yellow")
        
        # Scriviamo cosa ha trovato il retrieval in basso
        label_text = f"Retrieved: {retrieved_fname[:25]}... (Sim: {score:.2f})"
        d.rectangle([W+10, H-30, W+300, H-5], fill="black")
        d.text((W+15, H-25), label_text, fill="white")
        
        grid.save(os.path.join(args.output_dir, f"{real_key}_rag.png"))

    print(f"Finito. Immagini salvate in {args.output_dir}")

if __name__ == "__main__":
    main()