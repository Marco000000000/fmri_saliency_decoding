import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
import textwrap

# Add root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.clip_aligner import FmriToClipAligner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_end2end_S3")
    # Scaling: Usa lo stesso valore usato nel training (2.0) o aumentalo leggermente se il testo è corto
    parser.add_argument("--scale_factor", type=float, default=2.0, help="Amplificazione segnale fMRI")
    # Nota: Punta alla cartella del training End-to-End
    parser.add_argument("--model_path", type=str, default="trained_fmri_decoders/end2end_text") 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference End-to-End fMRI->Text (S{args.subject}) ---")

    # 1. CARICA MODELLO DECODER (GPT-2)
    # Usiamo lo stesso modello base, ma non tocchiamo i suoi pesi (sono standard)
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    print(f"Caricamento {model_name}...")
    
    caption_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    caption_model.config.pad_token_id = tokenizer.pad_token_id

    # 2. CARICA L'ALIGNER ADDESTRATO (End-to-End)
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    target_dim = caption_model.config.encoder.hidden_size # 768
    
    aligner = FmriToClipAligner(input_dim, clip_dim=target_dim).to(device)
    
    ckpt_path = os.path.join(args.model_path, f"S{args.subject}_{args.rois}", "best_text_aligner.pth")
    if not os.path.exists(ckpt_path):
        print(f"❌ ERRORE: Modello non trovato in {ckpt_path}")
        print("Assicurati di aver lanciato lo script '02_train_end2end_text.py'")
        return
        
    print(f"Caricamento pesi Aligner da: {ckpt_path}")
    aligner.load_state_dict(torch.load(ckpt_path, map_location=device))
    aligner.eval()

    # 3. FUNZIONE DI GENERAZIONE
    def generate_caption(fmri_vec):
        # A. Aligner
        emb = aligner(fmri_vec)
        
        # B. Scaling (Cruciale per End-to-End)
        # Deve essere coerente col training (dove abbiamo usato * 2.0)
        emb = emb * args.scale_factor
        
        # C. Inject in GPT-2
        # [Batch, Dim] -> [Batch, 1, Dim]
        encoder_outputs = BaseModelOutput(last_hidden_state=emb.unsqueeze(1))
        
        # D. Generazione
        # Parametri ottimizzati per evitare loop e frasi troppo corte
        output_ids = caption_model.generate(
            encoder_outputs=encoder_outputs,
            max_length=15,          # Le caption ImageNet sono brevi
            num_beams=5,            # Beam search per la qualità
            repetition_penalty=1.5, # Penalità ripetizioni leggermente più alta
            no_repeat_ngram_size=2, # Impedisce "a dog a dog"
            early_stopping=False
        )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 4. DATA LOADING
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
    
    print(f"Generazione su {len(unique_files)} immagini di test...")

    # 5. LOOP E VISUALIZZAZIONE
    try: font = ImageFont.truetype("arial.ttf", 18)
    except: font = ImageFont.load_default()

    for fname in tqdm(unique_files):
        trials = np.array(grouped[fname]) 
        
        # Mean Input (Il segnale più pulito)
        fmri_mean = torch.tensor(trials.mean(axis=0, keepdims=True)).float().to(device)
        
        # Single Trial (Per vedere la robustezza)
        fmri_single = torch.tensor(trials[0:1]).float().to(device)

        with torch.no_grad():
            cap_mean = generate_caption(fmri_mean)
            cap_single = generate_caption(fmri_single)

        # --- CREAZIONE IMMAGINE RISULTATO ---
        # Layout:
        # [ Immagine Reale ]
        # [ Box Testo Mean ]
        # [ Box Testo Single ]
        
        W, H = 300, 300
        text_h = 100
        
        # Carica immagine reale
        if fname in img_map and os.path.exists(img_map[fname]):
            try:
                img = Image.open(img_map[fname]).convert("RGB").resize((W, H))
            except: img = Image.new("RGB", (W, H), "gray")
        else: img = Image.new("RGB", (W, H), "gray")

        # Canvas: Larghezza immagine, Altezza img + 2 box testo
        final = Image.new("RGB", (W, H + text_h * 2), "white")
        final.paste(img, (0, 0))
        
        draw = ImageDraw.Draw(final)
        
        def draw_textbox(text, y_start, label, color):
            # Sfondo
            draw.rectangle([0, y_start, W, y_start+text_h], fill=color)
            # Label (es. "Mean Prediction")
            draw.text((10, y_start+5), label, fill="black", font=font)
            # Caption wrappata
            wrapped = textwrap.fill(text, width=35)
            draw.text((10, y_start+30), wrapped, fill="black", font=font)

        # Box 1: Mean (Verdino chiaro)
        draw_textbox(cap_mean, H, "MEAN Pred (High SNR)", "#e0ffe0")
        
        # Box 2: Single (Rossino chiaro)
        draw_textbox(cap_single, H+text_h, "SINGLE Pred (Noisy)", "#ffe0e0")

        save_name = os.path.splitext(fname)[0] + "_end2end.jpg"
        final.save(os.path.join(args.output_dir, save_name))

    print(f"Finito. Risultati salvati in: {args.output_dir}")

if __name__ == "__main__":
    main()