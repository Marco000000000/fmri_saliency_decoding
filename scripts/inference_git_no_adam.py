import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import textwrap  # <--- Per mandare a capo il testo
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from transformers.modeling_outputs import BaseModelOutput

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs

# ==========================================
# 1. MODELLO LINEARE
# ==========================================
class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.linear(x)

# ==========================================
# 2. GENERAZIONE TESTO
# ==========================================
def generate_caption(model, tokenizer, embedding, max_len=20):
    if len(embedding.shape) == 2:
        embedding = embedding.unsqueeze(1)
        
    encoder_outputs = BaseModelOutput(last_hidden_state=embedding)
    
    output_ids = model.generate(
        encoder_outputs=encoder_outputs,
        max_length=max_len,
        num_beams=5,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ==========================================
# 3. DISEGNO CON WRAPPING (A CAPO)
# ==========================================
def draw_multiline_comparison(real_img_path, gt_text, avg_text, single_text, output_path):
    """
    Crea un layout con immagine a sinistra e 3 blocchi di testo a destra.
    Gestisce il testo lungo mandandolo a capo.
    """
    W, H = 512, 512
    # Larghezza pannello testo
    text_W = 512 
    canvas = Image.new("RGB", (W + text_W, H), "black")
    
    # A. Foto Reale
    if os.path.exists(real_img_path):
        try:
            img = Image.open(real_img_path).convert("RGB").resize((W, H))
            canvas.paste(img, (0, 0))
        except: pass
        
    d = ImageDraw.Draw(canvas)
    try:
        font_header = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        font_text = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font_header = ImageFont.load_default()
        font_text = ImageFont.load_default()

    # Funzione helper per scrivere blocco
    def draw_block(title, text, y_start, color_title):
        # Disegna Titolo
        d.text((W + 20, y_start), title, fill=color_title, font=font_header)
        
        # Wrappa il testo (ogni 45 caratteri va a capo)
        lines = textwrap.wrap(text, width=45)
        
        current_y = y_start + 30
        for line in lines:
            d.text((W + 20, current_y), line, fill="white", font=font_text)
            current_y += 22 # Interlinea
        
        return current_y + 30 # Ritorna la Y per il prossimo blocco

    # 1. GROUND TRUTH (Cyan)
    next_y = draw_block("COMPUTER VISION (GT):", gt_text, 30, "cyan")
    
    # 2. INPUT AVG (Lime - Migliore)
    next_y = draw_block("BRAIN (Input Avg):", avg_text, next_y, "lime")
    
    # 3. SINGLE TRIAL (Orange - Rumoroso)
    draw_block("BRAIN (Single Trial):", single_text, next_y, "orange")
    
    canvas.save(output_path)

# ==========================================
# 4. MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_captioning_compare")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Inference Multi-Line (Avg vs Single) S{args.subject} ---")

    # Setup Modelli
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    caption_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    caption_model.eval()

    # Aligner
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    target_dim = caption_model.config.encoder.hidden_size
    
    aligner = SimpleLinearAligner(input_dim, output_dim=target_dim).to(device)
    
    # Usa il modello addestrato con Cosine Loss (quello più recente)
    model_path = f"trained_fmri_decoders/caption_aligner_cosine/S{args.subject}_{args.rois}/best_caption_aligner.pth"
    if not os.path.exists(model_path):
        print(f"❌ Manca il modello: {model_path}")
        return
        
    aligner.load_state_dict(torch.load(model_path, map_location=device))
    aligner.eval()

    # Dati
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
    
    print(f"Generating comparison for {len(unique_files)} images...")
    
    log_results = []

    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        
        # A. Ground Truth
        gt_caption = "N/A"
        if os.path.exists(img_map[fname]):
            try:
                i = Image.open(img_map[fname]).convert("RGB")
                pixel_values = processor(images=i, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    vit_out = caption_model.encoder(pixel_values).last_hidden_state
                    gt_caption = generate_caption(caption_model, tokenizer, vit_out[:, 0, :])
            except: pass

        # Prepare fMRI Data
        trials_np = np.array(grouped[fname])
        trials_tensor = torch.tensor(trials_np).float().to(device)
        
        with torch.no_grad():
            # B. Caso 1: MEDIA (Standard)
            # Prende tutti i trial, fa la media, riduce il rumore
            fmri_avg = trials_tensor.mean(dim=0, keepdim=True)
            emb_avg = aligner(fmri_avg)
            caption_avg = generate_caption(caption_model, tokenizer, emb_avg)
            
            # C. Caso 2: SINGOLO TRIAL (Real-time simulation)
            # Prende solo il primo trial. Molto più difficile.
            fmri_single = trials_tensor[0:1] 
            emb_single = aligner(fmri_single)
            caption_single = generate_caption(caption_model, tokenizer, emb_single)

        # Disegno output multiriga
        out_img_path = os.path.join(args.output_dir, f"{real_key}_compare.png")
        draw_multiline_comparison(img_map[fname], gt_caption, caption_avg, caption_single, out_img_path)
        
        log_line = f"FILE: {fname}\n GT: {gt_caption}\n AVG: {caption_avg}\n SNG: {caption_single}\n"
        log_results.append(log_line)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write("\n".join(log_results))

    print(f"Finito. Risultati in {args.output_dir}")

if __name__ == "__main__":
    main()