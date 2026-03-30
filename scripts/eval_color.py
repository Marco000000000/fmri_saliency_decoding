import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import bdpy
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class SimpleLinearAligner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
        self.sigmoid = nn.Sigmoid() # FIX 1: Forza i valori strettamente tra 0.0 (Nero) e 1.0 (Luce massima)
        
    def forward(self, x): 
        return self.sigmoid(self.linear(x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", type=str, default="ROI_VC")
    parser.add_argument("--palette_size", type=int, default=8) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    
    # Dati di TEST Naturali
    # h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ImageNetTest_fmriprep_volume_native.h5"
    # tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
    # img_dir = f"{base_dir}/data/GOD_Dataset/images/test"
    h5_path = f"{base_dir}/data/Shen2019/fmri/sub-0{args.subject}_ArtificialShapes_fmriprep_volume_native.h5"
    img_dir = f"{base_dir}/data/Shen2019/stimuli/ArtificialShapes/ArtificialImage"
    tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_ArtificialImage.tsv"

    
    out_dir = f"reconstructions/S{args.subject}_color_palette_eval"
    os.makedirs(out_dir, exist_ok=True)

    print(f"🧠 Caricamento Test Set S{args.subject}...")
    bdata = bdpy.BData(h5_path)
    X_raw = bdata.select(args.roi)
    
    stim_key = 'image_index' if 'image_index' in bdata.metadata.key else 'stimulus_id'
    labels = bdata.select(stim_key)[:, 0]
    unique_labels = np.unique(labels[labels > 0])
    
    # Averaging dei 1200 trial -> 50 immagini uniche
    averaged_fmri, output_ids = [], []
    for lbl in unique_labels:
        idx = [i for i in np.where(labels == lbl)[0] if i < X_raw.shape[0]]
        if len(idx) > 0:
            averaged_fmri.append(np.mean(X_raw[idx], axis=0))
            output_ids.append(int(lbl))
    X_test = torch.tensor(np.array(averaged_fmri), dtype=torch.float32).to(device)

    # Caricamento Modello Colore (Palette 8x8)
    target_dim = args.palette_size * args.palette_size * 3 # 8 * 8 * 3 = 192
    aligner_color = SimpleLinearAligner(X_test.shape[1], target_dim).to(device)
    
    # Puntiamo alla nuova cartella del training palette
    model_path = f"{base_dir}/trained_fmri_decoders/shen_color_palette/S{args.subject}_color_aligner.pth"
    if not os.path.exists(model_path):
        print(f"❌ Errore: Modello non trovato in {model_path}. Assicurati di aver completato il training.")
        return
        
    aligner_color.load_state_dict(torch.load(model_path, map_location=device))
    aligner_color.eval()

    # Mapping TSV
    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    id_to_filename = {int(row[3]): (str(row[0]) + ".JPEG" if not str(row[0]).endswith('.JPEG') else str(row[0])) for _, row in df_tsv.iterrows()}

    print(f"🚀 Generazione Color Palette (50 immagini)...")
    for fmri_vec, img_id in tqdm(zip(X_test, output_ids), total=len(output_ids)):
        with torch.no_grad():
            pred_flat = aligner_color(fmri_vec.unsqueeze(0)) # Shape: (1, 192)
            
            # Reshape a (8, 8, 3) 
            # Non serve più il clamp(0,1) o il min-max scaling perché la Sigmoide garantisce il range [0,1]
            pred_rgb = pred_flat.view(args.palette_size, args.palette_size, 3).cpu().numpy()
            
            # Converti in immagine PIL direttamente
            pred_img_8 = Image.fromarray((pred_rgb * 255).astype(np.uint8))
            
            # Upscale a 512x512 usando NEAREST
            pred_img_large = pred_img_8.resize((512, 512), Image.NEAREST)

        # Plot comparativo
        try:
            orig_fname = id_to_filename.get(img_id)
            if orig_fname and os.path.exists(os.path.join(img_dir, orig_fname)):
                orig_img = Image.open(os.path.join(img_dir, orig_fname)).convert("RGB").resize((512, 512))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(orig_img)
                ax1.set_title(f"Originale (ID {img_id})")
                ax1.axis("off")
                
                ax2.imshow(pred_img_large)
                ax2.set_title(f"Palette fMRI ({args.palette_size}x{args.palette_size})")
                ax2.axis("off")
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"color_palette_{img_id}.png"))
                plt.close()
        except Exception as e:
            print(f"⚠️ Errore ID {img_id}: {e}")

    print(f"✅ Finito! Guarda i risultati in: {out_dir}")

if __name__ == "__main__":
    main()