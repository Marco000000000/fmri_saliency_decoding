import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from diffusers import FluxPipeline
from huggingface_hub import login
# Add root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.config.roi_config import ALL_ROI_MAPPINGS, parse_roi_keys

def main():
    dataset_root = "data/GOD_Dataset"
    output_dir = "data/GOD_Dataset/flux_targets"
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda"
    
    # 1. Carica SOLO gli Encoder di Flux (usiamo la versione Schnell che è più leggera/veloce)
    # Nota: Carichiamo tutto in bfloat16 per risparmiare memoria
    print("Caricamento Flux Encoders (CLIP + T5)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        transformer=None,    # Non ci serve il modello di diffusione ora
        vae=None             # Non ci serve il VAE ora
    ).to(device)

    # Flux ha 2 encoder: CLIP (pooled) e T5 (sequence)
    # Dobbiamo estrarli entrambi.
    
    # 2. Carica i dati di Training per avere le label
    # (Usiamo la tua funzione esistente, ci servono solo i path delle immagini per capire le classi)
    print("Caricamento dataset...")
    # Dummy ROI load
    raw_data = load_fmri_image_pairs(
        os.path.join(dataset_root, "fmri_files", "training"),
        os.path.join(dataset_root, "images", "training"),
        "Subject3_ImageNetTraining.h5", 
        {"VC": ALL_ROI_MAPPINGS["VC"]}
    )
    
    # Mappiamo i file alle classi (per ImageNet/GOD il nome file o cartella aiuta)
    # Nel GOD dataset, dobbiamo estrarre il nome della classe. 
    # Assumiamo tu abbia un modo per mappare image_path -> "goldfish".
    # Per semplicità qui, useremo BLIP per generare una caption se non abbiamo le classi esplicite,
    # OPPURE (meglio) usiamo una stringa generica se non hai il mapping delle classi pronto.
    
    # ESEMPIO: Se hai una lista di classi ordinata come nel training:
    # Per ora simuliamo che ogni immagine abbia una label.
    # Nel GOD Dataset reale, bisogna mappare ID -> WordNet Label.
    
    print("Estrazione Embeddings...")
    
    # Dizionario per evitare di ricalcolare la stessa classe 100 volte
    cache_embeddings = {} 
    
    # Per questo esempio, usiamo una caption generica + ID per differenziare,
    # MA PER IL TUO CASO REALE: Sostituisci questo con la vera classe ImageNet.
    # Se non hai le classi pronte, dimmelo e aggiungiamo un passaggio con BLIP.
    
    # Usiamo BLIP per generare i prompt "reali" (Molto meglio per Flux)
    from transformers import BlipProcessor, BlipForConditionalGeneration
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    
    from PIL import Image
    
    for i, img_path in tqdm(enumerate(raw_data['image_paths']), total=len(raw_data['image_paths'])):
        fname = os.path.basename(img_path)
        save_name = os.path.splitext(fname)[0] + ".pt"
        save_path = os.path.join(output_dir, save_name)
        
        if os.path.exists(save_path):
            continue
            
        # 1. Genera Caption (Ground Truth Semantica)
        image = Image.open(img_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt").to(device)
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        # prompt = f"high quality photo of {caption}"
        prompt = caption # Flux preferisce naturalezza
        
       
        
            # 2. Estrai Flux Embeddings
        with torch.no_grad():
            # CORREZIONE: encode_prompt restituisce 3 valori, non 4.
            # 1. prompt_embeds (T5)
            # 2. pooled_prompt_embeds (CLIP)
            # 3. text_ids (Non ci servono)
            
            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt, 
                prompt_2=prompt,
                device=device
            )
            
            # Salviamo tutto in un dizionario
            data_to_save = {
                "t5": prompt_embeds.cpu(),
                "clip": pooled_prompt_embeds.cpu(),
                "caption": caption
            }
            
            torch.save(data_to_save, save_path)

    print(f"Fatto! Target salvati in {output_dir}")

if __name__ == "__main__":
    main()