import argparse
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    # MODIFICA QUI: Punta direttamente alla cartella di test
    parser.add_argument("--test_dir", type=str, default="data/GOD_Dataset/images/test", 
                        help="Cartella contenente solo le immagini di test")
    parser.add_argument("--output_file", type=str, default="data/GOD_Dataset/captions_test_gt.json", 
                        help="Dove salvare il JSON finale")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per la generazione")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Generazione GT Captions (Test Set Only) su {device} ---")

    # 1. Carica il Modello (Lo stesso usato per il Brain Decoding)
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    print(f"Caricamento modello: {model_name}...")
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # 2. Trova immagini nella cartella test
    image_paths = []
    print(f"Scansione cartella test: {args.test_dir}")
    
    if not os.path.exists(args.test_dir):
        print(f"❌ Errore: La cartella {args.test_dir} non esiste!")
        return

    # Walk ricorsivo (utile se le immagini di test sono divise in sottocartelle)
    for root, dirs, files in os.walk(args.test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    # Ordinamento per coerenza
    image_paths.sort()
    
    print(f"Trovate {len(image_paths)} immagini nel Test Set.")
    if len(image_paths) == 0:
        return

    # 3. Generazione in Batch
    results = {}
    
    # Parametri ottimizzati per qualità (Beam Search)
    gen_kwargs = {
        "max_length": 30,
        "num_beams": 5,
        "early_stopping": True,
        "repetition_penalty": 1.2
    }

    print("Inizio generazione didascalie...")
    for i in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[i : i + args.batch_size]
        images = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img)
                valid_paths.append(p)
            except Exception as e:
                print(f"Errore file {p}: {e}")

        if not images:
            continue

        # Preprocessing
        inputs = feature_extractor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)

        # Forward
        with torch.no_grad():
            output_ids = model.generate(pixel_values, **gen_kwargs)
        
        # Decode
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Salva risultati: NomeFile -> Caption
        for path, cap in zip(valid_paths, captions):
            fname = os.path.basename(path)
            results[fname] = cap.strip()

    # 4. Salvataggio
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Salvato: {args.output_file}")
    
    # Preview
    if len(results) > 0:
        k = list(results.keys())[0]
        print(f"Esempio: {k} -> '{results[k]}'")

if __name__ == "__main__":
    main()