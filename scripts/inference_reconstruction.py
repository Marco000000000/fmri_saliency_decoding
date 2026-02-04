import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torchvision.transforms as transforms
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
TEST_CLASSES = [
    'goldfish', 'great horned owl', 'iguana', 'duck', 'swan', 'conch', 'crab', 'killer whale', 
    'leopard', 'bat', 'fly', 'butterfly', 'goat', 'camel', 'llama', 'plane', 'wheelbarrow', 
    'beer mug', 'bowling ball', 'bulldozer', 'cannon', 'canoe', 'casket', 'wagon', 'cowboy hat', 
    'top hat', 'electric guitar', 'fire extinguisher', 'football helmet', 'piano', 'gravestone', 
    'hammock', 'harp', 'ipod', 'knob', 'post box', 'mandolin', 'kitchen', 'mosque', 
    'surgical instruments', 'coin', 'paper shredder', 'snowmobile', "sock", 'stained glass window', 
    'drum', 'umbrella', 'dvd player', 'washing machine', 'welding helmet'
]
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs
from src.models.clip_aligner import FmriToClipAligner
# Importiamo il nuovo ottimizzatore
from src.optim.feature_inversion import FeatureInversionPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="results_reconstruction_blind")
    parser.add_argument("--strength", type=float, default=0.6, help="SD strength (0.6 mantiene la struttura ottimizzata)")
    parser.add_argument("--opt_steps", type=int, default=200, help="Step di ottimizzazione immagine")
    parser.add_argument("--tv_weight", type=float, default=0.05, help="Peso Total Variation (più alto = più sfocato/pulito)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Models Setup
    # A. fMRI -> CLIP Aligner
    print("Caricamento modelli...")
    clip_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    
    aligner_clip = FmriToClipAligner(input_dim, clip_dim=clip_model.config.projection_dim).to(device)
    # Path del modello addestrato (assicurati di aver fatto il training prima!)
    aligner_path = f"trained_fmri_decoders/clip_aligner/S{args.subject}_{args.rois}/best_clip_aligner.pth"
    if not os.path.exists(aligner_path):
        print(f"ERRORE: Modello non trovato in {aligner_path}. Esegui train_clip_aligner.py prima.")
        return
    aligner_clip.load_state_dict(torch.load(aligner_path, map_location=device))
    aligner_clip.eval()

    # B. Inversion Pipeline
    inverter = FeatureInversionPipeline(clip_model, device, num_steps=args.opt_steps)

    # C. Stable Diffusion (Img2Img)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 2. Data Load (Solo fMRI e percorsi GT per confronto)
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

    print(f"Start Inference su {len(unique_files)} immagini (Blind Reconstruction)...")
    
    # Prompt fisso generico per aiutare la qualità (la semantica viene dall'immagine ottimizzata)
    # Oppure potresti usare il retrieval della classe se vuoi barare un po', ma proviamo "pure vision"
    base_prompt = "high quality, realistic photo, 8k, detailed"

    for fname in tqdm(unique_files):
        real_key = os.path.splitext(fname)[0]
        
        # Prepara dati fMRI
        trials = np.array(grouped[fname]) # [Num_Trials, Input_Dim]
        
        # --- MODALITÀ 1: Media dei Trial ---
        fmri_mean = torch.tensor(trials.mean(axis=0, keepdims=True)).float().to(device)
        gt_pil = Image.open(img_map[fname]).convert("RGB")
        gt_proc = clip_processor(images=gt_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            target_emb_mean = clip_model(**gt_proc).image_embeds
        # with torch.no_grad():
        #     target_emb_mean = aligner_clip(fmri_mean) # [1, 768]
            
        # 3. Ottimizzazione Immagine (INVERSIONE)
        # Generiamo l'immagine "sfocata" partendo da zero, guidati dall'embedding fMRI
        reconstructed_tensor = inverter.optimize_image(
            target_emb_mean, 
            image_size=512, # Generiamo a 512 per SD
            lambda_tv=args.tv_weight
        )
        
        # Converti tensor -> PIL
        rec_pil = transforms.ToPILImage()(reconstructed_tensor.squeeze(0).cpu())
        
        # 4. Refinement con Stable Diffusion
        # Usiamo l'immagine ricostruita come init_image
        gen_image = pipe(
            prompt=base_prompt,
            image=rec_pil,
            strength=args.strength, # Quanto SD può modificare l'input
            guidance_scale=7.5,
            num_inference_steps=50
        ).images[0]

        # 5. Visualization Grid
        W, H = 512, 512
        grid = Image.new("RGB", (W*3, H), "black")
        
        gt_img_path = img_map[fname]
        if os.path.exists(gt_img_path):
            gt_img = Image.open(gt_img_path).convert("RGB").resize((W,H))
            grid.paste(gt_img, (0, 0))
        
        grid.paste(rec_pil, (W, 0))
        grid.paste(gen_image, (2*W, 0))
        
        # Labels
        from PIL import ImageDraw
        d = ImageDraw.Draw(grid)
        d.text((10, 10), "GT (Reference)", fill="white")
        d.text((W+10, 10), "Optimized Input (From fMRI)", fill="yellow")
        d.text((2*W+10, 10), "Final SD Output", fill="green")
        
        grid.save(os.path.join(args.output_dir, f"{real_key}_blind.png"))
        
        # --- (Opzionale) MODALITÀ 2: Singoli Trial ---
        # Se vuoi vedere la varianza, puoi decommentare un loop simile su `trials[0]`, `trials[1]`...

    print("Finito.")

if __name__ == "__main__":
    main()