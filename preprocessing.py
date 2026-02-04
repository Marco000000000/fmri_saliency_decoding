import os
import argparse
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from PIL import Image
from tqdm import tqdm
import warnings

# Ignora i warning
warnings.filterwarnings("ignore")

from transformers import DPTImageProcessor, DPTForDepthEstimation
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    parser = argparse.ArgumentParser(description="Genera mappe GT a GRANA GROSSA (SAM + CLIP)")
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset", help="Root del dataset GOD")
    parser.add_argument("--sam_model", type=str, default="facebook/sam-vit-base", help="Modello SAM")
    return parser.parse_args()

# --- DATI ADE20K ---
def get_ade20k_palette():
    """Palette colori standard ADE20K (150 classi)."""
    return [
        0,0,0, # Background
        120, 120, 120, 180, 120, 120, 6, 230, 230, 80, 50, 50, 4, 200, 3, 120, 120, 80, 140, 140, 140, 204, 5, 255, 230, 230, 230, 4, 250, 7, 224, 5, 255, 235, 255, 7, 150, 5, 61, 120, 120, 70, 8, 255, 51, 255, 6, 82, 143, 255, 140, 204, 255, 4, 255, 51, 7, 204, 70, 3, 0, 102, 200, 61, 230, 250, 255, 6, 51, 11, 102, 255, 255, 7, 71, 255, 9, 224, 9, 7, 230, 220, 220, 220, 255, 9, 92, 112, 9, 255, 8, 255, 214, 7, 255, 224, 255, 184, 6, 10, 255, 71, 255, 41, 10, 7, 255, 255, 224, 255, 8, 102, 8, 255, 255, 61, 6, 255, 194, 7, 255, 122, 8, 0, 255, 20, 255, 8, 41, 255, 5, 153, 6, 51, 255, 235, 12, 255, 160, 150, 20, 0, 163, 255, 140, 140, 140, 250, 10, 15, 20, 255, 0, 31, 255, 0, 255, 31, 0, 255, 224, 0, 153, 255, 0, 0, 0, 255, 255, 71, 0, 0, 235, 255, 0, 173, 255, 31, 0, 255, 11, 200, 200, 255, 82, 0, 0, 255, 245, 0, 61, 255, 0, 255, 112, 0, 255, 133, 255, 0, 0, 255, 163, 0, 255, 102, 0, 194, 255, 0, 0, 143, 255, 51, 255, 0, 0, 82, 255, 0, 255, 41, 0, 255, 173, 10, 0, 255, 173, 255, 0, 0, 255, 153, 255, 92, 0, 255, 0, 255, 255, 0, 245, 255, 0, 102, 255, 173, 0, 255, 0, 20, 255, 184, 184, 0, 31, 255, 0, 255, 61, 0, 71, 255, 255, 0, 204, 0, 255, 194, 0, 255, 82, 0, 10, 255, 0, 112, 255, 51, 0, 255, 0, 194, 255, 0, 122, 255, 0, 255, 163, 255, 150, 20, 0, 255, 10, 0, 255, 122, 0, 255, 10, 255, 0, 0, 224, 255, 143, 255, 0, 82, 0, 255, 163, 255, 0, 255, 235, 0, 8, 184, 170, 133, 0, 255, 0, 255, 92, 184, 0, 255, 255, 0, 31, 0, 184, 255, 0, 214, 255, 255, 0, 112, 92, 255, 0, 0, 224, 255, 112, 224, 255, 70, 184, 160, 163, 0, 255, 153, 0, 255, 71, 255, 0, 255, 0, 163, 255, 204, 0, 255, 0, 143, 0, 255, 235, 133, 255, 0, 255, 0, 235, 245, 0, 255, 255, 0, 122, 255, 245, 0, 10, 190, 212, 214, 255, 0, 0, 204, 255, 20, 0, 255, 255, 255, 0, 0, 153, 255, 0, 41, 255, 0, 255, 204, 41, 0, 255, 41, 255, 0, 173, 0, 255, 0, 245, 255, 71, 0, 255, 122, 0, 255, 0, 255, 184, 0, 92, 255, 184, 255, 0, 0, 133, 255, 255, 214, 0, 25, 194, 194, 102, 255, 0, 92, 0, 255
    ]

def get_ade20k_classes():
    """Lista delle 150 classi ADE20K per CLIP."""
    return [
        "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk",
        "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa",
        "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
        "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink",
        "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow",
        "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
        "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
        "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television",
        "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
        "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
        "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
        "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture",
        "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
        "bulletin board", "shower", "radiator", "glass", "clock", "flag"
    ]

def main(args):
    print(f"Usando device: {device}")
    
    # --- 1. DEPTH (MiDaS) ---
    print("Caricamento MiDaS (Depth)...")
    depth_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device).eval()

    # --- 2. SAM (Segment Anything) ---
    # --- 2. SAM (Segment Anything) ---
    print(f"Caricamento SAM ({args.sam_model})...")
    
    # CONFIGURAZIONE BILANCIATA (Meno rumore, ma oggetti interi)
    sam_generator = pipeline(
        "mask-generation", 
        model=args.sam_model, 
        device=0 if torch.cuda.is_available() else -1, 
        points_per_batch=64,
        points_per_side=32,         # Torniamo a 32 per catturare dettagli come la staccionata
        pred_iou_thresh=0.82,       # ABBASSATO (era 0.88): Accetta forme un po' meno perfette
        stability_score_thresh=0.85,# ABBASSATO (era 0.95): Fondamentale per vedere il corpo del mulino!
        crop_n_layers=0             # Mantiene la visione d'insieme (evita i coriandoli nell'erba)
    )

    # --- 3. CLIP (Classificatore) ---
    print("Caricamento CLIP (openai/clip-vit-base-patch32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    ade_classes = get_ade20k_classes()
    print("Encoding classi ADE20K con CLIP...")
    with torch.no_grad():
        inputs_text = clip_processor(text=ade_classes, return_tensors="pt", padding=True).to(device)
        text_features = clip_model.get_text_features(**inputs_text)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True) 

    palette = get_ade20k_palette()
    img_root = os.path.join(args.dataset_root, "images")
    out_root = os.path.join(args.dataset_root, "derived_maps")

    for split in ["training", "test"]:
        input_dir = os.path.join(img_root, split)
        if not os.path.exists(input_dir): continue

        out_depth = os.path.join(out_root, "depth", split)
        out_seg = os.path.join(out_root, "segmentation", split)
        os.makedirs(out_depth, exist_ok=True)
        os.makedirs(out_seg, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Elaborazione {split}: {len(image_files)} immagini...")

        for filename in tqdm(image_files):
            try:
                img_path = os.path.join(input_dir, filename)
                name_no_ext = os.path.splitext(filename)[0]
                
                image_pil = Image.open(img_path).convert("RGB")
                original_size = image_pil.size
                
                # --- A. DEPTH ---
                if not os.path.exists(os.path.join(out_depth, name_no_ext + ".png")):
                    inputs_depth = depth_processor(images=image_pil, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs_depth = depth_model(**inputs_depth)
                        predicted_depth = outputs_depth.predicted_depth
                        
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1), size=original_size[::-1], mode="bicubic", align_corners=False
                    )
                    depth_map = prediction.squeeze().cpu().numpy()
                    depth_min, depth_max = depth_map.min(), depth_map.max()
                    depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
                    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                    Image.fromarray(depth_uint8).save(os.path.join(out_depth, name_no_ext + ".png"))

                # --- B. SAM + CLIP SEGMENTATION ---
                outputs_sam = sam_generator(image_pil) 
                
                # Gestione Lista vs Dict
                if isinstance(outputs_sam, list):
                    raw_masks = outputs_sam
                elif isinstance(outputs_sam, dict):
                    raw_masks = []
                    if 'masks' in outputs_sam:
                        for idx, m_img in enumerate(outputs_sam['masks']):
                             raw_masks.append({'mask': m_img})
                
                masks_data = []
                
                # ... (dentro il loop masks_data) ...
                
                # Calcoliamo la soglia di area dinamica
                total_pixels = original_size[0] * original_size[1]
                # Scendiamo allo 0.2% (es: su 512x512 sono circa 500 pixel). 
                # Abbastanza per togliere foglie singole, ma tiene la staccionata o parti del mulino.
                min_area_thresh = total_pixels * 0.002 
                
                for m_item in raw_masks:
                    # Estrai Maschera
                    if isinstance(m_item, dict):
                        mask_img = m_item['mask']
                        bbox = m_item.get('bbox', None)
                    else: continue

                    mask_arr = np.array(mask_img) > 0
                    area = np.sum(mask_arr)
                    
                    # FILTRO RUMORE (Foglie, sassi, artefatti)
                    if area < min_area_thresh: continue 

                    # ... (resto del codice uguale: calcolo bbox e append) ...

                    # Calcolo BBox se manca
                    if bbox is None:
                        y_indices, x_indices = np.where(mask_arr)
                        if len(y_indices) == 0: continue
                        x_min, x_max = np.min(x_indices), np.max(x_indices)
                        y_min, y_max = np.min(y_indices), np.max(y_indices)
                        w = x_max - x_min
                        h = y_max - y_min
                        bbox = [x_min, y_min, w, h]

                    masks_data.append({'mask': mask_arr, 'area': area, 'bbox': bbox})

                masks_data.sort(key=lambda x: x['area'], reverse=True)

                # Canvas finale
                final_seg_map = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)

                for m_data in masks_data:
                    mask_arr = m_data['mask']
                    bbox = m_data['bbox']
                    
                    x, y, w, h = bbox
                    x1 = max(0, int(x)); y1 = max(0, int(y))
                    x2 = min(original_size[0], int(x+w)); y2 = min(original_size[1], int(y+h))
                    
                    # Filtra ritagli piccolissimi per CLIP
                    if (x2 - x1) < 16 or (y2 - y1) < 16: continue
                    if x2 <= x1 or y2 <= y1: continue

                    crop = image_pil.crop((x1, y1, x2, y2))
                    
                    inputs_crop = clip_processor(images=crop, return_tensors="pt").to(device)
                    with torch.no_grad():
                        image_features = clip_model.get_image_features(**inputs_crop)
                        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                        
                        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        best_class_idx = similarity.argmax(dim=-1).item()
                        
                    # Dipingi la maschera
                    final_seg_map[mask_arr] = best_class_idx

                # Salva risultato
                seg_img = Image.fromarray(final_seg_map)
                seg_img.putpalette(get_ade20k_palette()[3:]) 
                seg_img = seg_img.convert("RGB")
                seg_img.save(os.path.join(out_seg, name_no_ext + ".png"))

            except Exception as e:
                print(f"Errore {filename}: {e}")

if __name__ == "__main__":
    main(get_args())