import numpy as np
import cv2
import pandas as pd
import os
import urllib.request
from PIL import Image
from scipy.spatial import cKDTree

# --- PALETTE SETUP ---
ADE20K_OFFICIAL_CSV_URL = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/data/objectInfo150.csv"

def get_ade20k_palette_official():
    csv_path = "objectInfo150.csv"
    if not os.path.exists(csv_path):
        try:
            urllib.request.urlretrieve(ADE20K_OFFICIAL_CSV_URL, csv_path)
        except:
            return get_fallback_palette()
    try:
        df = pd.read_csv(csv_path)
        colors = df[['R', 'G', 'B']].values
        if len(colors) != 150: return get_fallback_palette()
        return colors.astype(np.uint8)
    except:
        return get_fallback_palette()

def get_fallback_palette():
    np.random.seed(42)
    return np.random.randint(0, 255, (150, 3), dtype=np.uint8)

PALETTE_ARRAY = get_ade20k_palette_official()
COLOR_TREE = cKDTree(PALETTE_ARRAY)

# --- UTILS ---
def normalize_min_max(img_array):
    """
    Stira il contrasto (0-255).
    """
    if img_array.max() > img_array.min():
        return ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    return img_array.astype(np.uint8)

def guided_filter_channel(I, p, r, eps):
    """
    Algoritmo Guided Filter.
    Assicurarsi che I e p siano float32.
    """
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r)) # Usa CV_32F esplicitamente
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

    q = mean_a * I + mean_b
    return q

# --- CORE FUNCTIONS ---
def refine_jointly(depth_pil, seg_pil):
    """
    Raffina la Segmentation Map usando la Depth Map come guida.
    """
    # 1. Prepare Depth (Guida)
    # PIL -> Numpy -> MinMax -> Float32 [0,1]
    depth = np.array(depth_pil)
    depth = normalize_min_max(depth)
    # FIX: Cast esplicito a float32 dopo la divisione
    depth = (depth / 255.0).astype(np.float32)
    
    # Bilateral Filter richiede float32
    depth = cv2.bilateralFilter(depth, 9, 75, 75)
    
    # 2. Prepare Seg (Sorgente)
    seg = np.array(seg_pil)
    seg = normalize_min_max(seg)
    # FIX: Cast esplicito a float32
    seg = (seg / 255.0).astype(np.float32)
    
    # 3. Guided Filter
    r = 8       
    eps = 1e-3  
    
    refined_channels = []
    for i in range(3): 
        p = seg[:, :, i]
        q = guided_filter_channel(depth, p, r, eps)
        refined_channels.append(q)
    
    seg_refined = np.stack(refined_channels, axis=2)
    seg_refined = np.clip(seg_refined * 255, 0, 255).astype(np.uint8)
    
    # 4. Quantizzazione
    h, w, c = seg_refined.shape
    img_flat = seg_refined.reshape(-1, 3)
    _, indices = COLOR_TREE.query(img_flat, k=1)
    seg_final = PALETTE_ARRAY[indices].reshape(h, w, c).astype(np.uint8)
    
    # 5. Pulizia Finale
    seg_final = cv2.medianBlur(seg_final, 7)
    
    return Image.fromarray((depth * 255).astype(np.uint8)), Image.fromarray(seg_final)

def refine_depth(pil_img):
    img = np.array(pil_img)
    img = normalize_min_max(img)
    
    # CLAHE works on uint8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)
    
    # Bilateral Filter needs float32 if ranges are 0-1, or uint8 is fine too.
    # Qui usiamo uint8 diretto che OpenCV supporta
    img_clean = cv2.bilateralFilter(img_eq, 9, 75, 75)
    
    return Image.fromarray(img_clean)

def refine_segmentation(pil_img):
    return refine_jointly(pil_img.convert("L"), pil_img)[1]