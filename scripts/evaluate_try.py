import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm
import lpips
import torchvision.transforms.functional as F_vision
import torchvision.models as models
from transformers import CLIPImageProcessor, CLIPModel, CLIPVisionModelWithProjection

class AlexNetExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).eval()
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier[:-1] 
    def forward(self, x):
        with torch.no_grad():
            x2 = self.features[:6](x).flatten(1)
            x5 = self.features[:13](x).flatten(1)
            x_full = self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))
        return x2, x5, x_full

def normalize_embed(emb): return emb / emb.norm(p=2, dim=-1, keepdim=True)

def apply_smart_mask(img_pil, mask_np, bbox, mode):
    if mode == "full" or mask_np is None: return img_pil
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(img_pil.size, Image.NEAREST)
    blurred_img = img_pil.filter(ImageFilter.GaussianBlur(radius=30))
    if mode == "foreground":
        fg_isolated = Image.composite(img_pil, blurred_img, mask_pil.convert("L"))
        if bbox: return fg_isolated.crop(bbox).resize(img_pil.size, Image.BICUBIC)
        return fg_isolated
    elif mode == "background":
        inv_mask = ImageOps.invert(mask_pil.convert("L"))
        return Image.composite(img_pil, blurred_img, inv_mask)

def get_n_way(g_list, t_list):
    if len(g_list) == 0: return 0.0, 0.0
    sim = torch.matmul(torch.cat(g_list, dim=0), torch.cat(t_list, dim=0).T)
    t1, t5 = 0, 0
    for i in range(len(g_list)):
        rank = (torch.argsort(sim[i], descending=True) == i).nonzero(as_tuple=True)[0].item() + 1
        if rank == 1: t1 += 1
        if rank <= 5: t5 += 1
    return (t1 / len(g_list)) * 100, (t5 / len(g_list)) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["god", "shen"], required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--test_type", choices=["natural", "imagery"], default="natural")
    parser.add_argument("--gen_dir", type=str, required=True, help="Cartella con le immagini generate")
    parser.add_argument("--eval_mode", choices=["full", "foreground", "background"], default="full")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"
    masks_dir = f"{base_dir}/data/precomputed_binary_masks"
    
    if args.dataset == "shen":
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv"
        gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"
    else: 
        tsv_path = f"{base_dir}/data/Shen2019/stimuli/stimulus_NaturalImageTest.tsv" 
        gt_dir = f"{base_dir}/data/GOD_Dataset/images/test"
        ext = ".JPEG"

    df_tsv = pd.read_csv(tsv_path, sep='\t', header=None)
    id_to_filename, id_to_maskname = {}, {}
    for _, row in df_tsv.iterrows():
        img_id, fname = int(row[3]), str(row[0]).strip()
        if args.dataset == "god":
            key = f"{fname}{ext}" if not fname.endswith(ext) else fname
            id_to_filename[key] = key
            id_to_maskname[key] = f"{fname}.npy"
        else:
            key = f"id_{img_id}.png"
            id_to_filename[key] = f"{fname}{ext}" if not fname.endswith(ext) else fname
            id_to_maskname[key] = f"{fname}.npy"

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device).eval()
    clip_processor_b = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model_b = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    clip_processor_xl = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    clip_model_xl = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device).eval()
    alexnet = AlexNetExtractor().to(device)
    import torchvision.transforms as T
    alex_transform = T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    results = {'lpips': [], 'clip_b_g': [], 'clip_b_t': [], 'clip_xl_g': [], 'clip_xl_t': [], 'alex2_g': [], 'alex2_t': [], 'alex5_g': [], 'alex5_t': [], 'alex7_g': [], 'alex7_t': []}

    # MATCHING LOGIC (With PNG/JPEG extension fix for GOD)
    valid_pairs = []
    for f in os.listdir(args.gen_dir):
        if args.dataset == "god":
            gt_key = os.path.splitext(f)[0] + ".JPEG"
        else:
            gt_key = f
            
        if gt_key in id_to_filename:
            valid_pairs.append((f, gt_key)) 

    if not valid_pairs:
        print(f"❌ Nessuna immagine valida trovata in {args.gen_dir}")
        return

    print(f"Valutazione {len(valid_pairs)} immagini in {args.gen_dir} | MODO: {args.eval_mode.upper()}...")
    for gen_f, gt_key in tqdm(valid_pairs):
        gt_path = os.path.join(gt_dir, id_to_filename[gt_key])
        gen_path = os.path.join(args.gen_dir, gen_f)
        if not os.path.exists(gt_path) or not os.path.exists(gen_path): continue
        
        gt_pil = Image.open(gt_path).convert("RGB").resize((512, 512))
        gen_pil = Image.open(gen_path).convert("RGB").resize((512, 512))
        
        mask_np, bbox = None, None
        if args.test_type != "imagery":
            mask_path = os.path.join(masks_dir, id_to_maskname[gt_key])
            if os.path.exists(mask_path): 
                mask_np = np.load(mask_path).reshape(64, 64)
                bbox_64 = Image.fromarray((mask_np * 255).astype(np.uint8)).getbbox()
                if bbox_64: bbox = (bbox_64[0]*8, bbox_64[1]*8, bbox_64[2]*8, bbox_64[3]*8)

        gt_pil = apply_smart_mask(gt_pil, mask_np, bbox, args.eval_mode)

        with torch.no_grad():
            t_gt_lpips = (F_vision.to_tensor(gt_pil).to(device) * 2.0 - 1.0).unsqueeze(0)
            t_gen_lpips = (F_vision.to_tensor(gen_pil).to(device) * 2.0 - 1.0).unsqueeze(0)
            results['lpips'].append(loss_fn_vgg(t_gen_lpips, t_gt_lpips).item())

            results['clip_b_t'].append(normalize_embed(clip_model_b.get_image_features(**clip_processor_b(images=gt_pil, return_tensors="pt").to(device))).cpu())
            results['clip_b_g'].append(normalize_embed(clip_model_b.get_image_features(**clip_processor_b(images=gen_pil, return_tensors="pt").to(device))).cpu())

            results['clip_xl_t'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gt_pil, return_tensors="pt").to(device)).image_embeds).cpu())
            results['clip_xl_g'].append(normalize_embed(clip_model_xl(**clip_processor_xl(images=gen_pil, return_tensors="pt").to(device)).image_embeds).cpu())

            a2_gt, a5_gt, a7_gt = alexnet(alex_transform(gt_pil).unsqueeze(0).to(device))
            a2_gen, a5_gen, a7_gen = alexnet(alex_transform(gen_pil).unsqueeze(0).to(device))

            results['alex2_t'].append(normalize_embed(a2_gt).cpu()); results['alex2_g'].append(normalize_embed(a2_gen).cpu())
            results['alex5_t'].append(normalize_embed(a5_gt).cpu()); results['alex5_g'].append(normalize_embed(a5_gen).cpu())
            results['alex7_t'].append(normalize_embed(a7_gt).cpu()); results['alex7_g'].append(normalize_embed(a7_gen).cpu())

    avg_lpips = np.mean(results['lpips'])
    cb_t1, cb_t5 = get_n_way(results['clip_b_g'], results['clip_b_t'])
    cxl_t1, cxl_t5 = get_n_way(results['clip_xl_g'], results['clip_xl_t'])
    a2_t1, a2_t5 = get_n_way(results['alex2_g'], results['alex2_t'])
    a5_t1, a5_t5 = get_n_way(results['alex5_g'], results['alex5_t'])
    a7_t1, a7_t5 = get_n_way(results['alex7_g'], results['alex7_t'])

    print(f"\n--- RISULTATI {args.eval_mode.upper()} ---")
    print(f"LPIPS ↓: {avg_lpips:.4f}")
    print(f"CLIP-B:  {cb_t1:.2f}% / {cb_t5:.2f}%")
    print(f"CLIP-XL: {cxl_t1:.2f}% / {cxl_t5:.2f}%")
    print(f"Alex2:   {a2_t1:.2f}% / {a2_t5:.2f}%")
    print(f"Alex5:   {a5_t1:.2f}% / {a5_t5:.2f}%")
    print(f"Alex7:   {a7_t1:.2f}% / {a7_t5:.2f}%\n")

if __name__ == "__main__": main()