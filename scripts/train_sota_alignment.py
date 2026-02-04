import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.datasets.multimodal_dataset1 import MultimodalFmriDataset
from src.models.sota_wrappers import FmriToSota

def train(model, loader, opt, criterion, device, task):
    model.train() 
    total_loss = 0
    feature_criterion = torch.nn.MSELoss() 
    
    # PULIZIA
    torch.cuda.empty_cache()

    for batch in tqdm(loader, desc=f"Train {task}", leave=False):
        fmri = batch['fmri'].to(device)
        images = batch['image'].to(device)
        
        target_features = None
        target_depth = None # Per la pixel loss

        # --- 1. TEACHER (ORACLE) ---
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                if task == 'seg':
                    # SegFormer: Feature Loss su piramide
                    outputs = model.model.segformer.encoder(images, output_hidden_states=True)
                    target_features = [t.detach().float() for t in outputs.hidden_states[1:]]
                    
                elif task == 'depth':
                    # DPT: Pixel Loss (Confrontiamo output studente con output teacher)
                    # Resize DPT 384x384
                    images_dpt = torch.nn.functional.interpolate(images, size=(384, 384), mode='bilinear')
                    
                    # Teacher Output Completo (Depth Map)
                    teacher_out = model.model(images_dpt).predicted_depth
                    
                    # Resize a 512 per matchare lo studente (che fa resize finale)
                    target_depth = F.interpolate(teacher_out.unsqueeze(1), size=(512, 512), mode='bilinear').detach().float()
                    
                    del images_dpt
            del images
        
        # --- 2. STUDENT ---
        opt.zero_grad(set_to_none=True)
        
        if task == 'seg':
            # SegFormer: Feature Alignment
            pred_features = model.forward_fmri(fmri)
            loss = 0
            for pred_map, target_map in zip(pred_features, target_features):
                loss += feature_criterion(pred_map, target_map)
                
        elif task == 'depth':
            # DPT: End-to-End Pixel Loss
            # fMRI -> Generator -> Depth Decoder -> Mappa
            feat = model.forward_fmri(fmri)
            pred_depth = model.decode_from_features(feat) # Usa il nostro SimpleDepthDecoder
            
            # Loss direttamente sulla mappa di profondità
            loss = feature_criterion(pred_depth, target_depth)

        # --- 3. OPTIMIZE ---
        loss.backward()
        opt.step()
        
        total_loss += loss.item()
        
        del target_features
        del target_depth
        
    return total_loss / len(loader)

def validate_and_save_vis(model, loader, device, task, out_dir, epoch):
    model.eval()
    # Salva una visualizzazione di debug ogni tot epoche (o alla fine della prima)
    if epoch % 5 == 0 or epoch == 0:
        with torch.no_grad():
            batch = next(iter(loader))
            fmri = batch['fmri'].to(device)
            
            # Predici e Decodifica
            pred_feat = model.forward_fmri(fmri)
            
            # --- FIX: Aggiunta la 's' finale al metodo ---
            rec_map = model.decode_from_features(pred_feat)
            
            # Post-processing per visualizzazione
            if task == 'seg':
                # Argmax sugli indici delle classi [B, 150, H, W] -> [B, H, W]
                rec_map = torch.argmax(rec_map, dim=1).float() 
                # Normalizziamo 0-150 a 0-255 per vedere qualcosa
                rec_map = (rec_map / 150.0 * 255.0)
            
            elif task == 'depth':
                # Normalizziamo la depth map per visualizzarla [0, 255]
                d_min, d_max = rec_map.min(), rec_map.max()
                rec_map = (rec_map - d_min) / (d_max - d_min + 1e-8) * 255.0

            # Salva la prima immagine del batch
            from PIL import Image
            import numpy as np
            
            # Prendi la prima del batch, porta su CPU, converti a uint8
            img_np = rec_map[0].squeeze().cpu().numpy().astype(np.uint8)
            
            # Crea cartella debug se non esiste
            os.makedirs(f"{out_dir}/debug_vis", exist_ok=True)
            Image.fromarray(img_np).save(f"{out_dir}/debug_vis/{task}_ep{epoch}.png")
            
    # Rimetti in train mode
    model.train()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--task", type=str, required=True, choices=['depth', 'seg'])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--output_dir", type=str, default="trained_sota_models")
    args = parser.parse_args()

    device = torch.device("cuda")
    
    # Dataset
    roi_keys = ["VC"]
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        f"Subject{args.subject}_ImageNetTraining.h5", rois_dict
    )
    tr_d, _ = split_data_for_class(raw_data, 0.95)
    train_ds = MultimodalFmriDataset(tr_d['fmri'], tr_d['image_paths'], args.dataset_root, "training")
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

    # Modello
    model = FmriToSota(input_dim, task=args.task).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.MSELoss() # Allineamento feature puro

    os.makedirs(args.output_dir, exist_ok=True)

    for ep in range(args.epochs):
        loss = train(model, train_loader, opt, criterion, device, args.task)
        print(f"Ep {ep+1} Loss: {loss:.4f}")
        validate_and_save_vis(model, train_loader, device, args.task, args.output_dir, ep)
    
    # CORRETTO (Salva tutto quello che serve per l'inference):
    torch.save(model.state_dict(), f"{args.output_dir}/model_{args.task}_best.pth")
if __name__ == "__main__":
    main()