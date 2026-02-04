import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.config.roi_config import ALL_ROI_MAPPINGS, subject_dims, parse_roi_keys
from src.datasets.fmri_reconstruction_dataset import load_fmri_image_pairs, split_data_for_class
from src.models.clip_aligner import FmriToClipAligner

# Dataset Custom che include le Caption
class FmriCaptionDataset(Dataset):
    def __init__(self, fmri_data, image_paths, captions_dict, tokenizer, max_len=30):
        self.fmri = fmri_data
        self.image_paths = image_paths
        self.captions_dict = captions_dict
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        fmri = torch.tensor(self.fmri[idx]).float()
        path = self.image_paths[idx]
        fname = os.path.basename(path)
        
        # Recupera caption (o usa placeholder se fallisce)
        text = self.captions_dict.get(fname, "an image")
        
        # Tokenizzazione per GPT-2 (Target)
        tokenized = self.tokenizer(
            text, 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()
        
        # In HuggingFace, labels=-100 vengono ignorate dalla Loss
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100 

        return {
            "fmri": fmri,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--rois", type=str, default="VC")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4) # LR basso per fine-tuning delicato
    parser.add_argument("--dataset_root", type=str, default="data/GOD_Dataset")
    parser.add_argument("--captions_file", type=str, default="data/GOD_Dataset/captions_generated.json")
    parser.add_argument("--save_dir", type=str, default="trained_fmri_decoders/end2end_text")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training End-to-End fMRI -> Text (S{args.subject}) ---")

    # 1. Carica Caption Dictionary
    if not os.path.exists(args.captions_file):
        print("ERRORE: Devi prima eseguire lo script 01_generate_dataset_captions.py!")
        return
    with open(args.captions_file, "r") as f:
        captions_dict = json.load(f)

    # 2. Modello GPT-2 (Decoder)
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    caption_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    
    # Congeliamo GPT-2. Vogliamo solo addestrare l'interfaccia (Aligner)
    # Se hai molta GPU e dati, puoi scongelare gli ultimi layer, ma è rischioso.
    for param in caption_model.parameters():
        param.requires_grad = False
    caption_model.eval() 

    # 3. Aligner (Quello che addestriamo)
    roi_keys = parse_roi_keys(args.rois)
    input_dim = sum([subject_dims[k][args.subject-1] for k in roi_keys])
    target_dim = caption_model.config.encoder.hidden_size # 768
    
    aligner = FmriToClipAligner(input_dim, clip_dim=target_dim).to(device)
    optimizer = optim.AdamW(aligner.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Dataloaders
    rois_dict = {k: ALL_ROI_MAPPINGS[k] for k in roi_keys}
    train_h5 = f"Subject{args.subject}_ImageNetTraining.h5"
    raw_data = load_fmri_image_pairs(
        os.path.join(args.dataset_root, "fmri_files", "training"),
        os.path.join(args.dataset_root, "images", "training"),
        train_h5, rois_dict
    )
    
    train_dict, val_dict = split_data_for_class(raw_data, 0.95) # Più dati possibili per train
    
    train_ds = FmriCaptionDataset(train_dict['fmri'], train_dict['image_paths'], captions_dict, tokenizer)
    val_ds = FmriCaptionDataset(val_dict['fmri'], val_dict['image_paths'], captions_dict, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    save_path = os.path.join(args.save_dir, f"S{args.subject}_{args.rois}")
    os.makedirs(save_path, exist_ok=True)

    # 5. Training Loop
    best_val_loss = float('inf')
    
    for ep in range(args.epochs):
        aligner.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {ep+1}", leave=False):
            fmri = batch['fmri'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # A. Forward Aligner
            # fMRI -> Embedding
            fmri_emb = aligner(fmri)
            
            # B. Scaling Trick (Opzionale ma aiuta)
            # Moltiplichiamo per 2.0 per matchare la norma tipica di ViT
            fmri_emb = fmri_emb * 2.0 
            
            # C. Preparazione per GPT-2
            # GPT-2 vuole [Batch, Seq, Dim]
            encoder_outputs = BaseModelOutput(last_hidden_state=fmri_emb.unsqueeze(1))
            
            # D. Forward GPT-2 (Calcolo Loss)
            outputs = caption_model(
                encoder_outputs=encoder_outputs,
                labels=labels, # Qui avviene la magia: calcola la loss rispetto al testo vero
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        aligner.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                fmri = batch['fmri'].to(device)
                labels = batch['labels'].to(device)
                input_ids = batch['input_ids'].to(device)
                
                fmri_emb = aligner(fmri) * 2.0
                encoder_outputs = BaseModelOutput(last_hidden_state=fmri_emb.unsqueeze(1))
                
                outputs = caption_model(
                    encoder_outputs=encoder_outputs,
                    labels=labels,
                    decoder_input_ids=input_ids
                )
                val_loss += outputs.loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Ep {ep+1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}", end="")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(aligner.state_dict(), os.path.join(save_path, "best_text_aligner.pth"))
            print(" [SAVED *]")
        else:
            print("")

if __name__ == "__main__":
    main()