import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class FeatureInversionPipeline:
    def __init__(self, clip_model, device, step_size=0.02, num_steps=200):
        self.clip_model = clip_model
        self.device = device
        self.step_size = step_size
        self.num_steps = num_steps
        
        self.normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                              std=(0.26862954, 0.26130258, 0.27577711))
        
        # Augmentations: RandomAffine è fondamentale per la robustezza geometrica
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.3),
        ])

    def get_tv_loss(self, img):
        """Total Variation Loss: Liscia le differenze tra pixel adiacenti"""
        b, c, h, w = img.shape
        h_tv = torch.pow((img[:, :, 1:, :] - img[:, :, :h-1, :]), 2).mean()
        w_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, :w-1]), 2).mean()
        return h_tv + w_tv

    def get_color_correlation_loss(self, img):
        """
        Anti-Rainbow Loss: Costringe i canali RGB a muoversi insieme.
        Riduce drasticamente il rumore verde/viola.
        """
        # Calcola la media dei canali (immagine grayscale approssimata)
        mean_color = torch.mean(img, dim=1, keepdim=True)
        # Penalizza la varianza rispetto al grigio
        color_var = torch.pow(img - mean_color, 2).mean()
        return color_var

    def optimize_image(self, target_embedding, init_img=None, image_size=512, 
                       lambda_tv=2.5,        # Aumentato drasticamente per lisciare
                       lambda_color=1.5,     # Aggiunto: peso alto contro il rumore colorato
                       num_cuts=32):
        
        # --- Configurazione Piramide ---
        scales = [128, 224, image_size]  # Saltiamo 64, partiamo da 128 per forme migliori
        iterations = [150, 100, 80]      # Più iterazioni all'inizio
        
        target_embedding = F.normalize(target_embedding, p=2, dim=1).detach()
        
        # Inizializzazione
        current_size = scales[0]
        if init_img is None:
            # Partiamo da un grigio medio uniforme (meno bias del rumore random)
            img = torch.ones(1, 3, current_size, current_size).to(self.device) * 0.5
            # Aggiungiamo pochissimo rumore solo per sbloccare i gradienti
            img = img + torch.randn_like(img) * 0.01 
        else:
            img = F.interpolate(init_img.clone(), size=(current_size, current_size), mode='bilinear')

        img.requires_grad_(True)
        scaler = torch.cuda.amp.GradScaler() # Per velocità
        
        print(f" > Starting Robust De-noising Optimization...")

        for scale_idx, scale in enumerate(scales):
            steps = iterations[scale_idx]
            
            # LR decay: Più alto all'inizio, molto fine alla fine
            lr = self.step_size * (0.6 ** scale_idx)
            optimizer = optim.Adam([img], lr=lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            
            # Blur dinamico: molto forte all'inizio
            blur_sigma = max(0.5, 1.5 - (scale_idx * 0.5))

            for step in range(steps):
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    img_clamped = torch.clamp(img, 0, 1)
                    
                    # --- 1. CLIP Loss (Batching) ---
                    cuts = []
                    # Meno cut ma più mirati
                    n_cuts = num_cuts if scale < 224 else num_cuts // 2
                    
                    for _ in range(n_cuts):
                        # Se l'img è piccola, upscale prima del crop
                        if scale < 224:
                            scaled = F.interpolate(img_clamped, size=(224, 224), mode='bilinear')
                            cut = self.aug(scaled)
                        else:
                            cut = self.aug(img_clamped)
                            
                        if cut.shape[-1] != 224:
                            cut = F.interpolate(cut, size=(224, 224), mode='bilinear')
                        cuts.append(cut)
                    
                    batch = torch.cat(cuts, dim=0)
                    batch = self.normalize(batch)
                    embeds = self.clip_model(batch).image_embeds
                    embeds = F.normalize(embeds, p=2, dim=1)
                    
                    clip_loss = (1 - torch.cosine_similarity(embeds, target_embedding)).mean()
                    
                    # --- 2. Regularization Losses (Il segreto della pulizia) ---
                    loss_tv = self.get_tv_loss(img)
                    loss_color = self.get_color_correlation_loss(img)
                    loss_range = torch.mean(torch.abs(img - 0.5)) * 0.2 # Tiene i colori tenui
                    
                    # Totale pesato
                    total_loss = clip_loss + (lambda_tv * loss_tv) + (lambda_color * loss_color) + loss_range

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # --- 3. Active Smoothing (Pulizia forzata) ---
                # Ogni 5 step forziamo una sfocatura per uccidere i pixel singoli
                if step % 5 == 0:
                    with torch.no_grad():
                        k_size = 3
                        img.data = transforms.functional.gaussian_blur(img.data, kernel_size=k_size, sigma=blur_sigma)
                        img.data.clamp_(0, 1)
            
            # Upscale alla prossima risoluzione
            if scale_idx < len(scales) - 1:
                next_scale = scales[scale_idx + 1]
                with torch.no_grad():
                    img_new = F.interpolate(img.detach(), size=(next_scale, next_scale), mode='bicubic', align_corners=False)
                    img = img_new.clone().detach().requires_grad_(True)

        return img.detach()