import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, DPTForDepthEstimation

# --- 1. GENERATORE (fMRI -> Feature Map) ---
class FmriDecoderGenerator(nn.Module):
    """
    Sostituisce il layer Linear gigante con una rete CNN che risparmia memoria.
    fMRI -> Linear(Bottleneck) -> Reshape -> ConvTranspose Upsampling -> Target
    """
    def __init__(self, input_dim, start_dim, start_shape, target_shape, target_channels):
        super().__init__()
        self.start_shape = start_shape # es. (6, 6)
        self.start_dim = start_dim     # es. 768
        self.target_shape = target_shape # es. (24, 24)
        
        # Linear Projector (Low Rank)
        flat_start_dim = start_dim * start_shape[0] * start_shape[1]
        self.linear = nn.Linear(input_dim, flat_start_dim)
        
        # Upsampling Blocks
        layers = []
        current_h, current_w = start_shape
        
        while current_h < target_shape[0]:
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(start_dim, start_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(start_dim),
                nn.ReLU(inplace=True)
            ))
            current_h *= 2
            current_w *= 2
            
        layers.append(nn.Conv2d(start_dim, target_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.start_dim, self.start_shape[0], self.start_shape[1])
        x = self.decoder(x)
        return x

# --- 2. DECODER DEPTH CUSTOM (Sostituisce il Neck DPT rotto) ---
class SimpleDepthDecoder(nn.Module):
    """
    Prende le feature 768x24x24 e genera la Depth Map 384x384.
    Simula il comportamento del Head DPT ma senza richiedere le feature ResNet mancanti.
    """
    def __init__(self, in_channels=768):
        super().__init__()
        
        # Upsample progressivo: 24 -> 48 -> 96 -> 192 -> 384 (Factor 16)
        # Usiamo ConvTranspose per imparare l'upsample
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        ) # 48x48
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        ) # 96x96
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # 192x192
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 384x384
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x) # [B, 1, 384, 384]
        return x

# --- 3. WRAPPER PRINCIPALE ---
class FmriToSota(nn.Module):
    def __init__(self, input_dim, task='seg'):
        super().__init__()
        self.task = task
        
        if task == 'seg':
            print("Caricamento SegFormer (Float16)...")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                torch_dtype=torch.float16
            )
            # Student SegFormer (Usa Adapter Inverso)
            self.generator = FmriDecoderGenerator(
                input_dim=input_dim, start_dim=256, start_shape=(4, 4),
                target_shape=(16, 16), target_channels=256
            )
            from src.models.adapters import SegFormerAdapter
            self.adapter = SegFormerAdapter(in_channels=256)

        elif task == 'depth':
            print("Caricamento DPT (Float16)...")
            self.model = DPTForDepthEstimation.from_pretrained(
                "Intel/dpt-hybrid-midas",
                torch_dtype=torch.float16
            )
            # Student DPT (Usa Generator + Custom Decoder)
            self.generator = FmriDecoderGenerator(
                input_dim=input_dim, start_dim=768, start_shape=(6, 6),
                target_shape=(24, 24), target_channels=768
            )
            # Qui usiamo il nostro decoder custom invece di quello rotto di DPT
            self.depth_decoder = SimpleDepthDecoder(in_channels=768)

        # Congela Teacher SOTA
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward_fmri(self, fmri):
        # 1. Genera feature map dal bottleneck
        x = self.generator(fmri) # [B, C, H, W]
        
        # 2. SegFormer usa l'adapter per la piramide
        if self.task == 'seg':
            return self.adapter(x)
        else:
            # DPT ritorna la feature map diretta (24x24) che useremo nel decoder
            return x

    def decode_from_features(self, features):
        if self.task == 'seg':
            # Decoder SegFormer Originale (Funziona perché abbiamo la piramide corretta)
            features_fp16 = [f.to(dtype=torch.float16) for f in features]
            with torch.no_grad():
                logits = self.model.decode_head(features_fp16)
                logits = F.interpolate(logits, size=(512, 512), mode='bilinear')
            return logits.float()

        elif self.task == 'depth':
            # Decoder Depth Custom (Sostituisce DPT Neck/Head)
            # Input: features [B, 768, 24, 24] (Float32)
            # Output: depth [B, 1, 384, 384]
            depth_map = self.depth_decoder(features)
            
            # Resize a 512 per compatibilità
            depth_map = F.interpolate(depth_map, size=(512, 512), mode='bilinear')
            return depth_map