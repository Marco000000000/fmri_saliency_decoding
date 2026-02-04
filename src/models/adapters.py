import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerAdapter(nn.Module):
    """
    Prende il bottleneck (fMRI feature) e ricostruisce la piramide
    multiscala usando ConvTranspose (Inverse Convolution).
    """
    def __init__(self, in_channels=256):
        super().__init__()
        
        # SegFormer-B0 si aspetta canali: [32, 64, 160, 256]
        # Noi partiamo da 256 (il feature vector profondo)
        
        # Stage 4 (Il più profondo, 1/32): 16x16. Lo abbiamo già, serve solo adattare i canali se necessario.
        # Qui input è 256, output deve essere 256. Identity o Conv 1x1.
        self.stage4_conv = nn.Conv2d(in_channels, 256, kernel_size=1)
        
        # Stage 3 (1/16): 16x16 -> 32x32. Target channels: 160
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 160, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU()
        )
        
        # Stage 2 (1/8): 32x32 -> 64x64. Target channels: 64
        # Input qui è 160 (output dello stage 3)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(160, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Stage 1 (1/4): 64x64 -> 128x128. Target channels: 32
        # Input qui è 64 (output dello stage 2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: [Batch, 256, 16, 16] (Feature fMRI reshaped)
        Returns: [c1, c2, c3, c4] come si aspetta SegFormer
        """
        # 1. Stage 4 (Bottom) - 16x16
        c4 = self.stage4_conv(x)
        
        # 2. Stage 3 - Upsample a 32x32
        # Nota: Partiamo sempre da 'x' o a cascata? 
        # A cascata (FPN style) è meglio per propagare i dettagli.
        # Ma qui stiamo "inventando" dettagli, quindi usiamo x come seed per c3.
        c3 = self.up3(x)
        
        # 3. Stage 2 - Upsample a 64x64 (partendo da c3)
        c2 = self.up2(c3)
        
        # 4. Stage 1 - Upsample a 128x128 (partendo da c2)
        c1 = self.up1(c2)
        
        return [c1, c2, c3, c4]