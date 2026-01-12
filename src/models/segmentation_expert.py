import torch
import torch.nn as nn
from .saliency_decoder import EnhancedResNetDecoder

class MultiClassExpert(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.decoder_channels = 256
        self.h = 7
        self.w = 7
        
        self.linear = nn.Linear(input_dim, self.decoder_channels * self.h * self.w)
        self.decoder = EnhancedResNetDecoder(input_channels=256, upsample_stages=5)
        
        # Output: 3 Canali (uno per classe)
        self.decoder.final_conv = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], self.decoder_channels, self.h, self.w)
        logits, _ = self.decoder(x, return_pre_activation=False)
        return logits # [Batch, 3, 224, 224]