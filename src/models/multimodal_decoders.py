import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .saliency_decoder import EnhancedResNetDecoder

class ParallelAlexNetAligner(nn.Module):
    """
    Allineatore ottimizzato (Single Linear):
    Mappa fMRI -> AlexNet Features (256x6x6) usando un singolo layer lineare veloce.
    Include la testa di classificazione di AlexNet (congelata) per inferenza.
    """
    def __init__(self, input_dim):
        super().__init__()
        # AlexNet features size (256 * 6 * 6 = 9216)
        self.output_dim = 256 * 6 * 6
        
        # 1. Single Linear Layer (Molto più veloce di ParallelLinearModels)
        self.linear = nn.Linear(input_dim, self.output_dim)
        
        # 2. Classifier Part (Fixed from AlexNet)
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        original_alexnet = models.alexnet(weights=weights)
        
        self.classifier_head = nn.Sequential(
            original_alexnet.avgpool,
            nn.Flatten(),
            original_alexnet.classifier
        )
        
        # Congela il classificatore (non lo addestriamo)
        for p in self.classifier_head.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Mappa fMRI -> Feature Spaziali
        features = self.linear(x) # [B, 9216]
        
        # Reshape per il classificatore AlexNet [B, 256, 6, 6]
        features_spatial = features.view(-1, 256, 6, 6)
        
        # Classificazione (opzionale, usata per inferenza)
        logits = self.classifier_head(features_spatial)
        
        return features, logits

class MultiTaskDecoder(nn.Module):
    def __init__(self, input_dim, upsample_stages=5):
        super().__init__()
        self.decoder_channels = 256
        self.decoder_H = 7
        self.decoder_W = 7
        
        # Encoder condiviso
        self.linear = nn.Linear(input_dim, self.decoder_channels * self.decoder_H * self.decoder_W)
        
        # Heads
        self.depth_head = EnhancedResNetDecoder(input_channels=256, upsample_stages=upsample_stages)
        self.depth_head.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.seg_head = EnhancedResNetDecoder(input_channels=256, upsample_stages=upsample_stages)
        self.seg_head.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], self.decoder_channels, self.decoder_H, self.decoder_W)
        depth_logits, _ = self.depth_head(x, return_pre_activation=False)
        seg_logits, _ = self.seg_head(x, return_pre_activation=False)
        return depth_logits, seg_logits