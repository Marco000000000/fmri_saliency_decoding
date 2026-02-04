import torch
import torch.nn as nn

class FmriToClipAligner(nn.Module):
    def __init__(self, input_dim, clip_dim=768): # 768 per SD 1.5 text/image encoder
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, clip_dim)
            # Niente attivazione finale, gli embedding CLIP non sono bounded 0-1
        )

    def forward(self, x):
        return self.net(x)