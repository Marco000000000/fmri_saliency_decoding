import torch
import torch.nn as nn

class LinearLatentProjector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Il VAE di Stable Diffusion comprime 512x512 in 64x64 con 4 canali.
        # Shape target: [Batch, 4, 64, 64]
        self.target_shape = (4, 64, 64)
        self.output_dim = 4 * 64 * 64
        
        # Unico Layer Lineare: fMRI -> Spazio Latente
        self.linear = nn.Linear(input_dim, self.output_dim)

    def forward(self, x):
        # x: [Batch, fMRI_dim]
        x = self.linear(x)
        # Reshape per il decoder VAE: [Batch, 4, 64, 64]
        return x.view(-1, *self.target_shape)