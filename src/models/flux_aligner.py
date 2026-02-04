import torch
import torch.nn as nn

class FmriToFluxSequenceAligner(nn.Module):
    def __init__(self, input_dim, seq_len=77, t5_dim=4096, clip_dim=768, dropout=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.t5_dim = t5_dim
        
        # 1. compressione iniziale dell'fMRI
        self.stem = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. CLIP Head (Semplice vettore globale)
        self.clip_head = nn.Linear(2048, clip_dim)
        
        # 3. T5 Sequence Head (La parte complessa)
        # Vogliamo trasformare [Batch, 2048] -> [Batch, 77, 4096]
        # Usiamo un "bottleneck" transformer approach semplificato per memoria
        
        # Proiettiamo a dimensione (Seq_len * Latent)
        # Nota: 77 * 64 è gestibile. 
        self.seq_projector = nn.Linear(2048, seq_len * 64) 
        
        # Upscaler finale per ogni token: 64 -> 4096
        self.token_upscaler = nn.Linear(64, t5_dim)
        
        # Un piccolo Transformer Encoder per dare coerenza alla sequenza (sintassi)
        encoder_layer = nn.TransformerEncoderLayer(d_model=t5_dim, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.sequence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        # x: [Batch, Voxel]
        features = self.stem(x)
        
        # A. CLIP (Facile)
        clip_out = self.clip_head(features)
        
        # B. T5 Sequence
        # 1. Creiamo una sequenza grezza
        seq = self.seq_projector(features) # [Batch, 77*64]
        seq = seq.view(-1, self.seq_len, 64) # [Batch, 77, 64]
        
        # 2. Espandiamo alla dimensione piena di T5
        seq = self.token_upscaler(seq) # [Batch, 77, 4096]
        
        # 3. Raffiniamo con Transformer (il cervello impara la "grammatica" di T5)
        # Questo permette ai token di "parlarsi" tra loro
        t5_seq_out = self.sequence_transformer(seq) # [Batch, 77, 4096]
        
        return clip_out, t5_seq_out