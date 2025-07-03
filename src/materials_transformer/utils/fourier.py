import torch
import torch.nn as nn

class FourierMixer(nn.Module):
    """
    A simple token mixer using 2D FFT. The idea is to have it replace self-attention (Fnet).
    """
    def forward(self, x):
        # x has shape (B, L, D) where L is seq_len (T*N) and D is embed_dim
        # take the real part of the inverse FFT of the FFT of the input lol
        return torch.fft.ifft2(torch.fft.fft2(x, dim=(-2, -1))).real
    
class FNetEncoderLayer(nn.Module):
    """
    A single FNet block, designed to replace Multi-Head self-attention directly with a Fourier mixer
    """
    def __init__(self, embed_dim, mlp_ratio, dropout):
        super().__init__()
        self.mixer = FourierMixer()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, src):
        # identical structure to pre-LN transformer block
        src = src + self.mixer(self.norm1(src))
        src = src + self.mlp(self.norm2(src))
        return src