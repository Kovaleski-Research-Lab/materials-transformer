import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class FNOFilter1d(nn.Module):
    """
    1D Spectral Convolution layer a la Fourier Neural Operators
    """
    def __init__(self, in_channels, out_channels, num_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_modes = num_modes # number of fourier modes to keep
        
        # learnable weights for fourier modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, 
                                                            self.num_modes, dtype=torch.cfloat))
        
    def forward(self, x):
        B, L, D = x.shape
        x_fft = torch.fft.rfft(x, dim=1) # (B, L//2 + 1, D)
        
        # truncate lower modes and apply weights
        # (B, D, modes) x (D, D_out, modes) -> (B, D_out, modes)
        out_fft = torch.zeros(B, L // 2 + 1, self.out_channels, device=x.device, dtype=torch.cfloat)
        out_fft[:, :self.num_modes] = torch.einsum(
            "bmi,iom->bmo", 
            x_fft[:, :self.num_modes], 
            self.weights
        )
        
        # inverse fft back to token domain
        x_filtered = torch.fft.irfft(out_fft, n=L, dim=1)
        return x_filtered
    
class AdaptiveFNOFilter1d(nn.Module):
    """
    1D layer, closely following Adaptive Fourier Neural Operator (AFNO)
    """
    def __init__(self, channel_dim, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, channel_factor=1):
        super().__init__()
        assert channel_dim % num_blocks == 0, f"channel dim {channel_dim} needs to be divisible by num_blocks {num_blocks}"
        
        self.in_channels = channel_dim
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.in_channels // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.channel_factor = channel_factor
        self.scale = 0.02
        
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.channel_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.channel_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.channel_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        
    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, L, D = x.shape
        
        x = torch.fft.rfft(x, dim=1, norm="ortho")
        x = x.reshape(B, L // 2 + 1, self.num_blocks, self.block_size)
        
        o1_real = torch.zeros([B, L // 2 + 1, self.num_blocks, self.block_size * self.channel_factor], device=x.device)
        o1_imag = torch.zeros([B, L // 2 + 1, self.num_blocks, self.block_size * self.channel_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)
        
        num_modes = L // 2 + 1
        kept_modes = int(num_modes * self.hard_thresholding_fraction)
        
        o1_real[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )
        
        o1_imag[:, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )
        
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, L // 2 + 1, D)
        x = torch.fft.irfft(x, n=L, dim=1, norm="ortho")
        x = x.type(dtype)
        return x + bias

class FNOTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_blocks, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        #self.spectral_layer = SpectralConv1d(embed_dim, embed_dim, num_modes=num_blocks)
        self.filter = AdaptiveFNOFilter1d(embed_dim, num_blocks=num_blocks)
        #self.w = nn.Conv1d(embed_dim, embed_dim, 1) # Linear projection for skip connection
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        residual = x
        # pre-LayerNorm
        x = self.norm1(x)
        # apply spectral conv and add residual connection
        #residual = self.w(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.filter(x) # + residual
        x = self.mlp(self.norm2(x))
        x = x + residual
        return x