import torch
import torch.nn as nn
from .enc_dec import TemporalModelFactory



# ============================================================
# Dual-path Cross-Band Fusion
# ============================================================


class TemporalMixer(nn.Module):
    def __init__(self, channels: int, temporal_model_type: str, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.temporal = TemporalModelFactory(temporal_model_type, channels, num_blocks=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.temporal(self.norm(x))


class BandMixer(nn.Module):
    """
    Mix across K bands at each time step.
    Input/output: [B, K, C, T]
    """
    def __init__(self, channels: int, num_heads: int, ff_mult: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, c, t = x.shape
        xt = x.permute(0, 3, 1, 2).contiguous().view(b * t, k, c)

        h = self.norm1(xt)
        h, _ = self.attn(h, h, h, need_weights=False)
        xt = xt + h

        h = self.norm2(xt)
        h = self.ff(h)
        xt = xt + h

        x = xt.view(b, t, k, c).permute(0, 2, 3, 1).contiguous()
        return x


class DualPathFusionLayer(nn.Module):
    """
    1) temporal mixer within each band
    2) band mixer across bands
    """
    def __init__(self, channels: int, num_heads: int, ff_mult: int, temporal_model_type: str, dropout: float):
        super().__init__()
        self.temporal_mixer = TemporalMixer(channels, temporal_model_type, dropout)
        self.band_mixer = BandMixer(channels, num_heads=num_heads, ff_mult=ff_mult)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, k, c, t = x.shape

        # temporal mix in each band independently
        xt = x.view(b * k, c, t)
        xt = self.temporal_mixer(xt)
        x = xt.view(b, k, c, t)

        # mix across bands
        x = self.band_mixer(x)
        return x


class CrossBandFusionV2(nn.Module):
    def __init__(self, channels: int, num_heads: int, num_layers: int, ff_mult: int, temporal_model_type: str, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DualPathFusionLayer(channels, num_heads, ff_mult, temporal_model_type, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x