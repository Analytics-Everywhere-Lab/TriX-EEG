import torch
import torch.nn as nn
from typing import Dict
from .enc_dec import TemporalModelFactory
from .basic_blocks import ConvNormAct

# ============================================================
# Routed Denoiser
# ============================================================
class RoutedDenoiserBlockV2(nn.Module):
    def __init__(self, channels: int, num_blocks: int, temporal_model_type: str, dropout: float):
        super().__init__()
        self.pre = ConvNormAct(channels, channels, kernel_size=3)
        self.temporal = TemporalModelFactory(temporal_model_type, channels, num_blocks, dropout)
        self.post = ConvNormAct(channels, channels, kernel_size=3)
        # self.refiner = TinyUNet1D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        h = self.temporal(h)
        h = self.post(h)
        return x + h