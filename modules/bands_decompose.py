import math
import torch
import torch.nn as nn
import pywt
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


# ============================================================
# FFT Band Decomposition / Reconstruction
# ============================================================

class FFTBandDecomposition(nn.Module):
    def __init__(self, fs: float, signal_length: int, band_defs: Optional[List[Tuple[float, float]]] = None):
        super().__init__()
        self.fs = fs
        self.signal_length = signal_length
        self.band_defs = band_defs if band_defs is not None else self.default_band_defs(fs)
        self.num_bands = len(self.band_defs)

        freqs = torch.fft.rfftfreq(signal_length, d=1.0 / fs)
        masks = []
        nyq = fs / 2.0

        for low, high in self.band_defs:
            high_eff = min(high, nyq)
            if math.isclose(high_eff, nyq):
                mask = (freqs >= low) & (freqs <= high_eff)
            else:
                mask = (freqs >= low) & (freqs < high_eff)
            masks.append(mask.float())

        masks = torch.stack(masks, dim=0)
        self.register_buffer("freq_masks", masks, persistent=False)

    def default_band_defs(self, fs: float) -> List[Tuple[float, float]]:
        nyq = fs / 2.0
        return [
            (0.5, 4.0),
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 30.0),
            (30.0, 80.0),
            (80.0, nyq),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 1, f"Expected [B,1,T], got {x.shape}"
        b, _, t = x.shape
        if t != self.signal_length:
            raise ValueError(f"Expected signal_length={self.signal_length}, got T={t}")

        x_fft = torch.fft.rfft(x.squeeze(1), dim=-1)
        xk = x_fft.unsqueeze(1) * self.freq_masks.unsqueeze(0).to(x_fft.dtype)
        bands = torch.fft.irfft(xk, n=t, dim=-1)
        return bands

    def reconstruct(self, bands: torch.Tensor) -> torch.Tensor:
        return bands.sum(dim=1, keepdim=True)



if __name__ == "__main__":
    fs = 256.0
    signal_length = 512
    batch_size = 2
    x = torch.randn(batch_size, 10, signal_length)
    x = x.reshape(batch_size * 10, 1, signal_length)
    decomp = FFTBandDecomposition(fs, signal_length)
    recon = decomp.reconstruct
    
    bands = decomp(x)
    print(f"Input shape: {x.shape}")
    print(f"Decomposed bands shape: {bands.shape}")  # [B, 6, T]
    print(f"Band names: {decomp.BAND_NAMES}")
    
    x_rec = recon(bands)
    print(f"Reconstructed shape: {x_rec.shape}")  # [B, 1, T]
    
    # Calculate reconstruction error
    recon_error = torch.mean((x - x_rec) ** 2)
    print(f"Reconstruction MSE: {recon_error.item():.6f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x[0, 0].cpu().numpy(), label="Original")
    plt.plot(x_rec[0, 0].cpu().numpy(), label="Reconstructed", alpha=0.7)
    plt.title("Original vs Reconstructed EEG Signal")
    plt.legend()
    plt.subplot(2, 1, 2)
    for i in range(bands.shape[1]):
        plt.plot(bands[0, i].cpu().numpy() + i, label=decomp.BAND_NAMES[i])
    plt.title("Decomposed EEG Bands")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
