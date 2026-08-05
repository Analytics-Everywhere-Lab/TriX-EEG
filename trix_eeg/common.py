"""Shared tensor and spectral helpers for TriX-EEG."""

from __future__ import annotations

from collections.abc import Mapping

import torch


EPS = 1e-8
NEURAL_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


def as_bct(x: torch.Tensor) -> torch.Tensor:
    """Return a signal as [batch, channels, time]."""
    if x.ndim == 2:
        return x.unsqueeze(1)
    if x.ndim != 3:
        raise ValueError(f"Expected [B,T] or [B,C,T], got {tuple(x.shape)}")
    return x


def prediction_from_output(output) -> torch.Tensor:
    if isinstance(output, Mapping):
        if "y_hat" not in output:
            raise KeyError("Model output dictionary has no 'y_hat' prediction")
        output = output["y_hat"]
    if not torch.is_tensor(output):
        raise TypeError(f"Unsupported model output type: {type(output)}")
    return as_bct(output)


def predict(model, x: torch.Tensor) -> torch.Tensor:
    return prediction_from_output(model(as_bct(x)))


def flatten_signal(x: torch.Tensor) -> torch.Tensor:
    x = as_bct(x)
    return x.reshape(x.shape[0], -1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(flatten_signal(x), dim=-1)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    return flatten_signal(x).square().sum(dim=-1)


def correlation(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x_flat, y_flat = flatten_signal(x), flatten_signal(y)
    x_centered = x_flat - x_flat.mean(dim=-1, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=-1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=-1)
    denominator = (
        torch.linalg.vector_norm(x_centered, dim=-1)
        * torch.linalg.vector_norm(y_centered, dim=-1)
    )
    return numerator / denominator.clamp_min(eps)


def periodogram(x: torch.Tensor) -> torch.Tensor:
    """One-sided PSD used by the draft's spectral RRMSE formula."""
    x = as_bct(x)
    return torch.fft.rfft(x, dim=-1).abs().square() / x.shape[-1]


def spectral_band_power(
    x: torch.Tensor,
    fs: float,
    bands=NEURAL_BANDS,
) -> torch.Tensor:
    """Integrated periodogram power, returned as [B,C,num_bands]."""
    x = as_bct(x)
    psd = periodogram(x)
    frequencies = torch.fft.rfftfreq(x.shape[-1], 1.0 / float(fs), device=x.device)
    powers = []
    for low, high in bands.values():
        mask = (frequencies >= low) & (frequencies < high)
        if not bool(mask.any()):
            raise ValueError(f"No FFT bins in frequency band {low:g}-{high:g} Hz")
        if int(mask.sum()) == 1:
            power = psd[..., mask].squeeze(-1)
        else:
            power = torch.trapz(psd[..., mask], frequencies[mask], dim=-1)
        powers.append(power)
    return torch.stack(powers, dim=-1)


def finite_mean(values: torch.Tensor) -> float:
    values = values.detach().float()
    finite = torch.isfinite(values)
    return float(values[finite].mean().cpu()) if bool(finite.any()) else float("nan")
