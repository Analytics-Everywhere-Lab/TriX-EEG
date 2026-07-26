import torch
from typing import Dict, Mapping, Tuple
from utils import rms, compute_psd


EEG_FREQUENCY_BANDS: Mapping[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}



# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def batch_rrmse_temporal(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 3 and target.size(1) == 1:
        target = target.squeeze(1)

    num = rms(pred - target, dim=-1, eps=eps)
    den = rms(target, dim=-1, eps=eps)
    return num / (den + eps)


@torch.no_grad()
def batch_rrmse_spectral(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    psd_pred = compute_psd(pred)
    psd_tgt = compute_psd(target)

    num = rms(psd_pred - psd_tgt, dim=-1, eps=eps)
    den = rms(psd_tgt, dim=-1, eps=eps)
    return num / (den + eps)


@torch.no_grad()
def batch_cc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    if pred.dim() == 3 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 3 and target.size(1) == 1:
        target = target.squeeze(1)

    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    tgt_c = target - target.mean(dim=-1, keepdim=True)

    cov = torch.mean(pred_c * tgt_c, dim=-1)
    var_p = torch.mean(pred_c ** 2, dim=-1)
    var_t = torch.mean(tgt_c ** 2, dim=-1)

    return cov / (torch.sqrt(var_p * var_t) + eps)


@torch.no_grad()
def batch_snr_db(clean: torch.Tensor, noise: torch.Tensor, eps: float = 1e-8):
    if clean.dim() == 3 and clean.size(1) == 1:
        clean = clean.squeeze(1)
    if noise.dim() == 3 and noise.size(1) == 1:
        noise = noise.squeeze(1)

    p_signal = torch.mean(clean ** 2, dim=-1)
    p_noise = torch.mean(noise ** 2, dim=-1)
    return 10.0 * torch.log10((p_signal + eps) / (p_noise + eps))


@torch.no_grad()
def batch_snr_improvement(noisy: torch.Tensor, pred: torch.Tensor, clean: torch.Tensor):
    snr_in = batch_snr_db(clean, noisy - clean)
    snr_out = batch_snr_db(clean, pred - clean)
    return snr_out - snr_in


def _as_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"Expected [B,T] or [B,C,T], got {tuple(x.shape)}")
    return x


def _welch_psd(
    x: torch.Tensor,
    fs: float,
    nperseg: int = 256,
    overlap: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Welch PSD with density scaling; returns frequencies and [B,C,F]."""
    x = _as_bct(x)
    segment_length = min(int(nperseg), x.size(-1))
    if segment_length < 2:
        raise ValueError("Frequency metrics require at least two time samples")
    step = max(1, int(round(segment_length * (1.0 - overlap))))
    segments = x.unfold(-1, segment_length, step)
    segments = segments - segments.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        segment_length, periodic=True, dtype=x.dtype, device=x.device
    )
    spectrum = torch.fft.rfft(segments * window, dim=-1)
    psd = spectrum.abs().square() / (float(fs) * window.square().sum())
    if segment_length % 2 == 0:
        psd[..., 1:-1] *= 2.0
    else:
        psd[..., 1:] *= 2.0
    frequencies = torch.fft.rfftfreq(
        segment_length, d=1.0 / float(fs), device=x.device
    )
    return frequencies, psd.mean(dim=-2)


@torch.no_grad()
def batch_frequency_preservation(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: float,
    bands: Mapping[str, Tuple[float, float]] = EEG_FREQUENCY_BANDS,
    fmin: float = 0.5,
    fmax: float = 45.0,
    alpha_band: Tuple[float, float] = (8.0, 13.0),
    alpha_tolerance_hz: float = 1.0,
    nperseg: int = 256,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """
    Frequency-preservation metrics
    """
    frequencies, pred_psd = _welch_psd(pred, fs=fs, nperseg=nperseg)
    _, target_psd = _welch_psd(target, fs=fs, nperseg=nperseg)
    if pred_psd.shape != target_psd.shape:
        raise ValueError(
            f"Prediction and target PSD shapes differ: {pred_psd.shape} vs "
            f"{target_psd.shape}"
        )

    valid = (frequencies >= fmin) & (frequencies <= fmax)
    if not bool(valid.any()):
        raise ValueError(f"No PSD bins in requested range {fmin:g}-{fmax:g} Hz")
    pred_valid = pred_psd[..., valid]
    target_valid = target_psd[..., valid]
    dot = (pred_valid * target_valid).sum(dim=-1)
    denom = (
        torch.linalg.vector_norm(pred_valid, dim=-1)
        * torch.linalg.vector_norm(target_valid, dim=-1)
    )
    result = {"psd_similarity": (dot / denom.clamp_min(eps)).mean(dim=1)}

    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        if int(mask.sum()) < 2:
            raise ValueError(
                f"Band {name} ({low:g}-{high:g} Hz) has fewer than two PSD bins"
            )
        pred_power = torch.trapz(pred_psd[..., mask], frequencies[mask], dim=-1)
        target_power = torch.trapz(target_psd[..., mask], frequencies[mask], dim=-1)
        result[f"band_power_absolute_error_{name}"] = (
            pred_power - target_power
        ).abs().mean(dim=1)

    alpha = (frequencies >= alpha_band[0]) & (frequencies <= alpha_band[1])
    if not bool(alpha.any()):
        raise ValueError("No PSD bins in the alpha band")
    alpha_frequencies = frequencies[alpha]
    pred_peak = alpha_frequencies[pred_psd[..., alpha].argmax(dim=-1)]
    target_peak = alpha_frequencies[target_psd[..., alpha].argmax(dim=-1)]
    peak_error = (pred_peak - target_peak).abs()
    result["alpha_peak_error_hz"] = peak_error.mean(dim=1)
    result["alpha_peak_preservation_rate"] = (
        peak_error <= alpha_tolerance_hz
    ).to(pred_psd.dtype).mean(dim=1)
    return result
