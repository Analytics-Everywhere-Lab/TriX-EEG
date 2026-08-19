"""Level 1: clean-signal validity metrics and Denoise Score (DS)."""

from __future__ import annotations

import torch

from .aggregation import (
    LEGACY_AGGREGATION,
    validate_aggregation,
)

from .common import (
    EPS,
    NEURAL_BANDS,
    as_bct,
    correlation,
    flatten_signal,
    l2_norm,
    periodogram,
    spectral_band_power,
    squared_l2_norm,
)


@torch.no_grad()
def level1_signal_validity(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    clean: torch.Tensor,
    *,
    fs: float = 256.0,
    delta_snr_max: float = 20.0,
    band_power_error_max: float = 1.0,
    bands=NEURAL_BANDS,
    eps: float = EPS,
    aggregation: str = LEGACY_AGGREGATION,
) -> dict[str, torch.Tensor]:
    """Implement equations (3)-(13), returning one value per input window."""
    validate_aggregation(aggregation)
    noisy, prediction, clean = map(as_bct, (noisy, prediction, clean))
    if not (noisy.shape == prediction.shape == clean.shape):
        raise ValueError("noisy, prediction, and clean must have identical shapes")
    if delta_snr_max <= 0 or band_power_error_max <= 0:
        raise ValueError("Normalization tolerances must be positive")

    rrmse_t = l2_norm(prediction - clean) / l2_norm(clean).clamp_min(eps)
    clean_psd, prediction_psd = periodogram(clean), periodogram(prediction)
    psd_difference = flatten_signal(prediction_psd - clean_psd)
    psd_clean = flatten_signal(clean_psd)
    rrmse_s = (
        torch.linalg.vector_norm(psd_difference, dim=-1)
        / torch.linalg.vector_norm(psd_clean, dim=-1).clamp_min(eps)
    )
    cc = correlation(prediction, clean, eps)

    clean_energy = squared_l2_norm(clean)
    snr_in = 10.0 * torch.log10(
        clean_energy.clamp_min(eps) / squared_l2_norm(noisy - clean).clamp_min(eps)
    )
    snr_out = 10.0 * torch.log10(
        clean_energy.clamp_min(eps) / squared_l2_norm(prediction - clean).clamp_min(eps)
    )
    delta_snr = snr_out - snr_in

    clean_power = spectral_band_power(clean, fs, bands)
    prediction_power = spectral_band_power(prediction, fs, bands)
    relative_band_errors = (
        (prediction_power - clean_power).abs() / clean_power.clamp_min(eps)
    )
    macro_band_power_error = relative_band_errors.mean(dim=(1, 2))
    bps = 1.0 - torch.clamp(
        macro_band_power_error / band_power_error_max, 0.0, 1.0
    )

    utility_rrmse_t = 1.0 - torch.clamp(rrmse_t, 0.0, 1.0)
    utility_rrmse_s = 1.0 - torch.clamp(rrmse_s, 0.0, 1.0)
    utility_cc = torch.clamp(cc, 0.0, 1.0)
    utility_delta_snr = torch.clamp(delta_snr / delta_snr_max, 0.0, 1.0)
    ds = (
        utility_rrmse_t + utility_rrmse_s + utility_cc + utility_delta_snr + bps
    ) / 5.0

    result = {
        "rrmse_temporal": rrmse_t,
        "rrmse_spectral": rrmse_s,
        "cc": cc,
        "snr_in_db": snr_in,
        "snr_out_db": snr_out,
        "delta_snr_db": delta_snr,
        "macro_band_power_error": macro_band_power_error,
        "bps": bps,
        "utility_rrmse_temporal": utility_rrmse_t,
        "utility_rrmse_spectral": utility_rrmse_s,
        "utility_cc": utility_cc,
        "utility_delta_snr": utility_delta_snr,
        "ds_legacy_arithmetic": ds,
        "ds": ds,
    }
    for index, name in enumerate(bands):
        result[f"band_power_error_{name}"] = relative_band_errors[..., index].mean(dim=1)
    return result
