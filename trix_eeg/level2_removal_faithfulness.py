"""Level 2: artifact-removal faithfulness metrics and score (FS)."""

from __future__ import annotations

import torch

from .common import EPS, as_bct, correlation, flatten_signal


@torch.no_grad()
def level2_removal_faithfulness(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor | None = None,
    *,
    raer_error_max: float = 1.0,
    residual_ratio_max: float = 0.1,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Implement equations (14)-(23), returning one value per input window."""
    noisy, prediction, clean = map(as_bct, (noisy, prediction, clean))
    artifact = noisy - clean if artifact is None else as_bct(artifact)
    if not (noisy.shape == prediction.shape == clean.shape == artifact.shape):
        raise ValueError("All Level-2 signals must have identical shapes")
    if raer_error_max <= 0 or residual_ratio_max <= 0:
        raise ValueError("Normalization tolerances must be positive")

    removed = noisy - prediction
    removed_flat = flatten_signal(removed)
    clean_flat = flatten_signal(clean)
    artifact_flat = flatten_signal(artifact)
    removed_energy = removed_flat.square().sum(dim=-1)
    clean_energy = clean_flat.square().sum(dim=-1)
    artifact_energy = artifact_flat.square().sum(dim=-1)

    ras = torch.clamp(correlation(removed, artifact, eps), min=0.0)
    raer = torch.sqrt(removed_energy) / torch.sqrt(artifact_energy).clamp_min(eps)
    utility_raer = 1.0 - torch.clamp(
        (raer - 1.0).abs() / raer_error_max, 0.0, 1.0
    )

    projection_on_clean_coefficient = (
        (removed_flat * clean_flat).sum(dim=-1) / clean_energy.clamp_min(eps)
    )
    projected_clean_energy = (
        projection_on_clean_coefficient.square() * clean_energy
    )
    nlr = projected_clean_energy / removed_energy.clamp_min(eps)
    utility_nlr = 1.0 - torch.clamp(nlr, 0.0, 1.0)

    projection_on_artifact_coefficient = (
        (flatten_signal(prediction) * artifact_flat).sum(dim=-1)
        / artifact_energy.clamp_min(eps)
    )
    projected_artifact_energy = (
        projection_on_artifact_coefficient.square() * artifact_energy
    )
    # Equation (20) intentionally normalizes by true artifact energy.
    rar = projected_artifact_energy / artifact_energy.clamp_min(eps)
    utility_rar = 1.0 - torch.clamp(rar / residual_ratio_max, 0.0, 1.0)
    fs = (torch.clamp(ras, 0.0, 1.0) + utility_raer + utility_nlr + utility_rar) / 4.0
    return {
        "ras": ras,
        "raer": raer,
        "utility_raer": utility_raer,
        "nlr": nlr,
        "utility_nlr": utility_nlr,
        "rar": rar,
        "utility_rar": utility_rar,
        "fs": fs,
    }
