"""Level 2: artifact-removal faithfulness metrics and score (FS)."""

from __future__ import annotations

import torch

from .aggregation import (
    LEGACY_AGGREGATION,
    validate_aggregation,
)

from .common import EPS, as_bct, correlation, flatten_signal


@torch.no_grad()
def level2_removal_faithfulness(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    clean: torch.Tensor,
    artifact: torch.Tensor | None = None,
    *,
    raer_error_max: float = 1.0,
    are_error_max: float = 1.0,
    age_error_max: float = 1.0,
    cnl_error_max: float = 1.0,
    regression_ridge_fraction: float = 1e-6,
    residual_ratio_max: float = 0.1,
    eps: float = EPS,
    aggregation: str = LEGACY_AGGREGATION,
) -> dict[str, torch.Tensor]:
    """Return removed-component diagnostics and the five-component FS.
    The primary FS averages RAS, RAER, ARE, AGE, and CNL. 
     
    """
    validate_aggregation(aggregation)
    noisy, prediction, clean = map(as_bct, (noisy, prediction, clean))
    artifact = noisy - clean if artifact is None else as_bct(artifact)
    if not (noisy.shape == prediction.shape == clean.shape == artifact.shape):
        raise ValueError("All Level-2 signals must have identical shapes")
    if (
        raer_error_max <= 0
        or are_error_max <= 0
        or age_error_max <= 0
        or cnl_error_max <= 0
        or residual_ratio_max <= 0
        or regression_ridge_fraction < 0
    ):
        raise ValueError("Normalization tolerances must be positive")

    # Ingredients 
    removed = noisy - prediction             # removed = artifact + residual
    removed_flat = flatten_signal(removed)
    clean_flat = flatten_signal(clean)
    artifact_flat = flatten_signal(artifact)
    removed_energy = removed_flat.square().sum(dim=-1)
    clean_energy = clean_flat.square().sum(dim=-1)
    artifact_energy = artifact_flat.square().sum(dim=-1)

    # RAS
    ras = torch.clamp(correlation(removed, artifact, eps), min=0.0)

    # RAER
    raer = torch.sqrt(removed_energy) / torch.sqrt(artifact_energy).clamp_min(eps)
    utility_raer = 1.0 - torch.clamp(
        (raer - 1.0).abs() / raer_error_max, 0.0, 1.0
    )

    # ARE
    artifact_reconstruction_error = torch.sqrt(
        (removed_flat - artifact_flat).square().sum(dim=-1)
    ) / torch.sqrt(artifact_energy).clamp_min(eps)
    utility_are = 1.0 - torch.clamp(
        artifact_reconstruction_error / are_error_max, 0.0, 1.0
    )

    projection_on_clean_coefficient = (
        (removed_flat * clean_flat).sum(dim=-1) / clean_energy.clamp_min(eps)
    )
    projected_clean_energy = (
        projection_on_clean_coefficient.square() * clean_energy
    )
    nlr = projected_clean_energy / removed_energy.clamp_min(eps)
    utility_nlr = 1.0 - torch.clamp(nlr, 0.0, 1.0)

    #  Jointly decompose the removed component:
    #   removed = beta_artifact * artifact + beta_clean * clean + residual.
    design = torch.stack((artifact_flat, clean_flat), dim=-1)
    target = removed_flat.unsqueeze(-1)
    gram = design.transpose(1, 2) @ design
    rhs = design.transpose(1, 2) @ target
    gram_scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
    identity = torch.eye(2, dtype=gram.dtype, device=gram.device).unsqueeze(0)
    ridge = regression_ridge_fraction * gram_scale.clamp_min(eps)
    regularized_gram = gram + ridge[:, None, None] * identity
    beta = torch.linalg.solve(regularized_gram, rhs).squeeze(-1)
    beta_artifact = beta[:, 0]
    beta_clean = beta[:, 1]
    regression_residual_flat = removed_flat - (design @ beta.unsqueeze(-1)).squeeze(-1)
    regression_residual_ratio = (
        regression_residual_flat.square().sum(dim=-1)
        / artifact_energy.clamp_min(eps)
    )
    regression_condition_number = torch.linalg.cond(regularized_gram)

    age = (beta_artifact - 1.0).abs()
    utility_age = 1.0 - torch.clamp(age / age_error_max, 0.0, 1.0)
    cnl = beta_clean.abs()
    utility_cnl = 1.0 - torch.clamp(cnl / cnl_error_max, 0.0, 1.0)

    # Retain RAR v2 as a diagnostic.
    reconstruction_error_flat = flatten_signal(prediction - clean)
    projection_on_artifact_coefficient = (
        (reconstruction_error_flat * artifact_flat).sum(dim=-1)
        / artifact_energy.clamp_min(eps)
    )
    projected_artifact_energy = (
        projection_on_artifact_coefficient.square() * artifact_energy
    )
    rar_error_projection_v2 = projected_artifact_energy / artifact_energy.clamp_min(eps)
    utility_rar_error_projection_v2 = 1.0 - torch.clamp(
        rar_error_projection_v2 / residual_ratio_max, 0.0, 1.0
    )

    # Retain the original prediction-based RAR as a diagnostic. 
    legacy_projection_coefficient = (
        (flatten_signal(prediction) * artifact_flat).sum(dim=-1)
        / artifact_energy.clamp_min(eps)
    )
    legacy_projected_artifact_energy = (
        legacy_projection_coefficient.square() * artifact_energy
    )
    rar_prediction_legacy = (
        legacy_projected_artifact_energy / artifact_energy.clamp_min(eps)
    )
    utility_rar_prediction_legacy = 1.0 - torch.clamp(
        rar_prediction_legacy / residual_ratio_max, 0.0, 1.0
    )
    rar = rar_prediction_legacy
    utility_rar = utility_rar_prediction_legacy
    fs = (
        torch.clamp(ras, 0.0, 1.0)
        + utility_raer
        + utility_are
        + utility_age
        + utility_cnl
    ) / 5.0
    fs_prediction_rar_legacy = (
        torch.clamp(ras, 0.0, 1.0)
        + utility_raer
        + utility_are
        + utility_nlr
        + utility_rar_prediction_legacy
    ) / 5.0
    fs_error_projection_v2 = (
        torch.clamp(ras, 0.0, 1.0)
        + utility_raer
        + utility_are
        + utility_nlr
        + utility_rar_error_projection_v2
    ) / 5.0
    return {
        "ras": ras,
        "raer": raer,
        "utility_raer": utility_raer,
        "artifact_reconstruction_error": artifact_reconstruction_error,
        "utility_are": utility_are,
        "nlr": nlr,
        "utility_nlr": utility_nlr,
        "beta_artifact": beta_artifact,
        "beta_clean": beta_clean,
        "age": age,
        "utility_age": utility_age,
        "cnl": cnl,
        "utility_cnl": utility_cnl,
        "regression_residual_ratio": regression_residual_ratio,
        "regression_condition_number": regression_condition_number,
        "rar": rar,
        "utility_rar": utility_rar,
        "rar_error_projection_v2": rar_error_projection_v2,
        "utility_rar_error_projection_v2": utility_rar_error_projection_v2,
        "fs_error_projection_v2": fs_error_projection_v2,
        "rar_prediction_legacy": rar_prediction_legacy,
        "utility_rar_prediction_legacy": utility_rar_prediction_legacy,
        "fs_prediction_rar_legacy": fs_prediction_rar_legacy,
        "fs_legacy_arithmetic": fs,
        "fs": fs,
    }
