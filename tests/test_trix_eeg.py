import math

import pytest
import torch

from trix_eeg.final_score import final_trix_score
from trix_eeg.level1_signal_validity import level1_signal_validity
from trix_eeg.level2_removal_faithfulness import level2_removal_faithfulness
from trix_eeg.level3_explanation_reliability import (
    causal_deletion_effects,
    gradient_attribution,
    level3_explanation_reliability,
    smoothgrad_attribution,
    temporal_artifact_alignment,
)


class HalfDenoiser(torch.nn.Module):
    def forward(self, x):
        return {"y_hat": 0.5 * x}


def test_level1_perfect_reconstruction_has_unit_score():
    torch.manual_seed(1)
    clean = torch.randn(3, 1, 512)
    noisy = clean + 0.5 * torch.randn_like(clean)
    metrics = level1_signal_validity(noisy, clean, clean)
    assert torch.allclose(metrics["ds"], torch.ones(3), atol=1e-5)
    assert torch.allclose(metrics["bps"], torch.ones(3), atol=1e-5)


def test_level2_exact_orthogonal_artifact_is_faithful():
    torch.manual_seed(2)
    clean = torch.randn(2, 1, 512)
    artifact = torch.randn_like(clean)
    coefficient = (artifact * clean).sum(-1, keepdim=True) / clean.square().sum(-1, keepdim=True)
    artifact = artifact - coefficient * clean
    noisy = clean + artifact
    metrics = level2_removal_faithfulness(noisy, clean, clean, artifact)
    assert torch.allclose(metrics["fs"], torch.ones(2), atol=1e-5)
    assert torch.allclose(metrics["raer"], torch.ones(2), atol=1e-5)


def test_level3_metrics_are_bounded():
    torch.manual_seed(3)
    noisy = torch.randn(2, 1, 512)
    artifact = 0.25 * torch.randn_like(noisy)
    metrics = level3_explanation_reliability(
        HalfDenoiser(), noisy, artifact,
        deletion_fractions=(0.1,), random_deletion_repeats=1,
    )
    for key in ("taas", "bac", "caf", "ass", "rs"):
        assert bool(((metrics[key] >= 0) & (metrics[key] <= 1)).all())


def test_causal_deletion_diagnostics_have_expected_shape():
    torch.manual_seed(4)
    noisy = torch.randn(3, 1, 512)
    effects = causal_deletion_effects(
        HalfDenoiser(), noisy,
        deletion_fractions=(0.1, 0.2), random_deletion_repeats=1,
    )
    assert effects["d_top"].shape == (3, 2)
    assert effects["d_random"].shape == (3, 2)
    assert effects["caf_by_fraction"].shape == (3, 2)
    assert torch.isfinite(effects["d_top"]).all()
    assert bool(((effects["caf_by_fraction"] >= 0) & (effects["caf_by_fraction"] <= 1)).all())


def test_smoothgrad_is_reproducible_and_nonnegative():
    torch.manual_seed(5)
    noisy = torch.randn(2, 1, 512)
    first = smoothgrad_attribution(
        HalfDenoiser(), noisy, num_samples=4, noise_fraction=0.1, seed=9
    )
    second = smoothgrad_attribution(
        HalfDenoiser(), noisy, num_samples=4, noise_fraction=0.1, seed=9
    )
    assert first.shape == noisy.shape
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()
    assert bool((first >= 0).all())


def test_gradient_attribution_can_preserve_sign_for_visualization():
    noisy = torch.tensor([[[-2.0, -0.5, 0.5, 2.0]]])
    signed = gradient_attribution(HalfDenoiser(), noisy, absolute=False)
    magnitude = gradient_attribution(HalfDenoiser(), noisy)
    assert bool((signed[..., :2] < 0).all())
    assert bool((signed[..., 2:] > 0).all())
    assert torch.allclose(magnitude, signed.abs())


def test_taas_is_bounded_normalized_and_scale_invariant():
    torch.manual_seed(6)
    saliency = torch.rand(2, 1, 512)
    artifact = torch.randn(2, 1, 512)
    first = temporal_artifact_alignment(saliency, artifact)
    scaled = temporal_artifact_alignment(7.0 * saliency, 3.0 * artifact)
    assert first["taas"].shape == (2,)
    assert bool(((first["taas"] >= 0) & (first["taas"] <= 1)).all())
    assert torch.allclose(
        first["attribution_distribution"].sum(-1), torch.ones(2), atol=1e-6
    )
    assert torch.allclose(
        first["artifact_distribution"].sum(-1), torch.ones(2), atol=1e-6
    )
    assert torch.allclose(first["taas"], scaled["taas"], atol=1e-6)


def test_final_score_equal_and_weighted():
    assert final_trix_score(0.5, 0.5, 0.5) == pytest.approx(0.5)
    expected = math.sqrt(0.25 * 1.0)
    assert final_trix_score(0.25, 1.0, 0.1, weights=(1, 1, 0)) == pytest.approx(expected)
    assert final_trix_score(0.0, 1.0, 1.0) == 0.0
