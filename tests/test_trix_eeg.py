import math

import pytest
import torch
from torch.utils.data import Dataset

from trix_eeg.final_score import final_trix_score
from trix_eeg.aggregation import (
    AGGREGATION_CHOICES,
    LEGACY_AGGREGATION,
    validate_aggregation,
)
from trix_eeg.level1_signal_validity import level1_signal_validity
from trix_eeg.level2_removal_faithfulness import level2_removal_faithfulness
from trix_eeg.level3_explanation_reliability import (
    causal_deletion_effects,
    gradient_attribution,
    gradient_x_input_attribution,
    integrated_gradients_attribution,
    level3_explanation_reliability,
    smoothgrad_attribution,
    temporal_artifact_alignment,
)
from trix_eeg.sampling import stratified_subset_indices


class HalfDenoiser(torch.nn.Module):
    def forward(self, x):
        return {"y_hat": 0.5 * x}


class StratifiedToyDataset(Dataset):
    def __init__(self, strata=4, samples_per_stratum=10):
        self.samples = [
            {"snr_db": torch.tensor(float(stratum)), "value": index}
            for stratum in range(strata)
            for index in range(samples_per_stratum)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_stratified_subsets_are_balanced_reproducible_and_seeded():
    dataset = StratifiedToyDataset()
    first = stratified_subset_indices(dataset, 12, seed=42)
    repeated = stratified_subset_indices(dataset, 12, seed=42)
    second = stratified_subset_indices(dataset, 12, seed=43)

    assert first == repeated
    assert first != second
    assert len(first) == len(set(first)) == 12
    counts = {stratum: 0 for stratum in range(4)}
    for index in first:
        counts[int(dataset[index]["snr_db"].item())] += 1
    assert set(counts.values()) == {3}


def test_level1_perfect_reconstruction_has_unit_score():
    torch.manual_seed(1)
    clean = torch.randn(3, 1, 512)
    noisy = clean + 0.5 * torch.randn_like(clean)
    metrics = level1_signal_validity(noisy, clean, clean)
    assert torch.allclose(metrics["ds"], torch.ones(3), atol=1e-5)
    assert torch.allclose(metrics["bps"], torch.ones(3), atol=1e-5)


def test_only_legacy_arithmetic_aggregation_is_supported():
    assert AGGREGATION_CHOICES == (LEGACY_AGGREGATION,)
    assert validate_aggregation(LEGACY_AGGREGATION) == LEGACY_AGGREGATION
    with pytest.raises(ValueError, match="Unknown aggregation"):
        validate_aggregation("hierarchical_geometric")


def test_level1_arithmetic_formula():
    torch.manual_seed(14)
    clean = torch.randn(2, 1, 512)
    noisy = clean + 0.4 * torch.randn_like(clean)
    metrics = level1_signal_validity(noisy, 0.8 * clean, clean)
    expected = (
        metrics["utility_rrmse_temporal"]
        + metrics["utility_rrmse_spectral"]
        + metrics["utility_cc"]
        + metrics["utility_delta_snr"]
        + metrics["bps"]
    ) / 5.0
    assert torch.allclose(metrics["ds"], expected)
    assert torch.allclose(metrics["ds"], metrics["ds_legacy_arithmetic"])


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
    assert torch.allclose(
        metrics["artifact_reconstruction_error"], torch.zeros(2), atol=1e-5
    )
    assert torch.allclose(metrics["utility_are"], torch.ones(2), atol=1e-5)


def test_level2_arithmetic_formula():
    torch.manual_seed(15)
    clean = torch.randn(2, 1, 512)
    artifact = 0.5 * torch.randn_like(clean)
    noisy = clean + artifact
    metrics = level2_removal_faithfulness(
        noisy, clean + 0.1 * artifact, clean, artifact
    )
    expected = (
        metrics["ras"] + metrics["utility_raer"] + metrics["utility_are"]
        + metrics["utility_nlr"] + metrics["utility_rar"]
    ) / 5.0
    assert torch.allclose(metrics["fs"], expected)
    assert torch.allclose(metrics["fs"], metrics["fs_legacy_arithmetic"])


def test_artifact_reconstruction_error_penalizes_wrong_pairing():
    torch.manual_seed(8)
    clean = torch.randn(2, 1, 512)
    artifact = torch.randn_like(clean)
    coefficient = (
        (artifact * clean).sum(-1, keepdim=True)
        / clean.square().sum(-1, keepdim=True)
    )
    artifact = artifact - coefficient * clean
    noisy = clean + artifact
    correct = level2_removal_faithfulness(noisy, clean, clean, artifact)
    wrong = level2_removal_faithfulness(noisy, clean, clean, -artifact)
    assert torch.allclose(
        correct["artifact_reconstruction_error"], torch.zeros(2), atol=1e-5
    )
    assert torch.allclose(
        wrong["artifact_reconstruction_error"], 2.0 * torch.ones(2), atol=1e-5
    )
    assert bool((wrong["fs"] < correct["fs"]).all())


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
    expected_legacy = (
        metrics["taas"] + metrics["bac"] + metrics["caf"] + metrics["ass"]
    ) / 4.0
    assert torch.allclose(metrics["rs"], expected_legacy)
    assert torch.allclose(metrics["rs"], metrics["rs_legacy_arithmetic"])


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


def test_gradient_x_input_and_integrated_gradients_match_half_denoiser():
    noisy = torch.tensor([[[-2.0, -0.5, 0.5, 2.0]]])
    gradient_x_input = gradient_x_input_attribution(HalfDenoiser(), noisy)
    integrated_gradients = integrated_gradients_attribution(
        HalfDenoiser(), noisy, steps=8
    )
    assert torch.allclose(gradient_x_input, 0.5 * noisy.square(), atol=1e-6)
    assert torch.allclose(integrated_gradients, 0.25 * noisy.square(), atol=1e-6)


def test_level3_accepts_custom_attribution_function():
    torch.manual_seed(7)
    noisy = torch.randn(2, 1, 512)
    artifact = 0.25 * torch.randn_like(noisy)

    def constant_attribution(model, inputs, seed):
        return torch.ones_like(inputs)

    metrics = level3_explanation_reliability(
        HalfDenoiser(), noisy, artifact,
        attribution_fn=constant_attribution,
        deletion_fractions=(0.1,), random_deletion_repeats=1,
    )
    for key in ("taas", "bac", "caf", "ass", "rs"):
        assert torch.isfinite(metrics[key]).all()
        assert bool(((metrics[key] >= 0) & (metrics[key] <= 1)).all())


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
