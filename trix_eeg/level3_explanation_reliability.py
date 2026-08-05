"""Level 3: explanation reliability metrics and score (RS)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .common import (
    EPS,
    NEURAL_BANDS,
    as_bct,
    flatten_signal,
    predict,
    spectral_band_power,
)


def removed_energy(model, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = predict(model, noisy)
    energy = flatten_signal(as_bct(noisy) - prediction).square().sum(dim=-1)
    return prediction, energy


@torch.enable_grad()
def gradient_attribution(
    model, noisy: torch.Tensor, *, absolute: bool = True
) -> torch.Tensor:
    """Input gradient of g(y)=||y-f(y)||_2^2 (equations 24-25).

    Metrics use the default non-negative magnitude map. ``absolute=False`` is
    provided for visualizations that need to preserve gradient direction.
    """
    differentiable_input = as_bct(noisy).detach().clone().requires_grad_(True)
    _, energy = removed_energy(model, differentiable_input)
    gradient = torch.autograd.grad(
        energy.sum(), differentiable_input, create_graph=False, retain_graph=False
    )[0]
    gradient = gradient.detach()
    return gradient.abs() if absolute else gradient


@torch.enable_grad()
def smoothgrad_attribution(
    model,
    noisy: torch.Tensor,
    *,
    num_samples: int = 32,
    noise_fraction: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """Average absolute gradients over RMS-scaled Gaussian perturbations.

    The target remains ``g(y)=||y-f(y)||_2^2``. Averaging non-negative maps
    preserves the saliency convention in equation (24).
    """
    noisy = as_bct(noisy).detach()
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_fraction < 0:
        raise ValueError("noise_fraction must be non-negative")
    if noise_fraction == 0:
        return gradient_attribution(model, noisy)

    model.eval()
    generator = torch.Generator(device=noisy.device)
    generator.manual_seed(int(seed))
    noise_scale = noise_fraction * noisy.square().mean(dim=-1, keepdim=True).sqrt()
    accumulated = torch.zeros_like(noisy)
    for _ in range(int(num_samples)):
        perturbation = torch.randn(
            noisy.shape,
            dtype=noisy.dtype,
            device=noisy.device,
            generator=generator,
        ) * noise_scale
        accumulated += gradient_attribution(model, noisy + perturbation)
    return (accumulated / float(num_samples)).detach()


def _suppress_band(x: torch.Tensor, fs: float, low: float, high: float) -> torch.Tensor:
    x = as_bct(x)
    spectrum = torch.fft.rfft(x, dim=-1)
    frequencies = torch.fft.rfftfreq(x.shape[-1], 1.0 / float(fs), device=x.device)
    mask = (frequencies >= low) & (frequencies < high)
    filtered = spectrum.clone()
    filtered[..., mask] = 0
    return torch.fft.irfft(filtered, n=x.shape[-1], dim=-1)


def _delete_with_zero_baseline(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return as_bct(x).masked_fill(mask, 0.0)


def _top_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.Tensor:
    flattened = flatten_signal(scores)
    count = max(1, min(flattened.shape[-1], round(fraction * flattened.shape[-1])))
    indices = flattened.topk(count, dim=-1).indices
    mask = torch.zeros_like(flattened, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.reshape_as(as_bct(scores))


def _random_fraction_mask(
    x: torch.Tensor, fraction: float, generator: torch.Generator
) -> torch.Tensor:
    shape = flatten_signal(x).shape
    random_scores = torch.rand(shape, device=x.device, generator=generator)
    return _top_fraction_mask(random_scores.reshape_as(as_bct(x)), fraction)


def causal_deletion_effects(
    model,
    noisy: torch.Tensor,
    *,
    saliency: torch.Tensor | None = None,
    deletion_fractions: Sequence[float] = (0.1, 0.2, 0.3),
    random_deletion_repeats: int = 3,
    seed: int = 42,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Return Dtop, Drand, and CAF(k) from equations (34)-(36).

    Results have shape ``[batch, num_fractions]``. This public diagnostic is
    useful for plotting the causal quantities hidden by the final mean CAF.
    The deletion baseline and random protocol match
    :func:`level3_explanation_reliability` exactly.
    """
    noisy = as_bct(noisy)
    fractions = tuple(float(value) for value in deletion_fractions)
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("Deletion fractions must be non-empty and lie in (0,1)")
    if random_deletion_repeats <= 0:
        raise ValueError("random_deletion_repeats must be positive")
    model.eval()
    saliency = gradient_attribution(model, noisy) if saliency is None else as_bct(saliency)
    if saliency.shape != noisy.shape:
        raise ValueError("saliency and noisy must have identical shapes")

    top_effects, random_effects = [], []
    generator = torch.Generator(device=noisy.device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        _, base_energy = removed_energy(model, noisy)
        for fraction in fractions:
            top_mask = _top_fraction_mask(saliency, fraction)
            _, top_energy = removed_energy(
                model, _delete_with_zero_baseline(noisy, top_mask)
            )
            d_top = (base_energy - top_energy).abs() / base_energy.abs().clamp_min(eps)
            d_random = torch.zeros_like(d_top)
            for _ in range(random_deletion_repeats):
                random_mask = _random_fraction_mask(noisy, fraction, generator)
                _, random_energy = removed_energy(
                    model, _delete_with_zero_baseline(noisy, random_mask)
                )
                d_random += (
                    (base_energy - random_energy).abs()
                    / base_energy.abs().clamp_min(eps)
                )
            top_effects.append(d_top)
            random_effects.append(d_random / random_deletion_repeats)
    d_top = torch.stack(top_effects, dim=-1)
    d_random = torch.stack(random_effects, dim=-1)
    return {
        "d_top": d_top.detach(),
        "d_random": d_random.detach(),
        "caf_by_fraction": torch.clamp(d_top - d_random, 0.0, 1.0).detach(),
    }


def temporal_artifact_alignment(
    saliency: torch.Tensor,
    artifact: torch.Tensor,
    *,
    fs: float = 256.0,
    frequency_range: tuple[float, float] = (0.5, 45.0),
    n_fft: int = 128,
    hop_length: int = 32,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Compute Soft Temporal Artifact Alignment (TAAS).

    Artifact energy is marginalized over channels and the selected STFT
    frequencies. Attribution magnitudes are averaged over the same Hann-window
    frame locations and marginalized over channels. Both temporal profiles are
    L1-normalized before calculating total-variation similarity.
    """
    saliency, artifact = map(as_bct, (saliency, artifact))
    if saliency.shape != artifact.shape:
        raise ValueError("saliency and artifact must have identical shapes")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if n_fft <= 1 or n_fft > artifact.shape[-1]:
        raise ValueError("n_fft must be in [2, signal_length]")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    low, high = map(float, frequency_range)
    if not 0.0 <= low < high <= fs / 2.0:
        raise ValueError("frequency_range must satisfy 0 <= low < high <= fs/2")

    batch, channels, signal_length = artifact.shape
    window = torch.hann_window(
        n_fft, dtype=artifact.dtype, device=artifact.device
    )
    artifact_stft = torch.stft(
        artifact.reshape(batch * channels, signal_length),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    frequencies = torch.fft.rfftfreq(
        n_fft, d=1.0 / float(fs), device=artifact.device
    )
    frequency_mask = (frequencies >= low) & (frequencies <= high)
    if not bool(frequency_mask.any()):
        raise ValueError(
            f"No STFT bins in artifact frequency range {low:g}-{high:g} Hz"
        )
    frame_count = artifact_stft.shape[-1]
    artifact_energy = artifact_stft.abs().square().reshape(
        batch, channels, -1, frame_count
    )[..., frequency_mask, :].sum(dim=(1, 2))

    # Hann weighting mirrors the temporal support of each artifact STFT frame.
    saliency_frames = F.conv1d(
        saliency.abs().reshape(batch * channels, 1, signal_length),
        window.reshape(1, 1, -1),
        stride=hop_length,
    ).reshape(batch, channels, frame_count).sum(dim=1)

    artifact_distribution = artifact_energy / artifact_energy.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    attribution_distribution = saliency_frames / saliency_frames.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    taas = 1.0 - 0.5 * (
        attribution_distribution - artifact_distribution
    ).abs().sum(dim=-1)
    frame_times = (
        torch.arange(frame_count, device=artifact.device, dtype=artifact.dtype)
        * hop_length
        + n_fft / 2.0
    ) / float(fs)
    return {
        "taas": taas.clamp(0.0, 1.0),
        "attribution_distribution": attribution_distribution,
        "artifact_distribution": artifact_distribution,
        "artifact_energy": artifact_energy,
        "frame_times": frame_times,
    }


def level3_explanation_reliability(
    model,
    noisy: torch.Tensor,
    artifact: torch.Tensor,
    *,
    fs: float = 256.0,
    bands=NEURAL_BANDS,
    taas_frequency_range: tuple[float, float] = (0.5, 45.0),
    taas_n_fft: int = 128,
    taas_hop_length: int = 32,
    deletion_fractions: Sequence[float] = (0.1, 0.2, 0.3),
    random_deletion_repeats: int = 3,
    stability_noise_fraction: float = 0.01,
    seed: int = 42,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    """Implement equations (24)-(38) using reproducible gradient saliency.

    This implementation uses absolute input gradients, soft temporal artifact
    alignment, a zero deletion baseline, and RMS-scaled Gaussian perturbations.
    """
    noisy, artifact = map(as_bct, (noisy, artifact))
    if noisy.shape != artifact.shape:
        raise ValueError("noisy and artifact must have identical shapes")
    if stability_noise_fraction < 0:
        raise ValueError("Perturbation scale must be non-negative")
    if random_deletion_repeats <= 0:
        raise ValueError("random_deletion_repeats must be positive")
    fractions = tuple(float(value) for value in deletion_fractions)
    if not fractions or any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("Deletion fractions must be non-empty and lie in (0,1)")

    model.eval()
    saliency = gradient_attribution(model, noisy)
    temporal_alignment = temporal_artifact_alignment(
        saliency,
        artifact,
        fs=fs,
        frequency_range=taas_frequency_range,
        n_fft=taas_n_fft,
        hop_length=taas_hop_length,
        eps=eps,
    )
    taas = temporal_alignment["taas"]

    with torch.no_grad():
        prediction, base_energy = removed_energy(model, noisy)
        importance = []
        for low, high in bands.values():
            suppressed_prediction = predict(model, _suppress_band(noisy, fs, low, high))
            numerator = torch.linalg.vector_norm(
                flatten_signal(prediction - suppressed_prediction), dim=-1
            )
            denominator = torch.linalg.vector_norm(
                flatten_signal(prediction), dim=-1
            ).clamp_min(eps)
            importance.append(numerator / denominator)
        importance = torch.stack(importance, dim=-1)
        p_distribution = importance / importance.sum(dim=-1, keepdim=True).clamp_min(eps)
        artifact_power = spectral_band_power(artifact, fs, bands).sum(dim=1)
        q_distribution = artifact_power / artifact_power.sum(dim=-1, keepdim=True).clamp_min(eps)
        bac = 1.0 - 0.5 * (p_distribution - q_distribution).abs().sum(dim=-1)

        generator = torch.Generator(device=noisy.device)
        generator.manual_seed(int(seed))
        caf_by_fraction = []
        for fraction in fractions:
            top_mask = _top_fraction_mask(saliency, fraction)
            _, top_energy = removed_energy(
                model, _delete_with_zero_baseline(noisy, top_mask)
            )
            d_top = (base_energy - top_energy).abs() / base_energy.abs().clamp_min(eps)
            random_effect = torch.zeros_like(d_top)
            for _ in range(random_deletion_repeats):
                random_mask = _random_fraction_mask(noisy, fraction, generator)
                _, random_energy = removed_energy(
                    model, _delete_with_zero_baseline(noisy, random_mask)
                )
                random_effect += (
                    (base_energy - random_energy).abs()
                    / base_energy.abs().clamp_min(eps)
                )
            random_effect /= random_deletion_repeats
            caf_by_fraction.append(torch.clamp(d_top - random_effect, 0.0, 1.0))
        caf = torch.stack(caf_by_fraction, dim=-1).mean(dim=-1)

        noise_rms = noisy.square().mean(dim=-1, keepdim=True).sqrt()
        perturbation = torch.randn(
            noisy.shape, dtype=noisy.dtype, device=noisy.device, generator=generator
        ) * (stability_noise_fraction * noise_rms)
    perturbed_saliency = gradient_attribution(model, noisy + perturbation)
    ass = 1.0 - (
        flatten_signal(saliency - perturbed_saliency).abs().sum(dim=-1)
        / (
            flatten_signal(saliency).sum(dim=-1)
            + flatten_signal(perturbed_saliency).sum(dim=-1)
            + eps
        )
    )
    ass = torch.clamp(ass, 0.0, 1.0)
    rs = (taas + bac + caf + ass) / 4.0
    result = {
        "taas": taas.detach(),
        "bac": bac.detach(),
        "caf": caf.detach(),
        "ass": ass.detach(),
        "rs": rs.detach(),
    }
    for index, fraction in enumerate(fractions):
        result[f"caf_at_{fraction:g}"] = caf_by_fraction[index].detach()
    return result
