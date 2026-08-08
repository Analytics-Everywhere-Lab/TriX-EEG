"""Dataset-level streaming evaluation for the three TriX-EEG levels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .common import as_bct, predict
from .final_score import final_trix_score
from .level1_signal_validity import level1_signal_validity
from .level2_removal_faithfulness import level2_removal_faithfulness
from .level3_explanation_reliability import level3_explanation_reliability


@dataclass
class TriXConfig:
    fs: float = 256.0
    delta_snr_max: float = 20.0
    band_power_error_max: float = 1.0
    raer_error_max: float = 1.0
    are_error_max: float = 1.0
    residual_ratio_max: float = 0.1
    taas_frequency_range: tuple[float, float] = (0.5, 45.0)
    taas_n_fft: int = 128
    taas_hop_length: int = 32
    deletion_fractions: tuple[float, ...] = (0.1, 0.2, 0.3)
    random_deletion_repeats: int = 3
    stability_noise_fraction: float = 0.01
    level3_max_samples: int = 64
    level3_batch_size: int = 8
    seed: int = 42
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)


class _MetricAccumulator:
    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, torch.Tensor]):
        for name, values in metrics.items():
            values = values.detach().float().reshape(-1)
            finite = torch.isfinite(values)
            if bool(finite.any()):
                self.sums[name] = self.sums.get(name, 0.0) + float(values[finite].sum().cpu())
                self.counts[name] = self.counts.get(name, 0) + int(finite.sum())

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in self.sums
            if self.counts[name] > 0
        }


def _signals_from_batch(batch, device):
    noisy = as_bct(batch["x"].to(device, non_blocking=True).float())
    clean = as_bct(batch["clean"].to(device, non_blocking=True).float())
    artifact_value = batch.get("artifact", batch.get("noise"))
    artifact = noisy - clean if artifact_value is None else as_bct(
        artifact_value.to(device, non_blocking=True).float()
    )
    return noisy, clean, artifact


def _explanation_loader(loader, config: TriXConfig):
    dataset = loader.dataset
    requested = len(dataset) if config.level3_max_samples <= 0 else min(
        len(dataset), config.level3_max_samples
    )
    if requested == len(dataset):
        subset = dataset
    else:
        indices = torch.linspace(0, len(dataset) - 1, requested).round().long().tolist()
        subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=config.level3_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=loader.pin_memory,
    )


def evaluate_trix_model(model, loader, device, config: TriXConfig, *, description="TriX"):
    """Evaluate Level 1/2 on all windows and Level 3 on a fixed uniform subset."""
    model.eval()
    level1_accumulator = _MetricAccumulator()
    level2_accumulator = _MetricAccumulator()
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{description} L1/L2", leave=False):
            noisy, clean, artifact = _signals_from_batch(batch, device)
            prediction = predict(model, noisy)
            level1_accumulator.update(level1_signal_validity(
                noisy, prediction, clean, fs=config.fs,
                delta_snr_max=config.delta_snr_max,
                band_power_error_max=config.band_power_error_max,
            ))
            level2_accumulator.update(level2_removal_faithfulness(
                noisy, prediction, clean, artifact,
                raer_error_max=config.raer_error_max,
                are_error_max=config.are_error_max,
                residual_ratio_max=config.residual_ratio_max,
            ))
            total_samples += noisy.shape[0]

    level3_accumulator = _MetricAccumulator()
    explanation_samples = 0
    for batch_index, batch in enumerate(tqdm(
        _explanation_loader(loader, config), desc=f"{description} L3", leave=False
    )):
        noisy, _, artifact = _signals_from_batch(batch, device)
        level3_accumulator.update(level3_explanation_reliability(
            model, noisy, artifact, fs=config.fs,
            taas_frequency_range=config.taas_frequency_range,
            taas_n_fft=config.taas_n_fft,
            taas_hop_length=config.taas_hop_length,
            deletion_fractions=config.deletion_fractions,
            random_deletion_repeats=config.random_deletion_repeats,
            stability_noise_fraction=config.stability_noise_fraction,
            seed=config.seed + batch_index,
        ))
        explanation_samples += noisy.shape[0]

    level1 = level1_accumulator.means()
    level2 = level2_accumulator.means()
    level3 = level3_accumulator.means()
    tds = final_trix_score(
        level1["ds"], level2["fs"], level3["rs"], weights=config.weights
    )
    return {
        "num_samples_level1_level2": total_samples,
        "num_samples_level3": explanation_samples,
        "level1_signal_validity": level1,
        "level2_removal_faithfulness": level2,
        "level3_explanation_reliability": level3,
        "scores": {"ds": level1["ds"], "fs": level2["fs"], "rs": level3["rs"], "tds": tds},
        "settings": asdict(config),
    }
