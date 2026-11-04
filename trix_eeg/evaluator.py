"""Dataset-level streaming evaluation for the three TriX-EEG levels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .common import as_bct, predict
from .aggregation import LEGACY_AGGREGATION, validate_aggregation
from .final_score import final_trix_score
from .level1_signal_validity import level1_signal_validity
from .level2_removal_faithfulness import level2_removal_faithfulness
from .level3_explanation_reliability import level3_explanation_reliability
from .sampling import stratified_subset_indices


@dataclass
class TriXConfig:
    aggregation: str = LEGACY_AGGREGATION
    fs: float = 256.0
    delta_snr_max: float = 20.0
    band_power_error_max: float = 1.0
    raer_error_max: float = 1.0
    are_error_max: float = 1.0
    age_error_max: float = 1.0
    cnl_error_max: float = 1.0
    regression_ridge_fraction: float = 1e-6
    residual_ratio_max: float = 0.1
    level2_definition: str = "joint_regression_age_cnl_v4"
    rar_definition: str = "prediction_projection_legacy_diagnostic_only"
    taas_frequency_range: tuple[float, float] = (0.5, 45.0)
    taas_n_fft: int = 128
    taas_hop_length: int = 32
    deletion_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8)
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


def _explanation_loader(loader, config: TriXConfig, subset_seed: int):
    dataset = loader.dataset
    indices = stratified_subset_indices(
        dataset, config.level3_max_samples, subset_seed
    )
    if len(indices) == len(dataset):
        subset = dataset
    else:
        subset = Subset(dataset, indices)
    explanation_loader = DataLoader(
        subset,
        batch_size=config.level3_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=loader.pin_memory,
    )
    return explanation_loader, indices


def _evaluate_level1_level2(model, loader, device, config, description):
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
                aggregation=config.aggregation,
            ))
            level2_accumulator.update(level2_removal_faithfulness(
                noisy, prediction, clean, artifact,
                raer_error_max=config.raer_error_max,
                are_error_max=config.are_error_max,
                age_error_max=config.age_error_max,
                cnl_error_max=config.cnl_error_max,
                regression_ridge_fraction=config.regression_ridge_fraction,
                residual_ratio_max=config.residual_ratio_max,
                aggregation=config.aggregation,
            ))
            total_samples += noisy.shape[0]
    return level1_accumulator.means(), level2_accumulator.means(), total_samples


def evaluate_trix_level1_level2(
    model, loader, device, config: TriXConfig, *, description="TriX"
):
    """Evaluate deterministic Levels 1 and 2 over the complete loader."""
    model.eval()
    validate_aggregation(config.aggregation)
    return _evaluate_level1_level2(model, loader, device, config, description)


def _evaluate_level3(model, loader, device, config, description, seed):
    level3_accumulator = _MetricAccumulator()
    explanation_samples = 0
    explanation_loader, subset_indices = _explanation_loader(loader, config, seed)
    for batch_index, batch in enumerate(tqdm(
        explanation_loader, desc=f"{description} L3", leave=False
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
            seed=seed + batch_index,
            aggregation=config.aggregation,
        ))
        explanation_samples += noisy.shape[0]
    return level3_accumulator.means(), explanation_samples, subset_indices


def _scores(level1, level2, level3, config):
    tds = final_trix_score(
        level1["ds"], level2["fs"], level3["rs"], weights=config.weights
    )
    tds_legacy = final_trix_score(
        level1["ds_legacy_arithmetic"],
        level2["fs_legacy_arithmetic"],
        level3["rs_legacy_arithmetic"],
        weights=config.weights,
    )
    # For comparison only
    tds_prediction_rar_legacy = final_trix_score(
        level1["ds"],
        level2["fs_prediction_rar_legacy"],
        level3["rs"],
        weights=config.weights,
    )
    tds_error_projection_v2 = final_trix_score(
        level1["ds"],
        level2["fs_error_projection_v2"],
        level3["rs"],
        weights=config.weights,
    )
    return {
        "aggregation": config.aggregation,
        "ds": level1["ds"], "fs": level2["fs"], "rs": level3["rs"],
        "tds": tds,
        "ds_legacy_arithmetic": level1["ds_legacy_arithmetic"],
        "fs_legacy_arithmetic": level2["fs_legacy_arithmetic"],
        "rs_legacy_arithmetic": level3["rs_legacy_arithmetic"],
        "tds_legacy_arithmetic": tds_legacy,
        "fs_prediction_rar_legacy": level2["fs_prediction_rar_legacy"],
        "tds_prediction_rar_legacy": tds_prediction_rar_legacy,
        "fs_error_projection_v2": level2["fs_error_projection_v2"],
        "tds_error_projection_v2": tds_error_projection_v2,
    }


def evaluate_trix_model(model, loader, device, config: TriXConfig, *, description="TriX"):
    """Evaluate Level 1/2 fully and Level 3 on one seeded stratified subset."""
    model.eval()
    validate_aggregation(config.aggregation)
    level1, level2, total_samples = _evaluate_level1_level2(
        model, loader, device, config, description
    )
    level3, explanation_samples, subset_indices = _evaluate_level3(
        model, loader, device, config, description, config.seed
    )

    return {
        "num_samples_level1_level2": total_samples,
        "num_samples_level3": explanation_samples,
        "level3_subset_indices": subset_indices,
        "level1_signal_validity": level1,
        "level2_removal_faithfulness": level2,
        "level3_explanation_reliability": level3,
        "scores": _scores(level1, level2, level3, config),
        "settings": asdict(config),
    }


def _summarize_dicts(records):
    means, stds = {}, {}
    common_names = set.intersection(*(set(record) for record in records))
    for name in sorted(common_names):
        values = [record[name] for record in records]
        if all(isinstance(value, (int, float, bool)) for value in values):
            numeric = [float(value) for value in values]
            means[name] = statistics.mean(numeric)
            stds[name] = statistics.stdev(numeric) if len(numeric) > 1 else 0.0
        elif len(set(map(str, values))) == 1:
            means[name] = values[0]
    return means, stds


def evaluate_trix_model_multiseed(
    model, loader, device, config: TriXConfig, seeds, *, description="TriX"
):
    """Repeat Level 3 over seeded, stratified subsets of one fixed test split.

    Level 1 and Level 2 are deterministic for fixed inputs/checkpoints, so they are
    evaluated once and are not assigned uncertainty fields. Level 3 and the
    resulting TDS are summarized with the mean and sample standard deviation over
    ``seeds``.
    """
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) < 2:
        raise ValueError("Multi-seed evaluation requires at least two seeds")
    model.eval()
    validate_aggregation(config.aggregation)
    level1, level2, total_samples = _evaluate_level1_level2(
        model, loader, device, config, description
    )
    level3_runs, score_runs = [], []
    subset_indices_by_seed = {}
    explanation_samples = None
    for seed in seeds:
        level3, current_samples, subset_indices = _evaluate_level3(
            model, loader, device, config, f"{description} seed={seed}", seed
        )
        if explanation_samples is not None and current_samples != explanation_samples:
            raise RuntimeError("Level-3 sample count changed across seeds")
        explanation_samples = current_samples
        subset_indices_by_seed[str(seed)] = subset_indices
        level3_runs.append(level3)
        score_runs.append(_scores(level1, level2, level3, config))

    level3_mean, level3_std = _summarize_dicts(level3_runs)
    scores_mean, scores_std = _summarize_dicts(score_runs)
    stochastic_score_std = {
        name: value for name, value in scores_std.items()
        if name == "rs" or name == "tds"
        or name.startswith("rs_") or name.startswith("tds_")
    }
    settings = asdict(config)
    settings["evaluation_seeds"] = list(seeds)
    evaluation = {
        "num_samples_level1_level2": total_samples,
        "num_samples_level3": explanation_samples,
        "level3_subset_indices_by_seed": subset_indices_by_seed,
        "level1_signal_validity": level1,
        "level2_removal_faithfulness": level2,
        "level3_explanation_reliability": level3_mean,
        "scores": scores_mean,
        "settings": settings,
    }
    evaluation_std = {
        "level3_explanation_reliability": level3_std,
        "scores": stochastic_score_std,
    }
    return evaluation, evaluation_std
