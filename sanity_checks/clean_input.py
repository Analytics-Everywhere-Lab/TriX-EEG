#!/usr/bin/env python3
"""Run a trained denoiser on clean inputs with a zero-artifact reference."""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from sanity_checks.common import (
    MetricAccumulator,
    add_metric_scope,
    add_common_arguments,
    add_metric_arguments,
    base_metadata,
    build_dataset,
    load_model,
    make_loader,
    metric_record,
    select_device,
    set_seed,
    signals_from_batch,
    write_results,
)
from trix_eeg.common import flatten_signal, predict
from trix_eeg.level1_signal_validity import level1_signal_validity
from trix_eeg.level2_removal_faithfulness import level2_removal_faithfulness


def main(args) -> None:
    set_seed(args.seed)
    device = select_device(args.cpu)
    loader = make_loader(build_dataset(args), args)
    model, model_metadata = load_model(args, device)
    accumulator = MetricAccumulator()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Clean-input check"):
            _, clean, _ = signals_from_batch(batch, device)
            prediction = predict(model, clean)
            zero_artifact = torch.zeros_like(clean)
            level1 = level1_signal_validity(
                clean, prediction, clean, fs=args.fs,
                delta_snr_max=args.delta_snr_max,
                band_power_error_max=args.band_power_error_max,
                aggregation=args.aggregation,
            )
            level2 = level2_removal_faithfulness(
                clean, prediction, clean, zero_artifact,
                raer_error_max=args.raer_error_max,
                are_error_max=args.are_error_max,
                age_error_max=args.age_error_max,
                cnl_error_max=args.cnl_error_max,
                regression_ridge_fraction=args.regression_ridge_fraction,
                residual_ratio_max=args.residual_ratio_max,
                aggregation=args.aggregation,
            )
            removed = clean - prediction
            diagnostics = {
                "removed_l2": torch.linalg.vector_norm(flatten_signal(removed), dim=-1),
                "removed_relative_l2": (
                    torch.linalg.vector_norm(flatten_signal(removed), dim=-1)
                    / torch.linalg.vector_norm(flatten_signal(clean), dim=-1).clamp_min(1e-8)
                ),
            }
            accumulator.update({
                **{f"l1_{key}": level1[key] for key in (
                    "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db",
                    "macro_band_power_error", "bps", "ds",
                )},
                "l2_nlr": level2["nlr"],
                "l2_utility_nlr": level2["utility_nlr"],
                "l2_beta_clean": level2["beta_clean"],
                "l2_cnl": level2["cnl"],
                "l2_utility_cnl": level2["utility_cnl"],
                **diagnostics,
            })

    metadata = base_metadata(args, model_metadata)
    add_metric_scope(
        metadata,
        manipulated_inputs=["model_input_clean_instead_of_noisy", "zero_artifact"],
        reported_by_level={
            "level1": [
                "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db",
                "macro_band_power_error", "bps", "ds",
            ],
            "level2": ["beta_clean", "cnl", "utility_cnl", "nlr", "utility_nlr"],
            "diagnostic": ["removed_l2", "removed_relative_l2"],
        },
        omitted_invariants={
            "level2.artifact_required": (
                "RAS, RAER, ARE, AGE, RAR, and FS are undefined or misleading when artifact=0."
            ),
            "level3": "No attribution input is manipulated by this check.",
        },
    )
    metadata["expected_behavior"] = {
        "l1_ds": "high",
        "removed_l2_and_relative_l2": "very low",
        "l1_bps": "high",
        "l2_nlr": "low, but unstable/interpreted cautiously when removed energy is tiny",
        "l2_cnl": "low; directly measures removal proportional to the clean input",
        "artifact_required_metrics": "omitted because artifact=0",
    }
    write_results(
        args.output,
        [metric_record("clean_input_zero_artifact", accumulator.means())],
        metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_metric_arguments(parser)
    parser.add_argument("--output", default="runs/sanity_checks/clean_input.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
