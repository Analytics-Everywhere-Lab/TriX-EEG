#!/usr/bin/env python3
"""One-factor-at-a-time sensitivity sweep for TriX metric utility thresholds."""

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
from trix_eeg.common import predict
from trix_eeg.level1_signal_validity import level1_signal_validity
from trix_eeg.level2_removal_faithfulness import level2_removal_faithfulness


THRESHOLD_FIELDS = (
    "delta_snr_max",
    "band_power_error_max",
    "raer_error_max",
    "are_error_max",
    "residual_ratio_max",
)

UTILITY_BY_THRESHOLD = {
    "delta_snr_max": ("utility_delta_snr", "ds", "level1"),
    "band_power_error_max": ("bps", "ds", "level1"),
    "raer_error_max": ("utility_raer", "fs", "level2"),
    "are_error_max": ("utility_are", "fs", "level2"),
    "residual_ratio_max": ("utility_rar", "fs", "level2"),
}


def conditions(args):
    baseline = {name: float(getattr(args, name)) for name in THRESHOLD_FIELDS}
    yield "baseline", "baseline", None, baseline
    sweeps = {
        "delta_snr_max": args.delta_snr_max_values,
        "band_power_error_max": args.band_power_error_max_values,
        "raer_error_max": args.raer_error_max_values,
        "are_error_max": args.are_error_max_values,
        "residual_ratio_max": args.residual_ratio_max_values,
    }
    for name, values in sweeps.items():
        for value in values:
            value = float(value)
            if value <= 0:
                raise ValueError(f"{name} values must be positive")
            settings = dict(baseline)
            settings[name] = value
            yield f"{name}={value:g}", name, value, settings


def main(args) -> None:
    set_seed(args.seed)
    device = select_device(args.cpu)
    loader = make_loader(build_dataset(args), args)
    model, model_metadata = load_model(args, device)
    condition_list = list(conditions(args))
    accumulators = {label: MetricAccumulator() for label, _, _, _ in condition_list}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Threshold sensitivity"):
            noisy, clean, artifact = signals_from_batch(batch, device)
            prediction = predict(model, noisy)
            for label, threshold_name, _, settings in condition_list:
                level1 = level1_signal_validity(
                    noisy, prediction, clean, fs=args.fs,
                    delta_snr_max=settings["delta_snr_max"],
                    band_power_error_max=settings["band_power_error_max"],
                    aggregation=args.aggregation,
                )
                level2 = level2_removal_faithfulness(
                    noisy, prediction, clean, artifact,
                    raer_error_max=settings["raer_error_max"],
                    are_error_max=settings["are_error_max"],
                    residual_ratio_max=settings["residual_ratio_max"],
                    aggregation=args.aggregation,
                )
                if threshold_name == "baseline":
                    selected = {
                        "ds": level1["ds"],
                        "fs": level2["fs"],
                        "utility_delta_snr": level1["utility_delta_snr"],
                        "bps": level1["bps"],
                        "utility_raer": level2["utility_raer"],
                        "utility_are": level2["utility_are"],
                        "utility_rar": level2["utility_rar"],
                    }
                else:
                    utility, score, level = UTILITY_BY_THRESHOLD[threshold_name]
                    source = level1 if level == "level1" else level2
                    selected = {utility: source[utility], score: source[score]}
                accumulators[label].update(selected)

    records = [
        metric_record(
            label, accumulators[label].means(),
            threshold=name, threshold_value=value,
        )
        for label, name, value, _ in condition_list
    ]
    metadata = base_metadata(args, model_metadata)
    add_metric_scope(
        metadata,
        manipulated_inputs=["metric_normalization_threshold"],
        reported_by_level={
            "level1": ["utility_delta_snr", "bps", "ds"],
            "level2": ["utility_raer", "utility_are", "utility_rar", "fs"],
        },
        omitted_invariants={
            "raw_metrics": "Raw metrics do not consume normalization thresholds.",
            "level3": "No Level-3 threshold is swept by this experiment.",
        },
    )
    metadata["protocol"] = (
        "One-factor-at-a-time sweep. Each row reports only the utility and "
        "aggregate score that consume the manipulated threshold."
    )
    write_results(args.output, records, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_metric_arguments(parser)
    parser.add_argument("--delta-snr-max-values", nargs="+", type=float,
                        default=[10.0, 20.0, 30.0])
    parser.add_argument("--band-power-error-max-values", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--raer-error-max-values", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--are-error-max-values", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--residual-ratio-max-values", nargs="+", type=float,
                        default=[0.05, 0.1, 0.2])
    parser.add_argument("--output", default="runs/sanity_checks/metric_thresholds.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
