#!/usr/bin/env python3
"""Evaluate a freshly initialized model on DS, FS, RS, and TDS."""

from __future__ import annotations

import argparse
import copy
import gc

import torch

from sanity_checks.common import (
    add_metric_scope,
    add_common_arguments,
    add_metric_arguments,
    base_metadata,
    build_dataset,
    load_model,
    make_config,
    make_loader,
    metric_record,
    select_device,
    set_seed,
    write_results,
)
from trix_eeg.evaluator import evaluate_trix_model


def flatten_evaluation(evaluation):
    level1_names = (
        "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db",
        "macro_band_power_error", "bps", "ds",
    )
    level2_names = (
        "ras", "raer", "artifact_reconstruction_error",
        "beta_artifact", "age", "beta_clean", "cnl",
        "regression_residual_ratio", "regression_condition_number",
        "nlr", "rar",
        "rar_error_projection_v2", "rar_prediction_legacy", "fs",
        "fs_error_projection_v2", "fs_prediction_rar_legacy",
    )
    level3_names = ("taas", "bac", "caf", "ass", "rs")
    return {
        **{
            key: value for key, value in evaluation["scores"].items()
            if isinstance(value, (int, float))
        },
        **{
            f"l1_{key}": value
            for key, value in evaluation["level1_signal_validity"].items()
            if key in level1_names
        },
        **{
            f"l2_{key}": value
            for key, value in evaluation["level2_removal_faithfulness"].items()
            if key in level2_names
        },
        **{
            f"l3_{key}": value
            for key, value in evaluation["level3_explanation_reliability"].items()
            if key in level3_names
        },
    }


def main(args) -> None:
    set_seed(args.seed)
    device = select_device(args.cpu)
    # Level 1/2 use the complete fixed test set. The evaluator selects the
    # seed-specific stratified Level-3 subset internally.
    level12_args = copy.copy(args)
    level12_args.batch_size = args.level12_batch_size
    loader = make_loader(build_dataset(args), level12_args, max_samples=0)
    config = make_config(args)
    records = []
    trained_values = None
    if not args.untrained_only:
        trained_model, _ = load_model(args, device)
        trained_values = flatten_evaluation(evaluate_trix_model(
            trained_model, loader, device, config, description="Trained baseline"
        ))
        records.append(metric_record("trained", trained_values))
        del trained_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model, model_metadata = load_model(args, device, randomize=True)
    untrained_values = flatten_evaluation(evaluate_trix_model(
        model, loader, device, config, description="Untrained model"
    ))
    records.append(metric_record("untrained", untrained_values))
    if trained_values is not None:
        records.append(metric_record(
            "untrained_minus_trained",
            {name: untrained_values[name] - trained_values[name] for name in trained_values},
        ))
    metadata = base_metadata(args, model_metadata)
    add_metric_scope(
        metadata,
        manipulated_inputs=["model_parameters", "model_prediction", "attribution_map"],
        reported_by_level={
            "level1": [
                "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db",
                "macro_band_power_error", "bps", "ds",
            ],
            "level2": [
                "ras", "raer", "artifact_reconstruction_error",
                "beta_artifact", "age", "beta_clean", "cnl",
                "regression_residual_ratio", "regression_condition_number",
                "nlr", "rar",
                "rar_error_projection_v2", "rar_prediction_legacy", "fs",
                "fs_error_projection_v2", "fs_prediction_rar_legacy",
            ],
            "level3": ["taas", "bac", "caf", "ass", "rs"],
            "overall": ["tds"],
        },
    )
    metadata["expected_behavior"] = (
        "DS, FS, RS, and especially their geometric aggregate TDS should be lower "
        "than the matched trained checkpoint; this script intentionally does not "
        "load trained parameters."
    )
    write_results(args.output, records, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_metric_arguments(parser)
    parser.set_defaults(max_samples=64, batch_size=8)
    parser.add_argument(
        "--untrained-only", action="store_true",
        help="Skip the matched trained-checkpoint baseline.",
    )
    parser.add_argument(
        "--level12-batch-size", type=int, default=64,
        help="Batch size for complete-test-set Level-1/2 evaluation; Level 3 uses --batch-size.",
    )
    parser.add_argument("--output", default="runs/sanity_checks/random_model.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
