#!/usr/bin/env python3
"""Evaluate a freshly initialized model on DS, FS, RS, and TDS."""

from __future__ import annotations

import argparse
import gc

import torch

from sanity_checks.common import (
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
    return {
        **evaluation["scores"],
        **{
            f"l1_{key}": value
            for key, value in evaluation["level1_signal_validity"].items()
        },
        **{
            f"l2_{key}": value
            for key, value in evaluation["level2_removal_faithfulness"].items()
        },
        **{
            f"l3_{key}": value
            for key, value in evaluation["level3_explanation_reliability"].items()
        },
    }


def main(args) -> None:
    set_seed(args.seed)
    device = select_device(args.cpu)
    loader = make_loader(build_dataset(args), args)
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
    parser.add_argument("--output", default="runs/sanity_checks/random_model.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
