#!/usr/bin/env python3
"""Compare Level-3 metrics from real and randomly shuffled attribution maps."""

from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from sanity_checks.common import (
    MetricAccumulator,
    add_common_arguments,
    add_metric_arguments,
    base_metadata,
    build_dataset,
    level3_kwargs,
    load_model,
    make_config,
    make_loader,
    metric_record,
    select_device,
    set_seed,
    signals_from_batch,
    write_results,
)
from trix_eeg.common import as_bct, flatten_signal
from trix_eeg.level3_explanation_reliability import (
    gradient_attribution,
    level3_explanation_reliability,
)


def shuffled_gradient_attribution(model, noisy: torch.Tensor, seed: int) -> torch.Tensor:
    """Shuffle each sample's genuine gradient map over channel-time positions."""
    saliency = gradient_attribution(model, noisy)
    flat = flatten_signal(saliency)
    generator = torch.Generator(device=flat.device)
    generator.manual_seed(int(seed))
    shuffled = torch.empty_like(flat)
    for index in range(flat.shape[0]):
        permutation = torch.randperm(flat.shape[1], device=flat.device, generator=generator)
        shuffled[index] = flat[index, permutation]
    return shuffled.reshape_as(as_bct(saliency))


def attribution_dependent(metrics):
    return {
        name: value for name, value in metrics.items()
        if name in {"taas", "caf", "ass", "rs"} or name.startswith("caf_at_")
    }


def main(args) -> None:
    set_seed(args.seed)
    device = select_device(args.cpu)
    dataset = build_dataset(args)
    loader = make_loader(dataset, args)
    model, model_metadata = load_model(args, device)
    config = make_config(args)
    accumulators = {"real": MetricAccumulator(), "shuffled": MetricAccumulator()}

    for batch_index, batch in enumerate(tqdm(loader, desc="Random-attribution check")):
        noisy, _, artifact = signals_from_batch(batch, device)
        common = {
            **level3_kwargs(config),
            "seed": args.seed + 2 * batch_index,
        }
        accumulators["real"].update(attribution_dependent(
            level3_explanation_reliability(model, noisy, artifact, **common)
        ))
        accumulators["shuffled"].update(attribution_dependent(
            level3_explanation_reliability(
            model, noisy, artifact,
            attribution_fn=shuffled_gradient_attribution,
            perturbed_attribution_seed_offset=1,
            **common,
            )
        ))

    records = [
        metric_record(condition, accumulator.means())
        for condition, accumulator in accumulators.items()
    ]
    baseline, shuffled = (accumulators[name].means() for name in ("real", "shuffled"))
    records.append(metric_record(
        "shuffled_minus_real",
        {name: shuffled[name] - baseline[name] for name in baseline},
    ))
    metadata = base_metadata(args, model_metadata)
    metadata["expected_behavior"] = {
        "taas": "lower after shuffling",
        "caf": "lower or near zero after shuffling",
        "ass": "lower because baseline and perturbed maps are independently shuffled",
        "rs": "lower; invariant BAC is intentionally omitted from output",
    }
    write_results(args.output, records, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_metric_arguments(parser)
    parser.set_defaults(max_samples=64, batch_size=8)
    parser.add_argument("--output", default="runs/sanity_checks/random_attribution.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
