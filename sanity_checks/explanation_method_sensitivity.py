#!/usr/bin/env python3
"""Compare Level-3 conclusions across Gradient x Input, SmoothGrad, and IG."""

from __future__ import annotations

import argparse

from tqdm import tqdm

from sanity_checks.common import (
    MetricAccumulator,
    add_metric_scope,
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
from trix_eeg.level3_explanation_reliability import (
    gradient_x_input_attribution,
    integrated_gradients_attribution,
    level3_explanation_reliability,
    smoothgrad_attribution,
)


def main(args) -> None:
    set_seed(args.seed)
    device = select_device(args.cpu)
    loader = make_loader(build_dataset(args), args, subset_seed=args.seed)
    model, model_metadata = load_model(args, device)
    config = make_config(args)

    methods = {
        "gradient_x_input": lambda model, noisy, seed: gradient_x_input_attribution(model, noisy),
        "smoothgrad": lambda model, noisy, seed: smoothgrad_attribution(
            model, noisy, num_samples=args.smoothgrad_samples,
            noise_fraction=args.smoothgrad_noise_fraction, seed=seed,
            gradient_batch_size=args.attribution_batch_size,
        ),
        "integrated_gradients": lambda model, noisy, seed: integrated_gradients_attribution(
            model, noisy, steps=args.ig_steps,
            gradient_batch_size=args.attribution_batch_size,
        ),
    }
    accumulators = {name: MetricAccumulator() for name in methods}
    for batch_index, batch in enumerate(tqdm(loader, desc="XAI-method sensitivity")):
        noisy, _, artifact = signals_from_batch(batch, device)
        for name, attribution_fn in methods.items():
            metrics = level3_explanation_reliability(
                model, noisy, artifact,
                attribution_fn=attribution_fn,
                seed=args.seed + 2 * batch_index,
                **level3_kwargs(config),
            )
            accumulators[name].update({
                key: value for key, value in metrics.items()
                if key in {"taas", "caf", "ass", "rs"}
            })

    records = [
        metric_record(name, accumulator.means(), method=name)
        for name, accumulator in accumulators.items()
    ]
    metadata = base_metadata(args, model_metadata)
    add_metric_scope(
        metadata,
        manipulated_inputs=["attribution_method", "attribution_map"],
        reported_by_level={"level3": ["taas", "caf", "ass", "rs"]},
        omitted_invariants={
            "level1": "Model prediction and signal references are unchanged.",
            "level2": "Removed signal and artifact reference are unchanged.",
            "level3.bac": "BAC does not consume the attribution map.",
        },
    )
    metadata.update({
        "smoothgrad_samples": args.smoothgrad_samples,
        "smoothgrad_noise_fraction": args.smoothgrad_noise_fraction,
        "integrated_gradients_steps": args.ig_steps,
        "attribution_gradient_batch_size": args.attribution_batch_size,
        "expected_behavior": (
            "TAAS, CAF, ASS, and RS need not be identical, but model rankings and "
            "the qualitative reliability conclusion should remain consistent. "
            "Invariant BAC is intentionally omitted from output."
        ),
    })
    write_results(args.output, records, metadata)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    add_metric_arguments(parser)
    parser.set_defaults(max_samples=64, batch_size=8)
    parser.add_argument("--smoothgrad-samples", type=int, default=32)
    parser.add_argument("--smoothgrad-noise-fraction", type=float, default=0.1)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument(
        "--attribution-batch-size", type=int, default=4,
        help="Maximum backward-pass batch size inside SmoothGrad and IG.",
    )
    parser.add_argument("--output", default="runs/sanity_checks/explanation_methods.json")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
