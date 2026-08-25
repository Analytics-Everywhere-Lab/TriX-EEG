#!/usr/bin/env python3
"""Run and aggregate dependency-aware sanity checks over seeds."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


CHECKS = {
    "random_attribution": "sanity_checks.random_attribution",
    "random_model": "sanity_checks.random_model",
    "explanation_methods": "sanity_checks.explanation_method_sensitivity",
    "clean_input": "sanity_checks.clean_input",
    "metric_thresholds": "sanity_checks.metric_threshold_sensitivity",
    "wrong_artifact": "sanity_checks.wrong_artifact_pairing",
}
DETERMINISTIC_CHECKS = {"clean_input", "metric_thresholds"}
DATASETS = ("eeg_eog", "eeg_emg", "motion")
DONORS = {"eeg_eog": "eeg_emg", "eeg_emg": "eeg_eog", "motion": "eeg_eog"}


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--level12-batch-size", type=int, default=64)
    parser.add_argument(
        "--attribution-batch-size", type=int, default=8,
        help="Internal SmoothGrad/IG backward batch used by explanation-method checks.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Recompute seed files that already exist and parse as valid JSON.",
    )
    parser.add_argument(
        "--family", choices=("braanet", "benchmark"), default="benchmark",
        help="Model family passed to each sanity-check script.",
    )
    parser.add_argument(
        "--model", default="det",
        help="Benchmark model name; ignored when --family=braanet.",
    )
    parser.add_argument(
        "--model-label", default=None,
        help="Human-readable model name used in the final report.",
    )
    parser.add_argument(
        "--framework-label", default="joint-regression Level 2, legacy aggregation",
        help="Framework/metric-set description used in the final report.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if len(args.seeds) < 5:
        parser.error("At least five seeds are required")

    for seed in args.seeds:
        seed_dir = args.output_dir / "seeds" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            for prefix, script in CHECKS.items():
                output = seed_dir / f"{prefix}_{dataset}.json"
                first_seed_output = (
                    args.output_dir / "seeds" / f"seed_{args.seeds[0]}"
                    / f"{prefix}_{dataset}.json"
                )
                if (
                    seed != args.seeds[0]
                    and prefix in DETERMINISTIC_CHECKS
                    and first_seed_output.is_file()
                ):
                    payload = json.loads(first_seed_output.read_text())
                    payload["metadata"]["seed"] = seed
                    payload["metadata"].setdefault("arguments", {})["seed"] = seed
                    output.write_text(json.dumps(payload, indent=2))
                    shutil.copy2(first_seed_output.with_suffix(".csv"), output.with_suffix(".csv"))
                    print(f"= Reusing deterministic {first_seed_output} as {output}", flush=True)
                    continue
                if not args.no_resume and output.is_file():
                    try:
                        json.loads(output.read_text())
                    except (OSError, ValueError):
                        pass
                    else:
                        print(f"= Reusing {output}", flush=True)
                        continue
                command = [
                    sys.executable, "-m", script,
                    "--dataset", dataset,
                    "--family", args.family,
                    "--model", args.model,
                    "--aggregation", "legacy_arithmetic",
                    "--seed", str(seed),
                    "--data-seed", str(args.data_seed),
                    "--max-samples", str(args.max_samples),
                    "--batch-size", str(args.batch_size),
                    "--output", str(output),
                ]
                if args.cpu:
                    command.append("--cpu")
                if prefix == "wrong_artifact":
                    command.extend(("--donor-dataset", DONORS[dataset]))
                if prefix == "random_model":
                    command.extend(("--level12-batch-size", str(args.level12_batch_size)))
                if prefix == "explanation_methods":
                    command.extend((
                        "--attribution-batch-size", str(args.attribution_batch_size)
                    ))
                run(command)

    for dataset in DATASETS:
        for prefix in CHECKS:
            seed_json = [
                args.output_dir / "seeds" / f"seed_{seed}" / f"{prefix}_{dataset}.json"
                for seed in args.seeds
            ]
            output = args.output_dir / f"{prefix}_{dataset}.json"
            run([
                sys.executable, "-m", "sanity_checks.aggregate_multiseed",
                "--seed-json", *map(str, seed_json),
                "--seeds", *map(str, args.seeds),
                "--output", str(output),
            ])

    default_label = "BandRouteNet" if args.family == "braanet" else (
        "EEGDNet" if args.model == "det" else args.model
    )
    run([
        sys.executable, "-m", "sanity_checks.generate_levelwise_report",
        "--input-dir", str(args.output_dir),
        "--output", str(args.output_dir / "REPORT_LEVELWISE.md"),
        "--model-label", args.model_label or (
            f"{default_label} ({args.framework_label}, five subsets)"
        ),
    ])


if __name__ == "__main__":
    main()
