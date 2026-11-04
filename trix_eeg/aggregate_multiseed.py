#!/usr/bin/env python3
"""Aggregate repeated TriX-EEG evaluations as mean and sample standard deviation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from copy import deepcopy
from pathlib import Path


LEVEL_GROUPS = (
    "level1_signal_validity",
    "level2_removal_faithfulness",
    "level3_explanation_reliability",
    "scores",
)
CSV_METRICS = (
    "ds", "fs", "rs", "tds", "tds_error_projection_v2", "tds_prediction_rar_legacy",
    "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db", "bps",
    "ras", "raer", "artifact_reconstruction_error",
    "beta_artifact", "age", "utility_age", "beta_clean", "cnl", "utility_cnl",
    "regression_residual_ratio", "regression_condition_number", "nlr", "rar",
    "rar_error_projection_v2", "fs_error_projection_v2",
    "rar_prediction_legacy", "fs_prediction_rar_legacy",
    "taas", "bac", "caf", "ass",
)
STOCHASTIC_METRICS = {
    "rs", "tds", "tds_error_projection_v2", "tds_prediction_rar_legacy",
    "taas", "bac", "caf", "ass",
}


def _mean_std(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _result_key(row):
    return row["dataset"], row["family"], row["model"]


def aggregate_payloads(payloads, seeds):
    indexed = [
        {_result_key(row): row for row in payload["results"]}
        for payload in payloads
    ]
    keys = set(indexed[0])
    if any(set(index) != keys for index in indexed[1:]):
        raise ValueError("Seed runs do not contain identical model/dataset coverage")

    results = []
    for key in sorted(keys):
        rows = [index[key] for index in indexed]
        result = deepcopy(rows[0])
        evaluation = result["evaluation"]
        evaluation_std = {}
        for count_key in ("num_samples_level1_level2", "num_samples_level3"):
            values = [row["evaluation"][count_key] for row in rows]
            if len(set(values)) != 1:
                raise ValueError(f"Inconsistent {count_key} for {key}: {values}")

        for group in LEVEL_GROUPS:
            group_rows = [row["evaluation"][group] for row in rows]
            common_names = set.intersection(*(set(group_row) for group_row in group_rows))
            means, stds = {}, {}
            for name in sorted(common_names):
                values = [group_row[name] for group_row in group_rows]
                if all(isinstance(value, (int, float, bool)) for value in values):
                    means[name], stds[name] = _mean_std(values)
                elif len(set(map(str, values))) == 1:
                    means[name] = values[0]
            evaluation[group] = means
            if group == "level3_explanation_reliability":
                evaluation_std[group] = stds
            elif group == "scores":
                evaluation_std[group] = {
                    name: value for name, value in stds.items()
                    if name == "rs" or name == "tds"
                    or name.startswith("rs_") or name.startswith("tds_")
                }
        evaluation["settings"] = deepcopy(rows[0]["evaluation"]["settings"])
        evaluation["settings"]["seeds"] = list(seeds)
        evaluation["settings"].pop("seed", None)
        result["evaluation_std"] = evaluation_std
        results.append(result)

    metadata = {
        "aggregation_type": "mean_and_sample_std_across_seeds",
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "framework_settings": deepcopy(payloads[0]["metadata"]["framework_settings"]),
        "source_failures": [payload["metadata"].get("failures", []) for payload in payloads],
    }
    metadata["framework_settings"]["seeds"] = list(seeds)
    metadata["framework_settings"].pop("seed", None)
    return {"metadata": metadata, "results": results}


def _metric(row, name):
    evaluation = row["evaluation"]
    if name in evaluation["scores"]:
        return evaluation["scores"][name], row["evaluation_std"].get("scores", {}).get(name)
    for group in LEVEL_GROUPS[:-1]:
        if name in evaluation[group]:
            return evaluation[group][name], row["evaluation_std"].get(group, {}).get(name)
    raise KeyError(name)


def write_csv(payload, path):
    identity = (
        "dataset", "family", "model", "checkpoint", "checkpoint_epoch",
        "level1_level2_samples", "level3_samples",
    )
    columns = [*identity]
    for metric in CSV_METRICS:
        columns.append(metric)
        if metric in STOCHASTIC_METRICS:
            columns.append(f"{metric}_std")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in payload["results"]:
            evaluation = row["evaluation"]
            output = {
                "dataset": row["dataset"],
                "family": row["family"],
                "model": row["model"],
                "checkpoint": row["checkpoint"],
                "checkpoint_epoch": row.get("checkpoint_epoch"),
                "level1_level2_samples": evaluation["num_samples_level1_level2"],
                "level3_samples": evaluation["num_samples_level3"],
            }
            for metric in CSV_METRICS:
                mean, std = _metric(row, metric)
                if not math.isfinite(mean) or (
                    metric in STOCHASTIC_METRICS
                    and (std is None or not math.isfinite(std))
                ):
                    raise ValueError(f"Non-finite aggregate for {row['dataset']}/{row['model']}/{metric}")
                output[metric] = mean
                if metric in STOCHASTIC_METRICS:
                    output[f"{metric}_std"] = std
            writer.writerow(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-json", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if len(args.seed_json) != len(args.seeds):
        parser.error("--seed-json and --seeds must have the same length")
    payloads = [json.loads(path.read_text()) for path in args.seed_json]
    payload = aggregate_payloads(payloads, args.seeds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    write_csv(payload, args.output_csv)
    print(f"Wrote {args.output_json} and {args.output_csv}")


if __name__ == "__main__":
    main()
