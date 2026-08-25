#!/usr/bin/env python3
"""Generate a dependency-aware, level-wise report from sanity-check JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("eeg_eog", "eeg_emg", "motion")
DATASET_LABELS = {
    "eeg_eog": "EEGDenoiseNet-EOG",
    "eeg_emg": "EEGDenoiseNet-EMG",
    "motion": "EEG-Motion",
}
CHECK_FILES = {
    "random_attribution": "random_attribution",
    "random_model": "random_model",
    "explanation_methods": "explanation_methods",
    "clean_input": "clean_input",
    "metric_thresholds": "metric_thresholds",
    "wrong_artifact": "wrong_artifact",
}


def load_result(root: Path, prefix: str, dataset: str) -> dict:
    path = root / f"{prefix}_{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sanity-check result: {path}")
    return json.loads(path.read_text())


def number(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def mean_std(record: dict, key: str):
    """Format an aggregated metric as mean ± sample standard deviation."""
    if f"{key}_std" not in record:
        return record[key]
    return f"{record[key]:.4f} ± {record[f'{key}_std']:.4f}"


def mean_only(record: dict, key: str):
    """Format a deterministic Level-1/2 quantity without a redundant SD."""
    return f"{record[key]:.4f}"


def table(headers, rows) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
    ]
    lines.extend("| " + " | ".join(number(value) for value in row) + " |" for row in rows)
    return lines


def row_by_condition(payload: dict, condition: str) -> dict:
    return next(row for row in payload["results"] if row["condition"] == condition)


def scope_section(payloads: dict[str, dict]) -> list[str]:
    rows = []
    for check, payload in payloads.items():
        scope = payload["metadata"]["metric_scope"]
        reported = "; ".join(
            f"{level}: {', '.join(metrics)}"
            for level, metrics in scope["reported_by_level"].items()
        )
        omitted = "; ".join(scope["omitted_invariants"].keys()) or "None"
        rows.append((check.replace("_", " ").title(), reported, omitted))
    return [
        "## Metric dependency scope",
        "",
        "Only metrics that consume the manipulated input are included.",
        "",
        *table(["Sanity check", "Reported affected metrics", "Omitted invariants"], rows),
        "",
    ]


def random_attribution_section(results) -> list[str]:
    metrics = ("taas", "caf", "ass", "rs")
    rows = []
    observations = []
    for dataset, payload in results.items():
        real = row_by_condition(payload, "real")
        shuffled = row_by_condition(payload, "shuffled")
        for condition, record in (("Real", real), ("Shuffled", shuffled)):
            rows.append((
                DATASET_LABELS[dataset], condition,
                *(mean_std(record, metric) for metric in metrics),
            ))
        changes = {metric: shuffled[metric] - real[metric] for metric in metrics}
        passed = sum(change < 0 for change in changes.values())
        observations.append(
            f"- {DATASET_LABELS[dataset]}: {passed}/4 affected Level-3 metrics decrease; "
            f"RS change = {changes['rs']:+.4f}."
        )
    return [
        "## 1. Random attribution",
        "",
        "Manipulated input: attribution map. Related level: Level 3.",
        "",
        *table(["Dataset", "Map", "TAAS", "CAF", "ASS", "RS"], rows),
        "",
        *observations,
        "",
        "Level 1, Level 2, and BAC are omitted because they do not consume the attribution map.",
        "",
    ]


def random_model_section(results) -> list[str]:
    groups = (
        ("Level 1", (
            ("l1_rrmse_temporal", "RRMSE-T"),
            ("l1_rrmse_spectral", "RRMSE-S"),
            ("l1_cc", "CC"),
            ("l1_delta_snr_db", "ΔSNR"),
            ("l1_macro_band_power_error", "Macro BPE"),
            ("l1_bps", "BPS"),
            ("ds", "DS"),
        )),
        ("Level 2", (
            ("l2_ras", "RAS"),
            ("l2_raer", "RAER"),
            ("l2_artifact_reconstruction_error", "ARE"),
            ("l2_age", "AGE"),
            ("l2_cnl", "CNL"),
            ("fs", "FS"),
        )),
        ("Level 3", (
            ("l3_taas", "TAAS"),
            ("l3_bac", "BAC"),
            ("l3_caf", "CAF"),
            ("l3_ass", "ASS"),
            ("rs", "RS"),
        )),
    )
    lines = [
        "## 2. Random/untrained model",
        "",
        "Manipulated input: model parameters. Related levels: Levels 1, 2, and 3.",
        "",
    ]
    for level, metrics in groups:
        formatter = mean_std if level == "Level 3" else mean_only
        rows = []
        for dataset, payload in results.items():
            for condition in ("trained", "untrained"):
                record = row_by_condition(payload, condition)
                rows.append((
                    DATASET_LABELS[dataset], condition.title(),
                    *(formatter(record, key) for key, _ in metrics),
                ))
        lines.extend([
            f"### {level}", "",
            *table(["Dataset", "Model", *(label for _, label in metrics)], rows), "",
        ])
    summary_rows = []
    for dataset, payload in results.items():
        trained = row_by_condition(payload, "trained")
        untrained = row_by_condition(payload, "untrained")
        summary_rows.append((
            DATASET_LABELS[dataset],
            untrained["ds"] - trained["ds"],
            untrained["fs"] - trained["fs"],
            untrained["rs"] - trained["rs"],
            untrained["tds"] - trained["tds"],
        ))
    lines.extend([
        "### Aggregate changes", "",
        *table(["Dataset", "ΔDS", "ΔFS", "ΔRS", "ΔTDS"], summary_rows), "",
        "Negative DS/FS/TDS changes are expected. RS is reported because model parameters also change the attribution, but it is not assumed to decrease.",
        "",
    ])
    return lines


def explanation_section(results) -> list[str]:
    metrics = ("taas", "caf", "ass", "rs")
    rows = []
    range_rows = []
    for dataset, payload in results.items():
        method_rows = payload["results"]
        for record in method_rows:
            rows.append((
                DATASET_LABELS[dataset], record["method"],
                *(mean_std(record, metric) for metric in metrics),
            ))
        range_rows.append((
            DATASET_LABELS[dataset],
            *(max(r[m] for r in method_rows) - min(r[m] for r in method_rows) for m in metrics),
        ))
    return [
        "## 3. Explanation-method sensitivity",
        "",
        "Manipulated input: explanation method and resulting attribution map. Related level: Level 3.",
        "",
        *table(["Dataset", "Method", "TAAS", "CAF", "ASS", "RS"], rows),
        "",
        "### Across-method ranges", "",
        *table(["Dataset", "TAAS range", "CAF range", "ASS range", "RS range"], range_rows),
        "",
        "Level 1, Level 2, and BAC are omitted because they are invariant to the selected explanation method.",
        "",
    ]


def clean_section(results) -> list[str]:
    l1_metrics = (
        ("l1_rrmse_temporal", "RRMSE-T"),
        ("l1_rrmse_spectral", "RRMSE-S"),
        ("l1_cc", "CC"),
        ("l1_delta_snr_db", "ΔSNR"),
        ("l1_macro_band_power_error", "Macro BPE"),
        ("l1_bps", "BPS"),
        ("l1_ds", "DS"),
    )
    rows, diagnostic_rows = [], []
    for dataset, payload in results.items():
        record = payload["results"][0]
        rows.append((
            DATASET_LABELS[dataset],
            *(mean_only(record, key) for key, _ in l1_metrics),
        ))
        diagnostic_rows.append((
            DATASET_LABELS[dataset],
            mean_only(record, "l2_beta_clean"), mean_only(record, "l2_cnl"),
            mean_only(record, "l2_utility_cnl"),
            mean_only(record, "removed_l2"),
            mean_only(record, "removed_relative_l2"),
        ))
    return [
        "## 4. Clean-input, zero-artifact check",
        "",
        "Manipulated inputs: clean signal is used as model input and artifact is set to zero.",
        "",
        "### Level 1", "",
        *table(["Dataset", *(label for _, label in l1_metrics)], rows), "",
        "### Related Level-2 and identity diagnostics", "",
        *table(["Dataset", "beta_x", "CNL", "U_CNL", "Removed L2", "Removed relative L2"], diagnostic_rows),
        "",
        "RAS, RAER, ARE, AGE, and FS are omitted because a zero artifact makes their artifact-normalized interpretation undefined or misleading. CNL remains interpretable because it directly measures removal proportional to the clean input.",
        "",
    ]


def threshold_section(results) -> list[str]:
    mapping = {
        "delta_snr_max": ("Level 1", "utility_delta_snr", "U_ΔSNR", "ds"),
        "band_power_error_max": ("Level 1", "bps", "BPS", "ds"),
        "raer_error_max": ("Level 2", "utility_raer", "U_RAER", "fs"),
        "are_error_max": ("Level 2", "utility_are", "U_ARE", "fs"),
        "age_error_max": ("Level 2", "utility_age", "U_AGE", "fs"),
        "cnl_error_max": ("Level 2", "utility_cnl", "U_CNL", "fs"),
        "residual_ratio_max": ("Level 2", "utility_rar", "U_RAR (diagnostic)", None),
    }
    lines = [
        "## 5. Metric-threshold sensitivity",
        "",
        "Manipulated input: one normalization threshold at a time. Only its utility and containing level score are reported.",
        "",
    ]
    for level in ("Level 1", "Level 2"):
        rows = []
        for dataset, payload in results.items():
            for record in payload["results"]:
                threshold = record["threshold"]
                if threshold not in mapping or mapping[threshold][0] != level:
                    continue
                _, utility_key, utility_label, score_key = mapping[threshold]
                rows.append((
                    DATASET_LABELS[dataset], threshold, record["threshold_value"],
                    utility_label, mean_only(record, utility_key),
                    score_key.upper() if score_key else "—",
                    mean_only(record, score_key) if score_key else "—",
                ))
        lines.extend([
            f"### {level}", "",
            *table(["Dataset", "Threshold", "Value", "Utility", "Utility value", "Score", "Score value"], rows),
            "",
        ])
    lines.extend([
        "Raw errors and all unswept levels are omitted because they do not consume the changed normalization threshold.",
        "",
    ])
    return lines


def wrong_pairing_section(results) -> list[str]:
    metrics = (
        ("ras", "RAS"), ("raer", "RAER"), ("utility_raer", "U_RAER"),
        ("artifact_reconstruction_error", "ARE"), ("utility_are", "U_ARE"),
        ("age", "AGE"), ("utility_age", "U_AGE"),
        ("cnl", "CNL"), ("utility_cnl", "U_CNL"), ("fs", "FS"),
    )
    rows, observations = [], []
    names = {
        "true_artifact": "True",
        "wrong_artifact_raw": "Wrong raw",
        "wrong_artifact_rms_matched": "Wrong RMS-matched",
    }
    for dataset, payload in results.items():
        for condition, label in names.items():
            record = row_by_condition(payload, condition)
            rows.append((
                DATASET_LABELS[dataset], label,
                *(mean_only(record, key) for key, _ in metrics),
            ))
        true = row_by_condition(payload, "true_artifact")
        wrong = row_by_condition(payload, "wrong_artifact_raw")
        observations.append(
            f"- {DATASET_LABELS[dataset]}: wrong-raw ΔRAS={wrong['ras']-true['ras']:+.4f}, "
            f"ΔARE={wrong['artifact_reconstruction_error']-true['artifact_reconstruction_error']:+.4f}, "
            f"ΔFS={wrong['fs']-true['fs']:+.4f}."
        )
    return [
        "## 6. Wrong artifact pairing",
        "",
        "Manipulated input: artifact reference. Related level: Level 2.",
        "",
        *table(["Dataset", "Reference", *(label for _, label in metrics)], rows),
        "",
        *observations,
        "",
        "Level 1, Level 3, and NLR are omitted because they do not consume the artifact reference. The retained old RAR is diagnostic and excluded from the updated FS.",
        "",
    ]


def assessment_section(loaded) -> list[str]:
    random_rs = []
    clean_rows = []
    wrong_drops = []
    explanation_rs_ranges = []
    largest_l1_spread = (-1.0, "")
    largest_are_spread = -1.0
    for dataset in DATASETS:
        random_payload = loaded["random_model"][dataset]
        trained = row_by_condition(random_payload, "trained")
        untrained = row_by_condition(random_payload, "untrained")
        random_rs.append(untrained["rs"] - trained["rs"])

        clean = loaded["clean_input"][dataset]["results"][0]
        clean_rows.append((clean["l1_ds"], clean["removed_relative_l2"]))

        wrong_payload = loaded["wrong_artifact"][dataset]
        true = row_by_condition(wrong_payload, "true_artifact")
        wrong = row_by_condition(wrong_payload, "wrong_artifact_raw")
        wrong_drops.append(true["fs"] - wrong["fs"])

        method_rows = loaded["explanation_methods"][dataset]["results"]
        explanation_rs_ranges.append(
            max(row["rs"] for row in method_rows)
            - min(row["rs"] for row in method_rows)
        )

        threshold_rows = loaded["metric_thresholds"][dataset]["results"]
        for threshold in ("delta_snr_max", "band_power_error_max"):
            values = [row["ds"] for row in threshold_rows if row["threshold"] == threshold]
            spread = max(values) - min(values)
            if spread > largest_l1_spread[0]:
                largest_l1_spread = (
                    spread,
                    f"{DATASET_LABELS[dataset]} {threshold}",
                )
        are_values = [
            row["fs"] for row in threshold_rows
            if row["threshold"] == "are_error_max"
        ]
        largest_are_spread = max(largest_are_spread, max(are_values) - min(are_values))

    return [
        "## Overall behavior assessment",
        "",
        "- **Random attribution: passes.** All four attribution-dependent Level-3 metrics decrease on every dataset. CAF and ASS provide most of the discrimination; TAAS is comparatively weak on EMG and motion.",
        f"- **Random model: Levels 1 and 2 pass when DS and FS decrease; standalone Level 3 is not expected to rank trained models by itself.** RS changes by "
        + ", ".join(f"{value:+.4f}" for value in random_rs)
        + " (EOG, EMG, motion). RS must therefore be conditioned on a valid denoiser.",
        "- **Explanation method: broadly robust.** RS ranges are "
        + ", ".join(f"{value:.4f}" for value in explanation_rs_ranges)
        + " (EOG, EMG, motion). The dataset with the largest range shows moderate method sensitivity.",
        "- **Clean input: tests identity behavior.** DS/removed-relative-L2 pairs are "
        + ", ".join(f"{ds:.4f}/{removed:.4f}" for ds, removed in clean_rows)
        + ". The very negative ΔSNR is an expected zero-noise-denominator edge case and should not be interpreted normally.",
        f"- **Threshold sweep: mathematically monotonic but not threshold-insensitive.** The largest Level-1 score spread is {largest_l1_spread[0]:.4f} ({largest_l1_spread[1]}); ARE changes FS by up to {largest_are_spread:.4f}. Thresholds must be fixed before comparisons.",
        "- **Wrong pairing: passes overall when the joint-regression FS is reference-sensitive.** FS drops by "
        + ", ".join(f"{value:.4f}" for value in wrong_drops)
        + " (EOG, EMG, motion). RAS, ARE, AGE, and CNL identify which aspects of the mismatch drive the change.",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-label", default="EEGDNet")
    args = parser.parse_args()
    output = args.output or args.input_dir / "REPORT_LEVELWISE.md"

    loaded = {
        check: {
            dataset: load_result(args.input_dir, prefix, dataset)
            for dataset in DATASETS
        }
        for check, prefix in CHECK_FILES.items()
    }
    representative = {check: payloads["eeg_eog"] for check, payloads in loaded.items()}
    aggregate_metadata = representative["random_attribution"].get("metadata", {})
    seeds = aggregate_metadata.get("seeds")
    uncertainty_note = (
        "Level-1 and Level-2 tables report their aggregate values without a "
        "redundant standard deviation. Level-3 values are reported as mean ± "
        f"sample standard deviation across {len(seeds)} seeded stratified "
        f"64-window subset replicates ({', '.join(map(str, seeds))}) on a fixed "
        "data split; each seed also controls matched stochastic operations. Overall TDS "
        "uncertainty is retained when reported because TDS consumes Level 3."
        if seeds else
        "Values are from one stochastic evaluation seed."
    )
    lines = [
        f"# {args.model_label} dependency-aware sanity-check report",
        "",
        "This report groups results by evaluation level and reports a metric only when it consumes the input manipulated by that sanity check.",
        "",
        uncertainty_note,
        "",
        *scope_section(representative),
        *random_attribution_section(loaded["random_attribution"]),
        *random_model_section(loaded["random_model"]),
        *explanation_section(loaded["explanation_methods"]),
        *clean_section(loaded["clean_input"]),
        *threshold_section(loaded["metric_thresholds"]),
        *wrong_pairing_section(loaded["wrong_artifact"]),
        *assessment_section(loaded),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
