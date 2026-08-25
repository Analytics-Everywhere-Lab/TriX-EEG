#!/usr/bin/env python3
"""Run TriX-EEG on EEGDenoiseNet and Motion EEG checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import math
import os
import traceback
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_denoise.motion_data import make_motion_dataset, make_motion_splits
from eegdenoisenet_dataset import create_benchmark_datasets
from load_dataset import (
    MOTION_EEG_DATA_FOLDER,
    MOTION_EEG_SIGNAL_LENGTH,
    MOTION_EEG_SOURCE_FS,
    MOTION_EEG_TARGET_FS,
)
from trix_eeg.evaluator import (
    TriXConfig,
    evaluate_trix_level1_level2,
    evaluate_trix_model,
    evaluate_trix_model_multiseed,
)
from trix_eeg.aggregation import AGGREGATION_CHOICES, LEGACY_AGGREGATION


BENCHMARK_NAMES = (
    "simplecnn", "fcnn", "complexcnn", "rnnlstm",
    "deepseparator", "det", "lrr_unet", "gan",
)
AVAILABLE_DENOISERS = (*BENCHMARK_NAMES, "braanet")


def _load_level3_reuse(path, evaluation_seeds):
    if path is None:
        return None, None
    with open(path) as stream:
        payload = json.load(stream)
    source_seeds = payload.get("metadata", {}).get("evaluation_seeds")
    if list(source_seeds or ()) != list(evaluation_seeds or ()):
        raise ValueError(
            "--reuse-level3-json must use the same evaluation seeds: "
            f"source={source_seeds}, requested={evaluation_seeds}"
        )
    index = {
        (str(row["dataset"]), str(row["family"]), str(row["model"])): row
        for row in payload.get("results", ())
    }
    if not index:
        raise ValueError(f"No reusable evaluations found in {path}")
    return index, payload.get("metadata", {})


def _reuse_level3_evaluation(
    source_row, level1, level2, total_samples, config, checkpoint_path
):
    if os.path.normpath(str(source_row["checkpoint"])) != os.path.normpath(checkpoint_path):
        raise ValueError(
            "Level-3 reuse checkpoint mismatch: "
            f"source={source_row['checkpoint']}, current={checkpoint_path}"
        )
    source = source_row["evaluation"]
    source_scores = source["scores"]
    if not math.isclose(level1["ds"], float(source_scores["ds"]), abs_tol=1e-6):
        raise ValueError("Recomputed DS does not match the Level-3 source run")
    source_fs = float(source_scores["fs"])
    if source_fs <= 0.0:
        raise ValueError("Cannot rescale TDS from a non-positive source FS")

    source_tds = float(source_scores["tds"])
    rs = float(source_scores["rs"])
    tds_by_name = {
        "tds": source_tds * (float(level2["fs"]) / source_fs) ** (1.0 / 3.0),
        "tds_error_projection_v2": source_tds * (
            float(level2["fs_error_projection_v2"]) / source_fs
        ) ** (1.0 / 3.0),
        "tds_prediction_rar_legacy": source_tds * (
            float(level2["fs_prediction_rar_legacy"]) / source_fs
        ) ** (1.0 / 3.0),
    }
    scores = {
        "aggregation": config.aggregation,
        "ds": level1["ds"],
        "fs": level2["fs"],
        "rs": rs,
        "tds": tds_by_name["tds"],
        "ds_legacy_arithmetic": level1["ds_legacy_arithmetic"],
        "fs_legacy_arithmetic": level2["fs_legacy_arithmetic"],
        "rs_legacy_arithmetic": rs,
        "tds_legacy_arithmetic": tds_by_name["tds"],
        "fs_error_projection_v2": level2["fs_error_projection_v2"],
        "tds_error_projection_v2": tds_by_name["tds_error_projection_v2"],
        "fs_prediction_rar_legacy": level2["fs_prediction_rar_legacy"],
        "tds_prediction_rar_legacy": tds_by_name["tds_prediction_rar_legacy"],
    }
    settings = dict(config.__dict__)
    settings["evaluation_seeds"] = list(
        source.get("settings", {}).get("evaluation_seeds", ())
    )
    settings["level3_reused"] = True
    evaluation = {
        "num_samples_level1_level2": total_samples,
        "num_samples_level3": source["num_samples_level3"],
        "level1_signal_validity": level1,
        "level2_removal_faithfulness": level2,
        "level3_explanation_reliability": deepcopy(
            source["level3_explanation_reliability"]
        ),
        "scores": scores,
        "settings": settings,
    }
    for name in ("level3_subset_indices", "level3_subset_indices_by_seed"):
        if name in source:
            evaluation[name] = deepcopy(source[name])

    source_std = source_row.get("evaluation_std", {})
    source_score_std = source_std.get("scores", {})
    rs_std = float(source_score_std.get("rs", 0.0))
    tds_std = float(source_score_std.get("tds", 0.0))
    tds_std_by_name = {
        name: tds_std * value / source_tds if source_tds > 0 else 0.0
        for name, value in tds_by_name.items()
    }
    evaluation_std = {
        "level3_explanation_reliability": deepcopy(
            source_std.get("level3_explanation_reliability", {})
        ),
        "scores": {
            "rs": rs_std,
            "tds": tds_std_by_name["tds"],
            "rs_legacy_arithmetic": rs_std,
            "tds_legacy_arithmetic": tds_std_by_name["tds"],
            "tds_error_projection_v2": tds_std_by_name["tds_error_projection_v2"],
            "tds_prediction_rar_legacy": tds_std_by_name["tds_prediction_rar_legacy"],
        },
    }
    return evaluation, evaluation_std


def _loader(dataset, args, device):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def _eeg_dataset(noise_type, args):
    eeg = np.load(args.eeg_npy)
    eog = np.load(args.eog_npy) if noise_type == "EOG" else None
    emg = np.load(args.emg_npy) if noise_type == "EMG" else None
    return create_benchmark_datasets(
        eeg_array=eeg,
        eog_array=eog,
        emg_array=emg,
        noise_type=noise_type,
        train_snr_levels=args.snr_levels,
        val_snr_levels=args.snr_levels,
        test_snr_levels=args.snr_levels,
        seed=args.seed,
    )[2]


def _motion_dataset(args):
    motion_args = SimpleNamespace(
        data_folder=args.motion_data_folder,
        source_fs=args.motion_source_fs,
        target_fs=args.fs,
        signal_length=args.signal_length,
        stride=args.motion_stride,
        no_normalize=args.no_motion_normalize,
        train_ratio=args.motion_train_ratio,
        val_ratio=args.motion_val_ratio,
        seed=args.seed,
    )
    _, _, test_ids = make_motion_splits(motion_args)
    return make_motion_dataset(motion_args, test_ids), test_ids


def _experiments(dataset_name, args):
    selected = AVAILABLE_DENOISERS if args.denoiser == "all" else (args.denoiser,)
    if dataset_name in {"eeg_eog", "eeg_emg"}:
        suffix = dataset_name.removeprefix("eeg_")
        root = args.benchmark_checkpoint_root
        for name in BENCHMARK_NAMES:
            if name not in selected:
                continue
            path = os.path.join(root, f"{name}_{suffix}", "best.pt")
            yield "benchmark", name, path
        if "braanet" in selected:
            braa_subdir = suffix.upper()
            yield "braanet", "braanet", os.path.join(
                args.braanet_checkpoint_root, braa_subdir, "best.pt"
            )
    else:
        for name in BENCHMARK_NAMES:
            if name not in selected:
                continue
            yield "benchmark", name, os.path.join(
                args.motion_checkpoint_root, "benchmarks", name, "best.pt"
            )
        if "braanet" in selected:
            yield "braanet", "braanet", os.path.join(
                args.motion_checkpoint_root, "braanet", "best.pt"
            )


def _load_model(family, name, checkpoint_path, device):
    # Model implementations are optional for users evaluating an external model.
    # Delay this import so that parsing the protocol CLI has no baseline dependency.
    from eval_motion_dataset import build_benchmark, build_braanet, load_checkpoint

    checkpoint = load_checkpoint(checkpoint_path, device)
    if family == "braanet":
        model, _, config = build_braanet(checkpoint, device)
    else:
        model, _, config = build_benchmark(checkpoint, name, device)
    model.eval()
    return model, checkpoint, config


def _write_outputs(results, output_json, output_csv, metadata):
    output_dir = os.path.dirname(output_json) or "."
    os.makedirs(output_dir, exist_ok=True)
    csv_output_dir = os.path.dirname(output_csv) or "."
    os.makedirs(csv_output_dir, exist_ok=True)
    with open(output_json, "w") as stream:
        json.dump({"metadata": metadata, "results": results}, stream, indent=2)

    identity_columns = [
        "dataset", "family", "model", "checkpoint", "checkpoint_epoch",
        "level1_level2_samples", "level3_samples", "aggregation",
    ]
    score_columns = [
        "ds", "fs", "rs", "tds",
        "ds_legacy_arithmetic", "fs_legacy_arithmetic",
        "rs_legacy_arithmetic", "tds_legacy_arithmetic",
        "fs_prediction_rar_legacy", "tds_prediction_rar_legacy",
        "fs_error_projection_v2", "tds_error_projection_v2",
    ]
    metric_columns = [
        "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db", "bps",
        "ras", "raer", "artifact_reconstruction_error",
        "beta_artifact", "age", "beta_clean", "cnl",
        "regression_residual_ratio", "regression_condition_number",
        "nlr", "rar",
        "rar_prediction_legacy",
        "rar_error_projection_v2",
        "taas", "bac", "caf", "ass",
    ]
    stochastic_columns = [
        "rs", "tds", "rs_legacy_arithmetic", "tds_legacy_arithmetic",
        "tds_prediction_rar_legacy",
        "tds_error_projection_v2",
        "taas", "bac", "caf", "ass",
    ]
    columns = identity_columns + score_columns + metric_columns
    columns += [f"{name}_std" for name in stochastic_columns]
    with open(output_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            evaluation = result["evaluation"]
            level1 = evaluation["level1_signal_validity"]
            level2 = evaluation["level2_removal_faithfulness"]
            level3 = evaluation["level3_explanation_reliability"]
            evaluation_std = result.get("evaluation_std", {})
            level3_std = evaluation_std.get("level3_explanation_reliability", {})
            score_std = evaluation_std.get("scores", {})
            values = {
                "dataset": result["dataset"],
                "family": result["family"],
                "model": result["model"],
                "checkpoint": result["checkpoint"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "level1_level2_samples": evaluation["num_samples_level1_level2"],
                "level3_samples": evaluation["num_samples_level3"],
                **evaluation["scores"],
                **{key: level1.get(key) for key in (
                    "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db", "bps"
                )},
                **{key: level2.get(key) for key in (
                    "ras", "raer", "artifact_reconstruction_error",
                    "beta_artifact", "age", "beta_clean", "cnl",
                    "regression_residual_ratio", "regression_condition_number",
                    "nlr", "rar",
                    "rar_prediction_legacy",
                    "rar_error_projection_v2",
                )},
                **{key: level3.get(key) for key in ("taas", "bac", "caf", "ass")},
            }
            std_groups = (score_std, level3_std)
            for name in stochastic_columns:
                values[f"{name}_std"] = next(
                    (group[name] for group in std_groups if name in group), 0.0
                )
            writer.writerow(values)


def print_evaluation_profile(evaluation):
    """Print the three metric profiles and final score profile consistently."""
    profiles = (
        ("Level 1", "DS", evaluation["level1_signal_validity"], (
            "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db", "bps",
        )),
        ("Level 2", "FS", evaluation["level2_removal_faithfulness"], (
            "ras", "raer", "artifact_reconstruction_error",
            "age", "cnl", "rar_prediction_legacy",
            "rar_error_projection_v2",
        )),
        ("Level 3", "RS", evaluation["level3_explanation_reliability"], (
            "taas", "bac", "caf", "ass",
        )),
    )
    scores = evaluation["scores"]
    for label, score_label, metrics, names in profiles:
        values = " ".join(f"{name}={metrics[name]:.4f}" for name in names)
        print(f"  {label} ({score_label}={scores[score_label.lower()]:.4f}): {values}")
    print(
        "  Final: "
        f"DS={scores['ds']:.4f} FS={scores['fs']:.4f} "
        f"RS={scores['rs']:.4f} TDS={scores['tds']:.4f}"
    )


def main(args):
    validate_cli_args(args)
    if args.taas_n_fft <= 1 or args.taas_n_fft > args.signal_length:
        raise ValueError("--taas-n-fft must be in [2, signal length]")
    if args.taas_hop_length <= 0:
        raise ValueError("--taas-hop-length must be positive")
    if not 0.0 <= args.taas_frequency_low < args.taas_frequency_high <= args.fs / 2.0:
        raise ValueError("TAAS frequencies must satisfy 0 <= low < high <= fs/2")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    config = TriXConfig(
        aggregation=args.aggregation,
        fs=args.fs,
        delta_snr_max=args.delta_snr_max,
        band_power_error_max=args.band_power_error_max,
        raer_error_max=args.raer_error_max,
        are_error_max=args.are_error_max,
        age_error_max=args.age_error_max,
        cnl_error_max=args.cnl_error_max,
        regression_ridge_fraction=args.regression_ridge_fraction,
        residual_ratio_max=args.residual_ratio_max,
        taas_frequency_range=(args.taas_frequency_low, args.taas_frequency_high),
        taas_n_fft=args.taas_n_fft,
        taas_hop_length=args.taas_hop_length,
        deletion_fractions=tuple(args.deletion_fractions),
        random_deletion_repeats=args.random_deletion_repeats,
        stability_noise_fraction=args.stability_noise_fraction,
        level3_max_samples=args.level3_max_samples,
        level3_batch_size=args.level3_batch_size,
        seed=args.seed,
        weights=tuple(args.weights),
    )
    reuse_index, reuse_metadata = _load_level3_reuse(
        args.reuse_level3_json, args.evaluation_seeds
    )
    results, failures = [], []
    for dataset_name in args.datasets:
        print(f"\n========== {dataset_name} ==========")
        if dataset_name == "eeg_eog":
            dataset, extra_metadata = _eeg_dataset("EOG", args), {}
        elif dataset_name == "eeg_emg":
            dataset, extra_metadata = _eeg_dataset("EMG", args), {}
        elif dataset_name == "motion":
            dataset, test_ids = _motion_dataset(args)
            extra_metadata = {"test_subjects": test_ids}
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        loader = _loader(dataset, args, device)
        for family, name, default_checkpoint_path in _experiments(dataset_name, args):
            checkpoint_path = args.checkpoint or default_checkpoint_path
            if not os.path.isfile(checkpoint_path):
                message = f"Missing checkpoint: {checkpoint_path}"
                if args.skip_failed:
                    print(f"[skip] {message}")
                    failures.append({"dataset": dataset_name, "model": name, "error": message})
                    continue
                raise FileNotFoundError(message)
            print(f"[{dataset_name}] {family}/{name}: {checkpoint_path}")
            try:
                model, checkpoint, model_config = _load_model(
                    family, name, checkpoint_path, device
                )
                if int(model_config.signal_length) != args.signal_length:
                    raise ValueError(
                        f"Checkpoint signal length {model_config.signal_length} != "
                        f"evaluation length {args.signal_length}"
                    )
                if reuse_index is not None:
                    source_key = (dataset_name, family, name)
                    if source_key not in reuse_index:
                        raise ValueError(f"Missing reusable Level-3 result for {source_key}")
                    level1, level2, total_samples = evaluate_trix_level1_level2(
                        model, loader, device, config,
                        description=f"{dataset_name}/{name}",
                    )
                    evaluation, evaluation_std = _reuse_level3_evaluation(
                        reuse_index[source_key], level1, level2, total_samples,
                        config, checkpoint_path,
                    )
                elif args.evaluation_seeds:
                    evaluation, evaluation_std = evaluate_trix_model_multiseed(
                        model, loader, device, config, args.evaluation_seeds,
                        description=f"{dataset_name}/{name}",
                    )
                else:
                    evaluation = evaluate_trix_model(
                        model, loader, device, config,
                        description=f"{dataset_name}/{name}",
                    )
                    evaluation_std = None
                result = {
                    "dataset": dataset_name,
                    "family": family,
                    "model": name,
                    "checkpoint": checkpoint_path,
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    **extra_metadata,
                    "evaluation": evaluation,
                }
                if evaluation_std is not None:
                    result["evaluation_std"] = evaluation_std
                results.append(result)
                print_evaluation_profile(evaluation)
                _write_outputs(
                    results, args.output_json, args.output_csv,
                    {"device": str(device), "failures": failures},
                )
            except Exception as error:
                if not args.skip_failed:
                    raise
                print(f"[failed] {dataset_name}/{name}: {error}")
                traceback.print_exc()
                failures.append({
                    "dataset": dataset_name, "model": name,
                    "checkpoint": checkpoint_path, "error": str(error),
                })
            finally:
                if "model" in locals():
                    del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del loader, dataset
        gc.collect()

    if not results:
        raise RuntimeError("No evaluations completed")
    metadata = {
        "device": str(device),
        "datasets": args.datasets,
        "snr_levels": args.snr_levels,
        "framework_settings": config.__dict__,
        "evaluation_seeds": args.evaluation_seeds,
        "level3_reused_from": args.reuse_level3_json,
        "level3_source_framework_settings": (
            reuse_metadata.get("framework_settings") if reuse_metadata else None
        ),
        "uncertainty_scope": (
            "fixed dataset split/checkpoints; five seeded stratified Level-3 "
            "subsets with one matched stochastic-operation seed per subset"
            if args.evaluation_seeds else "single evaluation seed"
        ),
        "failures": failures,
    }
    _write_outputs(results, args.output_json, args.output_csv, metadata)
    print(f"\nSaved {len(results)} evaluations to {args.output_json} and {args.output_csv}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["eeg_eog", "eeg_emg", "motion"],
        choices=["eeg_eog", "eeg_emg", "motion"],
    )
    parser.add_argument(
        "--denoiser", default="all", choices=[*AVAILABLE_DENOISERS, "all"],
        help="Evaluate one of the nine built-in denoisers (default: all).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=("Override the inferred checkpoint path. This is intended for one "
              "denoiser and one dataset."),
    )
    parser.add_argument("--benchmark-checkpoint-root", default="runs/benchmark_compare")
    parser.add_argument("--braanet-checkpoint-root", default="runs/braanet")
    parser.add_argument("--motion-checkpoint-root", default="runs/motion_train")
    parser.add_argument("--eeg-npy", default="data/EEG_all_epochs.npy")
    parser.add_argument("--eog-npy", default="data/EOG_all_epochs.npy")
    parser.add_argument("--emg-npy", default="data/EMG_all_epochs.npy")
    parser.add_argument(
        "--snr-levels", nargs="+", type=float,
        default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
    )
    parser.add_argument("--motion-data-folder", default=MOTION_EEG_DATA_FOLDER)
    parser.add_argument("--motion-source-fs", type=float, default=MOTION_EEG_SOURCE_FS)
    parser.add_argument("--motion-stride", type=int, default=None)
    parser.add_argument("--motion-train-ratio", type=float, default=0.8)
    parser.add_argument("--motion-val-ratio", type=float, default=0.1)
    parser.add_argument("--no-motion-normalize", action="store_true")
    parser.add_argument("--fs", type=float, default=MOTION_EEG_TARGET_FS)
    parser.add_argument("--signal-length", type=int, default=MOTION_EEG_SIGNAL_LENGTH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-seeds", nargs="+", type=int, default=None,
        help=("Evaluate Level 3 on one seeded stratified subset per value, using "
              "the same value as its stochastic-operation seed; report mean/SD."),
    )
    parser.add_argument(
        "--reuse-level3-json", default=None,
        help=("Reuse Level-3 values, subset indices, and uncertainty from a "
              "compatible result JSON while recomputing complete Levels 1/2. "
              "Intended for changes confined to deterministic Level-2 metrics."),
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--delta-snr-max", type=float, default=20.0)
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_CHOICES,
        default=LEGACY_AGGREGATION,
        help="Level-score aggregation version; legacy remains the default.",
    )
    parser.add_argument("--band-power-error-max", type=float, default=1.0)
    parser.add_argument("--raer-error-max", type=float, default=1.0)
    parser.add_argument("--are-error-max", type=float, default=1.0)
    parser.add_argument("--age-error-max", type=float, default=1.0)
    parser.add_argument("--cnl-error-max", type=float, default=1.0)
    parser.add_argument("--regression-ridge-fraction", type=float, default=1e-6)
    parser.add_argument("--residual-ratio-max", type=float, default=0.1)
    parser.add_argument("--taas-frequency-low", type=float, default=0.5)
    parser.add_argument("--taas-frequency-high", type=float, default=45.0)
    parser.add_argument("--taas-n-fft", type=int, default=128)
    parser.add_argument("--taas-hop-length", type=int, default=32)
    parser.add_argument(
        "--deletion-fractions", nargs="+", type=float, default=[0.1, 0.2, 0.3]
    )
    parser.add_argument("--random-deletion-repeats", type=int, default=3)
    parser.add_argument("--stability-noise-fraction", type=float, default=0.01)
    parser.add_argument(
        "--level3-max-samples", type=int, default=64,
        help="Stratified explanation subset size per seed; <=0 evaluates all windows.",
    )
    parser.add_argument("--level3-batch-size", type=int, default=8)
    parser.add_argument("--weights", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--output-json", default="runs/trix_eeg/trix_eeg_results.json")
    parser.add_argument("--output-csv", default="runs/trix_eeg/trix_eeg_summary.csv")
    return parser


def validate_cli_args(args):
    if args.checkpoint and (args.denoiser == "all" or len(args.datasets) != 1):
        raise ValueError(
            "--checkpoint requires one --denoiser and exactly one --datasets value"
        )
    return args


if __name__ == "__main__":
    main(validate_cli_args(build_parser().parse_args()))
