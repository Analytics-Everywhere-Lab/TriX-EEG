#!/usr/bin/env python3
"""Run TriX-EEG on EEGDenoiseNet and Motion EEG checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import json
import os
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_denoise.motion_data import make_motion_dataset, make_motion_splits
from eegdenoisenet_dataset import create_benchmark_datasets
from eval_motion_dataset import build_benchmark, build_braanet, load_checkpoint
from load_dataset import (
    MOTION_EEG_DATA_FOLDER,
    MOTION_EEG_SIGNAL_LENGTH,
    MOTION_EEG_SOURCE_FS,
    MOTION_EEG_TARGET_FS,
)
from trix_eeg.evaluator import TriXConfig, evaluate_trix_model


BENCHMARK_NAMES = (
    "simplecnn", "fcnn", "complexcnn", "rnnlstm",
    "deepseparator", "det", "lrr_unet", "gan",
)


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
    if dataset_name in {"eeg_eog", "eeg_emg"}:
        suffix = dataset_name.removeprefix("eeg_")
        root = args.benchmark_checkpoint_root
        for name in BENCHMARK_NAMES:
            path = os.path.join(root, f"{name}_{suffix}", "best.pt")
            yield "benchmark", name, path
        braa_subdir = suffix.upper()
        yield "braanet", "braanet", os.path.join(
            args.braanet_checkpoint_root, braa_subdir, "best.pt"
        )
    else:
        for name in BENCHMARK_NAMES:
            yield "benchmark", name, os.path.join(
                args.motion_checkpoint_root, "benchmarks", name, "best.pt"
            )
        yield "braanet", "braanet", os.path.join(
            args.motion_checkpoint_root, "braanet", "best.pt"
        )


def _load_model(family, name, checkpoint_path, device):
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
    with open(output_json, "w") as stream:
        json.dump({"metadata": metadata, "results": results}, stream, indent=2)

    columns = [
        "dataset", "family", "model", "checkpoint", "checkpoint_epoch",
        "level1_level2_samples", "level3_samples", "ds", "fs", "rs", "tds",
        "rrmse_temporal", "rrmse_spectral", "cc", "delta_snr_db", "bps",
        "ras", "raer", "nlr", "rar", "taas", "bac", "caf", "ass",
    ]
    with open(output_csv, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            evaluation = result["evaluation"]
            level1 = evaluation["level1_signal_validity"]
            level2 = evaluation["level2_removal_faithfulness"]
            level3 = evaluation["level3_explanation_reliability"]
            writer.writerow({
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
                **{key: level2.get(key) for key in ("ras", "raer", "nlr", "rar")},
                **{key: level3.get(key) for key in ("taas", "bac", "caf", "ass")},
            })


def main(args):
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
        fs=args.fs,
        delta_snr_max=args.delta_snr_max,
        band_power_error_max=args.band_power_error_max,
        raer_error_max=args.raer_error_max,
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
        for family, name, checkpoint_path in _experiments(dataset_name, args):
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
                evaluation = evaluate_trix_model(
                    model, loader, device, config,
                    description=f"{dataset_name}/{name}",
                )
                result = {
                    "dataset": dataset_name,
                    "family": family,
                    "model": name,
                    "checkpoint": checkpoint_path,
                    "checkpoint_epoch": checkpoint.get("epoch"),
                    **extra_metadata,
                    "evaluation": evaluation,
                }
                results.append(result)
                scores = evaluation["scores"]
                print(
                    f"  DS={scores['ds']:.4f} FS={scores['fs']:.4f} "
                    f"RS={scores['rs']:.4f} TDS={scores['tds']:.4f}"
                )
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
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--delta-snr-max", type=float, default=20.0)
    parser.add_argument("--band-power-error-max", type=float, default=1.0)
    parser.add_argument("--raer-error-max", type=float, default=1.0)
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
        help="Uniform explanation subset size; <=0 evaluates every window.",
    )
    parser.add_argument("--level3-batch-size", type=int, default=8)
    parser.add_argument("--weights", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--output-json", default="runs/trix_eeg/trix_eeg_results.json")
    parser.add_argument("--output-csv", default="runs/trix_eeg/trix_eeg_summary.csv")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
