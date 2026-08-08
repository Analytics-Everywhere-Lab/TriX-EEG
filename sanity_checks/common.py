"""Shared command-line, loading, aggregation, and output helpers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import fields
from types import SimpleNamespace

import attr
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from eeg_denoise.motion_data import make_motion_dataset, make_motion_splits
from eegdenoisenet_dataset import create_benchmark_datasets
from eval_motion_dataset import build_benchmark, build_braanet, load_checkpoint
from configs.braanetv2_config import BRAANetV2Config
from load_dataset import (
    MOTION_EEG_DATA_FOLDER,
    MOTION_EEG_SIGNAL_LENGTH,
    MOTION_EEG_SOURCE_FS,
    MOTION_EEG_TARGET_FS,
)
from trix_eeg.common import as_bct
from trix_eeg.evaluator import TriXConfig
from model import BRAANetV2
from train_benchmark_EEGDenoiseNet import BenchmarkTrainConfig, build_model


DATASET_CHOICES = ("eeg_eog", "eeg_emg", "motion")
MODEL_CHOICES = (
    "simplecnn", "fcnn", "complexcnn", "rnnlstm",
    "deepseparator", "det", "lrr_unet", "gan",
)


class MetricAccumulator:
    """Finite sample-wise macro mean accumulator."""

    def __init__(self):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict[str, torch.Tensor]) -> None:
        for name, values in metrics.items():
            values = values.detach().float().reshape(-1)
            finite = torch.isfinite(values)
            if bool(finite.any()):
                self.sums[name] = self.sums.get(name, 0.0) + float(
                    values[finite].sum().cpu()
                )
                self.counts[name] = self.counts.get(name, 0) + int(finite.sum())

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / self.counts[name]
            for name in self.sums
            if self.counts[name]
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(cpu: bool) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="eeg_eog")
    parser.add_argument("--family", choices=("braanet", "benchmark"), default="braanet")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="det")
    parser.add_argument("--checkpoint", default=None)
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Uniform subset size; <=0 uses the entire test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")


def add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--delta-snr-max", type=float, default=20.0)
    parser.add_argument("--band-power-error-max", type=float, default=1.0)
    parser.add_argument("--raer-error-max", type=float, default=1.0)
    parser.add_argument("--are-error-max", type=float, default=1.0)
    parser.add_argument("--residual-ratio-max", type=float, default=0.1)
    parser.add_argument("--taas-frequency-low", type=float, default=0.5)
    parser.add_argument("--taas-frequency-high", type=float, default=45.0)
    parser.add_argument("--taas-n-fft", type=int, default=128)
    parser.add_argument("--taas-hop-length", type=int, default=32)
    parser.add_argument("--deletion-fractions", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3])
    parser.add_argument("--random-deletion-repeats", type=int, default=3)
    parser.add_argument("--stability-noise-fraction", type=float, default=0.01)


def make_config(args) -> TriXConfig:
    return TriXConfig(
        fs=args.fs,
        delta_snr_max=args.delta_snr_max,
        band_power_error_max=args.band_power_error_max,
        raer_error_max=args.raer_error_max,
        are_error_max=args.are_error_max,
        residual_ratio_max=args.residual_ratio_max,
        taas_frequency_range=(args.taas_frequency_low, args.taas_frequency_high),
        taas_n_fft=args.taas_n_fft,
        taas_hop_length=args.taas_hop_length,
        deletion_fractions=tuple(args.deletion_fractions),
        random_deletion_repeats=args.random_deletion_repeats,
        stability_noise_fraction=args.stability_noise_fraction,
        level3_max_samples=args.max_samples,
        level3_batch_size=args.batch_size,
        seed=args.seed,
    )


def build_dataset(args, dataset_name: str | None = None):
    name = args.dataset if dataset_name is None else dataset_name
    if name in {"eeg_eog", "eeg_emg"}:
        noise_type = "EOG" if name == "eeg_eog" else "EMG"
        eeg = np.load(args.eeg_npy)
        noise = np.load(args.eog_npy if noise_type == "EOG" else args.emg_npy)
        kwargs = {"eog_array": noise} if noise_type == "EOG" else {"emg_array": noise}
        return create_benchmark_datasets(
            eeg_array=eeg,
            noise_type=noise_type,
            train_snr_levels=args.snr_levels,
            val_snr_levels=args.snr_levels,
            test_snr_levels=args.snr_levels,
            seed=args.seed,
            **kwargs,
        )[2]
    if name == "motion":
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
        return make_motion_dataset(motion_args, test_ids)
    raise ValueError(f"Unknown dataset: {name}")


def uniform_subset(dataset, max_samples: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    indices = torch.linspace(0, len(dataset) - 1, max_samples).round().long().tolist()
    return Subset(dataset, indices)


def make_loader(dataset, args, *, max_samples: int | None = None) -> DataLoader:
    sample_limit = args.max_samples if max_samples is None else max_samples
    return DataLoader(
        uniform_subset(dataset, sample_limit),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.cpu and torch.cuda.is_available(),
    )


def signals_from_batch(batch, device):
    noisy = as_bct(batch["x"].to(device, non_blocking=True).float())
    clean = as_bct(batch["clean"].to(device, non_blocking=True).float())
    value = batch.get("artifact", batch.get("noise"))
    artifact = noisy - clean if value is None else as_bct(
        value.to(device, non_blocking=True).float()
    )
    return noisy, clean, artifact


def resolve_checkpoint(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    if args.dataset == "motion":
        suffix = "braanet/best.pt" if args.family == "braanet" else (
            f"benchmarks/{args.model}/best.pt"
        )
        return os.path.join(args.motion_checkpoint_root, suffix)
    artifact = args.dataset.removeprefix("eeg_")
    if args.family == "braanet":
        return os.path.join(args.braanet_checkpoint_root, artifact.upper(), "best.pt")
    return os.path.join(args.benchmark_checkpoint_root, f"{args.model}_{artifact}", "best.pt")


def load_model(args, device, *, randomize: bool = False):
    path = resolve_checkpoint(args)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = load_checkpoint(path, device)
    if args.family == "braanet":
        model_name = "braanet"
        if randomize:
            allowed = set(attr.fields_dict(BRAANetV2Config))
            config = BRAANetV2Config(**{
                key: value for key, value in checkpoint.get("config", {}).items()
                if key in allowed
            })
            set_seed(args.seed)
            model = BRAANetV2(config).to(device)
        else:
            model, _, config = build_braanet(checkpoint, device)
    else:
        model_name = args.model
        if randomize:
            allowed = {field.name for field in fields(BenchmarkTrainConfig)}
            config_data = {
                key: value for key, value in checkpoint.get("config", {}).items()
                if key in allowed
            }
            config_data["model_name"] = args.model
            config = BenchmarkTrainConfig(**config_data)
            set_seed(args.seed)
            model = build_model(config).to(device)
        else:
            model, _, config = build_benchmark(checkpoint, args.model, device)
    if int(config.signal_length) != args.signal_length:
        raise ValueError(
            f"Checkpoint signal_length={config.signal_length}, evaluation uses "
            f"{args.signal_length}"
        )
    model.eval()
    return model, {
        "family": args.family,
        "model": model_name,
        "checkpoint": path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "randomized_parameters": bool(randomize),
    }


def write_results(output: str, records: list[dict], metadata: dict) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as stream:
        json.dump({"metadata": metadata, "results": records}, stream, indent=2)
    csv_path = os.path.splitext(output)[0] + ".csv"
    columns = sorted({key for row in records for key in row})
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {output} and {csv_path}")


def base_metadata(args, model_metadata: dict | None = None) -> dict:
    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "device": str(select_device(args.cpu)),
        "arguments": vars(args),
    }
    if model_metadata:
        metadata.update(model_metadata)
    return metadata


def level3_kwargs(config: TriXConfig) -> dict:
    return {
        "fs": config.fs,
        "taas_frequency_range": config.taas_frequency_range,
        "taas_n_fft": config.taas_n_fft,
        "taas_hop_length": config.taas_hop_length,
        "deletion_fractions": config.deletion_fractions,
        "random_deletion_repeats": config.random_deletion_repeats,
        "stability_noise_fraction": config.stability_noise_fraction,
    }


def metric_record(condition: str, metrics: dict[str, float], **extra) -> dict:
    return {"condition": condition, **extra, **metrics}
