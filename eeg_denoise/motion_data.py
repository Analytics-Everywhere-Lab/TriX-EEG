"""Shared data and evaluation helpers for MotionEEGDataset experiments."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_dataset import MotionEEGDataset, split_motion_subject_ids
from metrics import (
    batch_cc,
    batch_frequency_preservation,
    batch_rrmse_spectral,
    batch_rrmse_temporal,
    batch_snr_improvement,
)
from utils import EvalTracker
from eeg_denoise.training import prediction_from_output


def make_motion_splits(args: Any):
    """Split Trial IDs before any window samples are constructed."""
    return split_motion_subject_ids(
        data_folder=args.data_folder,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        seed=args.seed,
    )


def make_motion_dataset(args: Any, subject_ids: list[int]) -> MotionEEGDataset:
    return MotionEEGDataset(
        data_folder=args.data_folder,
        subject_ids=subject_ids,
        split=None,
        source_fs=args.source_fs,
        target_fs=args.target_fs,
        signal_length=args.signal_length,
        stride=args.stride,
        normalize=not args.no_normalize,
    )


def make_motion_loader(
    dataset: MotionEEGDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def motion_batch_to_device(batch, device):
    return (
        batch["x"].to(device, non_blocking=True).float(),
        batch["clean"].to(device, non_blocking=True).float(),
        None,
    )


@torch.no_grad()
def evaluate_motion(
    model,
    criterion,
    loader,
    device,
    *,
    fs=256.0,
    description="Motion eval",
):
    """Evaluate clean reconstruction and recovered motion artifact."""
    model.eval()
    tracker = EvalTracker()
    artifact_totals = {
        "artifact_rrmse_temporal": 0.0,
        "artifact_rrmse_spectral": 0.0,
        "artifact_cc": 0.0,
    }
    frequency_totals = {}
    total = 0

    for batch in tqdm(loader, desc=description, leave=False):
        noisy = batch["x"].to(device, non_blocking=True).float()
        clean = batch["clean"].to(device, non_blocking=True).float()
        artifact = batch["artifact"].to(device, non_blocking=True).float()
        output = model(noisy)
        prediction = prediction_from_output(output)
        loss = criterion(output, clean)["loss"]
        batch_size = noisy.size(0)

        tracker.update(
            loss.item(),
            batch_rrmse_temporal(prediction, clean).mean().item(),
            batch_rrmse_spectral(prediction, clean).mean().item(),
            batch_cc(prediction, clean).mean().item(),
            batch_snr_improvement(noisy, prediction, clean).mean().item(),
            batch_size,
        )

        recovered_artifact = noisy - prediction
        artifact_totals["artifact_rrmse_temporal"] += (
            batch_rrmse_temporal(recovered_artifact, artifact).sum().item()
        )
        artifact_totals["artifact_rrmse_spectral"] += (
            batch_rrmse_spectral(recovered_artifact, artifact).sum().item()
        )
        artifact_totals["artifact_cc"] += (
            batch_cc(recovered_artifact, artifact).sum().item()
        )
        for name, values in batch_frequency_preservation(prediction, clean, fs=fs).items():
            frequency_totals[name] = frequency_totals.get(name, 0.0) + values.sum().item()
        total += batch_size

    metrics = tracker.as_dict()
    denominator = max(total, 1)
    metrics.update({name: value / denominator for name, value in artifact_totals.items()})
    metrics.update({name: value / denominator for name, value in frequency_totals.items()})
    return metrics

