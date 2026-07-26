"""Shared dataset construction for subject-wise CrossEEGDataset experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from load_dataset import (
    CROSS_EEG_NUM_SUBJECTS as NO_IDS,
    CrossEEGDataset,
    create_cross_eeg_dataset,
)


def split_subjects(
    num_subjects: int = NO_IDS,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic, disjoint subject-level train/val/test splits."""
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to <= 1")

    subjects = np.arange(1, num_subjects + 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    num_train = int(round(num_subjects * train_ratio))
    num_val = int(round(num_subjects * val_ratio))
    return (
        subjects[:num_train].astype(int).tolist(),
        subjects[num_train:num_train + num_val].astype(int).tolist(),
        subjects[num_train + num_val:].astype(int).tolist(),
    )


def make_dataset(args: Any, subject_ids: list[int]) -> CrossEEGDataset:
    """Build a CrossEEGDataset from the common training CLI arguments."""
    stride = args.stride if args.stride is not None else args.signal_length
    return create_cross_eeg_dataset(
        data_folder=args.data_folder,
        subject_ids=subject_ids,
        source_fs=args.source_fs,
        target_fs=args.target_fs,
        signal_length=args.signal_length,
        stride=stride,
        resample=not args.no_resample,
        normalize=not args.no_normalize,
    )


def make_loader(
    dataset: CrossEEGDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Build a DataLoader with CUDA pinning when appropriate."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def cross_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Move a CrossEEGDataset batch to the selected device."""
    return (
        batch["x"].to(device, non_blocking=True).float(),
        batch["clean"].to(device, non_blocking=True).float(),
        None,
    )
