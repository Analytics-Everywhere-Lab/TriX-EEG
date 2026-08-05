"""Loader for the paired MATLAB CrossEEGDataset."""

from __future__ import annotations

import os
from fractions import Fraction

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset


CROSS_EEG_DATA_FOLDER = "cross_data"
CROSS_EEG_SOURCE_FS = 200.0
CROSS_EEG_NUM_SUBJECTS = 54
CROSS_EEG_DATASET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]
CROSS_EEG_DEFAULT_CHANNELS = ["FP1", "FP2", "F3", "F4", "Fz", "F7", "F8", "T3", "T4"]


def _ensure_channels_first(data, n_channels=19):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")
    if data.shape[0] == n_channels:
        return data
    if data.shape[1] == n_channels:
        return data.T
    raise ValueError(f"Cannot infer channel dimension from shape {data.shape}")


def _ensure_1d(data):
    data = np.asarray(data, dtype=np.float32).squeeze()
    if data.ndim != 1:
        raise ValueError(f"Expected 1D EOG signal after squeeze, got shape {data.shape}")
    return data


def _crop_to_same_length(*arrays):
    lengths = [array.shape[1] if array.ndim == 2 else array.shape[0] for array in arrays]
    minimum_length = min(lengths)
    return [
        array[:, :minimum_length] if array.ndim == 2 else array[:minimum_length]
        for array in arrays
    ]


def _resample_if_needed(data, source_fs, target_fs):
    if np.isclose(source_fs, target_fs):
        return data.astype(np.float32, copy=False)
    ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(1000)
    axis = 1 if data.ndim == 2 else 0
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=axis).astype(np.float32)


def _normalize_with_reference(data, reference, eps=1e-8):
    """Scale signals using the contaminated reference channel standard deviation."""
    std = reference.std(axis=1, keepdims=True)
    return data / (std + eps)


class CrossEEGDataset(Dataset):
    """Multi-channel windows from the paired CrossEEGDataset MATLAB files."""

    def __init__(
        self,
        data_folder=CROSS_EEG_DATA_FOLDER,
        subject_ids=None,
        channel_names=None,
        source_fs=CROSS_EEG_SOURCE_FS,
        target_fs=256.0,
        signal_length=512,
        stride=None,
        resample=False,
        normalize=True,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.subject_ids = (
            list(subject_ids) if subject_ids is not None
            else list(range(1, CROSS_EEG_NUM_SUBJECTS + 1))
        )
        self.channel_names = (
            list(channel_names) if channel_names is not None
            else list(CROSS_EEG_DEFAULT_CHANNELS)
        )
        self.source_fs = float(source_fs)
        self.target_fs = float(target_fs)
        self.signal_length = int(signal_length)
        self.stride = int(stride) if stride is not None else self.signal_length
        self.resample = bool(resample)
        self.normalize = bool(normalize)
        self.samples = []

        if self.signal_length <= 0:
            raise ValueError("signal_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        unknown_channels = [
            channel for channel in self.channel_names
            if channel not in CROSS_EEG_DATASET_CHANNELS
        ]
        if unknown_channels:
            raise ValueError(f"Unknown channel names: {unknown_channels}")
        self._build_samples()

    def _build_samples(self):
        contaminated_data = loadmat(os.path.join(self.data_folder, "Contaminated_Data.mat"))
        pure_data = loadmat(os.path.join(self.data_folder, "Pure_Data.mat"))
        heog_data = loadmat(os.path.join(self.data_folder, "HEOG.mat"))
        veog_data = loadmat(os.path.join(self.data_folder, "VEOG.mat"))

        for subject_id in self.subject_ids:
            contaminated_key = f"sim{subject_id}_con"
            pure_key = f"sim{subject_id}_resampled"
            heog_key = f"heog_{subject_id}"
            veog_key = f"veog_{subject_id}"
            required = [
                (contaminated_key, contaminated_data), (pure_key, pure_data),
                (heog_key, heog_data), (veog_key, veog_data),
            ]
            missing = [key for key, source in required if key not in source]
            if missing:
                print(f"Subject {subject_id}: missing keys {missing}; skipping")
                continue

            contaminated = _ensure_channels_first(
                contaminated_data[contaminated_key], len(CROSS_EEG_DATASET_CHANNELS)
            )
            pure = _ensure_channels_first(
                pure_data[pure_key], len(CROSS_EEG_DATASET_CHANNELS)
            )
            heog = _ensure_1d(heog_data[heog_key])
            veog = _ensure_1d(veog_data[veog_key])
            contaminated, pure, _, _ = _crop_to_same_length(contaminated, pure, heog, veog)

            if self.resample:
                contaminated = _resample_if_needed(contaminated, self.source_fs, self.target_fs)
                pure = _resample_if_needed(pure, self.source_fs, self.target_fs)
            if self.normalize:
                pure = _normalize_with_reference(pure, contaminated)
                contaminated = _normalize_with_reference(contaminated, contaminated)

            num_samples = contaminated.shape[1]
            if num_samples < self.signal_length:
                print(
                    f"Subject {subject_id}: length {num_samples} < "
                    f"signal_length {self.signal_length}; skipping"
                )
                continue

            channel_indices = [
                CROSS_EEG_DATASET_CHANNELS.index(channel) for channel in self.channel_names
            ]
            for start in range(0, num_samples - self.signal_length + 1, self.stride):
                end = start + self.signal_length
                self.samples.append({
                    "x": contaminated[channel_indices, start:end].astype(np.float32),
                    "clean": pure[channel_indices, start:end].astype(np.float32),
                    "subject_id": subject_id,
                    "channel_indices": channel_indices,
                    "start": start,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "x": torch.from_numpy(sample["x"]),
            "clean": torch.from_numpy(sample["clean"]),
            "subject_id": torch.tensor(sample["subject_id"], dtype=torch.long),
            "channel_index": torch.tensor(sample["channel_indices"], dtype=torch.long),
            "start": torch.tensor(sample["start"], dtype=torch.long),
        }


def create_cross_eeg_dataset(
    *, data_folder=CROSS_EEG_DATA_FOLDER, subject_ids=None, channel_names=None,
    source_fs=CROSS_EEG_SOURCE_FS, target_fs=256.0, signal_length=512,
    stride=None, resample=False, normalize=True,
):
    return CrossEEGDataset(
        data_folder=data_folder, subject_ids=subject_ids, channel_names=channel_names,
        source_fs=source_fs, target_fs=target_fs, signal_length=signal_length,
        stride=stride, resample=resample, normalize=normalize,
    )


def create_cross_eeg_dataloader(
    *, batch_size=32, num_workers=0, shuffle=False, pin_memory=None,
    device=None, **dataset_kwargs,
):
    dataset = create_cross_eeg_dataset(**dataset_kwargs)
    if pin_memory is None:
        pin_memory = device is not None and device.type == "cuda"
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def parse_cross_subject_ids(value):
    if value is None or value.strip().lower() == "all":
        return None
    subject_ids = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", maxsplit=1)
            subject_ids.extend(range(int(start), int(end) + 1))
        else:
            subject_ids.append(int(part))
    return subject_ids

