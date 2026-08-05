from __future__ import annotations

import os
import re
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np
import torch
from numpy.polynomial import Chebyshev
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos
from torch.utils.data import DataLoader, Dataset


MOTION_EEG_DATA_FOLDER = "data_motion_artifact/EEG-csv"
MOTION_EEG_SOURCE_FS = 2048.0
MOTION_EEG_TARGET_FS = 256.0
MOTION_EEG_NUM_SUBJECTS = 23
MOTION_EEG_SIGNAL_LENGTH = 512  # 2 seconds at 256 Hz
MOTION_EEG_CLEAN_BASELINE_ORDER = 20
MOTION_EEG_CONTAMINATED_BASELINE_ORDER = 3


def adaptive_polynomial_baseline_correction(
    signal,
    *,
    polynomial_order,
    max_fit_points=8192,
    iterations=5,
    outlier_sigma=3.5,
):
    """Remove recording-level drift with a robust polynomial baseline.

    Motion excursions must not be absorbed into the estimated baseline.  The
    fit is therefore iteratively sigma-clipped using a MAD scale estimate.  A
    bounded, evenly spaced subset keeps the fit practical for the roughly
    nine-minute, 2048-Hz recordings, while the resulting polynomial is
    evaluated at every original sample.

    Following the preprocessing used for this dataset in MLMRS-Net, callers
    use a flexible high-order baseline for the clean reference and a lower
    order baseline for the contaminated channel so that genuine motion
    artifacts are retained for the denoising task.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1-D signal, got shape {signal.shape}")
    if signal.size == 0:
        return signal.copy()
    if polynomial_order < 0:
        raise ValueError("polynomial_order must be non-negative")

    # The recordings are DC coupled.  Remove the robust constant offset before
    # fitting so it cannot dominate the polynomial coefficients.
    dc_removed = signal - np.median(signal)
    if dc_removed.size <= polynomial_order:
        return dc_removed

    fit_count = min(int(max_fit_points), dc_removed.size)
    fit_indices = np.linspace(
        0, dc_removed.size - 1, num=fit_count, dtype=np.int64
    )
    x_fit = np.linspace(-1.0, 1.0, num=fit_count, dtype=np.float64)
    y_fit = dc_removed[fit_indices]
    inliers = np.isfinite(y_fit)
    required = polynomial_order + 1

    baseline_model = None
    for _ in range(max(1, int(iterations))):
        if int(inliers.sum()) < required:
            break
        baseline_model = Chebyshev.fit(
            x_fit[inliers], y_fit[inliers], deg=int(polynomial_order),
            domain=(-1.0, 1.0),
        )
        residual = y_fit - baseline_model(x_fit)
        residual_center = np.median(residual[inliers])
        mad = np.median(np.abs(residual[inliers] - residual_center))
        robust_sigma = 1.4826 * mad
        if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
            break
        updated = (
            np.isfinite(residual)
            & (np.abs(residual - residual_center) <= outlier_sigma * robust_sigma)
        )
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    if baseline_model is None:
        return dc_removed
    x_full = np.linspace(-1.0, 1.0, num=dc_removed.size, dtype=np.float64)
    return dc_removed - baseline_model(x_full)


def available_motion_subject_ids(data_folder=MOTION_EEG_DATA_FOLDER) -> list[int]:
    """Return the numeric Trial IDs available in ``data_folder``."""
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Motion EEG data folder does not exist: {data_folder}")

    subject_ids = []
    for filename in os.listdir(data_folder):
        match = re.fullmatch(r"Trial(\d+)\.csv", filename, flags=re.IGNORECASE)
        if match:
            subject_ids.append(int(match.group(1)))
    if not subject_ids:
        raise FileNotFoundError(f"No Trial*.csv files found in {data_folder}")
    return sorted(subject_ids)


def split_motion_subject_ids(
    subject_ids: Optional[Sequence[int]] = None,
    *,
    data_folder=MOTION_EEG_DATA_FOLDER,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic, disjoint trial-level train/val/test splits."""

    ids = np.asarray(
        available_motion_subject_ids(data_folder) if subject_ids is None else subject_ids,
        dtype=np.int64,
    )

    rng = np.random.default_rng(seed)
    ids = ids.copy()
    rng.shuffle(ids)
    num_train = int(round(len(ids) * train_ratio))
    num_val = int(round(len(ids) * val_ratio))
    return (
        ids[:num_train].astype(int).tolist(),
        ids[num_train:num_train + num_val].astype(int).tolist(),
        ids[num_train + num_val:].astype(int).tolist(),
    )


def preprocess_motion_eeg(
    signal,
    *,
    source_fs=MOTION_EEG_SOURCE_FS,
    target_fs=MOTION_EEG_TARGET_FS,
    lowcut=0.1,
    highcut=100.0,
    notch_frequencies=(60.0, 50.0),
    notch_q_factors=(12.0, 5.0),
    filter_order=4,
    baseline_order=MOTION_EEG_CLEAN_BASELINE_ORDER,
):
    """DC-remove, baseline-correct, filter, and resample a complete channel."""
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = 0.5 * float(source_fs)

    if not 0 < lowcut < highcut < nyquist:
        raise ValueError(
            f"Expected 0 < lowcut < highcut < {nyquist}, got {lowcut}, {highcut}"
        )
    baseline_corrected = adaptive_polynomial_baseline_correction(
        signal, polynomial_order=baseline_order
    )

    bandpass = butter(
        filter_order,
        [lowcut, highcut],
        btype="bandpass",
        fs=source_fs,
        output="sos",
    )
    filtered = sosfiltfilt(bandpass, baseline_corrected)
    if len(notch_frequencies) != len(notch_q_factors):
        raise ValueError("notch_frequencies and notch_q_factors must have equal lengths")
    for frequency, quality_factor in zip(notch_frequencies, notch_q_factors):
        if not 0 < frequency < nyquist:
            raise ValueError(f"Invalid notch frequency {frequency} for source_fs={source_fs}")
        b, a = iirnotch(frequency, Q=quality_factor, fs=source_fs)
        filtered = sosfiltfilt(tf2sos(b, a), filtered)

    ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(1000)
    return resample_poly(filtered, ratio.numerator, ratio.denominator).astype(np.float32)


class MotionEEGDataset(Dataset):
    """Two-second paired EEG windows grouped by Trial ID.

    Channel 1 is the clean reference
    Channel 2 is the motion-contaminated
    recording. 
    """

    def __init__(
        self,
        data_folder=MOTION_EEG_DATA_FOLDER,
        subject_ids: Optional[Sequence[int]] = None,
        split: Optional[str] = "train",
        source_fs=MOTION_EEG_SOURCE_FS,
        target_fs=MOTION_EEG_TARGET_FS,
        signal_length=MOTION_EEG_SIGNAL_LENGTH,
        stride=None,
        normalize=True,
        clean_baseline_order=MOTION_EEG_CLEAN_BASELINE_ORDER,
        contaminated_baseline_order=MOTION_EEG_CONTAMINATED_BASELINE_ORDER,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    ):
        super().__init__()
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be one of {None, 'train', 'val', 'test'}")

        self.data_folder = data_folder
        self.source_fs = float(source_fs)
        self.target_fs = float(target_fs)
        self.signal_length = int(signal_length)
        self.stride = self.signal_length if stride is None else int(stride)
        self.normalize = bool(normalize)
        self.normalization_eps = 1e-8
        self.clean_baseline_order = int(clean_baseline_order)
        self.contaminated_baseline_order = int(contaminated_baseline_order)


        candidate_ids = (
            available_motion_subject_ids(data_folder)
            if subject_ids is None
            else [int(subject_id) for subject_id in subject_ids]
        )
        if split is not None:
            split_ids = split_motion_subject_ids(
                candidate_ids,
                data_folder=data_folder,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
            candidate_ids = split_ids[{"train": 0, "val": 1, "test": 2}[split]]
        self.subject_ids = list(candidate_ids)

        self.records: list[dict] = []
        self.samples: list[tuple[int, int]] = []
        self._build_samples()

    def _build_samples(self):
        for subject_id in self.subject_ids:
            path = os.path.join(self.data_folder, f"Trial{subject_id}.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing motion EEG trial: {path}")

            # The files have no header: column 1 is clean EEG and column 2 is
            # motion-contaminated EEG (zero-based usecols 1 and 2).
            channels = np.loadtxt(
                path,
                delimiter=",",
                usecols=(1, 2),
                dtype=np.float32,
            )
            clean = preprocess_motion_eeg(
                channels[:, 0], source_fs=self.source_fs, target_fs=self.target_fs,
                baseline_order=self.clean_baseline_order,
            )
            contaminated = preprocess_motion_eeg(
                channels[:, 1], source_fs=self.source_fs, target_fs=self.target_fs,
                baseline_order=self.contaminated_baseline_order,
            )
            length = min(len(clean), len(contaminated))
            clean = clean[:length]
            contaminated = contaminated[:length]

            artifact = contaminated - clean
            scale = float(np.std(contaminated)) if self.normalize else 1.0
            mean = float(np.mean(contaminated)) if self.normalize else 0.0
            if not np.isfinite(scale) or scale < self.normalization_eps:
                scale = 1.0
            if not np.isfinite(mean):
                mean = 0.0

            if length < self.signal_length:
                continue
            record_index = len(self.records)
            self.records.append(
                {
                    "clean": clean.astype(np.float32, copy=False),
                    "x": contaminated.astype(np.float32, copy=False),
                    "artifact": artifact.astype(np.float32, copy=False),
                    "scale": scale,
                    "mean": mean,
                    "subject_id": subject_id,
                }
            )
            self.samples.extend(
                (record_index, start)
                for start in range(0, length - self.signal_length + 1, self.stride)
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record_index, start = self.samples[index]
        record = self.records[record_index]
        end = start + self.signal_length

        contaminated = record["x"][start:end]
        clean = record["clean"][start:end]
        artifact = record["artifact"][start:end]
        scale = record["scale"]
        mean = record["mean"]

        # Use one affine transform for the paired signals.  The shared mean
        # cancels in x - clean, so the artifact is scaled but not centered:
        # (x - mean) / scale = (clean - mean) / scale + artifact / scale.
        contaminated = (contaminated - mean) / scale
        clean = (clean - mean) / scale
        artifact = artifact / scale

        return {
            "x": torch.from_numpy(contaminated).unsqueeze(0),
            "clean": torch.from_numpy(clean).unsqueeze(0),
            "artifact": torch.from_numpy(artifact).unsqueeze(0),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "mean": torch.tensor(mean, dtype=torch.float32),
            "subject_id": torch.tensor(record["subject_id"], dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


def create_motion_eeg_datasets(
    *,
    data_folder=MOTION_EEG_DATA_FOLDER,
    subject_ids=None,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    **dataset_kwargs,
):
    """Split Trial IDs first, then create windows independently per split."""
    train_ids, val_ids, test_ids = split_motion_subject_ids(
        subject_ids,
        data_folder=data_folder,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    common = dict(data_folder=data_folder, split=None, **dataset_kwargs)
    return (
        MotionEEGDataset(subject_ids=train_ids, **common),
        MotionEEGDataset(subject_ids=val_ids, **common),
        MotionEEGDataset(subject_ids=test_ids, **common),
    )


def create_motion_eeg_dataloaders(
    *,
    batch_size=32,
    num_workers=0,
    pin_memory=False,
    **dataset_kwargs,
):
    train_set, val_set, test_set = create_motion_eeg_datasets(**dataset_kwargs)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_set, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_set, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory),
    )
