"""Synthetic-noise loader for the EEGDenoiseNet benchmark arrays."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGDenoiseNetDatasetBenchmark(Dataset):
    """Create fixed-SNR EOG, EMG, or mixed benchmark samples."""

    def __init__(
        self,
        eeg_data,
        eog_data=None,
        emg_data=None,
        split: str = "train",
        noise_type: str = "EOG",
        snr_levels: Optional[Sequence[float]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        return_unstandardized: bool = False,
    ):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train', 'val', 'test'}")
        noise_type = noise_type.upper()
        if noise_type not in {"EOG", "EMG", "MIXED"}:
            raise ValueError("noise_type must be one of {'EOG', 'EMG', 'MIXED'}")
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self.split = split
        self.noise_type = noise_type
        self.snr_levels = np.array(self._normalize_snr_levels(snr_levels), dtype=np.float32)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.return_unstandardized = return_unstandardized
        self.eeg_all = self._ensure_2d_numpy(eeg_data)
        self.eog_all = None if eog_data is None else self._ensure_2d_numpy(eog_data)
        self.emg_all = None if emg_data is None else self._ensure_2d_numpy(emg_data)

        if self.noise_type == "EOG" and self.eog_all is None:
            raise ValueError("eog_data must be provided when noise_type='EOG'")
        if self.noise_type == "EMG" and self.emg_all is None:
            raise ValueError("emg_data must be provided when noise_type='EMG'")
        if self.noise_type == "MIXED" and (self.eog_all is None or self.emg_all is None):
            raise ValueError("Both eog_data and emg_data must be provided when noise_type='MIXED'")
        self.samples = self._prepare_data()

    @staticmethod
    def _ensure_2d_numpy(arr):
        if isinstance(arr, str):
            if arr.endswith(".npy"):
                arr = np.load(arr)
            elif arr.endswith((".pt", ".pth")):
                arr = torch.load(arr, map_location="cpu")
                arr = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
            else:
                raise ValueError("Only .npy/.pt/.pth are supported")
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        if arr.ndim != 2:
            raise ValueError(f"Expected [N,T] or [N,1,T], got {arr.shape}")
        return arr.astype(np.float32)

    @staticmethod
    def _get_rms(x: np.ndarray) -> float:
        return math.sqrt(np.mean(np.square(x)) + 1e-12)

    @staticmethod
    def _scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        clean_rms = EEGDenoiseNetDatasetBenchmark._get_rms(clean)
        noise_rms = EEGDenoiseNetDatasetBenchmark._get_rms(noise)
        scale = clean_rms / (noise_rms * (10 ** (snr_db / 10.0)) + 1e-12)
        return noise * scale

    @staticmethod
    def _standardize_with_noisy_std(clean, noisy, noise_total):
        std_value = max(float(np.std(noisy)), 1e-8)
        return (
            (clean / std_value).astype(np.float32),
            (noisy / std_value).astype(np.float32),
            (noise_total / std_value).astype(np.float32),
            std_value,
        )

    def _split_indices(self, n: int):
        n_train = int(round(n * self.train_ratio))
        n_val = int(round(n * self.val_ratio))
        n_test = n - n_train - n_val
        if self.split == "train":
            return 0, n_train
        if self.split == "val":
            return n_train, n_train + n_val
        return n_train + n_val, n_train + n_val + n_test

    @staticmethod
    def _expand_by_reuse(arr, target_n, rng):
        n = arr.shape[0]
        if n == target_n:
            return arr
        if n > target_n:
            return arr[:target_n]
        reuse_idx = rng.choice(n, size=target_n - n, replace=True)
        return np.vstack([arr, arr[reuse_idx]])

    def _prepare_eog_source(self, rng):
        eeg = self.eeg_all[rng.permutation(self.eeg_all.shape[0])]
        eog = self.eog_all[rng.permutation(self.eog_all.shape[0])]
        target_n = eog.shape[0]
        start, end = self._split_indices(target_n)
        return eeg[:target_n][start:end], eog[start:end]

    def _prepare_emg_source(self, rng):
        eeg = self.eeg_all[rng.permutation(self.eeg_all.shape[0])]
        emg = self.emg_all[rng.permutation(self.emg_all.shape[0])]
        target_n = emg.shape[0]
        eeg = self._expand_by_reuse(eeg, target_n, rng)
        start, end = self._split_indices(target_n)
        return eeg[start:end], emg[start:end]

    def _prepare_mixed_source(self, rng):
        eeg = self.eeg_all[rng.permutation(self.eeg_all.shape[0])]
        eog = self.eog_all[rng.permutation(self.eog_all.shape[0])]
        emg = self.emg_all[rng.permutation(self.emg_all.shape[0])]
        target_n = max(eog.shape[0], emg.shape[0])
        eeg = self._expand_by_reuse(eeg, target_n, rng)
        eog = self._expand_by_reuse(eog, target_n, rng)
        emg = self._expand_by_reuse(emg, target_n, rng)
        start, end = self._split_indices(target_n)
        return eeg[start:end], eog[start:end], emg[start:end]

    def _make_single_noise_samples(self, eeg, noise):
        samples = []
        for snr_db in self.snr_levels:
            for clean, artifact in zip(eeg, noise):
                artifact_scaled = self._scale_noise_to_snr(clean, artifact, float(snr_db))
                noisy = clean + artifact_scaled
                clean_std, noisy_std, noise_std, std_value = self._standardize_with_noisy_std(
                    clean, noisy, artifact_scaled
                )
                item = {
                    "x": torch.from_numpy(noisy_std).unsqueeze(0),
                    "clean": torch.from_numpy(clean_std).unsqueeze(0),
                    "noise": torch.from_numpy(noise_std).unsqueeze(0),
                    "snr_db": torch.tensor(float(snr_db), dtype=torch.float32),
                }
                if self.return_unstandardized:
                    item.update({
                        "x_raw": torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0),
                        "clean_raw": torch.from_numpy(clean.astype(np.float32)).unsqueeze(0),
                        "noise_raw": torch.from_numpy(artifact_scaled.astype(np.float32)).unsqueeze(0),
                        "std_value": torch.tensor(std_value, dtype=torch.float32),
                    })
                samples.append(item)
        return samples

    def _make_mixed_samples(self, eeg, eog, emg):
        samples = []
        for snr_db in self.snr_levels:
            for clean, eog_i, emg_i in zip(eeg, eog, emg):
                eog_scaled = self._scale_noise_to_snr(clean, eog_i, float(snr_db))
                emg_scaled = self._scale_noise_to_snr(clean, emg_i, float(snr_db))
                noise_total = eog_scaled + emg_scaled
                noisy = clean + noise_total
                clean_std, noisy_std, noise_std, std_value = self._standardize_with_noisy_std(
                    clean, noisy, noise_total
                )
                item = {
                    "x": torch.from_numpy(noisy_std).unsqueeze(0),
                    "clean": torch.from_numpy(clean_std).unsqueeze(0),
                    "noise": torch.from_numpy(noise_std).unsqueeze(0),
                    "eog": torch.from_numpy((eog_scaled / std_value).astype(np.float32)).unsqueeze(0),
                    "emg": torch.from_numpy((emg_scaled / std_value).astype(np.float32)).unsqueeze(0),
                    "snr_db": torch.tensor(float(snr_db), dtype=torch.float32),
                }
                if self.return_unstandardized:
                    item.update({
                        "x_raw": torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0),
                        "clean_raw": torch.from_numpy(clean.astype(np.float32)).unsqueeze(0),
                        "noise_raw": torch.from_numpy(noise_total.astype(np.float32)).unsqueeze(0),
                        "eog_raw": torch.from_numpy(eog_scaled.astype(np.float32)).unsqueeze(0),
                        "emg_raw": torch.from_numpy(emg_scaled.astype(np.float32)).unsqueeze(0),
                        "std_value": torch.tensor(std_value, dtype=torch.float32),
                    })
                samples.append(item)
        return samples

    def _prepare_data(self):
        rng = np.random.default_rng(self.seed)
        if self.noise_type == "EOG":
            return self._make_single_noise_samples(*self._prepare_eog_source(rng))
        if self.noise_type == "EMG":
            return self._make_single_noise_samples(*self._prepare_emg_source(rng))
        return self._make_mixed_samples(*self._prepare_mixed_source(rng))

    @staticmethod
    def _normalize_snr_levels(snr_levels):
        if snr_levels is None:
            return [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        if isinstance(snr_levels, str):
            return [float(value.strip()) for value in snr_levels.split(",")]
        return list(snr_levels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def create_benchmark_datasets(
    eeg_array, eog_array=None, emg_array=None, noise_type="EOG",
    train_snr_levels=None, val_snr_levels=None, test_snr_levels=None,
    seed=42, return_unstandardized=False,
):
    common = dict(
        eeg_data=eeg_array, eog_data=eog_array, emg_data=emg_array,
        noise_type=noise_type, seed=seed,
        return_unstandardized=return_unstandardized,
    )
    return (
        EEGDenoiseNetDatasetBenchmark(split="train", snr_levels=train_snr_levels, **common),
        EEGDenoiseNetDatasetBenchmark(split="val", snr_levels=val_snr_levels, **common),
        EEGDenoiseNetDatasetBenchmark(split="test", snr_levels=test_snr_levels, **common),
    )


def create_benchmark_dataloaders(
    eeg_array, eog_array=None, emg_array=None, noise_type="EOG",
    train_snr_levels=None, val_snr_levels=None, test_snr_levels=None,
    seed=42, batch_size=32, num_workers=0, return_unstandardized=False,
):
    datasets = create_benchmark_datasets(
        eeg_array=eeg_array, eog_array=eog_array, emg_array=emg_array,
        noise_type=noise_type, train_snr_levels=train_snr_levels,
        val_snr_levels=val_snr_levels, test_snr_levels=test_snr_levels,
        seed=seed, return_unstandardized=return_unstandardized,
    )
    return (
        DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(datasets[1], batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(datasets[2], batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )

