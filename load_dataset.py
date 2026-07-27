
import os
import yaml
import numpy as np
import torch
import math
from fractions import Fraction
from scipy.io import loadmat
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Sequence


CROSS_EEG_DATA_FOLDER = "cross_data"
CROSS_EEG_SOURCE_FS = 200.0
CROSS_EEG_NUM_SUBJECTS = 54
CROSS_EEG_DATASET_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4",
    "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4",
    "T5", "T6", "Fz", "Cz", "Pz",
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
    return resample_poly(
        data,
        ratio.numerator,
        ratio.denominator,
        axis=axis,
    ).astype(np.float32)


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
            list(subject_ids)
            if subject_ids is not None
            else list(range(1, CROSS_EEG_NUM_SUBJECTS + 1))
        )
        self.channel_names = (
            list(channel_names)
            if channel_names is not None
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
            channel
            for channel in self.channel_names
            if channel not in CROSS_EEG_DATASET_CHANNELS
        ]
        if unknown_channels:
            raise ValueError(f"Unknown channel names: {unknown_channels}")

        self._build_samples()

    def _build_samples(self):
        contaminated_data = loadmat(
            os.path.join(self.data_folder, "Contaminated_Data.mat")
        )
        pure_data = loadmat(os.path.join(self.data_folder, "Pure_Data.mat"))
        heog_data = loadmat(os.path.join(self.data_folder, "HEOG.mat"))
        veog_data = loadmat(os.path.join(self.data_folder, "VEOG.mat"))

        for subject_id in self.subject_ids:
            contaminated_key = f"sim{subject_id}_con"
            pure_key = f"sim{subject_id}_resampled"
            heog_key = f"heog_{subject_id}"
            veog_key = f"veog_{subject_id}"
            required = [
                (contaminated_key, contaminated_data),
                (pure_key, pure_data),
                (heog_key, heog_data),
                (veog_key, veog_data),
            ]
            missing = [key for key, source in required if key not in source]
            if missing:
                print(f"Subject {subject_id}: missing keys {missing}; skipping")
                continue

            contaminated = _ensure_channels_first(
                contaminated_data[contaminated_key],
                len(CROSS_EEG_DATASET_CHANNELS),
            )
            pure = _ensure_channels_first(
                pure_data[pure_key],
                len(CROSS_EEG_DATASET_CHANNELS),
            )
            heog = _ensure_1d(heog_data[heog_key])
            veog = _ensure_1d(veog_data[veog_key])
            contaminated, pure, _, _ = _crop_to_same_length(
                contaminated,
                pure,
                heog,
                veog,
            )

            if self.resample:
                contaminated = _resample_if_needed(
                    contaminated, self.source_fs, self.target_fs
                )
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
                CROSS_EEG_DATASET_CHANNELS.index(channel)
                for channel in self.channel_names
            ]
            for start in range(
                0,
                num_samples - self.signal_length + 1,
                self.stride,
            ):
                end = start + self.signal_length
                self.samples.append(
                    {
                        "x": contaminated[channel_indices, start:end].astype(np.float32),
                        "clean": pure[channel_indices, start:end].astype(np.float32),
                        "subject_id": subject_id,
                        "channel_indices": channel_indices,
                        "channel_index": (
                            channel_indices[0]
                            if len(channel_indices) == 1
                            else list(channel_indices)
                        ),
                        "channel_names": list(self.channel_names),
                        "start": start,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "x": torch.from_numpy(sample["x"]),
            "clean": torch.from_numpy(sample["clean"]),
            "subject_id": torch.tensor(sample["subject_id"], dtype=torch.long),
            "channel_index": torch.tensor(
                sample["channel_indices"],
                dtype=torch.long,
            ),
            "start": torch.tensor(sample["start"], dtype=torch.long),
        }


def create_cross_eeg_dataset(
    *,
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
    return CrossEEGDataset(
        data_folder=data_folder,
        subject_ids=subject_ids,
        channel_names=channel_names,
        source_fs=source_fs,
        target_fs=target_fs,
        signal_length=signal_length,
        stride=stride,
        resample=resample,
        normalize=normalize,
    )


def create_cross_eeg_dataloader(
    *,
    batch_size=32,
    num_workers=0,
    shuffle=False,
    pin_memory=None,
    device=None,
    **dataset_kwargs,
):
    """Create EEG-EOG dataset and DataLoader configuration."""
    dataset = create_cross_eeg_dataset(**dataset_kwargs)
    if pin_memory is None:
        pin_memory = device is not None and device.type == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
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


class EEGDenoiseNetDatasetBenchmark(Dataset):
    """
    EEGDenoiseNet dataset 

    Noise_type:
        - "EOG"
        - "EMG"
        - "MIXED"   

    - split into train / val / test with ratio 80 / 10 / 10
    - each split is expanded by 10 fixed SNR levels:
        [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2] dB
    - EOG mode:
        use 3400 EEG + 3400 EOG (truncate EEG to EOG count)
    - EMG mode:
        reuse some EEG to match EMG count

    MIXED mode extension:
        - align EEG, EOG, EMG to a common count
        - use the same 10 SNR levels
        - for each SNR level s:
              noisy = clean + scaled_eog(s) + scaled_emg(s)
    """


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
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                else:
                    arr = np.asarray(arr)
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
        """
        10*log10(rms(clean) / rms(noise_scaled)) = snr_db
        """
        clean_rms = EEGDenoiseNetDatasetBenchmark._get_rms(clean)
        noise_rms = EEGDenoiseNetDatasetBenchmark._get_rms(noise)
        scale = clean_rms / (noise_rms * (10 ** (snr_db / 10.0)) + 1e-12)
        return noise * scale

    @staticmethod
    def _standardize_with_noisy_std(clean: np.ndarray, noisy: np.ndarray, noise_total: np.ndarray):
        std_value = np.std(noisy)
        if std_value < 1e-8:
            std_value = 1e-8

        clean_std = clean / std_value
        noisy_std = noisy / std_value
        noise_std = noise_total / std_value

        return (
            clean_std.astype(np.float32),
            noisy_std.astype(np.float32),
            noise_std.astype(np.float32),
            float(std_value),
        )

    def _split_indices(self, n: int):
        n_train = int(round(n * self.train_ratio))
        n_val = int(round(n * self.val_ratio))
        n_test = n - n_train - n_val

        if self.split == "train":
            return 0, n_train
        elif self.split == "val":
            return n_train, n_train + n_val
        else:
            return n_train + n_val, n_train + n_val + n_test

    @staticmethod
    def _expand_by_reuse(arr: np.ndarray, target_n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Randomly reuse rows until reaching target_n.
        """
        n = arr.shape[0]
        if n == target_n:
            return arr
        if n > target_n:
            return arr[:target_n]

        extra = target_n - n
        reuse_idx = rng.choice(n, size=extra, replace=True)
        return np.vstack([arr, arr[reuse_idx]])

    def _prepare_eog_source(self, rng: np.random.Generator):
        """
        EOG benchmark:
        - use 3400 EOG
        - use 3400 EEG by truncating EEG 
        """
        eeg = self.eeg_all[rng.permutation(self.eeg_all.shape[0])]
        eog = self.eog_all[rng.permutation(self.eog_all.shape[0])]

        target_n = eog.shape[0]  # 3400 in original dataset
        eeg = eeg[:target_n]

        start, end = self._split_indices(target_n)
        eeg = eeg[start:end]
        eog = eog[start:end]
        return eeg, eog

    def _prepare_emg_source(self, rng: np.random.Generator):
        """
        EMG benchmark:
        - EMG count > EEG count
        - randomly reuse some EEG to match EMG count
        """
        eeg = self.eeg_all[rng.permutation(self.eeg_all.shape[0])]
        emg = self.emg_all[rng.permutation(self.emg_all.shape[0])]

        target_n = emg.shape[0]  # 5598 in original dataset
        eeg = self._expand_by_reuse(eeg, target_n, rng)

        start, end = self._split_indices(target_n)
        eeg = eeg[start:end]
        emg = emg[start:end]
        return eeg, emg

    def _prepare_mixed_source(self, rng: np.random.Generator):
        """
        MIXED extension:
        - target_n = max(EOG, EMG)
        """
        eeg = self.eeg_all[rng.permutation(self.eeg_all.shape[0])]
        eog = self.eog_all[rng.permutation(self.eog_all.shape[0])]
        emg = self.emg_all[rng.permutation(self.emg_all.shape[0])]

        target_n = max(eog.shape[0], emg.shape[0])

        eeg = self._expand_by_reuse(eeg, target_n, rng)
        eog = self._expand_by_reuse(eog, target_n, rng)
        emg = self._expand_by_reuse(emg, target_n, rng)

        start, end = self._split_indices(target_n)
        eeg = eeg[start:end]
        eog = eog[start:end]
        emg = emg[start:end]
        return eeg, eog, emg

    def _make_single_noise_samples(self, eeg: np.ndarray, noise: np.ndarray):
        samples = []

        for snr_db in self.snr_levels:
            for i in range(eeg.shape[0]):
                clean = eeg[i]
                art = noise[i]

                art_scaled = self._scale_noise_to_snr(clean, art, float(snr_db))
                noisy = clean + art_scaled

                clean_std, noisy_std, noise_std, std_value = self._standardize_with_noisy_std(
                    clean, noisy, art_scaled
                )

                item = {
                    "x": torch.from_numpy(noisy_std).unsqueeze(0),
                    "clean": torch.from_numpy(clean_std).unsqueeze(0),
                    "noise": torch.from_numpy(noise_std).unsqueeze(0),
                    "snr_db": torch.tensor(float(snr_db), dtype=torch.float32),
                }

                if self.return_unstandardized:
                    item["x_raw"] = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
                    item["clean_raw"] = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
                    item["noise_raw"] = torch.from_numpy(art_scaled.astype(np.float32)).unsqueeze(0)
                    item["std_value"] = torch.tensor(std_value, dtype=torch.float32)

                samples.append(item)

        return samples

    def _make_mixed_samples(self, eeg: np.ndarray, eog: np.ndarray, emg: np.ndarray):
        """
        MIXED extension:
        use the same SNR level for both EOG and EMG at each expansion step.
        """
        samples = []

        for snr_db in self.snr_levels:
            for i in range(eeg.shape[0]):
                clean = eeg[i]
                eog_i = eog[i]
                emg_i = emg[i]

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
                    item["x_raw"] = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)
                    item["clean_raw"] = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
                    item["noise_raw"] = torch.from_numpy(noise_total.astype(np.float32)).unsqueeze(0)
                    item["eog_raw"] = torch.from_numpy(eog_scaled.astype(np.float32)).unsqueeze(0)
                    item["emg_raw"] = torch.from_numpy(emg_scaled.astype(np.float32)).unsqueeze(0)
                    item["std_value"] = torch.tensor(std_value, dtype=torch.float32)

                samples.append(item)

        return samples

    def _prepare_data(self):
        """
        Create samples according to noise_type and split.
        """
        rng = np.random.default_rng(self.seed)

        if self.noise_type == "EOG":
            eeg, eog = self._prepare_eog_source(rng)
            return self._make_single_noise_samples(eeg, eog)

        elif self.noise_type == "EMG":
            eeg, emg = self._prepare_emg_source(rng)
            return self._make_single_noise_samples(eeg, emg)

        else:  # MIXED
            eeg, eog, emg = self._prepare_mixed_source(rng)
            return self._make_mixed_samples(eeg, eog, emg)
        
    def _normalize_snr_levels(self, snr_levels):
        if snr_levels is None:
            return [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
        if isinstance(snr_levels, str):
            return [float(x.strip()) for x in snr_levels.split(",")]
        return list(snr_levels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]



def create_benchmark_datasets(
    eeg_array,
    eog_array=None,
    emg_array=None,
    noise_type: str = "EOG",
    train_snr_levels=None,
    val_snr_levels=None,
    test_snr_levels=None,
    seed: int = 42,
    return_unstandardized: bool = False,
):
    train_set = EEGDenoiseNetDatasetBenchmark(
        eeg_data=eeg_array,
        eog_data=eog_array,
        emg_data=emg_array,
        split="train",
        noise_type=noise_type,
        snr_levels=train_snr_levels,
        seed=seed,
        return_unstandardized=return_unstandardized,
    )

    val_set = EEGDenoiseNetDatasetBenchmark(
        eeg_data=eeg_array,
        eog_data=eog_array,
        emg_data=emg_array,
        split="val",
        noise_type=noise_type,
        snr_levels=val_snr_levels,
        seed=seed,
        return_unstandardized=return_unstandardized,
    )

    test_set = EEGDenoiseNetDatasetBenchmark(
        eeg_data=eeg_array,
        eog_data=eog_array,
        emg_data=emg_array,
        split="test",
        noise_type=noise_type,
        snr_levels=test_snr_levels,
        seed=seed,
        return_unstandardized=return_unstandardized,
    )

    return train_set, val_set, test_set


def create_benchmark_dataloaders(
    eeg_array,
    eog_array=None,
    emg_array=None,
    noise_type: str = "EOG",
    train_snr_levels=None,
    val_snr_levels=None,
    test_snr_levels=None,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
    return_unstandardized: bool = False,
):
    train_set, val_set, test_set = create_benchmark_datasets(
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=noise_type,
        train_snr_levels=train_snr_levels,
        val_snr_levels=val_snr_levels,
        test_snr_levels=test_snr_levels,
        seed=seed,
        return_unstandardized=return_unstandardized,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "configs", "config.yaml")

    with open(file_path, "r") as stream:
        config_data = yaml.safe_load(stream)

    dataset_path = config_data["dataset"]["path"]

    EEG_all_epochs = np.load(os.path.join(dataset_path, "EEG_all_epochs.npy"))
    EOG_all_epochs = np.load(os.path.join(dataset_path, "EOG_all_epochs.npy"))
    EMG_all_epochs = np.load(os.path.join(dataset_path, "EMG_all_epochs.npy"))

    noise_type = config_data["dataset"].get("noise_type", "EOG")   # EOG / EMG / MIXED
    batch_size = config_data["training"].get("batch_size", 32)
    num_workers = config_data["training"].get("num_workers", 0)
    seed = config_data["training"].get("seed", 42)

    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        eeg_array=EEG_all_epochs,
        eog_array=EOG_all_epochs,
        emg_array=EMG_all_epochs,
        noise_type=noise_type,
        train_snr_levels=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],  
        val_snr_levels=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
        test_snr_levels=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],   # fixed-SNR test set for plotting one point
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        return_unstandardized=False,
    )

    print(f"Noise type: {noise_type}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size:   {len(val_loader.dataset)}")
    print(f"Test size:  {len(test_loader.dataset)}")

    batch = next(iter(train_loader))
    print("x shape:", batch["x"].shape)
    print("clean shape:", batch["clean"].shape)
    print("noise shape:", batch["noise"].shape)
    print("snr_db shape:", batch["snr_db"].shape)

    if noise_type == "MIXED":
        print("eog shape:", batch["eog"].shape)
        print("emg shape:", batch["emg"].shape)
