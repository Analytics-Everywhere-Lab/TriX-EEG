"""Compatibility facade for the three dataset-specific loader modules.

Existing training scripts can continue importing from ``load_dataset``. New
code may import directly from the corresponding dataset module.
"""

from cross_eeg_dataset import (
    CROSS_EEG_DATA_FOLDER,
    CROSS_EEG_DATASET_CHANNELS,
    CROSS_EEG_DEFAULT_CHANNELS,
    CROSS_EEG_NUM_SUBJECTS,
    CROSS_EEG_SOURCE_FS,
    CrossEEGDataset,
    create_cross_eeg_dataloader,
    create_cross_eeg_dataset,
    parse_cross_subject_ids,
)
from eegdenoisenet_dataset import (
    EEGDenoiseNetDatasetBenchmark,
    create_benchmark_dataloaders,
    create_benchmark_datasets,
)
from motion_eeg_dataset import (
    MOTION_EEG_DATA_FOLDER,
    MOTION_EEG_NUM_SUBJECTS,
    MOTION_EEG_SIGNAL_LENGTH,
    MOTION_EEG_SOURCE_FS,
    MOTION_EEG_TARGET_FS,
    MotionEEGDataset,
    available_motion_subject_ids,
    create_motion_eeg_dataloaders,
    create_motion_eeg_datasets,
    preprocess_motion_eeg,
    split_motion_subject_ids,
)


__all__ = [
    "CROSS_EEG_DATA_FOLDER",
    "CROSS_EEG_DATASET_CHANNELS",
    "CROSS_EEG_DEFAULT_CHANNELS",
    "CROSS_EEG_NUM_SUBJECTS",
    "CROSS_EEG_SOURCE_FS",
    "CrossEEGDataset",
    "create_cross_eeg_dataloader",
    "create_cross_eeg_dataset",
    "parse_cross_subject_ids",
    "EEGDenoiseNetDatasetBenchmark",
    "create_benchmark_dataloaders",
    "create_benchmark_datasets",
    "MOTION_EEG_DATA_FOLDER",
    "MOTION_EEG_NUM_SUBJECTS",
    "MOTION_EEG_SIGNAL_LENGTH",
    "MOTION_EEG_SOURCE_FS",
    "MOTION_EEG_TARGET_FS",
    "MotionEEGDataset",
    "available_motion_subject_ids",
    "create_motion_eeg_dataloaders",
    "create_motion_eeg_datasets",
    "preprocess_motion_eeg",
    "split_motion_subject_ids",
]
