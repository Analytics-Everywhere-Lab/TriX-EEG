# BandRouteNet

This repository contains official implementation of the paper: ``BandRouteNet: An Adaptive Band Routing Neural Network for EEG Artifact Removal``


## Setup

Use Python 3.10 or newer. Install the PyTorch build appropriate for your CPU or
CUDA environment, followed by the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Run commands from the repository root. Pass `--cpu` where supported to disable
CUDA.

## Data

- **EEGDenoiseNet**: single-channel clean EEG mixed with EOG, EMG, or both.
- **EEG-EOG dataset**: paired multi-channel contaminated and clean EEG,
  separated with subject-level train/validation/test splits.

### EEGDenoiseNet

The default directory is `data/`:

```text
data/EEG_all_epochs.npy
data/EOG_all_epochs.npy
data/EMG_all_epochs.npy
```

The original dataset documentation and preprocessing reference code are under
`EEGdenoiseNet/': https://github.com/ncclabsustech/EEGdenoiseNet

### EEG-EOG dataset

The default directory is `cross_data/`:

```text
cross_data/Contaminated_Data.mat
cross_data/Pure_Data.mat
cross_data/HEOG.mat
cross_data/VEOG.mat
```

## EEGDenoiseNet: training and evaluation

### Train BandRouteNet

```bash
python train_braanet_EEGDenoiseNet.py \
  --noise-type EOG \
  --epochs 15 \
  --save-dir runs/braanet/EOG
```

`--noise-type` accepts `EOG`, `EMG`, or `MIXED`. The train, validation, and
test SNR ranges default to `-7,-6,-5,-4,-3,-2,-1,0,1,2`.

Examples for the other artifact settings:

```bash
python train_braanet_EEGDenoiseNet.py \
  --noise-type EMG \
  --save-dir runs/braanet/EMG

python train_braanet_EEGDenoiseNet.py \
  --noise-type MIXED \
  --save-dir runs/braanet/Mixed
```
```

### Evaluate BandRouteNet

```bash
python eval_EEGDenoiseNet.py \
  --ckpt_path runs/braanet/EOG/best.pt \
  --noise_type EOG \
  --save_json_path runs/braanet/EOG/test_metrics_by_snr.json
```


## EEG-EOG dataset: training and evaluation

### Train BandRouteNet

```bash
python train_braanet_cross_dataset.py \
  --epochs 100 \
  --save-dir runs/cross_train/braanet_eog
```


- Use `--no-resample` to retain the original 200 Hz signals and
`--no-normalize` to disable contaminated-reference normalization.

- EEG-EOG dataset baseline training preserves the original sampling rate by default.
Pass `--resample` to resample from `--source-fs` to `--target-fs`.


### Evaluate BandRouteNet

```bash
python eval_cross_dataset.py \
  --eval-target braanet \
  --ckpt-path runs/cross_train/braanet_eog/best.pt \
  --subject-ids all \
  --save-json runs/cross_train/braanet_eog/cross_eval.json \
  --save-csv runs/cross_train/braanet_eog/cross_eval.csv
```

Evaluate the same paired EEG-EOG dataset windows across controlled SNR levels from
−7 to 2 dB and save a routing visualization:

```bash
MPLBACKEND=Agg python eval_cross_dataset.py \
  --eval-target braanet \
  --ckpt-path runs/cross_train/braanet_eog/best.pt \
  --subject-ids all \
  --eval-by-snr \
  --snr-min -7 \
  --snr-max 2 \
  --snr-save-json runs/cross_train/braanet_eog/metrics_by_snr.json \
  --snr-save-csv runs/cross_train/braanet_eog/metrics_by_snr.csv \
  --routing-visualization \
  --routing-save-path figures/routing_vis_cross_dataset.png
```

Controlled-SNR evaluation calculates the artifact residual as
`contaminated − clean`, scales that residual independently for each window and
channel to the requested SNR, and evaluates `clean + scaled residual`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Project structure

```text
.
├── train_braanet_EEGDenoiseNet.py       # BandRouteNet + EEGDenoiseNet training
├── train_benchmark_EEGDenoiseNet.py     # Baseline + EEGDenoiseNet train/eval
├── eval_EEGDenoiseNet.py                # BandRouteNet + EEGDenoiseNet evaluation
├── train_braanet_cross_dataset.py       # BandRouteNet + EEG-EOG dataset training
├── train_benchmark_cross_dataset.py     # Baseline + EEG-EOG dataset training
├── eval_cross_dataset.py                # EEG-EOG dataset evaluation
├── load_dataset.py                      # Both datasets and loader factories
├── model.py                             # BandRouteNet
├── configs/                             # Model configuration
├── modules/                             # BandRouteNet layers and losses
├── benchmark_models/                    # Baseline architectures
├── eeg_denoise/                         # Shared training/data infrastructure
├── experiments/                         # Additional experiment workflows
├── visualization/                       # Plots and qualitative analyses
└── tests/                               # Automated tests
```

Generated checkpoints and metrics are written under `runs/`. Figures are
written under `figures/`.
