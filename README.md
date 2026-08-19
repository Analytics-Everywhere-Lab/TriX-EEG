# Reconstruction is Not Enough: Towards Trustworthy and Explainable Evaluation of EEG Artifact Removal

TriX-EEG evaluates EEG artifact-removal models from three complementary
perspectives: whether the denoised output is valid, whether the correct artifact
was removed, and whether the model's explanation is reliable. The framework
contains 14 metrics organized into three levels and one overall TriX-EEG
Denoising Score (TDS).

## TriX-EEG framework

For a contaminated EEG window `y`, paired clean reference `x`, and artifact
`a = y - x`, a denoiser predicts `x_hat = f(y)`. The component removed by the
model is `a_hat = y - x_hat`. Metrics are computed per window and then
macro-averaged, giving every valid window equal weight.

| Level | Question | Metrics | Score |
|---|---|---|---|
| Level 1: Signal Validity | Does `x_hat` remain a valid clean EEG signal? | Temporal RRMSE, spectral RRMSE, correlation coefficient (CC), SNR improvement, band-power score (BPS) | DS |
| Level 2: Removal Faithfulness | Does `a_hat` match the true artifact without removing neural activity or leaving artifact behind? | Removed Artifact Similarity (RAS), Removed Artifact Energy Ratio (RAER), Artifact Reconstruction Error (ARE), Neural Leakage Ratio (NLR), Residual Artifact Ratio (RAR) | FS |
| Level 3: Explanation Reliability | Is the attribution artifact-aligned, intervention-consistent, causal, and stable? | Temporal Artifact Alignment Score (TAAS), Band Artifact Consistency (BAC), Causal Attribution Faithfulness (CAF), Attribution Stability Score (ASS) | RS |

All component metrics are converted to utilities in `[0, 1]`, where higher is
better. TriX-EEG uses one fixed aggregation policy—the original legacy
aggregation:

```text
DS  = mean(U_RRMSE-T, U_RRMSE-S, U_CC, U_delta-SNR, BPS)
FS  = mean(RAS, U_RAER, U_ARE, U_NLR, U_RAR)
RS  = mean(TAAS, BAC, CAF, ASS)
TDS = (DS * FS * RS) ** (1 / 3)
```

The arithmetic means make each metric within a level equally interpretable.
The geometric mean used for TDS limits compensation across levels: strong
reconstruction cannot fully conceal poor removal faithfulness or unreliable
explanations. 

The default explanation target is the energy removed by the denoiser,
`g(y) = ||y - f(y)||_2^2`; its attribution is the absolute input gradient
`|dg(y)/dy|`.

### Evaluation protocol

- Levels 1 and 2 are deterministic for fixed inputs and checkpoints and are
  evaluated once over the complete held-out test set.
- Level 3 requires gradients and repeated model interventions. It is evaluated
  over five stratified subsets of 64 windows using seeds `42 43 44 45 46`.
- Each seed selects a different subset and controls the corresponding stochastic
  operations. Every model receives the same indices and operation seed, enabling
  paired comparisons.
- Level-3 metrics, RS, and TDS are reported as mean and sample standard deviation
  across the five subsets. Levels 1 and 2 are reported as single values without
  artificial zero-error bars.

EEGDenoiseNet subsets are stratified across the ten SNR conditions; EEG-Motion
subsets are stratified across the three held-out trials. The complete protocol,
including metric definitions and fixed thresholds, is documented in
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

### Run the complete benchmark

The following command evaluates all available model checkpoints on
EEGDenoiseNet-EOG, EEGDenoiseNet-EMG, and EEG-Motion:

```bash
python evaluate_trix_eeg.py \
  --datasets eeg_eog eeg_emg motion \
  --evaluation-seeds 42 43 44 45 46 \
  --level3-max-samples 64 \
  --output-json runs/trix_eeg_are_multisubset/trix_eeg_results.json \
  --output-csv runs/trix_eeg_are_multisubset/trix_eeg_summary.csv \
  --cpu
```

Use `--level3-max-samples 0` to evaluate Level 3 on the complete test set. This
is substantially more expensive because attribution gradients and causal
interventions require repeated forward and backward passes.

The JSON output preserves configuration, checkpoints, subset indices,
level-wise metrics, aggregate scores, and uncertainty estimates. The CSV file
provides the corresponding flat summary.

### Sanity checks

Six dependency-aware checks test whether each metric responds to the input it
is designed to consume:

1. shuffled attribution;
2. randomly initialized model;
3. explanation-method sensitivity using Gradient × Input, SmoothGrad, and
   Integrated Gradients;
4. clean input with no artifact;
5. metric-threshold sensitivity;
6. incorrect artifact pairing.

Run all checks for EEGDNet over the five matched subsets:

```bash
python -m sanity_checks.run_multiseed_eegdnet \
  --output-dir runs/sanity_checks/eegdnet_legacy_multisubset \
  --seeds 42 43 44 45 46 \
  --max-samples 64 \
  --cpu
```

The generated `REPORT_LEVELWISE.md` reports only metrics that consume the
quantity manipulated by a given check. Expected invariants are omitted instead
of being incorrectly interpreted as failures.

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

- **EEGDenoiseNet-EOG/EMG**: single-channel clean EEG mixed with EOG or EMG at
  ten SNR conditions from -7 to 2 dB. The original dataset documentation and preprocessing reference code are under
`EEGdenoiseNet/` (source: <https://github.com/ncclabsustech/EEGdenoiseNet>).
- **EEG-Motion**: paired contaminated and clean recordings segmented into
  512-sample windows after filtering and resampling to 256 Hz. The original dataset are available at <https://physionet.org/content/motion-artifact/1.0.0/>.

## Tests

```bash
python -m pytest -q tests
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
├── evaluate_trix_eeg.py                 # Complete three-level benchmark runner
├── load_dataset.py                      # Both datasets and loader factories
├── model.py                             # BandRouteNet
├── trix_eeg/                            # Metric and aggregation implementation
├── sanity_checks/                       # Six dependency-aware sanity checks
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
