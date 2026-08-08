# TriX-EEG implementation

This directory implements all equations in `TriX_EEG_Draft_v3.pdf` as
sample-wise metrics. Dataset summaries are macro means over windows, and the
final TDS is the geometric mean of the three summarized level scores.

## Level analysis

- `level1_signal_validity.py` implements equations 3-13. It measures temporal
  reconstruction, periodogram reconstruction, waveform correlation, SNR gain,
  and preservation of delta/theta/alpha/beta/gamma power. Its DS answers:
  *does the output remain a valid clean EEG signal?*
- `level2_removal_faithfulness.py` implements equations 14-23. It compares
  `noisy - prediction` to the paired true artifact in morphology and energy,
  adds normalized Artifact Reconstruction Error (ARE), then measures
  projection-based neural leakage and residual artifact. FS is the mean of
  RAS, RAER utility, ARE utility, NLR utility, and RAR utility. It answers:
  *did the model remove the correct component rather than merely produce a
  plausible waveform?*
- `level3_explanation_reliability.py` implements equations 24-38. It uses the
  absolute input gradient of removed energy as temporal attribution, then
  tests soft temporal artifact alignment, spectral intervention
  consistency, causal deletion, and perturbation stability. Its RS answers:
  *is the explanation localized, spectrally meaningful, causal, and stable?*
- `final_score.py` implements equations 39-43. The geometric mean prevents a
  strong level from compensating for a failed level. It also supports the
  weighted generalization in equation 40.

## Protocol choices left open by the draft

The PDF deliberately describes several values as "predefined" rather than
fixing them. The runner records these choices in every JSON result. Defaults:

- `delta_snr_max = 20 dB`
- `band_power_error_max = 1.0`
- `raer_error_max = 1.0`
- `are_error_max = 1.0`
- `residual_ratio_max = 0.1`
- TAAS = 128-sample Hann STFT frames with a 32-sample hop over 0.5-45 Hz
- attribution = absolute gradient of `||y - f(y)||^2` with respect to input
- BAC band suppression = FFT-bin deletion for 0.5-45 Hz neural bands
- CAF = zero-baseline deletion at 10%, 20%, and 30%, with three matched random
  deletions per percentage
- ASS perturbation = zero-mean Gaussian noise at 1% of input RMS

## Running

Run every requested checkpoint and dataset:

```bash
python evaluate_trix_eeg.py --cpu
```

Levels 1-2 use the complete held-out dataset. Level 3 defaults to a uniformly
spaced 64-window subset because it requires gradients and many model
interventions. Use `--level3-max-samples 0` for every held-out window.

Calculate a final score independently:

```bash
python -m trix_eeg.final_score 0.8 0.7 0.6
```

The runner writes a complete machine-readable JSON file and a flat CSV summary
under `runs/trix_eeg/` by default.

## Metric sanity experiments

Six independent scripts live in `sanity_checks/`. Each writes both JSON
(including protocol metadata and expected behavior) and a flat CSV under
`runs/sanity_checks/`.

```bash
# 1. Real attribution versus a channel-time shuffled attribution
python -m sanity_checks.random_attribution --dataset eeg_eog --family braanet --cpu

# 2. Same architecture with freshly initialized, untrained parameters
python -m sanity_checks.random_model --dataset eeg_eog --family braanet --cpu

# 3. Gradient x Input, SmoothGrad, and zero-baseline Integrated Gradients
python -m sanity_checks.explanation_method_sensitivity --dataset eeg_eog --family braanet --cpu

# 4. Clean input with an explicit zero-artifact reference
python -m sanity_checks.clean_input --dataset eeg_eog --family braanet --cpu

# 5. One-factor-at-a-time normalization-threshold sweep
python -m sanity_checks.metric_threshold_sensitivity --dataset eeg_eog --family braanet --cpu

# 6. EOG samples evaluated against EMG artifact references
python -m sanity_checks.wrong_artifact_pairing --dataset eeg_eog \
  --donor-dataset eeg_emg --family braanet --cpu
```

Use `--family benchmark --model det` (or another benchmark name) to select a
benchmark checkpoint. `--checkpoint PATH` overrides automatic checkpoint
resolution. `--max-samples N` makes a deterministic uniform subset for a quick
run; a value of zero uses the complete test set. Level-3 experiments default to
64 samples because SmoothGrad and Integrated Gradients require repeated forward
and backward passes.

Two invariances are expected from the definitions and are recorded in output
metadata: BAC does not use the temporal attribution map, so attribution
shuffling does not change it; NLR does not use the artifact reference, so wrong
artifact pairing does not change it. The wrong-pairing script reports both raw
donor artifacts and RMS-matched donors, allowing morphology sensitivity to be
separated from an artifact-energy mismatch.

Plot the Level-3 TAAS, BAC, CAF, and ASS comparisons:

```bash
python visualization/plot_trix_level3_metrics.py
```

This writes an all-dataset overview and one detailed PNG/PDF pair per dataset
under `figures/trix_eeg/`. Use `--datasets motion` or `--models braanet det gan`
to plot a subset.

Compare the three level scores using model-wise radar charts:

```bash
python visualization/plot_trix_level_scores_radar.py
```

This produces a three-dataset overview plus one detailed DS/FS/RS/TDS radar chart
per dataset under `figures/trix_eeg_are/radar/`, using the ARE-extended results
by default. The default paper-friendly radial grid resolution
is 0.2 and the default range is 0.2-1.0 for every dataset.
Customize these using `--radial-step` and repeatable overrides such as
`--dataset-range motion 0.2 0.8`; `--radial-min` and `--radial-max` apply one
common range to all datasets. Use
`--models braanet det gan` to reduce visual overlap, and add
`--annotate-values` when plotting a small model subset. Values outside a custom
range are placed at the nearest boundary; downward and upward triangles identify
values below and above the displayed range, respectively.

Visualize BPS together with macro, delta, theta, alpha, beta, and gamma
relative band-power errors across every model and dataset:

```bash
python visualization/plot_trix_band_power_metrics.py
```

The resulting dashboard uses one model order across datasets, annotated exact
values, and a shared logarithmic error scale so the smaller EEG/EOG and EEG/EMG
errors remain visible beside the larger motion-artifact errors. By default it
reads the ARE-extended full-framework results and writes PNG/PDF output under
`figures/trix_eeg/band_power/`. Use `--sort-models declared`, `--datasets`, or
`--models` to customize the comparison.

Generate model-wise gradient-saliency localization and causal-deletion
diagnostics:

```bash
python visualization/plot_trix_level3_saliency.py --cpu
```

The raw-gradient figures overlay normalized signed gradient saliency, noisy EEG,
true artifact, and the soft STFT artifact-energy profile used by TAAS. By default, models process two adjacent
2-second windows separately and their maps are concatenated into one 4-second
view. Configure this with `--sample-index` and `--num-windows`. Separate
SmoothGrad figures average 32 absolute-gradient maps perturbed at 10% of input
RMS and report the resulting SG-TAAS. Configure them with
`--smoothgrad-samples` and `--smoothgrad-noise-fraction`. The causal figures
compare top-attribution deletion against matched random deletion at 10%, 20%,
and 30%, reporting mean ± SEM over 16 uniformly spaced windows. Change these
using `--sample-index`, `--causal-samples`, and `--deletion-fractions`.

The same command also writes a combined temporal heatmap per dataset. It shows
noisy EEG and artifact over the soft artifact-energy heatmap, followed by the
SmoothGrad map for every model on a shared time axis. Heatmap rows are normalized
independently, and model labels report their TAAS values.
