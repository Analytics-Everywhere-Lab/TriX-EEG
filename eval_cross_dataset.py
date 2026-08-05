import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.braanetv2_config import BRAANetV2Config
from load_dataset import (
    CROSS_EEG_DATA_FOLDER as DATA_FOLDER,
    CROSS_EEG_DATASET_CHANNELS as DATASET_CHANNELS,
    CROSS_EEG_DEFAULT_CHANNELS as CHANNELS,
    CROSS_EEG_NUM_SUBJECTS as NO_IDS,
    CROSS_EEG_SOURCE_FS as CROSS_FS,
    CrossEEGDataset,
    create_cross_eeg_dataloader,
    parse_cross_subject_ids,
)
from metrics import (
    EEG_FREQUENCY_BANDS,
    batch_cc,
    batch_frequency_preservation,
    batch_rrmse_spectral,
    batch_rrmse_temporal,
    batch_snr_improvement,
)
from model import BRAANetV2
from train_benchmark_EEGDenoiseNet import BenchmarkTrainConfig, build_model


BENCHMARK_EOG_SPECS = [
    {
        "name": "simplecnn_eog",
        "model_name": "simplecnn",
        "ckpt_path": "runs/cross_train/benchmarks/simplecnn_eog/best.pt",
    },
    {
        "name": "fcnn_eog",
        "model_name": "fcnn",
        "ckpt_path": "runs/cross_train/benchmarks/fcnn_eog/best.pt",
    },
    {
        "name": "complex_cnn_eog",
        "model_name": "complexcnn",
        "ckpt_path": "runs/cross_train/benchmarks/complexcnn_eog/best.pt",
    },
    {
        "name": "rnn_lstm_eog",
        "model_name": "rnnlstm",
        "ckpt_path": "runs/cross_train/benchmarks/rnnlstm_eog/best.pt",
    },
    {
        "name": "deepseparator_eog",
        "model_name": "deepseparator",
        "ckpt_path": "runs/cross_train/benchmarks/deepseparator_eog/best.pt",
    },
    {
        "name": "det_eog",
        "model_name": "det",
        "ckpt_path": "runs/cross_train/benchmarks/det_eog/best.pt",
    },
    {
        "name": "lrr_unet_eog",
        "model_name": "lrr_unet",
        "ckpt_path": "runs/cross_train/benchmarks/lrr_unet_eog/best.pt",
    },
    {
        "name": "gan_eog",
        "model_name": "gan",
        "ckpt_path": "runs/cross_train/benchmarks/gan_eog/best.pt",
    },
]


# =========================
# Utility functions
# =========================

def _mean(values):
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _metric_dict(rrmse_t, rrmse_s, cc, snr_imp):
    return {
        "rrmse_temporal": _mean(rrmse_t),
        "rrmse_spectral": _mean(rrmse_s),
        "cc": _mean(cc),
        "snr_improvement_db": _mean(snr_imp),
    }


# =========================
# Dataset
# =========================

# =========================
# Model loading / evaluation
# =========================

def load_model_from_checkpoint(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = BRAANetV2Config()

    for key, value in ckpt.get("config", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if cfg.band_defs is not None:
        cfg.band_defs = [tuple(band) for band in cfg.band_defs]

    model = BRAANetV2(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def resolve_existing_path(primary_path, fallback_paths=None):
    if os.path.exists(primary_path):
        return primary_path

    for path in fallback_paths or []:
        if os.path.exists(path):
            return path

    paths = [primary_path] + list(fallback_paths or [])
    raise FileNotFoundError(f"None of these checkpoint paths exist: {paths}")


def load_benchmark_model_from_checkpoint(model_name, ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    cfg = BenchmarkTrainConfig(model_name=model_name)
    for key, value in ckpt.get("config", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    cfg.model_name = model_name
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def evaluate_cross_dataset(model, loader, device):
    model.eval()

    all_rrmse_t = []
    all_rrmse_s = []
    all_cc = []
    all_snr_imp = []

    by_subject = defaultdict(lambda: defaultdict(list))
    by_channel = defaultdict(lambda: defaultdict(list))

    for batch in tqdm(loader, desc="Cross-dataset eval", leave=False):
        noisy = batch["x"].to(device, non_blocking=True).float()
        clean = batch["clean"].to(device, non_blocking=True).float()

        out = model(noisy)
        pred = out["y_hat"] if isinstance(out, dict) else out

        rrmse_t = batch_rrmse_temporal(pred, clean).detach().cpu().numpy()
        rrmse_s = batch_rrmse_spectral(pred, clean).detach().cpu().numpy()
        cc = batch_cc(pred, clean).detach().cpu().numpy()
        snr_imp = batch_snr_improvement(noisy, pred, clean).detach().cpu().numpy()

        rrmse_t_sample = rrmse_t.mean(axis=1) if rrmse_t.ndim == 2 else rrmse_t
        rrmse_s_sample = rrmse_s.mean(axis=1) if rrmse_s.ndim == 2 else rrmse_s
        cc_sample = cc.mean(axis=1) if cc.ndim == 2 else cc
        snr_imp_sample = snr_imp.mean(axis=1) if snr_imp.ndim == 2 else snr_imp

        all_rrmse_t.extend(rrmse_t_sample.tolist())
        all_rrmse_s.extend(rrmse_s_sample.tolist())
        all_cc.extend(cc_sample.tolist())
        all_snr_imp.extend(snr_imp_sample.tolist())

        subject_ids = batch["subject_id"].cpu().numpy().tolist()
        channel_indices = batch["channel_index"].cpu().numpy()

        for i, subject_id in enumerate(subject_ids):
            bucket = by_subject[str(subject_id)]
            bucket["rrmse_t"].append(float(rrmse_t_sample[i]))
            bucket["rrmse_s"].append(float(rrmse_s_sample[i]))
            bucket["cc"].append(float(cc_sample[i]))
            bucket["snr_imp"].append(float(snr_imp_sample[i]))

        if channel_indices.ndim == 1:
            channel_indices = channel_indices[:, None]
        for i in range(channel_indices.shape[0]):
            for j, channel_index in enumerate(channel_indices[i]):
                channel_name = DATASET_CHANNELS[int(channel_index)]
                bucket = by_channel[channel_name]
                metric_j = j if rrmse_t.ndim == 2 else 0
                bucket["rrmse_t"].append(float(rrmse_t[i, metric_j] if rrmse_t.ndim == 2 else rrmse_t[i]))
                bucket["rrmse_s"].append(float(rrmse_s[i, metric_j] if rrmse_s.ndim == 2 else rrmse_s[i]))
                bucket["cc"].append(float(cc[i, metric_j] if cc.ndim == 2 else cc[i]))
                bucket["snr_imp"].append(float(snr_imp[i, metric_j] if snr_imp.ndim == 2 else snr_imp[i]))

    summary = _metric_dict(all_rrmse_t, all_rrmse_s, all_cc, all_snr_imp)
    summary["num_windows"] = len(all_rrmse_t)

    subject_metrics = {
        subject_id: _metric_dict(
            values["rrmse_t"],
            values["rrmse_s"],
            values["cc"],
            values["snr_imp"],
        )
        for subject_id, values in sorted(by_subject.items(), key=lambda item: int(item[0]))
    }
    channel_metrics = {
        channel: _metric_dict(
            values["rrmse_t"],
            values["rrmse_s"],
            values["cc"],
            values["snr_imp"],
        )
        for channel, values in by_channel.items()
    }

    return {
        "summary": summary,
        "by_subject": subject_metrics,
        "by_channel": channel_metrics,
    }


def save_summary_csv(path, summary):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "rrmse_temporal",
        "rrmse_spectral",
        "cc",
        "snr_improvement_db",
        "num_windows",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in fieldnames})


def save_model_summary_csv(path, model_summaries):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "model",
        "checkpoint",
        "rrmse_temporal",
        "rrmse_spectral",
        "cc",
        "snr_improvement_db",
        "num_windows",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_summaries:
            writer.writerow({key: row[key] for key in fieldnames})


def parse_subject_ids(value):
    return parse_cross_subject_ids(value)


SNR_BAND_KEYS = [
    f"band_power_absolute_error_{band}" for band in EEG_FREQUENCY_BANDS
]
SNR_METRIC_KEYS = [
    "rrmse_temporal",
    "rrmse_spectral",
    "cc",
    "snr_improvement_db",
    "psd_similarity",
    *SNR_BAND_KEYS,
    "macro_band_power_error",
]


class CrossEEGMultiSNRDataset(Dataset):
    """Create EEG-EOG dataset with a specific SNR level - for evaluation"""

    def __init__(self, base_dataset, snr_levels, eps=1e-12):
        if not snr_levels:
            raise ValueError("At least one SNR level is required.")
        self.base_dataset = base_dataset
        self.snr_levels = [int(level) for level in snr_levels]
        self.eps = float(eps)

    def __len__(self):
        return len(self.base_dataset) * len(self.snr_levels)

    def __getitem__(self, index):
        snr_index, sample_index = divmod(index, len(self.base_dataset))
        sample = self.base_dataset[sample_index]
        clean = sample["clean"].float()
        residual = sample["x"].float() - clean
        snr_db = float(self.snr_levels[snr_index])

        signal_power = clean.square().mean(dim=-1, keepdim=True)
        residual_power = residual.square().mean(dim=-1, keepdim=True)
        target_residual_power = signal_power / (10.0 ** (snr_db / 10.0))
        scale = torch.sqrt(
            target_residual_power / residual_power.clamp_min(self.eps)
        )
        return {
            "x": clean + residual * scale,
            "clean": clean,
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


def _sample_average(values):
    return values.mean(dim=1) if values.ndim == 2 else values


@torch.no_grad()
def evaluate_cross_dataset_by_snr(model, loader, device, fs):
    """Return CrossEEG metrics grouped by controlled SNR."""
    model.eval()
    totals = {}
    counts = {}

    for batch in tqdm(loader, desc="Cross-dataset SNR eval", leave=False):
        noisy = batch["x"].to(device, non_blocking=True).float()
        clean = batch["clean"].to(device, non_blocking=True).float()
        snrs = batch["snr_db"].cpu().to(torch.int64)
        output = model(noisy)
        prediction = output["y_hat"] if isinstance(output, dict) else output

        metrics = {
            "rrmse_temporal": _sample_average(
                batch_rrmse_temporal(prediction, clean)
            ),
            "rrmse_spectral": _sample_average(
                batch_rrmse_spectral(prediction, clean)
            ),
            "cc": _sample_average(batch_cc(prediction, clean)),
            "snr_improvement_db": _sample_average(
                batch_snr_improvement(noisy, prediction, clean)
            ),
        }
        metrics.update(batch_frequency_preservation(prediction, clean, fs=fs))
        metrics["macro_band_power_error"] = torch.stack(
            [metrics[key] for key in SNR_BAND_KEYS],
            dim=0,
        ).mean(dim=0)

        cpu_metrics = {
            key: value.detach().cpu()
            for key, value in metrics.items()
        }
        for snr in snrs.unique().tolist():
            mask = snrs == snr
            counts[snr] = counts.get(snr, 0) + int(mask.sum())
            bucket = totals.setdefault(
                snr,
                {key: 0.0 for key in SNR_METRIC_KEYS},
            )
            for key in SNR_METRIC_KEYS:
                bucket[key] += float(cpu_metrics[key][mask].sum())

    return {
        str(snr): {
            key: totals[snr][key] / counts[snr]
            for key in SNR_METRIC_KEYS
        } | {"num_windows": counts[snr]}
        for snr in sorted(totals)
    }


def save_snr_results(json_path, csv_path, results, config):
    """Save per-model, per-SNR metrics as JSON and a flat CSV table."""
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w") as file:
            json.dump({"config": config, "results": results}, file, indent=2)

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fieldnames = ["model", "snr_db", *SNR_METRIC_KEYS, "num_windows"]
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for model_name, model_results in results.items():
                for snr, values in model_results.items():
                    writer.writerow(
                        {"model": model_name, "snr_db": int(snr), **values}
                    )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    braanet_model = None
    braanet_cfg = None

    if args.eval_target in {"braanet", "all"}:
        braanet_model, braanet_cfg, _ = load_model_from_checkpoint(args.ckpt_path, device)

    target_fs = args.target_fs
    if target_fs is None and braanet_cfg is not None:
        target_fs = float(braanet_cfg.fs)
    if target_fs is None:
        target_fs = 256.0

    signal_length = args.signal_length
    if signal_length is None and braanet_cfg is not None:
        signal_length = int(braanet_cfg.signal_length)
    if signal_length is None:
        signal_length = 512

    stride = args.stride if args.stride is not None else signal_length

    loader = create_cross_eeg_dataloader(
        data_folder=args.data_folder,
        subject_ids=parse_subject_ids(args.subject_ids),
        source_fs=args.source_fs,
        target_fs=target_fs,
        signal_length=signal_length,
        stride=stride,
        resample=not args.no_resample,
        normalize=not args.no_normalize,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    dataset = loader.dataset

    if len(dataset) == 0:
        raise RuntimeError("No cross-dataset windows were created.")

    shared_config = {
        "ckpt_path": args.ckpt_path,
        "data_folder": args.data_folder,
        "source_fs": args.source_fs,
        "target_fs": target_fs,
        "signal_length": signal_length,
        "stride": stride,
        "resample": not args.no_resample,
        "normalize": not args.no_normalize,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
    }
    snr_results = {}
    snr_loader = None
    if args.eval_by_snr:
        if args.snr_min > args.snr_max:
            raise ValueError("--snr-min must be less than or equal to --snr-max")
        snr_levels = list(range(args.snr_min, args.snr_max + 1))
        snr_dataset = CrossEEGMultiSNRDataset(dataset, snr_levels)
        snr_loader = DataLoader(
            snr_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        shared_config["snr_levels_db"] = snr_levels
        shared_config["snr_construction"] = (
            "clean + scaled (contaminated - clean) residual per window/channel"
        )

    if args.eval_target in {"braanet", "all"}:
        results = evaluate_cross_dataset(braanet_model, loader, device)
        results["config"] = dict(shared_config)

        if args.save_json:
            os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
            with open(args.save_json, "w") as f:
                json.dump(results, f, indent=2)

        if args.save_csv:
            save_summary_csv(args.save_csv, results["summary"])

        summary = results["summary"]
        print("\n========== CROSS-DATASET EOG EVALUATION: BRAANET ==========")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Dataset:    {args.data_folder}")
        print(f"Windows:    {summary['num_windows']}")
        print(f"RRMSE_t:    {summary['rrmse_temporal']:.6f}")
        print(f"RRMSE_s:    {summary['rrmse_spectral']:.6f}")
        print(f"CC:         {summary['cc']:.6f}")
        print(f"SNR_imp:    {summary['snr_improvement_db']:.6f} dB")
        if args.save_json:
            print(f"Saved JSON: {args.save_json}")
        if args.save_csv:
            print(f"Saved CSV:  {args.save_csv}")
        if snr_loader is not None:
            snr_results["braanet"] = evaluate_cross_dataset_by_snr(
                braanet_model,
                snr_loader,
                device,
                fs=target_fs,
            )

    if args.eval_target in {"benchmarks", "all"}:
        benchmark_results = {}
        benchmark_summaries = []

        for spec in BENCHMARK_EOG_SPECS:
            if args.benchmark_checkpoint_root:
                ckpt_path = os.path.join(
                    args.benchmark_checkpoint_root,
                    f"{spec['model_name']}_eog",
                    "best.pt",
                )
                ckpt_path = resolve_existing_path(ckpt_path)
            else:
                ckpt_path = resolve_existing_path(
                    spec["ckpt_path"], spec.get("fallback_paths")
                )
            print(f"\n========== CROSS-DATASET EOG EVALUATION: {spec['name']} ==========")
            print(f"Checkpoint: {ckpt_path}")

            model, cfg, _ = load_benchmark_model_from_checkpoint(
                model_name=spec["model_name"],
                ckpt_path=ckpt_path,
                device=device,
            )
            results = evaluate_cross_dataset(model, loader, device)
            results["config"] = {
                **shared_config,
                "model_name": spec["model_name"],
                "checkpoint": ckpt_path,
                "training_config": cfg.__dict__,
            }
            benchmark_results[spec["name"]] = results

            summary = results["summary"]
            row = {
                "model": spec["name"],
                "checkpoint": ckpt_path,
                "rrmse_temporal": summary["rrmse_temporal"],
                "rrmse_spectral": summary["rrmse_spectral"],
                "cc": summary["cc"],
                "snr_improvement_db": summary["snr_improvement_db"],
                "num_windows": summary["num_windows"],
            }
            benchmark_summaries.append(row)

            print(f"Windows: {summary['num_windows']}")
            print(f"RRMSE_t: {summary['rrmse_temporal']:.6f}")
            print(f"RRMSE_s: {summary['rrmse_spectral']:.6f}")
            print(f"CC:      {summary['cc']:.6f}")
            print(f"SNR_imp: {summary['snr_improvement_db']:.6f} dB")
            if snr_loader is not None:
                snr_results[spec["name"]] = evaluate_cross_dataset_by_snr(
                    model,
                    snr_loader,
                    device,
                    fs=target_fs,
                )

        payload = {
            "config": shared_config,
            "summaries": benchmark_summaries,
            "results": benchmark_results,
        }

        if args.benchmark_save_json:
            os.makedirs(os.path.dirname(args.benchmark_save_json) or ".", exist_ok=True)
            with open(args.benchmark_save_json, "w") as f:
                json.dump(payload, f, indent=2)
        if args.benchmark_save_csv:
            save_model_summary_csv(args.benchmark_save_csv, benchmark_summaries)

        print("\n========== BENCHMARK CROSS-DATASET SUMMARY ==========")
        for row in benchmark_summaries:
            print(
                f"{row['model']:<18} | "
                f"RRMSE_t={row['rrmse_temporal']:.6f} | "
                f"RRMSE_s={row['rrmse_spectral']:.6f} | "
                f"CC={row['cc']:.6f} | "
                f"SNR_imp={row['snr_improvement_db']:.6f} dB"
            )
        if args.benchmark_save_json:
            print(f"Saved benchmark JSON: {args.benchmark_save_json}")
        if args.benchmark_save_csv:
            print(f"Saved benchmark CSV:  {args.benchmark_save_csv}")

    if snr_loader is not None:
        save_snr_results(
            args.snr_save_json,
            args.snr_save_csv,
            snr_results,
            shared_config,
        )
        print("\n========== CONTROLLED-SNR SUMMARY ==========")
        for model_name, model_results in snr_results.items():
            for snr, values in model_results.items():
                print(
                    f"{model_name:<18} | SNR={int(snr):>3} dB | "
                    f"RRMSE_t={values['rrmse_temporal']:.6f} | "
                    f"RRMSE_s={values['rrmse_spectral']:.6f} | "
                    f"CC={values['cc']:.6f} | "
                    f"SNR_imp={values['snr_improvement_db']:.6f} dB"
                )
        if args.snr_save_json:
            print(f"Saved SNR JSON: {args.snr_save_json}")
        if args.snr_save_csv:
            print(f"Saved SNR CSV:  {args.snr_save_csv}")

    if args.routing_visualization:
        if braanet_model is None:
            raise ValueError(
                "--routing-visualization requires --eval-target braanet or all"
            )
        from visualization.evaluation_cross_dataset import (
            collect_consecutive_visualization_segments,
            plot_routing_visualization,
        )

        routing = collect_consecutive_visualization_segments(
            model=braanet_model,
            loader=loader,
            device=device,
            num_segments=args.routing_num_segments,
            sample_index_in_batch=args.routing_sample_index,
            channel_index=args.routing_channel_index,
        )
        plot_routing_visualization(
            routing,
            fs=target_fs,
            band_defs=braanet_cfg.band_defs,
            save_path=args.routing_save_path,
            show=args.show_routing,
        )
        print(f"Saved routing visualization: {args.routing_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BRAANet or baseline checkpoints on paired CrossEEG data, "
            "controlled SNR levels, and optional BRAANet routing."
        )
    )
    parser.add_argument("--eval-target", type=str, default="braanet",
                        choices=["braanet", "benchmarks", "all"])
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="runs/cross_train/braanet_eog/best.pt",
    )
    parser.add_argument("--data-folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--source-fs", type=float, default=CROSS_FS)
    parser.add_argument("--target-fs", type=float, default=None)
    parser.add_argument("--signal-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--subject-ids", type=str, default="all",
                        help="Subject ids, e.g. 'all', '1,2,3', or '1-10'.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-resample", action="store_true",
                        help="Keep cross_data at source-fs instead of resampling to checkpoint fs.")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Use raw contaminated/pure signals instead of contaminated-reference z-scoring.")
    parser.add_argument("--save-json", type=str, default="runs/braanet/EOG/cross_dataset_eval.json")
    parser.add_argument("--save-csv", type=str, default="runs/braanet/EOG/cross_dataset_eval_summary.csv")
    parser.add_argument("--benchmark-save-json", type=str,
                        default="runs/benchmark_compare/cross_dataset_eog_eval.json")
    parser.add_argument("--benchmark-save-csv", type=str,
                        default="runs/benchmark_compare/cross_dataset_eog_eval_summary.csv")
    parser.add_argument(
        "--benchmark-checkpoint-root",
        type=str,
        default=None,
        help=(
            "Optional directory containing <model>_eog/best.pt baseline "
            "checkpoints; otherwise use the built-in benchmark paths."
        ),
    )
    parser.add_argument(
        "--eval-by-snr",
        action="store_true",
        help="Evaluate after scaling each paired CrossEEG residual to controlled SNR levels.",
    )
    parser.add_argument("--snr-min", type=int, default=-7)
    parser.add_argument("--snr-max", type=int, default=2)
    parser.add_argument(
        "--snr-save-json",
        default="runs/cross_eval/cross_dataset_by_snr.json",
    )
    parser.add_argument(
        "--snr-save-csv",
        default="runs/cross_eval/cross_dataset_by_snr.csv",
    )
    parser.add_argument(
        "--routing-visualization",
        action="store_true",
        help="Save a BRAANet band-routing visualization from the evaluated CrossEEG loader.",
    )
    parser.add_argument(
        "--routing-save-path",
        default="figures/routing_vis_cross_dataset.png",
    )
    parser.add_argument("--routing-num-segments", type=int, default=2)
    parser.add_argument("--routing-sample-index", type=int, default=0)
    parser.add_argument("--routing-channel-index", type=int, default=0)
    parser.add_argument("--show-routing", action="store_true")
    main(parser.parse_args())
