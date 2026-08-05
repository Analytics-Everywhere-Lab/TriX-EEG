"""Evaluate BRAANet and benchmark checkpoints on held-out MotionEEG trials."""

import argparse
import json
import os
from dataclasses import fields

import attr
import torch

from configs.braanetv2_config import BRAANetV2Config
from eeg_denoise.motion_data import (
    evaluate_motion,
    make_motion_dataset,
    make_motion_loader,
    make_motion_splits,
)
from eeg_denoise.training import select_device, set_seed
from load_dataset import (
    MOTION_EEG_DATA_FOLDER as DATA_FOLDER,
    MOTION_EEG_SIGNAL_LENGTH as SIGNAL_LENGTH,
    MOTION_EEG_SOURCE_FS as SOURCE_FS,
    MOTION_EEG_TARGET_FS as TARGET_FS,
)
from model import BRAANetV2
from modules.loss import BRAALossV2
from train_benchmark_EEGDenoiseNet import (
    BenchmarkTrainConfig,
    GenericDenoisingLoss,
    build_model,
)
from train_benchmark_motion_dataset import BENCHMARK_MODEL_NAMES, parse_model_names


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_braanet(checkpoint, device):
    allowed = set(attr.fields_dict(BRAANetV2Config))
    config_data = {
        key: value for key, value in checkpoint.get("config", {}).items()
        if key in allowed
    }
    cfg = BRAANetV2Config(**config_data)
    model = BRAANetV2(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, BRAALossV2(cfg, model.band_decomp), cfg


def build_benchmark(checkpoint, model_name, device):
    allowed = {field.name for field in fields(BenchmarkTrainConfig)}
    config_data = {
        key: value for key, value in checkpoint.get("config", {}).items()
        if key in allowed
    }
    config_data["model_name"] = model_name
    cfg = BenchmarkTrainConfig(**config_data)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = GenericDenoisingLoss(
        loss_name=cfg.loss_name,
        l1_weight=cfg.l1_weight,
        spectral_weight=cfg.spectral_weight,
    )
    return model, criterion, cfg


def selected_experiments(args):
    experiments = []
    if args.family in {"braanet", "all"}:
        experiments.append((
            "braanet",
            "braanet",
            os.path.join(args.checkpoint_root, "braanet", "best.pt"),
        ))
    if args.family in {"benchmark", "all"}:
        for model_name in parse_model_names(args.model_name):
            experiments.append((
                "benchmark",
                model_name,
                os.path.join(args.checkpoint_root, "benchmarks", model_name, "best.pt"),
            ))
    if args.checkpoint:
        if len(experiments) != 1:
            raise ValueError("--checkpoint can only be used when exactly one model is selected")
        family, name, _ = experiments[0]
        experiments[0] = (family, name, args.checkpoint)
    return experiments


def main(args):
    set_seed(args.seed)
    device = select_device(args.cpu)
    _, _, test_ids = make_motion_splits(args)
    test_set = make_motion_dataset(args, test_ids)
    test_loader = make_motion_loader(
        test_set, args.batch_size, args.num_workers, False, device
    )

    results = []
    for family, model_name, checkpoint_path in selected_experiments(args):
        if not os.path.isfile(checkpoint_path):
            if args.skip_missing:
                print(f"[skip] Missing checkpoint: {checkpoint_path}")
                continue
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        if family == "braanet":
            model, criterion, cfg = build_braanet(checkpoint, device)
        else:
            model, criterion, cfg = build_benchmark(checkpoint, model_name, device)
        if int(cfg.signal_length) != args.signal_length:
            raise ValueError(
                f"Checkpoint signal_length={cfg.signal_length}, but dataset uses "
                f"signal_length={args.signal_length}"
            )

        metrics = evaluate_motion(
            model, criterion, test_loader, device,
            fs=args.target_fs, description=f"Evaluate {model_name}",
        )
        result = {
            "family": family,
            "model_name": model_name,
            "checkpoint": checkpoint_path,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "test_subjects": test_ids,
            "test_windows": len(test_set),
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"\n{model_name}: "
            f"RRMSE_t={metrics['rrmse_temporal']:.4f} | "
            f"RRMSE_s={metrics['rrmse_spectral']:.4f} | "
            f"CC={metrics['cc']:.4f} | "
            f"SNR improvement={metrics['snr_improvement_db']:.4f} dB | "
            f"Artifact_CC={metrics['artifact_cc']:.4f}"
        )
        print(json.dumps(metrics, indent=2))

    if not results:
        raise FileNotFoundError("No selected checkpoints were found")
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(results, stream, indent=2)
    print(f"Saved evaluation: {args.output}")


def build_parser():
    parser = argparse.ArgumentParser("Evaluate models on MotionEEGDataset")
    parser.add_argument("--family", default="all",
                        choices=["braanet", "benchmark", "all"])
    parser.add_argument("--model-name", default="all",
                        help=f"Benchmark model(s): {','.join(BENCHMARK_MODEL_NAMES)} or all")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit checkpoint for one selected model")
    parser.add_argument("--checkpoint-root", default="runs/motion_train")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--output", default="runs/motion_train/evaluation.json")
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--source-fs", type=float, default=SOURCE_FS)
    parser.add_argument("--target-fs", type=float, default=TARGET_FS)
    parser.add_argument("--signal-length", type=int, default=SIGNAL_LENGTH)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
