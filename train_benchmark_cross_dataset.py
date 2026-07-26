import argparse
import json
import os
import time

import torch

from train_benchmark_EEGDenoiseNet import (
    BenchmarkTrainConfig,
    GenericDenoisingLoss,
    build_model,
)
from load_dataset import (
    CROSS_EEG_DATA_FOLDER as DATA_FOLDER,
    CROSS_EEG_NUM_SUBJECTS as NO_IDS,
    CROSS_EEG_SOURCE_FS as CROSS_FS,
)
from eeg_denoise.cross_data import (
    cross_batch_to_device,
    make_dataset,
    make_loader,
    split_subjects,
)
from eeg_denoise.training import (
    evaluate,
    load_model_checkpoint,
    model_selection_score,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
)


BENCHMARK_MODEL_NAMES = [
    "simplecnn",
    "fcnn",
    "complexcnn",
    "rnnlstm",
    "deepseparator",
    "det",
    "lrr_unet",
]


def build_benchmark_config(args, model_name, input_channels=1):
    return BenchmarkTrainConfig(
        model_name=model_name,
        signal_length=args.signal_length,
        input_channels=input_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        cpu=args.cpu,
        loss_name=args.loss_name,
        l1_weight=args.l1_weight,
        spectral_weight=args.spectral_weight,
        det_patch_len=args.det_patch_len,
        det_depth=args.det_depth,
        det_heads=args.det_heads,
        det_dropout=args.det_dropout,
        det_emb_dropout=args.det_emb_dropout,
        lrr_stages=args.lrr_stages,
        lrr_base_ch=args.lrr_base_ch,
        lrr_netn_layers=args.lrr_netn_layers,
        lrr_netr_layers=args.lrr_netr_layers,
        deepseparator_indicator=args.deepseparator_indicator,
    )


def train_model(args, model_name, splits, datasets, loaders, device):
    train_set, val_set, test_set = datasets
    cfg = build_benchmark_config(args, model_name, input_channels=len(train_set.channel_names))
    model = build_model(cfg).to(device)
    criterion = GenericDenoisingLoss(
        loss_name=args.loss_name,
        l1_weight=args.l1_weight,
        spectral_weight=args.spectral_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

    save_dir = os.path.join(args.save_root, f"{model_name}_eog")
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")

    train_ids, val_ids, test_ids = splits
    train_loader, val_loader, test_loader = loaders

    history = {
        "splits": {
            "train_subjects": train_ids,
            "val_subjects": val_ids,
            "test_subjects": test_ids,
            "train_windows": len(train_set),
            "val_windows": len(val_set),
            "test_windows": len(test_set),
        },
        "train": [],
        "val": [],
        "test": [],
    }

    best_score = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, args.grad_clip,
            batch_adapter=cross_batch_to_device,
        )
        val_metrics = evaluate(
            model, criterion, val_loader, device,
            batch_adapter=cross_batch_to_device,
        )
        test_metrics = evaluate(
            model, criterion, test_loader, device,
            batch_adapter=cross_batch_to_device,
        )
        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history["test"].append(test_metrics)

        score = model_selection_score(val_metrics)
        is_best = score < best_score
        if is_best:
            best_score = score
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, cfg, history)
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_score, cfg, history)

        print(
            f"[{model_name}][Epoch {epoch:03d}/{args.epochs:03d}] "
            f"time={time.time() - start:.1f}s | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_score={score:.4f} | "
            f"val_RRMSE_t={val_metrics['rrmse_temporal']:.4f} | "
            f"val_RRMSE_s={val_metrics['rrmse_spectral']:.4f} | "
            f"val_CC={val_metrics['cc']:.4f} | "
            f"val_SNRimp={val_metrics['snr_improvement_db']:.4f} | "
            f"test_RRMSE_t={test_metrics['rrmse_temporal']:.4f} | "
            f"test_RRMSE_s={test_metrics['rrmse_spectral']:.4f} | "
            f"test_CC={test_metrics['cc']:.4f} | "
            f"test_SNRimp={test_metrics['snr_improvement_db']:.4f} "
            f"{'BEST' if is_best else ''}"
        )

    load_model_checkpoint(best_path, model, device)
    final_test_metrics = evaluate(
        model, criterion, test_loader, device,
        batch_adapter=cross_batch_to_device,
    )
    history["final_test"] = final_test_metrics

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "model_name": model_name}, f, indent=2)
    with open(os.path.join(save_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    return {
        "model_name": model_name,
        "save_dir": save_dir,
        "best_path": best_path,
        "final_test_metrics": final_test_metrics,
    }


def parse_model_names(value):
    if value.strip().lower() == "all":
        return BENCHMARK_MODEL_NAMES
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in BENCHMARK_MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown benchmark models: {unknown}")
    return names


def main(args):
    set_seed(args.seed)
    device = select_device(args.cpu)
    os.makedirs(args.save_root, exist_ok=True)

    train_ids, val_ids, test_ids = split_subjects(
        num_subjects=args.num_subjects,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_set = make_dataset(args, train_ids)
    val_set = make_dataset(args, val_ids)
    test_set = make_dataset(args, test_ids)

    train_loader = make_loader(train_set, args.batch_size, args.num_workers, True, device)
    val_loader = make_loader(val_set, args.batch_size, args.num_workers, False, device)
    test_loader = make_loader(test_set, args.batch_size, args.num_workers, False, device)

    splits = (train_ids, val_ids, test_ids)
    datasets = (train_set, val_set, test_set)
    loaders = (train_loader, val_loader, test_loader)

    results = []
    for model_name in parse_model_names(args.model_name):
        print(f"\n========== TRAIN {model_name} ON CROSS_DATA ==========")
        results.append(train_model(args, model_name, splits, datasets, loaders, device))

    summary_path = os.path.join(args.save_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n========== FINAL TEST SUMMARY ==========")
    for result in results:
        metrics = result["final_test_metrics"]
        print(
            f"{result['model_name']:<14} | "
            f"RRMSE_t={metrics['rrmse_temporal']:.4f} | "
            f"RRMSE_s={metrics['rrmse_spectral']:.4f} | "
            f"CC={metrics['cc']:.4f} | "
            f"SNRimp={metrics['snr_improvement_db']:.4f}"
        )
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train benchmark EOG denoisers on cross_data")
    parser.add_argument("--model-name", type=str, default="all",
                        help="One of simplecnn,fcnn,complexcnn,rnnlstm,deepseparator,det,lrr_unet or all.")
    parser.add_argument("--data-folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--source-fs", type=float, default=CROSS_FS)
    parser.add_argument("--target-fs", type=float, default=256.0)
    parser.add_argument("--signal-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None)
    resampling = parser.add_mutually_exclusive_group()
    resampling.add_argument(
        "--resample",
        dest="no_resample",
        action="store_false",
        help="Resample CrossEEG signals from source-fs to target-fs.",
    )
    resampling.add_argument(
        "--no-resample",
        dest="no_resample",
        action="store_true",
        help="Keep signals at source-fs (default, for backward compatibility).",
    )
    parser.set_defaults(no_resample=True)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--num-subjects", type=int, default=NO_IDS)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--loss-name", type=str, default="mse",
                        choices=["mse", "l1", "smoothl1", "mse_l1"])
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--deepseparator-indicator", type=str, default="zeros",
                        choices=["zeros", "ones", "noisy", "abs_noisy"])

    parser.add_argument("--det-patch-len", type=int, default=64)
    parser.add_argument("--det-depth", type=int, default=6)
    parser.add_argument("--det-heads", type=int, default=8)
    parser.add_argument("--det-dropout", type=float, default=0.1)
    parser.add_argument("--det-emb-dropout", type=float, default=0.1)

    parser.add_argument("--lrr-stages", type=int, default=3)
    parser.add_argument("--lrr-base-ch", type=int, default=32)
    parser.add_argument("--lrr-netn-layers", type=int, default=4)
    parser.add_argument("--lrr-netr-layers", type=int, default=4)

    parser.add_argument("--save-root", type=str, default="runs/cross_train_multichannel/benchmarks")
    main(parser.parse_args())
