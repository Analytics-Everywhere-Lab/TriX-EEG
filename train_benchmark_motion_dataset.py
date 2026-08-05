"""Train benchmark denoisers on subject-independent MotionEEGDataset splits."""

import argparse
import json
import os
import time

import torch

from benchmark_models.GAN.training import StateBundle, train_gan_one_epoch
from eeg_denoise.motion_data import (
    evaluate_motion,
    make_motion_dataset,
    make_motion_loader,
    make_motion_splits,
    motion_batch_to_device,
)
from eeg_denoise.training import (
    load_model_checkpoint,
    model_selection_score,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
)
from load_dataset import (
    MOTION_EEG_DATA_FOLDER as DATA_FOLDER,
    MOTION_EEG_SIGNAL_LENGTH as SIGNAL_LENGTH,
    MOTION_EEG_SOURCE_FS as SOURCE_FS,
    MOTION_EEG_TARGET_FS as TARGET_FS,
)
from train_benchmark_EEGDenoiseNet import (
    BenchmarkTrainConfig,
    GenericDenoisingLoss,
    build_model,
)


BENCHMARK_MODEL_NAMES = [
    "simplecnn", "fcnn", "complexcnn", "rnnlstm",
    "deepseparator", "det", "lrr_unet", "gan",
]


def parse_model_names(value):
    if value.strip().lower() == "all":
        return list(BENCHMARK_MODEL_NAMES)
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in BENCHMARK_MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown benchmark models: {unknown}")
    if not names:
        raise ValueError("At least one benchmark model must be selected")
    return names


def build_config(args, model_name):
    return BenchmarkTrainConfig(
        model_name=model_name,
        signal_length=args.signal_length,
        input_channels=1,
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
        gan_hidden_dim=args.gan_hidden_dim,
        gan_num_layers=args.gan_num_layers,
        gan_tanh_output=args.gan_tanh_output,
        gan_adversarial_weight=args.gan_adversarial_weight,
        gan_reconstruction_weight=args.gan_reconstruction_weight,
    )


def train_model(args, model_name, splits, datasets, loaders, device):
    cfg = build_config(args, model_name)
    model = build_model(cfg).to(device)
    criterion = GenericDenoisingLoss(
        loss_name=args.loss_name,
        l1_weight=args.l1_weight,
        spectral_weight=args.spectral_weight,
    )
    if model_name == "gan":
        generator_optimizer = torch.optim.Adam(
            model.model.generator.parameters(), lr=args.lr, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            model.model.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999)
        )
        generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            generator_optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
        discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            discriminator_optimizer, T_max=args.epochs, eta_min=args.lr_min
        )
        optimizer = StateBundle(
            generator=generator_optimizer, discriminator=discriminator_optimizer
        )
        scheduler = StateBundle(
            generator=generator_scheduler, discriminator=discriminator_scheduler
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr_min
        )

    save_dir = os.path.join(args.save_root, model_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")
    train_ids, val_ids, test_ids = splits
    train_set, val_set, test_set = datasets
    train_loader, val_loader, test_loader = loaders
    history = {
        "splits": {
            "train_subjects": train_ids, "val_subjects": val_ids,
            "test_subjects": test_ids, "train_windows": len(train_set),
            "val_windows": len(val_set), "test_windows": len(test_set),
        },
        "train": [], "val": [], "test": [],
    }
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        if model_name == "gan":
            train_metrics = train_gan_one_epoch(
                model, train_loader, generator_optimizer, discriminator_optimizer,
                device, batch_adapter=motion_batch_to_device,
                grad_clip=args.grad_clip,
                adversarial_weight=args.gan_adversarial_weight,
                reconstruction_weight=args.gan_reconstruction_weight,
            )
        else:
            train_metrics = train_one_epoch(
                model, criterion, train_loader, optimizer, device, args.grad_clip,
                batch_adapter=motion_batch_to_device,
            )
        val_metrics = evaluate_motion(
            model, criterion, val_loader, device, fs=args.target_fs,
            description=f"{model_name} validation",
        )
        test_metrics = evaluate_motion(
            model, criterion, test_loader, device, fs=args.target_fs,
            description=f"{model_name} test",
        )
        if model_name == "gan":
            generator_scheduler.step()
            discriminator_scheduler.step()
        else:
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
            f"time={time.time() - start:.1f}s | train_loss={train_metrics['loss']:.4f} | "
            f"train_RRMSE_s={train_metrics['rrmse_spectral']:.4f} | "
            f"val_score={score:.4f} | val_RRMSE_t={val_metrics['rrmse_temporal']:.4f} | "
            f"val_RRMSE_s={val_metrics['rrmse_spectral']:.4f} | "
            f"val_SNRimp={val_metrics['snr_improvement_db']:.4f} dB | "
            f"test_RRMSE_t={test_metrics['rrmse_temporal']:.4f} | "
            f"test_RRMSE_s={test_metrics['rrmse_spectral']:.4f} | "
            f"test_CC={test_metrics['cc']:.4f} | "
            f"test_SNRimp={test_metrics['snr_improvement_db']:.4f} dB | "
            f"artifact_CC={test_metrics['artifact_cc']:.4f} "
            f"{'BEST' if is_best else ''}"
        )

    load_model_checkpoint(best_path, model, device)
    final_metrics = evaluate_motion(
        model, criterion, test_loader, device, fs=args.target_fs,
        description=f"Final {model_name} test",
    )
    history["final_test"] = final_metrics
    for filename, value in (
        ("history.json", history),
        ("config.json", {**vars(args), "model_name": model_name}),
        ("final_test_metrics.json", final_metrics),
    ):
        with open(os.path.join(save_dir, filename), "w") as stream:
            json.dump(value, stream, indent=2)
    return {
        "model_name": model_name,
        "checkpoint": best_path,
        "final_test_metrics": final_metrics,
    }


def main(args):
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(args.seed)
    device = select_device(args.cpu)
    os.makedirs(args.save_root, exist_ok=True)
    splits = make_motion_splits(args)
    datasets = tuple(make_motion_dataset(args, ids) for ids in splits)
    loaders = (
        make_motion_loader(datasets[0], args.batch_size, args.num_workers, True, device),
        make_motion_loader(datasets[1], args.batch_size, args.num_workers, False, device),
        make_motion_loader(datasets[2], args.batch_size, args.num_workers, False, device),
    )

    results = []
    for model_name in parse_model_names(args.model_name):
        print(f"\n========== TRAIN {model_name} ON MOTION EEG ==========")
        results.append(train_model(args, model_name, splits, datasets, loaders, device))
    summary_path = os.path.join(args.save_root, "summary.json")
    with open(summary_path, "w") as stream:
        json.dump(results, stream, indent=2)
    print(json.dumps(results, indent=2))


def build_parser():
    parser = argparse.ArgumentParser("Train benchmark models on MotionEEGDataset")
    parser.add_argument("--model-name", default="all",
                        help="Comma-separated benchmark models or 'all'.")
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--source-fs", type=float, default=SOURCE_FS)
    parser.add_argument("--target-fs", type=float, default=TARGET_FS)
    parser.add_argument("--signal-length", type=int, default=SIGNAL_LENGTH)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
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
    parser.add_argument("--save-root", default="runs/motion_train/benchmarks")
    parser.add_argument("--loss-name", default="mse",
                        choices=["mse", "l1", "smoothl1", "mse_l1"])
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--deepseparator-indicator", default="zeros",
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
    parser.add_argument("--gan-hidden-dim", type=int, default=50)
    parser.add_argument("--gan-num-layers", type=int, default=2)
    parser.add_argument("--gan-tanh-output", action="store_true")
    parser.add_argument("--gan-adversarial-weight", type=float, default=1.0)
    parser.add_argument("--gan-reconstruction-weight", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
