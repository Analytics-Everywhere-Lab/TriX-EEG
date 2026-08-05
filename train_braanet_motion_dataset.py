"""Train BRAANet on subject-independent MotionEEGDataset splits."""

import argparse
import json
import os
import time
import torch

from configs.braanetv2_config import BRAANetV2Config
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
from model import BRAANetV2
from modules.loss import BRAALossV2


def build_config(args):
    cfg = BRAANetV2Config(
        fs=args.target_fs,
        signal_length=args.signal_length,
        input_channels=1,
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        routing_hidden=args.routing_hidden,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_denoiser_blocks=args.num_denoiser_blocks,
        num_fusion_layers=args.num_fusion_layers,
        num_fusion_heads=args.num_fusion_heads,
        fusion_ff_mult=args.fusion_ff_mult,
        dropout=args.dropout,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        temporal_model_type=args.temporal_model_type,
        use_fullband_branch=not args.no_fullband_branch,
        use_fullband_conditioning=not args.no_fullband_conditioning,
        use_artifact_routing=not args.no_artifact_routing,
        use_cross_band_fusion=not args.no_cross_band_fusion,
        use_band_embedding=not args.no_band_embedding,
        fullband_num_blocks=args.num_fullband_blocks,
    )
    for name in (
        "lambda_time", "lambda_diff", "lambda_spec", "lambda_logspec",
        "lambda_band", "lambda_cbc", "lambda_ratio", "lambda_cov",
        "lambda_route", "lambda_route_tv", "lambda_route_entropy",
        "lambda_alpha_smooth", "lambda_alpha_sparse", "lambda_fullband_res",
        "eps",
    ):
        setattr(cfg, name, getattr(args, name))
    return cfg


def main(args):
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(args.seed)
    device = select_device(args.cpu)
    os.makedirs(args.save_dir, exist_ok=True)

    splits = make_motion_splits(args)
    datasets = tuple(make_motion_dataset(args, ids) for ids in splits)
    loaders = (
        make_motion_loader(datasets[0], args.batch_size, args.num_workers, True, device),
        make_motion_loader(datasets[1], args.batch_size, args.num_workers, False, device),
        make_motion_loader(datasets[2], args.batch_size, args.num_workers, False, device),
    )

    cfg = build_config(args)
    model = BRAANetV2(cfg).to(device)
    criterion = BRAALossV2(cfg, model.band_decomp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    train_ids, val_ids, test_ids = splits
    train_set, val_set, test_set = datasets
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
        "train": [], "val": [], "test": [],
    }
    best_score = float("inf")
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, args.grad_clip,
            batch_adapter=motion_batch_to_device,
        )
        val_metrics = evaluate_motion(
            model, criterion, val_loader, device, fs=args.target_fs,
            description="Motion validation",
        )
        test_metrics = evaluate_motion(
            model, criterion, test_loader, device, fs=args.target_fs,
            description="Motion test",
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
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"time={time.time() - start:.1f}s | train_loss={train_metrics['loss']:.4f} | "
            f"train_RRMSE_s={train_metrics['rrmse_spectral']:.4f} | "
            f"val_score={score:.4f} | val_RRMSE_t={val_metrics['rrmse_temporal']:.4f} | "
            f"val_RRMSE_s={val_metrics['rrmse_spectral']:.4f} | "
            f"val_CC={val_metrics['cc']:.4f} | "
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
        description="Final motion test",
    )
    history["final_test"] = final_metrics
    for filename, value in (
        ("history.json", history),
        ("config.json", vars(args)),
        ("final_test_metrics.json", final_metrics),
    ):
        with open(os.path.join(args.save_dir, filename), "w") as stream:
            json.dump(value, stream, indent=2)
    print(json.dumps(final_metrics, indent=2))
    print(f"Best checkpoint: {best_path}")


def build_parser():
    parser = argparse.ArgumentParser("Train BRAANet on MotionEEGDataset")
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--source-fs", type=float, default=SOURCE_FS)
    parser.add_argument("--target-fs", type=float, default=TARGET_FS)
    parser.add_argument("--signal-length", type=int, default=SIGNAL_LENGTH)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-dir", default="runs/motion_train/braanet")

    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--routing-hidden", type=int, default=64)
    parser.add_argument("--num-encoder-blocks", type=int, default=2)
    parser.add_argument("--num-decoder-blocks", type=int, default=2)
    parser.add_argument("--num-denoiser-blocks", type=int, default=2)
    parser.add_argument("--num-fullband-blocks", type=int, default=2)
    parser.add_argument("--num-fusion-layers", type=int, default=1)
    parser.add_argument("--num-fusion-heads", type=int, default=4)
    parser.add_argument("--fusion-ff-mult", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-type", default="inception", choices=["residual", "inception"])
    parser.add_argument("--decoder-type", default="inception", choices=["residual", "inception"])
    parser.add_argument("--temporal-model-type", default="gru", choices=["tcn", "gru"])
    parser.add_argument("--no-fullband-branch", action="store_true")
    parser.add_argument("--no-fullband-conditioning", action="store_true")
    parser.add_argument("--no-artifact-routing", action="store_true")
    parser.add_argument("--no-cross-band-fusion", action="store_true")
    parser.add_argument("--no-band-embedding", action="store_true")

    for name, default in (
        ("lambda-time", 1.0), ("lambda-diff", 0.0), ("lambda-spec", 0.0),
        ("lambda-logspec", 0.0), ("lambda-band", 0.0), ("lambda-cbc", 0.0),
        ("lambda-ratio", 0.0), ("lambda-cov", 0.0), ("lambda-route", 0.0),
        ("lambda-route-tv", 0.0), ("lambda-route-entropy", 0.0),
        ("lambda-alpha-smooth", 0.0), ("lambda-alpha-sparse", 0.0),
        ("lambda-fullband-res", 0.0), ("eps", 1e-8),
    ):
        parser.add_argument(f"--{name}", type=float, default=default)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
