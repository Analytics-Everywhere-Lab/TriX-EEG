import argparse
import json
import os
import time

import torch

from configs.braanetv2_config import BRAANetV2Config
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
from model import BRAANetV2
from modules.loss import BRAALossV2


def build_config(args):
    cfg = BRAANetV2Config(
        fs=args.target_fs,
        signal_length=args.signal_length,
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
    cfg.lambda_time = args.lambda_time
    cfg.lambda_diff = args.lambda_diff
    cfg.lambda_spec = args.lambda_spec
    cfg.lambda_logspec = args.lambda_logspec
    cfg.lambda_band = args.lambda_band
    cfg.lambda_cbc = args.lambda_cbc
    cfg.lambda_ratio = args.lambda_ratio
    cfg.lambda_cov = args.lambda_cov
    cfg.lambda_route = args.lambda_route
    cfg.lambda_route_tv = args.lambda_route_tv
    cfg.lambda_route_entropy = args.lambda_route_entropy
    cfg.lambda_alpha_smooth = args.lambda_alpha_smooth
    cfg.lambda_alpha_sparse = args.lambda_alpha_sparse
    cfg.lambda_fullband_res = args.lambda_fullband_res
    cfg.eps = args.eps
    return cfg


def main(args):
    set_seed(args.seed)
    device = select_device(args.cpu)
    os.makedirs(args.save_dir, exist_ok=True)

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

    cfg = build_config(args)
    cfg.input_channels = len(train_set.channel_names)
    model = BRAANetV2(cfg).to(device)
    criterion = BRAALossV2(cfg, model.band_decomp)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

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
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

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
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
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

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.save_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    print("\n========== FINAL TEST ==========")
    print(json.dumps(final_test_metrics, indent=2))
    print(f"Saved to: {args.save_dir}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train BRAANet EOG denoiser on cross_data")
    parser.add_argument("--data-folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--source-fs", type=float, default=CROSS_FS)
    parser.add_argument("--target-fs", type=float, default=256.0)
    parser.add_argument("--signal-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--no-resample", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--num-subjects", type=int, default=NO_IDS)
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
    parser.add_argument("--encoder-type", type=str, default="inception", choices=["residual", "inception"])
    parser.add_argument("--decoder-type", type=str, default="inception", choices=["residual", "inception"])
    parser.add_argument("--temporal-model-type", type=str, default="gru", choices=["tcn", "gru"])
    parser.add_argument("--no-fullband-branch", action="store_true")
    parser.add_argument("--no-fullband-conditioning", action="store_true")
    parser.add_argument("--no-artifact-routing", action="store_true")
    parser.add_argument("--no-cross-band-fusion", action="store_true")
    parser.add_argument("--no-band-embedding", action="store_true")

    parser.add_argument("--lambda-time", type=float, default=1.0)
    parser.add_argument("--lambda-diff", type=float, default=0.0)
    parser.add_argument("--lambda-spec", type=float, default=0.0)
    parser.add_argument("--lambda-logspec", type=float, default=0.0)
    parser.add_argument("--lambda-band", type=float, default=0.0)
    parser.add_argument("--lambda-cbc", type=float, default=0.0)
    parser.add_argument("--lambda-ratio", type=float, default=0.0)
    parser.add_argument("--lambda-cov", type=float, default=0.0)
    parser.add_argument("--lambda-route", type=float, default=0.0)
    parser.add_argument("--lambda-route-tv", type=float, default=0.0)
    parser.add_argument("--lambda-route-entropy", type=float, default=0.0)
    parser.add_argument("--lambda-alpha-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-alpha-sparse", type=float, default=0.0)
    parser.add_argument("--lambda-fullband-res", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--save-dir", type=str, default="runs/cross_train/braanet_eog")
    main(parser.parse_args())
