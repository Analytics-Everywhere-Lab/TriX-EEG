import os
import json
import time
import argparse

import numpy as np
import torch
from torchinfo import summary
from model import BRAANetV2
from modules.loss import BRAALossV2
from load_dataset import create_benchmark_dataloaders
from utils import EvalTracker, parse_batch
from metrics import (
    batch_cc,
    batch_frequency_preservation,
    batch_rrmse_spectral,
    batch_rrmse_temporal,
    batch_snr_improvement,
)
from configs.braanetv2_config import BRAANetV2Config
from eeg_denoise.training import (
    model_selection_score,
    save_checkpoint,
    select_device,
    set_seed,
)
from tqdm import tqdm


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, criterion, loader, optimizer, device, grad_clip=None):
    model.train()
    tracker = EvalTracker()

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        noisy, clean, _ = parse_batch(batch)
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        model_out = model(noisy)
        losses = criterion(model_out, clean)
        loss = losses["loss"]

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        pred = model_out["y_hat"].detach()

        rrmse_t = batch_rrmse_temporal(pred, clean).mean().item()
        rrmse_s = batch_rrmse_spectral(pred, clean).mean().item()
        cc = batch_cc(pred, clean).mean().item()
        snr_imp = batch_snr_improvement(noisy, pred, clean).mean().item()

        tracker.update(loss.item(), rrmse_t, rrmse_s, cc, snr_imp, noisy.size(0))

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "rrmse_t": f"{rrmse_t:.3f}",
            "rrmse_s": f"{rrmse_s:.3f}",
            "cc": f"{cc:.3f}",
            "snr": f"{snr_imp:.2f}"
        })

    return tracker.as_dict()


@torch.no_grad()
def evaluate(model, criterion, loader, device, by_snr=False, fs=256.0):
    model.eval()
    tracker = EvalTracker()
    snr_buckets = {}
    frequency_totals = {}
    num_frequency_samples = 0

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        noisy, clean, snr = parse_batch(batch)
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        model_out = model(noisy)
        losses = criterion(model_out, clean)
        pred = model_out["y_hat"]

        rrmse_t_all = batch_rrmse_temporal(pred, clean)
        rrmse_s_all = batch_rrmse_spectral(pred, clean)
        cc_all = batch_cc(pred, clean)
        snr_imp_all = batch_snr_improvement(noisy, pred, clean)
        frequency_all = batch_frequency_preservation(pred, clean, fs=fs)
        for name, values in frequency_all.items():
            frequency_totals[name] = frequency_totals.get(name, 0.0) + values.sum().item()
        num_frequency_samples += noisy.size(0)

        tracker.update(
            losses["loss"].item(),
            rrmse_t_all.mean().item(),
            rrmse_s_all.mean().item(),
            cc_all.mean().item(),
            snr_imp_all.mean().item(),
            noisy.size(0),
        )

        pbar.set_postfix({
            "loss": f"{losses['loss'].item():.4f}",
            "rrmse_t": f"{rrmse_t_all.mean().item():.3f}",
            "rrmse_s": f"{rrmse_s_all.mean().item():.3f}",
            "cc": f"{cc_all.mean().item():.3f}",
            "snr": f"{snr_imp_all.mean().item():.2f}"
        })

        if by_snr and snr is not None:
            if snr.dim() > 1:
                snr = snr.view(-1)
            snr = snr.cpu().numpy().tolist()
            for i, s in enumerate(snr):
                s_key = round(float(s))
                if s_key not in snr_buckets:
                    snr_buckets[s_key] = {
                        "rrmse_t": [],
                        "rrmse_s": [],
                        "cc": [],
                        "snr_imp": [],
                        **{name: [] for name in frequency_all},
                    }
                snr_buckets[s_key]["rrmse_t"].append(float(rrmse_t_all[i].item()))
                snr_buckets[s_key]["rrmse_s"].append(float(rrmse_s_all[i].item()))
                snr_buckets[s_key]["cc"].append(float(cc_all[i].item()))
                snr_buckets[s_key]["snr_imp"].append(float(snr_imp_all[i].item()))
                for name, values in frequency_all.items():
                    snr_buckets[s_key][name].append(float(values[i].item()))

    metrics = tracker.as_dict()
    denom = max(num_frequency_samples, 1)
    metrics.update({name: total / denom for name, total in frequency_totals.items()})

    if by_snr and len(snr_buckets) > 0:
        metrics["by_snr"] = {}
        for s in sorted(snr_buckets.keys()):
            metrics["by_snr"][str(s)] = {
                "rrmse_temporal": float(np.mean(snr_buckets[s]["rrmse_t"])),
                "rrmse_spectral": float(np.mean(snr_buckets[s]["rrmse_s"])),
                "cc": float(np.mean(snr_buckets[s]["cc"])),
                "snr_improvement_db": float(np.mean(snr_buckets[s]["snr_imp"])),
                **{
                    name: float(np.mean(snr_buckets[s][name]))
                    for name in frequency_totals
                },
            }

    return metrics


# ============================================================
# Main
# ============================================================

def main(args):
    set_seed(args.seed)

    device = select_device(args.cpu)
    os.makedirs(args.save_dir, exist_ok=True)

    eeg_array = np.load(args.eeg_npy)
    eog_array = np.load(args.eog_npy) if args.eog_npy else None
    emg_array = np.load(args.emg_npy) if args.emg_npy else None

    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=args.noise_type,
        
        train_snr_levels=args.train_snr_levels,
        val_snr_levels=args.val_snr_levels,
        test_snr_levels=args.test_snr_levels, 
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_unstandardized=args.return_unstandardized,
    )
    
    print(f'Create train loader with SNR levels: {args.train_snr_levels}')
    print(f'Create val loader with SNR levels: {args.val_snr_levels}')
    print(f'Create test loader with SNR levels: {args.test_snr_levels}')
          

    cfg = BRAANetV2Config(
        fs=args.fs,
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
        fullband_num_blocks=args.num_fullband_blocks,
    )

    # attach loss hyperparameters expected by BRAALossV2
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
    cfg.route_target_mean = args.route_target_mean
    cfg.lambda_alpha_smooth = args.lambda_alpha_smooth
    cfg.lambda_alpha_sparse = args.lambda_alpha_sparse
    cfg.lambda_fullband_res = args.lambda_fullband_res
    cfg.eps = args.eps
    
    print("Config:" , cfg)

    model = BRAANetV2(cfg).to(device)
    dummy = torch.randn(8, 6, 512).to(device)
    print(summary(model, input_size=(8, 1, 512), device=str(device)))

    criterion = BRAALossV2(cfg, model.band_decomp)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min,
    )

    history = {
        "train": [],
        "val": [],
        "test": [],
    }

    best_score = float("inf")
    best_path = os.path.join(args.save_dir, "best.pt")
    last_path = os.path.join(args.save_dir, "last.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # update_loss_weights(cfg, epoch, args.epochs)

        train_metrics = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            by_snr=args.eval_by_snr,
            fs=cfg.fs,
        )

        scheduler.step()

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        test_metrics = evaluate(
            model=model,
            criterion=criterion,
            loader=test_loader,
            device=device,
            by_snr=args.eval_by_snr,
            fs=cfg.fs,
        )

        history["test"].append(test_metrics)

        score = model_selection_score(val_metrics)
        is_best = score < best_score
        if is_best:
            best_score = score
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, cfg, history)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_score, cfg, history)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"time={dt:.1f}s | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_rrmse_t={val_metrics['rrmse_temporal']:.4f} | "
            f"val_rrmse_s={val_metrics['rrmse_spectral']:.4f} | "
            f"val_cc={val_metrics['cc']:.4f} | "
            f"val_snr_imp={val_metrics['snr_improvement_db']:.4f} dB | "
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_rrmse_t={test_metrics['rrmse_temporal']:.4f} | "
            f"test_rrmse_s={test_metrics['rrmse_spectral']:.4f} | "
            f"test_cc={test_metrics['cc']:.4f} | "
            f"test_snr_imp={test_metrics['snr_improvement_db']:.4f} dB | "
            f"{'BEST' if is_best else ''}"
        )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        criterion=criterion,
        loader=test_loader,
        device=device,
        by_snr=args.eval_by_snr,
        fs=cfg.fs,
    )
    history["test"] = test_metrics

    print("\n========== TEST ==========")
    print(json.dumps(test_metrics, indent=2))

    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved to: {args.save_dir}")
    print(f"Best checkpoint: {best_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train BRAANet on EEGDenoiseNet benchmark")

    # data
    parser.add_argument("--eeg-npy", type=str, default="data/EEG_all_epochs.npy")
    parser.add_argument("--eog-npy", type=str, default="data/EOG_all_epochs.npy")
    parser.add_argument("--emg-npy", type=str, default="data/EMG_all_epochs.npy")
    parser.add_argument("--noise-type", type=str, required=True, choices=["EOG", "EMG", "MIXED"])
    parser.add_argument("--train-snr-levels", type=str, default="-7,-6,-5,-4,-3,-2,-1,0,1,2")
    parser.add_argument("--val-snr-levels", type=str, default="-7,-6,-5,-4,-3,-2,-1,0,1,2")
    parser.add_argument("--test-snr-levels", type=str, default="-7,-6,-5,-4,-3,-2,-1,0,1,2")
    parser.add_argument("--return-unstandardized", default=False)

    # training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # signal
    parser.add_argument("--fs", type=float, default=256.0)
    parser.add_argument("--signal-length", type=int, default=512)

    # model
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

    parser.add_argument("--encoder-type", type=str, default="inception",
                        choices=["residual", "inception"])
    parser.add_argument("--decoder-type", type=str, default="inception",
                        choices=["residual", "inception"])
    parser.add_argument("--temporal-model-type", type=str, default="gru",
                        choices=["tcn", "gru"])

    parser.add_argument("--no-fullband-branch", action="store_true", default=False)
    parser.add_argument("--no-fullband-conditioning", action="store_true", default=False)

    # loss
    parser.add_argument("--lambda-time", type=float, default=1.0)
    parser.add_argument("--lambda-diff", type=float, default=0.0)
    parser.add_argument("--lambda-spec", type=float, default=0.0)
    parser.add_argument("--lambda-logspec", type=float, default=0.0)
    parser.add_argument("--lambda-band", type=float, default=0.0)
    parser.add_argument("--lambda-cbc", type=float, default=0.0)
    parser.add_argument("--lambda-ratio", type=float, default=0.0)
    parser.add_argument("--lambda-cov", type=float, default=0.0)

    parser.add_argument("--lambda-route", type=float, default=0.0)
    parser.add_argument("--route-target-mean", type=float, default=0.0)
    parser.add_argument("--lambda-route-tv", type=float, default=0.0)
    parser.add_argument("--lambda-route-entropy", type=float, default=0.0)

    parser.add_argument("--lambda-alpha-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-alpha-sparse", type=float, default=0.0)
    parser.add_argument("--lambda-fullband-res", type=float, default=0.0)

    parser.add_argument("--eps", type=float, default=1e-8)

    # eval
    parser.add_argument("--eval-by-snr", action="store_true", default=False)

    # save
    parser.add_argument("--save-dir", type=str, default="./runs/braanet")

    args = parser.parse_args()
    main(args)
