
# Evaluation functions
import torch
import numpy as np
from model import BRAANetV2
from configs.braanetv2_config import BRAANetV2Config
from modules.loss import BRAALossV2
from load_dataset import create_benchmark_dataloaders
from train_braanet_EEGDenoiseNet import evaluate
from train_benchmark_EEGDenoiseNet import (
    evaluate_checkpoint_across_snrs as evaluate_benchmark_checkpoint_across_snrs,
)
from tqdm import tqdm
import json
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def _unpack_batch(batch):
    """
    Robustly unpack a batch from DataLoader.

    Supported formats:
    - dict from benchmark loader:
        x, clean, noise, snr_db, ...
    - tuple/list:
        (x, y) or (x, y, meta...)
    """
    if isinstance(batch, dict):
        x = batch["x"]

        # Prefer clean target from your dataset
        if "clean" in batch:
            y = batch["clean"]
        elif "y" in batch:
            y = batch["y"]
        else:
            y = None

        return x, y, batch

    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            x = batch[0]
            y = batch[1]
            meta = batch[2:] if len(batch) > 2 else None
            return x, y, meta

    raise ValueError(f"Unsupported batch format: {type(batch)}")


@torch.no_grad()
def collect_consecutive_visualization_segments(
    model,
    loader,
    device,
    num_segments=3,
    sample_index_in_batch=0,
):
    """
    Collect multiple consecutive windows and concatenate them in time.

    Returns:
    - x            [T_total]
    - x_bands      [K, T_total]
    - routing_mean [K, T_total]
    - y_hat_bands  [K, T_total]
    - y_hat_band   [T_total]
    - y_hat        [T_total]
    - y            [T_total] or None
    """
    model.eval()

    xs = []
    ys = []
    x_bands_list = []
    routing_list = []
    y_hat_bands_list = []
    y_hat_band_list = []
    y_hat_list = []

    collected = 0

    for batch in loader:
        x, y, meta = _unpack_batch(batch)
        x = x.to(device)

        if y is not None:
            y = y.to(device)

        out = model(x)

        bsz = x.size(0)
        if sample_index_in_batch >= bsz:
            raise ValueError(
                f"sample_index_in_batch={sample_index_in_batch} >= batch size {bsz}"
            )

        i = sample_index_in_batch

        xs.append(x[i, 0].detach().cpu())

        if y is not None:
            ys.append(y[i, 0].detach().cpu())

        x_bands_list.append(out["x_bands"][i].detach().cpu())

        # routing_gates: [B, K, C, T] -> mean over latent channels => [K, T]
        routing_mean = out["routing_gates"][i].mean(dim=1).detach().cpu()
        # routing_list.append(1 - routing_mean)
        routing_list.append(routing_mean)

        y_hat_bands_list.append(out["y_hat_bands"][i].detach().cpu())
        y_hat_band_list.append(out["y_hat_band"][i, 0].detach().cpu())
        y_hat_list.append(out["y_hat"][i, 0].detach().cpu())

        collected += 1
        if collected >= num_segments:
            break

    if collected == 0:
        raise RuntimeError("No samples collected from loader.")

    result = {
        "x": torch.cat(xs, dim=-1),
        "x_bands": torch.cat(x_bands_list, dim=-1),
        "routing_mean": torch.cat(routing_list, dim=-1),
        "y_hat_bands": torch.cat(y_hat_bands_list, dim=-1),
        "y_hat_band": torch.cat(y_hat_band_list, dim=-1),
        "y_hat": torch.cat(y_hat_list, dim=-1),
        "y": torch.cat(ys, dim=-1) if len(ys) > 0 else None,
    }
    return result

def plot_routing_visualization(
    vis_dict,
    fs=256.0,
    band_defs=None,
    save_path=None,
    show=True,
    include_clean_target=True,
    include_final_yhat=True,
    figsize=(14, 8),
):
    """
    1) Input noisy signal + final denoised signal + clean target
    2) Routing heatmap
    """
    x = vis_dict["x"].numpy()
    routing_mean = vis_dict["routing_mean"].numpy()
    y_hat = vis_dict["y_hat"].numpy()
    y = vis_dict["y"].numpy() if vis_dict["y"] is not None else None

    K, T_total = routing_mean.shape
    t = np.arange(T_total) / fs

    if band_defs is None:
        band_labels = [f"Band {k+1}" for k in range(K)]
    else:
        band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma", "Higher"]
        if len(band_defs) == len(band_names):
            band_labels = [
                f"{name}"
                for (low, high), name in zip(band_defs, band_names)
            ]
        else:
            band_labels = [
                f"Band {k+1} ({low:g}-{high:g} Hz)"
                for k, (low, high) in enumerate(band_defs)
            ]

    fig = plt.figure(figsize=figsize)

    # 2 rows only: main plot + heatmap
    gs = GridSpec(
        2, 2,
        width_ratios=[40, 1],
        height_ratios=[1.2, 1.6],
        hspace=0.35,
        wspace=0.08,
    )

    x_min = t[0]
    x_max = t[-1]
    dt = 1.0 / fs
    extent = [t[0] - 0.5 * dt, t[-1] + 0.5 * dt, -0.5, K - 0.5]

    # 1. Signals
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x, linewidth=1.5, alpha=0.3, label="Input noisy signal", color="blue")

    if include_final_yhat:
        ax1.plot(t, y_hat, linewidth=1.5, linestyle="--", label="Final denoised signal", color="orange")

    if include_clean_target and y is not None:
        ax1.plot(t, y, linewidth=1.2, alpha=0.9, label="Clean target signal", color="green")

    # ax1.set_title("Signal Reconstruction")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim(x_min, x_max)
    ax1.legend(fontsize = 10)
    ax1.grid(True, alpha=0.25)

    # 2. Routing heatmap
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])

    im = ax2.imshow(
        routing_mean,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )
    # ax2.set_title("Routing Gate Heatmap")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Band")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(-0.5, K - 0.5)
    ax2.set_yticks(np.arange(K))
    ax2.set_yticklabels(band_labels)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Gate value")

    ax_dummy = fig.add_subplot(gs[0, 1])
    ax_dummy.axis("off")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



@torch.no_grad()
def visualize_checkpoint_routing(
    ckpt_path,
    eeg_array,
    eog_array=None,
    emg_array=None,
    noise_type="EOG",
    test_snr_level=-7,
    batch_size=32,
    num_workers=0,
    seed=42,
    return_unstandardized=False,
    device=None,
    num_segments=3,
    sample_index_in_batch=0,
    save_path=None,
    show=True,
):
    """
    End-to-end helper:
    - load model
    - build test loader at a chosen SNR
    - concatenate several consecutive windows
    - plot routing-related tensors
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion, cfg, ckpt = load_model_and_criterion_from_ckpt(ckpt_path, device)

    _, _, test_loader = create_benchmark_dataloaders(
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=noise_type,
        train_snr_levels=[test_snr_level],
        val_snr_levels=[test_snr_level],
        test_snr_levels=[test_snr_level],
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        return_unstandardized=return_unstandardized,
    )

    vis_dict = collect_consecutive_visualization_segments(
        model=model,
        loader=test_loader,
        device=device,
        num_segments=num_segments,
        sample_index_in_batch=sample_index_in_batch,
    )

    plot_routing_visualization(
        vis_dict=vis_dict,
        fs=cfg.fs,
        band_defs=cfg.band_defs,
        save_path=save_path,
        show=show,
        include_clean_target=True,
        include_final_yhat=True,
        
    )

    return vis_dict


def load_model_and_criterion_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_dict = ckpt["config"]
    cfg = BRAANetV2Config(**cfg_dict)

    model = BRAANetV2(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = BRAALossV2(cfg, model.band_decomp)
    return model, criterion, cfg, ckpt


def evaluate_checkpoint_one_snr(
    ckpt_path,
    eeg_array,
    eog_array=None,
    emg_array=None,
    noise_type="EOG",
    test_snr_level=-7,
    batch_size=32,
    num_workers=0,
    seed=42,
    return_unstandardized=False,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, criterion, cfg, ckpt = load_model_and_criterion_from_ckpt(ckpt_path, device)

    _, _, test_loader = create_benchmark_dataloaders(
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=noise_type,
        train_snr_levels=[test_snr_level],
        val_snr_levels=[test_snr_level],
        test_snr_levels=[test_snr_level],
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        return_unstandardized=return_unstandardized,
    )

    metrics = evaluate(
        model=model,
        criterion=criterion,
        loader=test_loader,
        device=device,
        by_snr=True,
        fs=cfg.fs,
    )

    return metrics


def evaluate_checkpoint_across_snrs(
    ckpt_path,
    eeg_array,
    eog_array=None,
    emg_array=None,
    noise_type="EOG",
    snr_levels=None,
    batch_size=32,
    num_workers=0,
    seed=42,
    return_unstandardized=False,
    device=None,
    save_json_path=None,
):
    if snr_levels is None:
        snr_levels = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}
    rows = []

    for snr in snr_levels:
        metrics = evaluate_checkpoint_one_snr(
            ckpt_path=ckpt_path,
            eeg_array=eeg_array,
            eog_array=eog_array,
            emg_array=emg_array,
            noise_type=noise_type,
            test_snr_level=snr,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            return_unstandardized=return_unstandardized,
            device=device,
        )

        all_results[str(snr)] = metrics

        rows.append({
            "snr_db": snr,
            "loss": metrics["loss"],
            "rrmse_temporal": metrics["rrmse_temporal"],
            "rrmse_spectral": metrics["rrmse_spectral"],
            "cc": metrics["cc"],
            "snr_improvement_db": metrics["snr_improvement_db"],
            **{
                key: value
                for key, value in metrics.items()
                if key.startswith("band_power_absolute_error_")
                or key in {
                    "psd_similarity",
                    "alpha_peak_error_hz",
                    "alpha_peak_preservation_rate",
                }
            },
        })

        print(f"SNR {snr:>2} dB | "
              f"RRMSE_t={metrics['rrmse_temporal']:.4f} | "
              f"RRMSE_s={metrics['rrmse_spectral']:.4f} | "
              f"CC={metrics['cc']:.4f} | "
              f"SNRimp={metrics['snr_improvement_db']:.4f}")

    # Pandas is only needed by the BRAANet report path. Keeping it lazy lets
    # benchmark-only evaluation (including GAN) run without that dependency.
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("snr_db").reset_index(drop=True)

    if save_json_path is not None:
        os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    return df, all_results



if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Evaluate BRAANetV2 checkpoint across SNR levels")
    args.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint to evaluate")
    args.add_argument("--eeg_array_path", type=str, default="data/EEG_all_epochs.npy", help="Path to EEG epochs numpy array")
    args.add_argument("--eog_array_path", type=str, default="data/EOG_all_epochs.npy", help="Path to EOG epochs numpy array (for EOG or MIXED noise)")
    args.add_argument("--emg_array_path", type=str, default="data/EMG_all_epochs.npy", help="Path to EMG epochs numpy array (for EMG or MIXED noise)")
    args.add_argument("--noise_type", type=str, choices=["EOG", "EMG", "MIXED"], default="EMG", help="Type of noise to evaluate against")
    args.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    args.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args.add_argument("--save_json_path", type=str, default="./test_metrics_by_snr.json", help="Path to save the detailed metrics by SNR in JSON format")
    args.add_argument("--test_snr_level", type=int, default=-2, help="SNR level to use for routing visualization")
    args.add_argument(
        "--benchmark_model_name",
        type=str,
        default=None,
        choices=["deepseparator", "fcnn", "rnnlstm", "simplecnn", "complexcnn", "det", "lrr_unet", "gan"],
        help="Evaluate a benchmark checkpoint instead of BRAANetV2 (includes GAN).",
    )
    args = args.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg_array = np.load(args.eeg_array_path)
    eog_array = np.load(args.eog_array_path)   # for EOG or MIXED
    emg_array = np.load(args.emg_array_path)   # for EMG or MIXED

    ckpt_path = args.ckpt_path
    noise_type = args.noise_type

    if args.benchmark_model_name is not None:
        _, raw_results = evaluate_benchmark_checkpoint_across_snrs(
            ckpt_path=ckpt_path,
            model_name=args.benchmark_model_name,
            eeg_array=eeg_array,
            eog_array=eog_array,
            emg_array=emg_array,
            noise_type=noise_type,
            snr_levels=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            device=device,
            save_json_path=args.save_json_path,
        )
        print("\n========== BENCHMARK CROSS-SNR RESULTS ==========")
        print(json.dumps(raw_results, indent=2))
        raise SystemExit(0)

    # Evaluate across SNRs and print summary table
    df_report, raw_results = evaluate_checkpoint_across_snrs(
        ckpt_path=ckpt_path,
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=args.noise_type,
        snr_levels=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
        save_json_path=args.save_json_path,
    )

    print("\nSummary Table:")
    print(df_report)


    # Visualize routing for a chosen SNR level
    vis_dict = visualize_checkpoint_routing(
        ckpt_path=ckpt_path,
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=noise_type,
        test_snr_level=-2,
        batch_size=args.batch_size,
        num_workers=0,
        seed=42,
        return_unstandardized=False,
        device=device,
        num_segments=2,                # 3 x 2s = 6 seconds
        sample_index_in_batch=25,
        save_path=f"./figures/routing_vis_{noise_type}.png",
        show=True,
    )
