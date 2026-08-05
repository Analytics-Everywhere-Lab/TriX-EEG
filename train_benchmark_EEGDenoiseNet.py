import os
import json
import time
import random
from dataclasses import dataclass, asdict
import argparse
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from load_dataset import create_benchmark_dataloaders
from benchmark_models.dataset_benchmark.dataset_benchmark_models import fcNN, RNN_lstm, simple_CNN, Complex_CNN
from benchmark_models.EEGDnet.Network_structure import DeT
from benchmark_models.LRR_Unet.LRR_Unet import LRRUNet
from benchmark_models.deep_separator.network import DeepSeparator
from benchmark_models.GAN.model import EEGDeNoiseGAN
from benchmark_models.GAN.training import StateBundle, train_gan_one_epoch
from torch.nn import functional as F
from torchinfo import summary
from warnings import filterwarnings
from metrics import batch_frequency_preservation
filterwarnings('ignore')

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================

def parse_snr_levels(x: Optional[Any]) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, str):
        x = x.strip()
        if len(x) == 0:
            return None
        return [int(v.strip()) for v in x.split(',')]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in x]
    raise TypeError(f"Unsupported SNR format: {type(x)}")


def ensure_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected [B,T] or [B,1,T], got {tuple(x.shape)}")


def ensure_bt(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        if x.size(1) == 1:
            return x.squeeze(1)
        return x
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected [B,T] or [B,1,T], got {tuple(x.shape)}")


def rms(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + eps)


def compute_psd(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = ensure_bt(x)
    fft = torch.fft.rfft(x, dim=-1)
    psd = (fft.real ** 2 + fft.imag ** 2) / x.size(-1)
    return psd + eps


class EvalTracker:
    def __init__(self):
        self.total = 0
        self.loss = 0.0
        self.rrmse_temporal = 0.0
        self.rrmse_spectral = 0.0
        self.cc = 0.0
        self.snr_improvement_db = 0.0

    def update(self, loss, rrmse_t, rrmse_s, cc, snr_imp, n):
        self.total += int(n)
        self.loss += float(loss) * n
        self.rrmse_temporal += float(rrmse_t) * n
        self.rrmse_spectral += float(rrmse_s) * n
        self.cc += float(cc) * n
        self.snr_improvement_db += float(snr_imp) * n

    def as_dict(self) -> Dict[str, float]:
        denom = max(self.total, 1)
        return {
            "loss": self.loss / denom,
            "rrmse_temporal": self.rrmse_temporal / denom,
            "rrmse_spectral": self.rrmse_spectral / denom,
            "cc": self.cc / denom,
            "snr_improvement_db": self.snr_improvement_db / denom,
        }


# ============================================================
# Batch parsing
# ============================================================

def parse_batch(batch):
    snr = None
    indicator = None

    if isinstance(batch, dict):
        noisy = None
        clean = None

        noisy_keys = ["noisy", "x", "contaminated", "input", "mixed", "artifacted"]
        clean_keys = ["clean", "target", "y", "eeg", "ground_truth", "gt"]
        indicator_keys = ["indicator", "ind", "mask", "label"]

        for k in noisy_keys:
            if k in batch:
                noisy = batch[k]
                break

        for k in clean_keys:
            if k in batch:
                clean = batch[k]
                break

        for k in indicator_keys:
            if k in batch:
                indicator = batch[k]
                break

        if "snr" in batch:
            snr = batch["snr"]
        elif "snr_db" in batch:
            snr = batch["snr_db"]

        if noisy is None or clean is None:
            raise KeyError(f"Could not parse batch dict keys. Found keys={list(batch.keys())}")

    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must have at least (noisy, clean)")

        noisy = batch[0]

        if len(batch) >= 3 and torch.is_tensor(batch[1]) and torch.is_tensor(batch[2]):
            # DeepSeparator-style dataset: (input, indicator, target)
            indicator = batch[1]
            clean = batch[2]
            if len(batch) >= 4 and torch.is_tensor(batch[3]):
                snr = batch[3]
        else:
            clean = batch[1]
            if len(batch) >= 3 and torch.is_tensor(batch[2]):
                snr = batch[2]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    return ensure_bct(noisy).float(), ensure_bct(clean).float(), snr, indicator


def gan_batch_to_device(batch, device):
    """Adapt an EEGDenoiseNet batch for the shared GAN training loop."""
    noisy, clean, snr, _ = parse_batch(batch)
    return (
        noisy.to(device, non_blocking=True),
        clean.to(device, non_blocking=True),
        snr,
    )


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def batch_rrmse_temporal(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred = ensure_bt(pred)
    target = ensure_bt(target)
    num = rms(pred - target, dim=-1, eps=eps)
    den = rms(target, dim=-1, eps=eps)
    return num / (den + eps)


@torch.no_grad()
def batch_rrmse_spectral(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    psd_pred = compute_psd(pred, eps=eps)
    psd_tgt = compute_psd(target, eps=eps)
    num = rms(psd_pred - psd_tgt, dim=-1, eps=eps)
    den = rms(psd_tgt, dim=-1, eps=eps)
    return num / (den + eps)


@torch.no_grad()
def batch_cc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred = ensure_bt(pred)
    target = ensure_bt(target)

    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    tgt_c = target - target.mean(dim=-1, keepdim=True)

    cov = torch.mean(pred_c * tgt_c, dim=-1)
    var_p = torch.mean(pred_c ** 2, dim=-1)
    var_t = torch.mean(tgt_c ** 2, dim=-1)
    return cov / (torch.sqrt(var_p * var_t) + eps)


@torch.no_grad()
def batch_snr_db(clean: torch.Tensor, noise: torch.Tensor, eps: float = 1e-8):
    clean = ensure_bt(clean)
    noise = ensure_bt(noise)
    p_signal = torch.mean(clean ** 2, dim=-1)
    p_noise = torch.mean(noise ** 2, dim=-1)
    return 10.0 * torch.log10((p_signal + eps) / (p_noise + eps))


@torch.no_grad()
def batch_snr_improvement(noisy: torch.Tensor, pred: torch.Tensor, clean: torch.Tensor):
    snr_in = batch_snr_db(clean, noisy - clean)
    snr_out = batch_snr_db(clean, pred - clean)
    return snr_out - snr_in


# ============================================================
# Model wrappers
# ============================================================

class BenchmarkModelAdapter(nn.Module):
    """
    Wraps each benchmark model so the rest of the pipeline can always do:
        out = model(noisy_bct)
        pred = out["y_hat"]   # [B,1,T]
    """
    def __init__(
        self,
        model_name: str,
        signal_length: int = 512,
        input_channels: int = 1,
        deepseparator_indicator: str = "noisy",
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.signal_length = signal_length
        self.input_channels = input_channels
        self.deepseparator_indicator = deepseparator_indicator

        if self.model_name == "deepseparator":
            self.model = DeepSeparator()
        elif self.model_name == "fcnn":
            self.model = fcNN(signal_length, input_channels=input_channels)
        elif self.model_name == "rnnlstm":
            self.model = RNN_lstm(signal_length, input_channels=input_channels)
        elif self.model_name == "simplecnn":
            self.model = simple_CNN(signal_length, input_channels=input_channels)
        elif self.model_name == "complexcnn":
            self.model = Complex_CNN(signal_length, input_channels=input_channels)
        elif self.model_name == "det":
            self.model = DeT(seq_len=signal_length, patch_len=64, depth=6, heads=8, dropout=0.1, emb_dropout=0.1)
        elif self.model_name == "lrr_unet":
            self.model = LRRUNet(stages=3, channels=1, base_ch=32, netn_layers=4, netr_layers=4)
        elif self.model_name == "gan":
            self.model = EEGDeNoiseGAN(input_channels=input_channels)
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def _build_indicator(self, x_bt: torch.Tensor, indicator: Optional[torch.Tensor] = None) -> torch.Tensor:
        if indicator is not None:
            indicator = indicator.float().to(x_bt.device)
            if indicator.dim() == 1:
                indicator = indicator.unsqueeze(1)
            elif indicator.dim() == 3 and indicator.size(-1) == 1:
                indicator = indicator.squeeze(-1)
            if indicator.dim() != 2:
                raise ValueError(f"DeepSeparator indicator must be [B] or [B,1], got {tuple(indicator.shape)}")
            if indicator.size(1) not in (1, x_bt.size(1) if x_bt.dim() == 3 else 1):
                raise ValueError(f"DeepSeparator indicator second dim must be 1 or C, got {tuple(indicator.shape)}")
            return indicator

        if self.deepseparator_indicator == "zeros":
            shape = x_bt.shape[:-1] if x_bt.dim() == 3 else (x_bt.size(0), 1)
            return torch.zeros(*shape, device=x_bt.device, dtype=x_bt.dtype)
        if self.deepseparator_indicator == "ones":
            shape = x_bt.shape[:-1] if x_bt.dim() == 3 else (x_bt.size(0), 1)
            return torch.ones(*shape, device=x_bt.device, dtype=x_bt.dtype)
        if self.deepseparator_indicator == "abs_noisy":
            return x_bt.abs().mean(dim=-1, keepdim=True)
        if self.deepseparator_indicator == "noisy":
            return x_bt.mean(dim=-1, keepdim=True)
        raise ValueError(
            f"Unsupported deepseparator_indicator={self.deepseparator_indicator}. "
            "Use one of ['zeros', 'ones', 'noisy', 'abs_noisy']."
        )

    def forward(self, x_bct: torch.Tensor, indicator: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x_bct = ensure_bct(x_bct)
        batch_size, channels, signal_length = x_bct.shape
        flatten_channels = channels != self.input_channels and self.input_channels == 1
        if channels != self.input_channels and not flatten_channels:
            raise ValueError(f"Expected C={self.input_channels}, got C={channels}")
        if flatten_channels:
            x_bct_model = x_bct.reshape(batch_size * channels, 1, signal_length)
            if indicator is not None and indicator.dim() >= 2:
                if indicator.size(1) == 1:
                    indicator = indicator.expand(batch_size, channels, *indicator.shape[2:])
                indicator = indicator.reshape(batch_size * channels, *indicator.shape[2:])
        else:
            x_bct_model = x_bct

        def restore(y: torch.Tensor) -> torch.Tensor:
            y = ensure_bct(y)
            if flatten_channels:
                y = y.reshape(batch_size, channels, signal_length)
            return y

        x_bct = x_bct_model
        x_bt = ensure_bt(x_bct)

        if self.model_name == "deepseparator":
            indicator = self._build_indicator(x_bt, indicator=indicator)
            y_hat = self.model(x_bt, indicator)
            return {"y_hat": restore(y_hat)}

        if self.model_name in {"fcnn", "det"}:
            y_hat = self.model(x_bt)
            return {"y_hat": restore(y_hat)}

        if self.model_name in {"rnnlstm", "simplecnn", "complexcnn", "novelcnn"}:
            x_seq = x_bt.unsqueeze(-1) if x_bt.dim() == 2 else x_bt
            y_hat = self.model(x_seq)
            return {"y_hat": restore(y_hat)}

        if self.model_name == "lrr_unet":
            out = self.model(x_bct)
            if not isinstance(out, dict) or "denoised" not in out:
                raise ValueError("LRRUNet is expected to return a dict with key 'denoised'.")
            return {
                "y_hat": restore(out["denoised"]),
                **out,
            }

        if self.model_name == "gan":
            return {"y_hat": restore(self.model(x_bct))}
        
        raise RuntimeError("Unreachable branch in BenchmarkModelAdapter.forward")


# ============================================================
# Losses
# ============================================================

class GenericDenoisingLoss(nn.Module):
    def __init__(self, loss_name: str = "mse", l1_weight: float = 0.0, spectral_weight: float = 0.0):
        super().__init__()
        self.loss_name = loss_name.lower()
        self.l1_weight = float(l1_weight)
        self.spectral_weight = float(spectral_weight)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, model_out: Dict[str, torch.Tensor], clean: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = ensure_bct(model_out["y_hat"])
        clean = ensure_bct(clean)

        if self.loss_name == "mse":
            base = self.mse(pred, clean)
        elif self.loss_name == "l1":
            base = self.l1(pred, clean)
        elif self.loss_name == "smoothl1":
            base = F.smooth_l1_loss(pred, clean)
        elif self.loss_name == "mse_l1":
            base = self.mse(pred, clean) + self.l1_weight * self.l1(pred, clean)
        else:
            raise ValueError(f"Unsupported loss_name: {self.loss_name}")

        spec = torch.tensor(0.0, device=pred.device)
        if self.spectral_weight > 0:
            spec = self.mse(compute_psd(pred), compute_psd(clean))

        total = base + self.spectral_weight * spec
        return {
            "loss": total,
            "loss_base": base.detach(),
            "loss_spectral": spec.detach(),
        }


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, criterion, loader, optimizer, device, grad_clip=None):
    model.train()
    tracker = EvalTracker()
    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        noisy, clean, _, indicator = parse_batch(batch)
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if indicator is not None:
            indicator = indicator.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        model_out = model(noisy, indicator=indicator)
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
            "snr": f"{snr_imp:.2f}",
        })

    return tracker.as_dict()


@torch.no_grad()
def evaluate(model, criterion, loader, device, by_snr=False, fs=256.0):
    model.eval()
    tracker = EvalTracker()
    snr_buckets = {}
    frequency_totals = {}
    num_frequency_samples = 0
    pbar = tqdm(loader, desc="Eval", leave=False)

    for batch in pbar:
        noisy, clean, snr, indicator = parse_batch(batch)
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        if indicator is not None:
            indicator = indicator.to(device, non_blocking=True)

        model_out = model(noisy, indicator=indicator)
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
            "snr": f"{snr_imp_all.mean().item():.2f}",
        })

        if by_snr and snr is not None:
            if snr.dim() > 1:
                snr = snr.view(-1)
            for i, s in enumerate(snr.cpu().numpy().tolist()):
                s_key = round(float(s))
                snr_buckets.setdefault(s_key, {
                    "rrmse_t": [],
                    "rrmse_s": [],
                    "cc": [],
                    "snr_imp": [],
                    **{name: [] for name in frequency_all},
                })
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
        for s in sorted(snr_buckets):
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
# Checkpointing
# ============================================================

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, cfg, history):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_score": best_score,
        "config": asdict(cfg),
        "history": history,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ============================================================
# Score selection
# ============================================================

def model_selection_score(metrics):
    return metrics["rrmse_temporal"] + metrics["rrmse_spectral"] - metrics["cc"]

def print_model_summary(model, cfg, device):
    print("\n========== MODEL SUMMARY ==========")
    try:
        dummy_input = torch.randn(1, cfg.input_channels, cfg.signal_length).to(device)
        print(summary(
            model,
            input_data=dummy_input,
            device=str(device),
        ))
    except Exception as e:
        print(f"[WARN] torchinfo.summary failed: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total params:         {total_params:,}")
    print(f"Trainable params:     {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    print("===================================\n")

# ============================================================
# Config
# ============================================================

@dataclass
class BenchmarkTrainConfig:
    model_name: str = "fcnn"
    eeg_npy: str = "data/EEG_all_epochs.npy"
    eog_npy: Optional[str] = "data/EOG_all_epochs.npy"
    emg_npy: Optional[str] = "data/EMG_all_epochs.npy"
    noise_type: str = "EOG"
    train_snr_levels: Optional[Sequence[int]] = (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2)
    val_snr_levels: Optional[Sequence[int]] = (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2)
    test_snr_levels: Optional[Sequence[int]] = (-7, -6, -5, -4, -3, -2, -1, 0, 1, 2)
    return_unstandardized: bool = False

    epochs: int = 10
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    cpu: bool = False

    signal_length: int = 512
    input_channels: int = 1
    eval_by_snr: bool = False
    save_dir: str = "./runs/benchmark_compare/fcnn"

    # generic loss
    loss_name: str = "mse"
    l1_weight: float = 0.0
    spectral_weight: float = 0.0

    # DeT settings
    det_patch_len: int = 64
    det_depth: int = 6
    det_heads: int = 8
    det_dropout: float = 0.1
    det_emb_dropout: float = 0.1

    # LRR-UNet settings
    lrr_stages: int = 3
    lrr_base_ch: int = 32
    lrr_netn_layers: int = 4
    lrr_netr_layers: int = 4

    # DeepSeparator setting
    deepseparator_indicator: str = "zeros"

    # EEG-DeNoiseGAN settings
    gan_hidden_dim: int = 50
    gan_num_layers: int = 2
    gan_tanh_output: bool = False
    gan_adversarial_weight: float = 1.0
    gan_reconstruction_weight: float = 1.0


# ============================================================
# Builders
# ============================================================

def build_model(cfg: BenchmarkTrainConfig) -> nn.Module:
    model = BenchmarkModelAdapter(
        model_name=cfg.model_name,
        signal_length=cfg.signal_length,
        input_channels=cfg.input_channels,
        deepseparator_indicator=cfg.deepseparator_indicator,
    )

    # overwrite submodel hyperparameters where needed
    if cfg.model_name.lower() == "det":
        model.model = DeT(
            seq_len=cfg.signal_length,
            patch_len=cfg.det_patch_len,
            depth=cfg.det_depth,
            heads=cfg.det_heads,
            dropout=cfg.det_dropout,
            emb_dropout=cfg.det_emb_dropout,
        )
    elif cfg.model_name.lower() == "lrr_unet":
        model.model = LRRUNet(
            stages=cfg.lrr_stages,
            channels=cfg.input_channels,
            base_ch=cfg.lrr_base_ch,
            netn_layers=cfg.lrr_netn_layers,
            netr_layers=cfg.lrr_netr_layers,
        )
    elif cfg.model_name.lower() == "gan":
        model.model = EEGDeNoiseGAN(
            input_channels=cfg.input_channels,
            hidden_dim=cfg.gan_hidden_dim,
            num_layers=cfg.gan_num_layers,
            tanh_output=cfg.gan_tanh_output,
        )
    return model


def create_loaders_from_cfg(cfg: BenchmarkTrainConfig):
    eeg_array = np.load(cfg.eeg_npy)
    eog_array = np.load(cfg.eog_npy) if cfg.eog_npy else None
    emg_array = np.load(cfg.emg_npy) if cfg.emg_npy else None

    train_snr_levels = parse_snr_levels(cfg.train_snr_levels)
    val_snr_levels = parse_snr_levels(cfg.val_snr_levels)
    test_snr_levels = parse_snr_levels(cfg.test_snr_levels)

    loaders = create_benchmark_dataloaders(
        eeg_array=eeg_array,
        eog_array=eog_array,
        emg_array=emg_array,
        noise_type=cfg.noise_type,
        train_snr_levels=train_snr_levels,
        val_snr_levels=val_snr_levels,
        test_snr_levels=test_snr_levels,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        return_unstandardized=cfg.return_unstandardized,
    )
    return loaders, eeg_array, eog_array, emg_array


# ============================================================
# Main training
# ============================================================

def train_and_evaluate(cfg: BenchmarkTrainConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    os.makedirs(cfg.save_dir, exist_ok=True)

    (train_loader, val_loader, test_loader), eeg_array, eog_array, emg_array = create_loaders_from_cfg(cfg)

    print(f"Create train loader with SNR levels: {parse_snr_levels(cfg.train_snr_levels)}")
    print(f"Create val loader with SNR levels: {parse_snr_levels(cfg.val_snr_levels)}")
    print(f"Create test loader with SNR levels: {parse_snr_levels(cfg.test_snr_levels)}")

    model = build_model(cfg).to(device)
    print_model_summary(model, cfg, device)
    criterion = GenericDenoisingLoss(
        loss_name=cfg.loss_name,
        l1_weight=cfg.l1_weight,
        spectral_weight=cfg.spectral_weight,
    )

    if cfg.model_name.lower() == "gan":
        # The notebook uses independent Adam optimizers with GAN betas.
        generator_optimizer = torch.optim.Adam(
            model.model.generator.parameters(), lr=cfg.lr, betas=(0.5, 0.999)
        )
        discriminator_optimizer = torch.optim.Adam(
            model.model.discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.999)
        )
        generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            generator_optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
        )
        discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            discriminator_optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
        )
        optimizer = StateBundle(
            generator=generator_optimizer, discriminator=discriminator_optimizer
        )
        scheduler = StateBundle(
            generator=generator_scheduler, discriminator=discriminator_scheduler
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    history = {"train": [], "val": [], "test": []}
    best_score = float("inf")
    best_path = os.path.join(cfg.save_dir, "best.pt")
    last_path = os.path.join(cfg.save_dir, "last.pt")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        if cfg.model_name.lower() == "gan":
            train_metrics = train_gan_one_epoch(
                model,
                train_loader,
                generator_optimizer,
                discriminator_optimizer,
                device,
                batch_adapter=gan_batch_to_device,
                grad_clip=cfg.grad_clip,
                adversarial_weight=cfg.gan_adversarial_weight,
                reconstruction_weight=cfg.gan_reconstruction_weight,
            )
        else:
            train_metrics = train_one_epoch(model, criterion, train_loader, optimizer, device, grad_clip=cfg.grad_clip)
        val_metrics = evaluate(model, criterion, val_loader, device, by_snr=cfg.eval_by_snr)
        test_metrics = evaluate(model, criterion, test_loader, device, by_snr=cfg.eval_by_snr)
        if cfg.model_name.lower() == "gan":
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

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] "
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

    load_checkpoint(best_path, model, map_location=device)
    final_test_metrics = evaluate(model, criterion, test_loader, device, by_snr=cfg.eval_by_snr)

    with open(os.path.join(cfg.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(cfg.save_dir, "final_test_metrics.json"), "w") as f:
        json.dump(final_test_metrics, f, indent=2)

    print("\n========== TEST ==========")
    print(json.dumps(final_test_metrics, indent=2))
    print(f"\nSaved to: {cfg.save_dir}")
    print(f"Best checkpoint: {best_path}")

    return {
        "config": cfg,
        "best_path": best_path,
        "history": history,
        "final_test_metrics": final_test_metrics,
        "arrays": {
            "eeg": eeg_array,
            "eog": eog_array,
            "emg": emg_array,
        },
    }


# ============================================================
# Cross-SNR evaluation for a saved checkpoint
# ============================================================

@torch.no_grad()
def evaluate_checkpoint_one_snr(
    ckpt_path: str,
    model_name: str,
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
    signal_length=512,
    loss_name="mse",
    l1_weight=0.0,
    spectral_weight=0.0,
    deepseparator_indicator="noisy",
    det_patch_len=64,
    det_depth=6,
    det_heads=8,
    det_dropout=0.1,
    det_emb_dropout=0.1,
    lrr_stages=3,
    lrr_base_ch=32,
    lrr_netn_layers=4,
    lrr_netr_layers=4,
    gan_hidden_dim=50,
    gan_num_layers=2,
    gan_tanh_output=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = BenchmarkTrainConfig(
        model_name=model_name,
        signal_length=signal_length,
        deepseparator_indicator=deepseparator_indicator,
        det_patch_len=det_patch_len,
        det_depth=det_depth,
        det_heads=det_heads,
        det_dropout=det_dropout,
        det_emb_dropout=det_emb_dropout,
        lrr_stages=lrr_stages,
        lrr_base_ch=lrr_base_ch,
        lrr_netn_layers=lrr_netn_layers,
        lrr_netr_layers=lrr_netr_layers,
        gan_hidden_dim=gan_hidden_dim,
        gan_num_layers=gan_num_layers,
        gan_tanh_output=gan_tanh_output,
    )
    # Prefer architecture settings stored with the checkpoint. This is
    # essential for non-default GAN hidden sizes/layer counts and is harmless
    # for older checkpoints without a config mapping.
    try:
        checkpoint_metadata = torch.load(
            ckpt_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint_metadata = torch.load(ckpt_path, map_location="cpu")
    for key, value in checkpoint_metadata.get("config", {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.model_name = model_name
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    criterion = GenericDenoisingLoss(loss_name=loss_name, l1_weight=l1_weight, spectral_weight=spectral_weight)

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

    return evaluate(model, criterion, test_loader, device, by_snr=True)


@torch.no_grad()
def evaluate_checkpoint_across_snrs(
    ckpt_path: str,
    model_name: str,
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
    **kwargs,
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
            model_name=model_name,
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
            **kwargs,
        )
        all_results[str(snr)] = metrics
        rows.append({
            "snr_db": snr,
            "loss": metrics["loss"],
            "rrmse_temporal": metrics["rrmse_temporal"],
            "rrmse_spectral": metrics["rrmse_spectral"],
            "cc": metrics["cc"],
            "snr_improvement_db": metrics["snr_improvement_db"],
        })
        print(
            f"SNR {snr:>2} dB | "
            f"RRMSE_t={metrics['rrmse_temporal']:.4f} | "
            f"RRMSE_s={metrics['rrmse_spectral']:.4f} | "
            f"CC={metrics['cc']:.4f} | "
            f"SNRimp={metrics['snr_improvement_db']:.4f}"
        )

    if save_json_path is not None:
        with open(save_json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    return rows, all_results

# Argparse
# ============================================================
# Argparse CLI
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in {"yes", "true", "t", "1", "y"}:
        return True
    if v in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def snr_arg(v):
    """
    Examples:
        "-7,-6,-5,-4,-3,-2,-1,0,1,2"
        "all"
        ""
    """
    if v is None:
        return None
    v = str(v).strip().lower()
    if v == "" or v == "none":
        return None
    if v == "all":
        return [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    return [int(x.strip()) for x in v.split(",")]


def add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["deepseparator", "fcnn", "rnnlstm", "simplecnn", "complexcnn", "det", "lrr_unet", "gan", 'novelcnn'])

    parser.add_argument("--eeg_npy", type=str, default="data/EEG_all_epochs.npy")
    parser.add_argument("--eog_npy", type=str, default="data/EOG_all_epochs.npy")
    parser.add_argument("--emg_npy", type=str, default="data/EMG_all_epochs.npy")
    parser.add_argument("--noise_type", type=str, default="EOG", choices=["EOG", "EMG", "MIXED"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--signal_length", type=int, default=512)
    parser.add_argument("--return_unstandardized", type=str2bool, default=False)

    # generic loss
    parser.add_argument("--loss_name", type=str, default="mse",
                        choices=["mse", "l1", "smoothl1", "mse_l1"])
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--spectral_weight", type=float, default=0.0)

    # DeepSeparator
    parser.add_argument("--deepseparator_indicator", type=str, default="zeros",
                        choices=["zeros", "ones", "noisy", "abs_noisy"])

    # DeT
    parser.add_argument("--det_patch_len", type=int, default=64)
    parser.add_argument("--det_depth", type=int, default=6)
    parser.add_argument("--det_heads", type=int, default=8)
    parser.add_argument("--det_dropout", type=float, default=0.1)
    parser.add_argument("--det_emb_dropout", type=float, default=0.1)

    # LRR-UNet
    parser.add_argument("--lrr_stages", type=int, default=3)
    parser.add_argument("--lrr_base_ch", type=int, default=32)
    parser.add_argument("--lrr_netn_layers", type=int, default=4)
    parser.add_argument("--lrr_netr_layers", type=int, default=4)

    # EEG-DeNoiseGAN
    parser.add_argument("--gan_hidden_dim", type=int, default=50)
    parser.add_argument("--gan_num_layers", type=int, default=2)
    parser.add_argument("--gan_tanh_output", type=str2bool, default=False)
    parser.add_argument("--gan_adversarial_weight", type=float, default=1.0)
    parser.add_argument("--gan_reconstruction_weight", type=float, default=1.0)


def build_train_config_from_args(args) -> BenchmarkTrainConfig:
    return BenchmarkTrainConfig(
        model_name=args.model_name,
        eeg_npy=args.eeg_npy,
        eog_npy=args.eog_npy,
        emg_npy=args.emg_npy,
        noise_type=args.noise_type,

        train_snr_levels=args.train_snr_levels,
        val_snr_levels=args.val_snr_levels,
        test_snr_levels=args.test_snr_levels,
        return_unstandardized=args.return_unstandardized,

        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        cpu=args.cpu,

        signal_length=args.signal_length,
        eval_by_snr=args.eval_by_snr,
        save_dir=args.save_dir,

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


def parser_main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate EEG denoising benchmark models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------------------------
    # train
    # ---------------------------
    train_parser = subparsers.add_parser("train", help="Run train_and_evaluate")
    add_shared_args(train_parser)

    train_parser.add_argument("--train_snr_levels", type=snr_arg, default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
                              help='Comma-separated SNRs, e.g. "-7,-6,-5,-4,-3,-2,-1,0,1,2" or "all"')
    train_parser.add_argument("--val_snr_levels", type=snr_arg, default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2])
    train_parser.add_argument("--test_snr_levels", type=snr_arg, default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2])

    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--lr_min", type=float, default=1e-6)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--eval_by_snr", type=str2bool, default=False)
    train_parser.add_argument("--save_dir", type=str, required=True)

    # ---------------------------
    # eval_ckpt
    # ---------------------------
    eval_parser = subparsers.add_parser("eval_ckpt", help="Run evaluate_checkpoint_across_snrs")
    add_shared_args(eval_parser)

    eval_parser.add_argument("--ckpt_path", type=str, required=True)
    eval_parser.add_argument("--snr_levels", type=snr_arg, default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2],
                             help='Comma-separated SNRs, e.g. "-7,-6,-5,-4,-3,-2,-1,0,1,2" or "all"')
    eval_parser.add_argument("--save_json_path", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = parser_main()
    args = parser.parse_args()

    if args.command == "train":
        cfg = build_train_config_from_args(args)
        train_and_evaluate(cfg)

    elif args.command == "eval_ckpt":
        set_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

        eeg_array = np.load(args.eeg_npy)
        eog_array = np.load(args.eog_npy) if args.eog_npy else None
        emg_array = np.load(args.emg_npy) if args.emg_npy else None

        rows, all_results = evaluate_checkpoint_across_snrs(
            ckpt_path=args.ckpt_path,
            model_name=args.model_name,
            eeg_array=eeg_array,
            eog_array=eog_array,
            emg_array=emg_array,
            noise_type=args.noise_type,
            snr_levels=args.snr_levels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            return_unstandardized=args.return_unstandardized,
            device=device,
            save_json_path=args.save_json_path,

            signal_length=args.signal_length,
            loss_name=args.loss_name,
            l1_weight=args.l1_weight,
            spectral_weight=args.spectral_weight,
            deepseparator_indicator=args.deepseparator_indicator,

            det_patch_len=args.det_patch_len,
            det_depth=args.det_depth,
            det_heads=args.det_heads,
            det_dropout=args.det_dropout,
            det_emb_dropout=args.det_emb_dropout,

            lrr_stages=args.lrr_stages,
            lrr_base_ch=args.lrr_base_ch,
            lrr_netn_layers=args.lrr_netn_layers,
            lrr_netr_layers=args.lrr_netr_layers,
            gan_hidden_dim=args.gan_hidden_dim,
            gan_num_layers=args.gan_num_layers,
            gan_tanh_output=args.gan_tanh_output,
        )

        print("\n========== CROSS-SNR RESULTS ==========")
        print(json.dumps(all_results, indent=2))
