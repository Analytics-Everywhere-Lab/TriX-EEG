"""Reusable training primitives shared by BRAANet and baseline scripts."""

from __future__ import annotations

import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from tqdm import tqdm

from metrics import (
    batch_cc,
    batch_frequency_preservation,
    batch_rrmse_spectral,
    batch_rrmse_temporal,
    batch_snr_improvement,
)
from utils import EvalTracker, parse_batch

BatchAdapter = Callable[[Any], tuple[torch.Tensor, torch.Tensor, Any]]


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(force_cpu: bool = False) -> torch.device:
    """Select CUDA when available unless CPU execution was requested."""
    return torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")


def move_batch_to_device(
    batch: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Parse a supported batch and move its signals to ``device``."""
    noisy, clean, snr = parse_batch(batch)
    return (
        noisy.to(device, non_blocking=True),
        clean.to(device, non_blocking=True),
        snr,
    )


def prediction_from_output(output: Any) -> torch.Tensor:
    """Return a prediction from either a tensor or a model output mapping."""
    if isinstance(output, Mapping):
        return output["y_hat"]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def model_selection_score(metrics: Mapping[str, float]) -> float:
    """Lower-is-better validation score used by all neural experiments."""
    return (
        metrics["rrmse_temporal"]
        + metrics["rrmse_spectral"]
        - metrics["cc"]
    )


def _update_metrics(
    tracker: EvalTracker,
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    clean: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    tracker.update(
        loss.item(),
        batch_rrmse_temporal(prediction, clean).mean().item(),
        batch_rrmse_spectral(prediction, clean).mean().item(),
        batch_cc(prediction, clean).mean().item(),
        batch_snr_improvement(noisy, prediction, clean).mean().item(),
        noisy.size(0),
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: Callable[[Any, torch.Tensor], Mapping[str, torch.Tensor]],
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
    batch_adapter: BatchAdapter = move_batch_to_device,
    description: str = "Train",
) -> dict[str, float]:
    """Train a model for one epoch and return sample-weighted metrics."""
    model.train()
    tracker = EvalTracker()

    for batch in tqdm(loader, desc=description, leave=False):
        noisy, clean, _ = batch_adapter(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(noisy)
        losses = criterion(output, clean)
        loss = losses["loss"]
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        _update_metrics(
            tracker,
            noisy,
            prediction_from_output(output).detach(),
            clean,
            loss,
        )

    return tracker.as_dict()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: Callable[[Any, torch.Tensor], Mapping[str, torch.Tensor]],
    loader: Any,
    device: torch.device,
    *,
    batch_adapter: BatchAdapter = move_batch_to_device,
    include_frequency: bool = False,
    fs: float = 256.0,
    description: str = "Eval",
) -> dict[str, float]:
    """Evaluate a model with the common time, spectral, CC, and SNR metrics."""
    model.eval()
    tracker = EvalTracker()
    frequency_totals: dict[str, float] = {}
    frequency_samples = 0

    for batch in tqdm(loader, desc=description, leave=False):
        noisy, clean, _ = batch_adapter(batch, device)
        output = model(noisy)
        losses = criterion(output, clean)
        prediction = prediction_from_output(output)
        _update_metrics(tracker, noisy, prediction, clean, losses["loss"])

        if include_frequency:
            values = batch_frequency_preservation(prediction, clean, fs=fs)
            for name, value in values.items():
                frequency_totals[name] = frequency_totals.get(name, 0.0) + value.sum().item()
            frequency_samples += noisy.size(0)

    metrics = tracker.as_dict()
    if include_frequency:
        denominator = max(frequency_samples, 1)
        metrics.update(
            {name: total / denominator for name, total in frequency_totals.items()}
        )
    return metrics


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_score: float,
    config: Any,
    history: Mapping[str, Any],
) -> None:
    """Persist a checkpoint using one stable format across all models."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config) if is_dataclass(config) else dict(vars(config))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_score": best_score,
            "config": config_dict,
            "history": history,
        },
        path,
    )


def load_model_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
) -> Mapping[str, Any]:
    """Load a project checkpoint while supporting older PyTorch versions."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

