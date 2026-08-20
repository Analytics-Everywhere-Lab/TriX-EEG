"""Training helpers for the least-squares EEG-DeNoiseGAN benchmark."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from tqdm import tqdm

from metrics import (
    batch_cc,
    batch_rrmse_spectral,
    batch_rrmse_temporal,
    batch_snr_improvement,
)
from utils import EvalTracker


class StateBundle:
    """Expose several optimizer/scheduler states through one checkpoint slot."""

    def __init__(self, **items: Any) -> None:
        self.items = items

    def state_dict(self) -> dict[str, Any]:
        return {name: item.state_dict() for name, item in self.items.items()}


def train_gan_one_epoch(
    model: nn.Module,
    loader: Any,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_adapter: Callable[[Any, torch.device], tuple[torch.Tensor, torch.Tensor, Any]],
    grad_clip: float | None = None,
    adversarial_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    description: str = "Train GAN",
) -> dict[str, float]:
    """Train discriminator and generator once for every paired EEG batch."""
    model.train()
    tracker = EvalTracker()
    mse = nn.MSELoss()
    discriminator_loss_total = 0.0
    adversarial_loss_total = 0.0
    reconstruction_loss_total = 0.0
    sample_count = 0

    for batch in tqdm(loader, desc=description, leave=False):
        noisy, clean, _ = batch_adapter(batch, device)
        batch_size = noisy.size(0)

        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            generated = model(noisy)["y_hat"]
        real_score = model.model.discriminate(clean)
        fake_score = model.model.discriminate(generated)
        discriminator_loss = 0.5 * (
            mse(real_score, torch.ones_like(real_score))
            + mse(fake_score, torch.zeros_like(fake_score))
        )
        discriminator_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.model.discriminator.parameters(), grad_clip)
        discriminator_optimizer.step()

        generator_optimizer.zero_grad(set_to_none=True)
        for parameter in model.model.discriminator.parameters():
            parameter.requires_grad_(False)
        generated = model(noisy)["y_hat"]
        generated_score = model.model.discriminate(generated)
        adversarial_loss = mse(generated_score, torch.ones_like(generated_score))
        reconstruction_loss = mse(generated, clean)
        generator_loss = (
            adversarial_weight * adversarial_loss
            + reconstruction_weight * reconstruction_loss
        )
        generator_loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.model.generator.parameters(), grad_clip)
        generator_optimizer.step()
        for parameter in model.model.discriminator.parameters():
            parameter.requires_grad_(True)

        prediction = generated.detach()
        tracker.update(
            generator_loss.item(),
            batch_rrmse_temporal(prediction, clean).mean().item(),
            batch_rrmse_spectral(prediction, clean).mean().item(),
            batch_cc(prediction, clean).mean().item(),
            batch_snr_improvement(noisy, prediction, clean).mean().item(),
            batch_size,
        )
        discriminator_loss_total += discriminator_loss.item() * batch_size
        adversarial_loss_total += adversarial_loss.item() * batch_size
        reconstruction_loss_total += reconstruction_loss.item() * batch_size
        sample_count += batch_size

    metrics = tracker.as_dict()
    denominator = max(sample_count, 1)
    metrics.update(
        discriminator_loss=discriminator_loss_total / denominator,
        adversarial_loss=adversarial_loss_total / denominator,
        reconstruction_loss=reconstruction_loss_total / denominator,
    )
    return metrics

