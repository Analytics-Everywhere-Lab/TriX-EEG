from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """LSTM denoiser from EEG-DeNoiseGAN.

    Inputs and outputs use the project-wide ``[batch, channels, time]`` shape.
    """

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 50,
        num_layers: int = 2,
        tanh_output: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        output_layers: list[nn.Module] = [
            nn.Linear(self.hidden_dim, self.input_channels)
        ]
        if tanh_output:
            output_layers.append(nn.Tanh())
        self.output = nn.Sequential(*output_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got {tuple(x.shape)}")
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} channels, got {x.size(1)}"
            )
        sequence = x.transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        return self.output(sequence).transpose(1, 2)


class Discriminator(nn.Module):
    """Four-stage convolutional least-squares GAN discriminator."""

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 3, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(3, 5, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(5, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(8, 10, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=2),
            nn.AdaptiveAvgPool1d(13),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10 * 13, 50),
            nn.Linear(50, 50),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class EEGDeNoiseGAN(nn.Module):
    """Generator/discriminator container with a denoiser-compatible forward."""

    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 50,
        num_layers: int = 2,
        tanh_output: bool = False,
    ) -> None:
        super().__init__()
        self.generator = Generator(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            tanh_output=tanh_output,
        )
        self.discriminator = Discriminator(input_channels=input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)
