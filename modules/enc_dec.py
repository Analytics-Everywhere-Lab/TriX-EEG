import torch
import torch.nn as nn
from .basic_blocks import ConvNormAct, ResidualDilatedBlock, InceptionBlock1D, TemporalTCNBlock, TemporalGRUBlock



#============================================================
# Encoder / Decoder families
# ============================================================


class ResidualEncoder1D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        layers = [ConvNormAct(in_ch, hidden_ch, kernel_size=5)]
        dilations = [1, 2, 4, 8]
        for i in range(num_blocks):
            layers.append(ResidualDilatedBlock(hidden_ch, dilation=dilations[i % len(dilations)], dropout=dropout))
        layers.append(nn.Conv1d(hidden_ch, out_ch, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualDecoder1D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        layers = [ConvNormAct(in_ch, hidden_ch, kernel_size=3)]
        dilations = [1, 2, 4, 8]
        for i in range(num_blocks):
            layers.append(ResidualDilatedBlock(hidden_ch, dilation=dilations[i % len(dilations)], dropout=dropout))
        layers.append(nn.Conv1d(hidden_ch, out_ch, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Inception families
class InceptionEncoder1D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        layers = [ConvNormAct(in_ch, hidden_ch, kernel_size=5)]
        for _ in range(num_blocks):
            layers += [InceptionBlock1D(hidden_ch, hidden_ch), nn.Dropout(dropout)]
        layers.append(nn.Conv1d(hidden_ch, out_ch, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InceptionDecoder1D(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        layers = [ConvNormAct(in_ch, hidden_ch, kernel_size=3)]
        for _ in range(num_blocks):
            layers += [InceptionBlock1D(hidden_ch, hidden_ch), nn.Dropout(dropout)]
        layers.append(nn.Conv1d(hidden_ch, out_ch, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# Encoder warper
class EncoderFactory(nn.Module):
    def __init__(self, kind: str, in_ch: int, hidden_ch: int, out_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        if kind == "residual":
            self.net = ResidualEncoder1D(in_ch, hidden_ch, out_ch, num_blocks, dropout)
        elif kind == "inception":
            self.net = InceptionEncoder1D(in_ch, hidden_ch, out_ch, num_blocks, dropout)
        else:
            raise ValueError(f"Unsupported encoder_type: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Decoder warper
class DecoderFactory(nn.Module):
    def __init__(self, kind: str, in_ch: int, hidden_ch: int, out_ch: int, num_blocks: int, dropout: float):
        super().__init__()
        if kind == "residual":
            self.net = ResidualDecoder1D(in_ch, hidden_ch, out_ch, num_blocks, dropout)
        elif kind == "inception":
            self.net = InceptionDecoder1D(in_ch, hidden_ch, out_ch, num_blocks, dropout)
        else:
            raise ValueError(f"Unsupported decoder_type: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalModelFactory(nn.Module):
    def __init__(self, kind: str, channels: int, num_blocks: int, dropout: float):
        super().__init__()
        if kind == "tcn":
            self.net = TemporalTCNBlock(channels, num_blocks=num_blocks, dropout=dropout)
        elif kind == "gru":
            self.net = TemporalGRUBlock(channels)
        else:
            raise ValueError(f"Unsupported temporal_model_type: {kind}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
