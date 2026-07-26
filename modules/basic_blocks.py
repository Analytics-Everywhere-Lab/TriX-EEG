import torch
import torch.nn as nn
from typing import Optional


# ============================================================
# Basic blocks
# ============================================================
class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, groups: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, groups=groups)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, kernel_size=3, dilation=dilation),
            nn.Dropout(dropout),
            ConvNormAct(channels, channels, kernel_size=3, dilation=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bottleneck_ch: Optional[int] = None):
        super().__init__()
        mid = bottleneck_ch or max(out_ch // 4, 8)
        self.pre = nn.Conv1d(in_ch, mid, kernel_size=1)
        branch_out = out_ch // 4

        self.b1 = ConvNormAct(mid, branch_out, kernel_size=3)
        self.b2 = ConvNormAct(mid, branch_out, kernel_size=5)
        self.b3 = ConvNormAct(mid, branch_out, kernel_size=7)
        self.b4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvNormAct(in_ch, branch_out, kernel_size=1),
        )
        self.post = ConvNormAct(branch_out * 4, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = self.pre(x)
        y = torch.cat([self.b1(xp), self.b2(xp), self.b3(xp), self.b4(x)], dim=1)
        return self.post(y)


class TemporalTCNBlock(nn.Module):
    def __init__(self, channels: int, num_blocks: int, dropout: float = 0.1):
        super().__init__()
        dilations = [1, 2, 4, 8]
        self.net = nn.Sequential(*[
            ResidualDilatedBlock(channels, dilation=dilations[i % len(dilations)], dropout=dropout)
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalGRUBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.gru = nn.GRU(input_size=channels, hidden_size=channels // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h, _ = self.gru(h)
        h = h.transpose(1, 2)
        h = self.proj(h)
        return x + h

