import torch
import torch.nn as nn


# ============================================================
# Artifact Routing: temporal gate with local + global context
# ============================================================
class ArtifactRoutingGateV2(nn.Module):
    """
    Input: x [B, C, T]
    Output: gate [B, C, T]

    Design:
    - Local temporal gate via depthwise temporal conv
    - Global channel context via GAP -> MLP
    - Combined into dynamic time-varying routing mask
    """
    def __init__(self, channels: int, hidden: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2

        self.local_dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.local_pw = nn.Conv1d(channels, channels, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )

        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_ctx = self.local_pw(self.local_dw(x))
        global_ctx = self.global_mlp(self.global_pool(x))
        g = self.out(local_ctx + global_ctx)
        return g