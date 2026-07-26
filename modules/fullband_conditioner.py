import torch
import torch.nn as nn
from typing import Dict
from .enc_dec import EncoderFactory, DecoderFactory, TemporalModelFactory
from .basic_blocks import ConvNormAct


# ============================================================
# Full-band Conditioning
# ============================================================
class FullBandConditioner(nn.Module):
    """
    Produces:
    - full-band latent zf [B, C, T]
    - dynamic alpha_t [B, 1, T]
    - optional conditioning shift/scale for band latents: Gamma [B, C, T], Beta [B, C, T]
    """
    def __init__(self, cfg):
        super().__init__()
        self.encoder = EncoderFactory(
            cfg.encoder_type,
            in_ch=1,
            hidden_ch=cfg.base_channels,
            out_ch=cfg.latent_channels,
            num_blocks=cfg.fullband_num_blocks,
            dropout=cfg.dropout,
        )
        self.temporal = TemporalModelFactory(
            cfg.temporal_model_type,
            channels=cfg.latent_channels,
            num_blocks=max(2, cfg.fullband_num_blocks // 2),
            dropout=cfg.dropout,
        )
        self.decoder = DecoderFactory(
            cfg.decoder_type,
            in_ch=cfg.latent_channels,
            hidden_ch=cfg.base_channels,
            out_ch=1,
            num_blocks=cfg.fullband_num_blocks,
            dropout=cfg.dropout,
        )
        self.alpha_head = nn.Sequential(
            ConvNormAct(cfg.latent_channels, cfg.latent_channels, kernel_size=3),
            nn.Conv1d(cfg.latent_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.cond_proj = nn.Conv1d(cfg.latent_channels, 2 * cfg.latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        zf = self.temporal(self.encoder(x))
        df = self.decoder(zf)
        alpha_t = self.alpha_head(zf)
        cond = self.cond_proj(zf)
        gamma, beta = torch.chunk(cond, chunks=2, dim=1)
        return {
            "zf": zf,
            "df": df,
            "alpha_t": alpha_t,
            "gamma": gamma,
            "beta": beta,
        }
