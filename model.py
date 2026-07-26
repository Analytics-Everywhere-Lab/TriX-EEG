from typing import Dict
import torch
import torch.nn as nn
from torchinfo import summary
from modules.bands_decompose import FFTBandDecomposition
from modules.enc_dec import EncoderFactory, DecoderFactory
from modules.crossband_fusion import CrossBandFusionV2
from modules.router import ArtifactRoutingGateV2
from modules.denoiser import RoutedDenoiserBlockV2
from modules.fullband_conditioner import FullBandConditioner
from configs.braanetv2_config import BRAANetV2Config



class BRAANetV2(nn.Module):
    def __init__(self, cfg: BRAANetV2Config):
        super().__init__()
        self.cfg = cfg
        if cfg.band_defs is None:
            cfg.band_defs = cfg.default_band_defs

        self.band_decomp = FFTBandDecomposition(cfg.fs, cfg.signal_length, cfg.band_defs)
        self.num_bands = len(cfg.band_defs)

        self.band_encoder = EncoderFactory(
            cfg.encoder_type,
            in_ch=1,
            hidden_ch=cfg.base_channels,
            out_ch=cfg.latent_channels,
            num_blocks=cfg.num_encoder_blocks,
            dropout=cfg.dropout,
        )
        self.band_decoder = DecoderFactory(
            cfg.decoder_type,
            in_ch=cfg.latent_channels,
            hidden_ch=cfg.base_channels,
            out_ch=1,
            num_blocks=cfg.num_decoder_blocks,
            dropout=cfg.dropout,
        )

        self.use_band_embedding = cfg.use_band_embedding
        if self.use_band_embedding:
            self.band_embed = nn.Parameter(torch.randn(self.num_bands, cfg.latent_channels, 1) * 0.02)
        else:
            self.register_parameter("band_embed", None)

        self.use_artifact_routing = cfg.use_artifact_routing
        if self.use_artifact_routing:
            self.routing_gate = ArtifactRoutingGateV2(
                channels=cfg.latent_channels,
                hidden=cfg.routing_hidden,
                kernel_size=7,
            )
        else:
            self.routing_gate = None

        self.band_denoiser = RoutedDenoiserBlockV2(
            channels=cfg.latent_channels,
            num_blocks=cfg.num_denoiser_blocks,
            temporal_model_type=cfg.temporal_model_type,
            dropout=cfg.dropout,
        )
        

        self.use_cross_band_fusion = cfg.use_cross_band_fusion
        if self.use_cross_band_fusion:
            self.cross_band_fusion = CrossBandFusionV2(
                channels=cfg.latent_channels,
                num_heads=cfg.num_fusion_heads,
                num_layers=cfg.num_fusion_layers,
                ff_mult=cfg.fusion_ff_mult,
                temporal_model_type=cfg.temporal_model_type,
                dropout=cfg.dropout,
            )
        else:
            self.cross_band_fusion = nn.Identity()

        self.use_fullband_branch = cfg.use_fullband_branch
        self.use_fullband_conditioning = cfg.use_fullband_conditioning
        if self.use_fullband_branch:
            self.fullband = FullBandConditioner(cfg)
        else:
            self.fullband = None

    def condition_band_latent(self, z: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        # z, gamma, beta: [B, C, T]
        return z * (1.0 + torch.tanh(gamma)) + beta

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert x.dim() == 3, f"Expected [B,C,T], got {x.shape}"
        b, input_channels, t = x.shape
        if t != self.cfg.signal_length:
            raise ValueError(f"Expected T={self.cfg.signal_length}, got {t}")
        x = x.reshape(b * input_channels, 1, t)

        x_bands = self.band_decomp(x)  # [B, K, T]
        if self.use_fullband_branch:
            full_out = self.fullband(x)
            zf = full_out["zf"]
            df = full_out["df"]
            alpha_t = full_out["alpha_t"]
            gamma = full_out["gamma"]
            beta = full_out["beta"]
        else:
            zf = None
            df = None
            alpha_t = None
            gamma = None
            beta = None

        band_latents = []
        routing_gates = []

        for k in range(self.num_bands):
            xb = x_bands[:, k:k + 1, :]
            ek = self.band_encoder(xb)
            if self.use_band_embedding:
                ek = ek + self.band_embed[k:k + 1]

            if self.use_fullband_branch and self.use_fullband_conditioning:
                ek = self.condition_band_latent(ek, gamma, beta)

            fk = self.band_denoiser(ek)        # [B, C, T]
            if self.use_artifact_routing:
                gk = self.routing_gate(ek)     # [B, C, T]
                zk = ek + gk * (fk - ek)
            else:
                gk = torch.ones_like(ek)
                zk = fk

            band_latents.append(zk)
            routing_gates.append(gk)

        z = torch.stack(band_latents, dim=1)       # [B, K, C, T]
        routing_gates = torch.stack(routing_gates, dim=1)  # [B, K, C, T]
        z = self.cross_band_fusion(z)

        y_hat_bands = []
        for k in range(self.num_bands):
            dk = self.band_decoder(z[:, k, :, :])
            y_hat_bands.append(dk)

        y_hat_bands = torch.cat(y_hat_bands, dim=1)       # [B, K, T]
        y_hat_band = y_hat_bands.sum(dim=1, keepdim=True) # [B, 1, T]

        if self.use_fullband_branch:
            y_hat = y_hat_band + alpha_t * df
        else:
            y_hat = y_hat_band

        def restore_channels(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            if tensor.dim() == 3 and tensor.size(1) == 1:
                return tensor.reshape(b, input_channels, tensor.size(-1))
            if tensor.dim() == 3:
                return tensor.reshape(b, input_channels, *tensor.shape[1:])
            if tensor.dim() == 4:
                return tensor.reshape(b, input_channels, *tensor.shape[1:])
            return tensor

        y_hat = y_hat.reshape(b, input_channels, t)
        y_hat_band = y_hat_band.reshape(b, input_channels, t)
        if input_channels > 1:
            y_hat_bands = y_hat_bands.reshape(b, input_channels, self.num_bands, t)
            x_bands = x_bands.reshape(b, input_channels, self.num_bands, t)
            routing_gates = routing_gates.reshape(b, input_channels, self.num_bands, *routing_gates.shape[2:])

        return {
            "y_hat": y_hat,
            "y_hat_band": y_hat_band,
            "y_hat_bands": y_hat_bands,
            "x_bands": x_bands,
            "routing_gates": routing_gates,
            "alpha_t": restore_channels(alpha_t),
            "zf": restore_channels(zf),
            "df": restore_channels(df),
        }



if __name__ == "__main__":
    # Example
    cfg = BRAANetV2Config()
    model = BRAANetV2(cfg)
    x = torch.randn(8, cfg.input_channels, 512)
    out = model(x)

    print("Output shapes:", out['y_hat'].shape, out['y_hat_band'].shape, out['y_hat_bands'].shape)

    for k, v in out.items():
        if v is None:
            print(k, None)
        else:
            print(k, tuple(v.shape))
    
    print(summary(model, input_data=x))
