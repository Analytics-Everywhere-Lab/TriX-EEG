from attr import dataclass
from typing import List, Optional, Tuple


@dataclass
class BRAANetV2Config:
    fs: float = 256.0
    signal_length: int = 512
    input_channels: int = 9
    band_defs: Optional[List[Tuple[float, float]]] = None

    nyq: float = fs / 2.0

    default_band_defs: List[Tuple[float, float]] = [
        (0.5, 4.0),
        (4.0, 8.0),
        (8.0, 12.0),
        (12.0, 30.0),
        (30.0, 80.0),
        (80.0, nyq),
    ]

    base_channels: int = 32
    latent_channels: int = 64
    routing_hidden: int = 64

    num_encoder_blocks: int = 2
    num_decoder_blocks: int = 2
    num_denoiser_blocks: int = 2
    fullband_num_blocks: int = 2
    num_fusion_layers: int = 1
    num_fusion_heads: int = 4
    fusion_ff_mult: int = 2

    dropout: float = 0.1

    encoder_type: str = "inception"   # residual | inception
    decoder_type: str = "inception"   # residual | inception
    temporal_model_type: str = "gru" # tcn | gru

    use_fullband_branch: bool = True
    use_fullband_conditioning: bool = True
    use_artifact_routing: bool = True
    use_cross_band_fusion: bool = True
    use_band_embedding: bool = True


    # Loss
    # 01/ Time loss
    lambda_time: float = 1.0
    lambda_diff: float = 0.0
    # 02/ Spectral loss
    lambda_spec: float = 0.0
    lambda_logspec: float = 0.0

    # 03/ Band-wise reconstruction loss
    lambda_band: float = 0.0
    lambda_cbc: float = 0.0
    lambda_ratio: float = 0.0
    lambda_cov: float = 0.0
    
    # 04/ Routing regularization
    lambda_route: float = 0.0
    lambda_route: float = 0.0
    lambda_route_tv: float = 0.0
    lambda_route_entropy: float = 0.0
    lambda_route_target_mean: float = 0.0
    lambda_alpha_smooth: float = 0.0
    
    # 05/ Full-band residual supervision
    lambda_alpha: float = 0.0
    lambda_fullband_res: float = 0.0
    lambda_alpha_sparse: float = 0.0
    
    # 06/ Correlation loss
    lambda_corr: float = 0.0
    
    eps: float = 1e-8
