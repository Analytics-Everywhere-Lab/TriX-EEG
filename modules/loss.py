import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Tuple
from typing import Optional


class BRAALossV2(nn.Module):
    """
    Losses used:
    - time-domain L1
    - derivative L1
    - spectral magnitude + log magnitude
    - band-wise reconstruction
    - cross-band consistency:
        * normalized band energy ratio
        * covariance across bands
    - temporal routing regularization for dynamic gates [B, K, C, T]
    - optional full-band residual branch supervision

    Expected cfg fields:
    - lambda_time, lambda_diff
    - lambda_spec, lambda_logspec
    - lambda_band, lambda_cbc
    - lambda_ratio, lambda_cov
    - lambda_route
    - eps

    """
    def __init__(self, cfg, band_decomp):
        super().__init__()
        self.cfg = cfg
        self.band_decomp = band_decomp

    @staticmethod
    def temporal_diff(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, T], [B, K, T], or [B, T]
        """
        if x.dim() == 2:
            return x[:, 1:] - x[:, :-1]
        if x.dim() == 3:
            return x[:, :, 1:] - x[:, :, :-1]
        raise ValueError(f"Unsupported shape: {x.shape}")

    @staticmethod
    def stft_mag(
        x: torch.Tensor,
        n_fft: int = 128,
        hop_length: int = 64,
        win_length: int = 128,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        x: [B, 1, T] or [B, T]
        return: [B, F, TT]
        """
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.dim() == 3:
            x = x.reshape(x.size(0) * x.size(1), x.size(2))
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
            center=True,
        )
        return torch.abs(X).clamp_min(eps)

    def time_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # l_time = F.l1_loss(y_hat, y)
        l_time = F.mse_loss(y_hat, y)
        l_diff = F.l1_loss(self.temporal_diff(y_hat), self.temporal_diff(y))
        return self.cfg.lambda_time * l_time + self.cfg.lambda_diff * l_diff

    def spectral_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mag_hat = self.stft_mag(y_hat, eps=self.cfg.eps)
        mag_y = self.stft_mag(y, eps=self.cfg.eps)

        l_mag = F.mse_loss(mag_hat, mag_y)
        l_log = F.l1_loss(torch.log(mag_hat + self.cfg.eps), torch.log(mag_y + self.cfg.eps))

        return self.cfg.lambda_spec * l_mag + self.cfg.lambda_logspec * l_log

    def band_recon_loss(self, y_hat_bands: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y_hat_bands: [B, K, T] or [B, C, K, T]
        y: [B, 1, T] or [B, C, T]
        """
        if y_hat_bands.dim() == 4:
            y_hat_bands = y_hat_bands.reshape(-1, y_hat_bands.size(-2), y_hat_bands.size(-1))
        if y.dim() == 3 and y.size(1) != 1:
            y = y.reshape(-1, 1, y.size(-1))
        y_bands = self.band_decomp(y)
        return F.l1_loss(y_hat_bands, y_bands)

    def band_energy_ratio(self, bands: torch.Tensor) -> torch.Tensor:
        """
        bands: [B, K, T]
        return: [B, K]
        """
        energy = (bands ** 2).sum(dim=-1)
        ratio = energy / (energy.sum(dim=1, keepdim=True) + self.cfg.eps)
        return ratio

    def covariance_across_bands(self, bands: torch.Tensor) -> torch.Tensor:
        """
        bands: [B, K, T]
        return: [B, K, K]
        """
        x = bands - bands.mean(dim=-1, keepdim=True)
        cov = x @ x.transpose(1, 2)
        cov = cov / max(bands.size(-1) - 1, 1)
        return cov

    def cross_band_consistency_loss(self, y_hat_bands: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y_hat_bands.dim() == 4:
            y_hat_bands = y_hat_bands.reshape(-1, y_hat_bands.size(-2), y_hat_bands.size(-1))
        if y.dim() == 3 and y.size(1) != 1:
            y = y.reshape(-1, 1, y.size(-1))
        y_bands = self.band_decomp(y)

        ratio_hat = self.band_energy_ratio(y_hat_bands)
        ratio_y = self.band_energy_ratio(y_bands)
        l_ratio = F.l1_loss(ratio_hat, ratio_y)

        cov_hat = self.covariance_across_bands(y_hat_bands)
        cov_y = self.covariance_across_bands(y_bands)
        l_cov = F.mse_loss(cov_hat, cov_y)

        return self.cfg.lambda_ratio * l_ratio + self.cfg.lambda_cov * l_cov

    def routing_loss(self, routing_gates: torch.Tensor) -> torch.Tensor:
        """
        routing_gates: [B, K, C, T]
        """
        g = routing_gates.clamp(1e-6, 1.0 - 1e-6)

        target_mean = getattr(self.cfg, "route_target_mean", 0.30)
        lambda_tv = getattr(self.cfg, "lambda_route_tv", 0.0)
        lambda_entropy = getattr(self.cfg, "lambda_route_entropy", 0.0)

        # keep average routing activity near a target instead of trivially shrinking to zero
        l_mean = (g.mean() - target_mean).abs()

        # temporal smoothness for dynamic routing masks
        if g.size(-1) > 1:
            l_tv = (g[..., 1:] - g[..., :-1]).abs().mean()
        else:
            l_tv = g.new_tensor(0.0)

        # low entropy => sharper / more decisive gates
        entropy = -(g * torch.log(g) + (1.0 - g) * torch.log(1.0 - g))
        l_entropy = entropy.mean()

        return l_mean + lambda_tv * l_tv + lambda_entropy * l_entropy

    def alpha_loss(self, alpha_t: Optional[torch.Tensor]) -> torch.Tensor:
        """
        alpha_t: [B, 1, T]
        Optional regularization for full-band dynamic mixing.
        """
        if alpha_t is None:
            return torch.tensor(0.0, device=self.band_decomp.freq_masks.device)

        lambda_alpha_smooth = getattr(self.cfg, "lambda_alpha_smooth", 0.0)
        lambda_alpha_sparse = getattr(self.cfg, "lambda_alpha_sparse", 0.0)

        l_smooth = (alpha_t[..., 1:] - alpha_t[..., :-1]).abs().mean() if alpha_t.size(-1) > 1 else alpha_t.new_tensor(0.0)
        l_sparse = alpha_t.mean()
        return lambda_alpha_smooth * l_smooth + lambda_alpha_sparse * l_sparse

    def fullband_residual_loss(
        self,
        y_hat_band: torch.Tensor,
        y: torch.Tensor,
        df: Optional[torch.Tensor],
        alpha_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Optional supervision for the full-band residual branch.
        """
        
        lambda_fullband_res = getattr(self.cfg, "lambda_fullband_res", 0.0)
        if lambda_fullband_res <= 0.0 or df is None or alpha_t is None:
            return y_hat_band.new_tensor(0.0)

        target_residual = y - y_hat_band
        pred_residual = alpha_t * df
        return lambda_fullband_res * F.mse_loss(pred_residual, target_residual)

    def forward(self, model_out: Dict[str, torch.Tensor], y: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_hat = model_out["y_hat"]
        y_hat_band = model_out["y_hat_band"]
        y_hat_bands = model_out["y_hat_bands"]
        routing_gates = model_out["routing_gates"]
        alpha_t = model_out.get("alpha_t", None)
        df = model_out.get("df", None)

        l_time = self.time_loss(y_hat, y)
        l_spec = self.spectral_loss(y_hat, y)
        l_band = self.band_recon_loss(y_hat_bands, y)
        l_cbc = self.cross_band_consistency_loss(y_hat_bands, y)
        l_route = self.routing_loss(routing_gates)
        l_alpha = self.alpha_loss(alpha_t)
        l_fullband = self.fullband_residual_loss(y_hat_band, y, df, alpha_t)

        total = (
             self.cfg.lambda_time * l_time
            + self.cfg.lambda_spec * l_spec
            + self.cfg.lambda_band * l_band
            + self.cfg.lambda_cbc * l_cbc
            + self.cfg.lambda_route * l_route
            + self.cfg.lambda_alpha * l_alpha
            + self.cfg.lambda_fullband_res * l_fullband
        )

        return {
            "loss": total,
            "loss_time": l_time,
            "loss_spec": l_spec,
            "loss_band": l_band,
            "loss_cbc": l_cbc,
            "loss_route": l_route,
            "loss_alpha": l_alpha,
            "loss_fullband": l_fullband,
        }
