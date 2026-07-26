import torch
from typing import Tuple, Dict


# ------------------- Utils for metrics -------------------
def rms(x: torch.Tensor, dim=-1, eps: float = 1e-8):
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + eps)


def compute_psd(x: torch.Tensor):
    """
    x: [B, C, T], [B, 1, T], or [B, T]
    return: PSD with the same leading dimensions.
    """
    if x.dim() == 3 and x.size(1) == 1:
        x = x.squeeze(1)
    X = torch.fft.rfft(x, dim=-1)
    psd = (X.real ** 2 + X.imag ** 2) / x.size(-1)
    return psd



def compute_corr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean Pearson correlation over batch.
    """
    p = pred.squeeze(1)
    t = target.squeeze(1)
    p = p - p.mean(dim=-1, keepdim=True)
    t = t - t.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    den = torch.sqrt((p.pow(2).sum(dim=-1) * t.pow(2).sum(dim=-1)).clamp_min(eps))
    return (num / den.clamp_min(eps)).mean()



def compute_snr_db(clean: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Signal-to-noise ratio in dB, where noise = estimate - clean.
    """
    signal_power = clean.pow(2).mean(dim=-1)
    noise_power = (estimate - clean).pow(2).mean(dim=-1)
    return (10.0 * torch.log10((signal_power + eps) / (noise_power + eps))).mean()



def compute_snr_improvement(noisy: torch.Tensor, denoised: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    SNR improvement = SNR(denoised, clean) - SNR(noisy, clean)
    """
    snr_in = compute_snr_db(clean, noisy)
    snr_out = compute_snr_db(clean, denoised)
    return snr_out - snr_in



def compute_psd_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 difference between power spectra.
    """
    p1 = torch.fft.rfft(pred.squeeze(1), dim=-1).abs().pow(2)
    p2 = torch.fft.rfft(target.squeeze(1), dim=-1).abs().pow(2)
    p1 = p1 / (p1.sum(dim=-1, keepdim=True) + 1e-8)
    p2 = p2 / (p2.sum(dim=-1, keepdim=True) + 1e-8)
    return (p1 - p2).abs().mean()



def compute_bandpower_relative_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    fs: int,
    band: Tuple[float, float],
    eps: float = 1e-8,
) -> torch.Tensor:
    p_pred = fft_bandpower(pred, fs, band)
    p_tgt = fft_bandpower(target, fs, band)
    return (((p_pred - p_tgt).abs()) / (p_tgt.abs() + eps)).mean()



def compute_band_metrics(pred: torch.Tensor, target: torch.Tensor, fs: int) -> Dict[str, torch.Tensor]:
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }
    out: Dict[str, torch.Tensor] = {}
    for name, band in bands.items():
        out[f"bp_re_{name}"] = compute_bandpower_relative_error(pred, target, fs, band)
    return out


def fft_bandpower(x: torch.Tensor, fs: int, band: Tuple[float, float]) -> torch.Tensor:
    """
    x: [B, 1, T] -> returns [B]
    """
    spec = torch.fft.rfft(x.squeeze(1), dim=-1)
    power = spec.abs().pow(2)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    if mask.sum() == 0:
        return torch.zeros(x.shape[0], device=x.device)
    return power[:, mask].mean(dim=-1)



#------------------------Utils for training and evaluation-------------------------

# ============================================================
# Batch parsing
# ============================================================

def parse_batch(batch):
    """
    Flexible parser for different dataset output formats.

    Expected target:
        noisy EEG: [B, 1, T] or [B, T]
        clean EEG: [B, 1, T] or [B, T]

    Supported formats:
    1) dict with keys like:
       - noisy / x / contaminated / input
       - clean / target / y / eeg
       - optional snr
    2) tuple/list:
       - (noisy, clean)
       - (noisy, clean, snr)
       - (noisy, clean, meta)
    """
    snr = None

    if isinstance(batch, dict):
        noisy = None
        clean = None

        noisy_keys = ["noisy", "x", "contaminated", "input", "mixed", "artifacted"]
        clean_keys = ["clean", "target", "y", "eeg", "ground_truth", "gt"]

        for k in noisy_keys:
            if k in batch:
                noisy = batch[k]
                break

        for k in clean_keys:
            if k in batch:
                clean = batch[k]
                break

        if "snr" in batch:
            snr = batch["snr"]
        elif "snr_db" in batch:
            snr = batch["snr_db"]

        if noisy is None or clean is None:
            raise KeyError(f"Could not parse batch dict keys. Found keys={list(batch.keys())}")

    elif isinstance(batch, (tuple, list)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list must have at least (noisy, clean)")
        noisy = batch[0]
        clean = batch[1]
        if len(batch) >= 3 and torch.is_tensor(batch[2]):
            snr = batch[2]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if noisy.dim() == 2:
        noisy = noisy.unsqueeze(1)
    if clean.dim() == 2:
        clean = clean.unsqueeze(1)

    return noisy.float(), clean.float(), snr


class EvalTracker:
    def __init__(self):
        self.total = 0
        self.loss = 0.0
        self.rrmse_temporal = 0.0
        self.rrmse_spectral = 0.0
        self.cc = 0.0
        self.snr_improvement_db = 0.0

    def update(self, loss, rrmse_t, rrmse_s, cc, snr_imp, n):
        self.total += int(n)
        self.loss += float(loss) * n
        self.rrmse_temporal += float(rrmse_t) * n
        self.rrmse_spectral += float(rrmse_s) * n
        self.cc += float(cc) * n
        self.snr_improvement_db += float(snr_imp) * n

    def as_dict(self) -> Dict[str, float]:
        denom = max(self.total, 1)
        return {
            "loss": self.loss / denom,
            "rrmse_temporal": self.rrmse_temporal / denom,
            "rrmse_spectral": self.rrmse_spectral / denom,
            "cc": self.cc / denom,
            "snr_improvement_db": self.snr_improvement_db / denom,
        }
