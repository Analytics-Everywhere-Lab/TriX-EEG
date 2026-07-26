import torch



def mad_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, 1, T]
    """
    median = x.median(dim=-1, keepdim=True).values
    mad = (x - median).abs().median(dim=-1, keepdim=True).values
    return (x - median) / (mad + eps)

def z_score_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, 1, T]
    """
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


def std_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: [B, 1, T]
    """
    std = x.std(dim=-1, keepdim=True)
    return x / (std + eps)

