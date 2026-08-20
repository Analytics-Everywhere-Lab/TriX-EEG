import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


# https://pmc.ncbi.nlm.nih.gov/articles/PMC12559029/
# ---------------------------------------------------------
# Basic blocks
# ---------------------------------------------------------
class ConvBNReLU1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock1d(nn.Module):
    """
    Simple residual block for 1D signals.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# ---------------------------------------------------------
# Net-D: Denoising EEG estimation module
# ---------------------------------------------------------
class DownBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.res = ResidualBlock1d(out_ch)
        self.conv = ConvBNReLU1d(out_ch, out_ch)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.proj(x)
        x = self.res(x)
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class Bottleneck1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.res = ResidualBlock1d(out_ch)
        self.conv = ConvBNReLU1d(out_ch, out_ch)

    def forward(self, x):
        x = self.proj(x)
        x = self.res(x)
        x = self.conv(x)
        return x


class UpBlock1d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.tconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBNReLU1d(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.tconv(x)

        # handle odd lengths safely
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[..., :skip.size(-1)]

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class NetD(nn.Module):
    """
    1D U-Net-like module for clean EEG estimation.
    Input:  [B, C, T]
    Output: [B, C, T] 
    """
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.down1 = DownBlock1d(in_ch, c1)
        self.down2 = DownBlock1d(c1, c2)
        self.down3 = DownBlock1d(c2, c3)
        self.bottleneck = Bottleneck1d(c3, c4)

        self.up3 = UpBlock1d(c4, c3, c3)
        self.up2 = UpBlock1d(c3, c2, c2)
        self.up1 = UpBlock1d(c2, c1, c1)

        self.final = nn.Sequential(
            nn.Conv1d(c1, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, out_ch, kernel_size=1),
        )

    def forward(self, x):
        s1, x = self.down1(x)
        s2, x = self.down2(x)
        s3, x = self.down3(x)

        x = self.bottleneck(x)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        x = self.final(x)
        return x


# ---------------------------------------------------------
# Net-N: Noisy estimation module
# ---------------------------------------------------------
class NetN(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=32, num_layers=4, out_ch=1):
        super().__init__()
        layers = []
        ch_in = in_ch
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(ch_in, hidden_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = hidden_ch
        layers.append(nn.Conv1d(ch_in, out_ch, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Net-R: Raw EEG synthesis module
# ---------------------------------------------------------
class NetR(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=32, num_layers=4, out_ch=1):
        super().__init__()
        layers = []
        ch_in = in_ch
        for i in range(num_layers - 1):
            layers.append(nn.Conv1d(ch_in, hidden_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(hidden_ch))
            layers.append(nn.ReLU(inplace=True))
            ch_in = hidden_ch
        layers.append(nn.Conv1d(ch_in, out_ch, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# One unfolding stage
# ---------------------------------------------------------
class LRRUNetStage(nn.Module):
    def __init__(self, channels=1, base_ch=32, netn_layers=4, netr_layers=4, init_rho=0.5):
        super().__init__()
        self.net_d = NetD(in_ch=channels, base_ch=base_ch, out_ch=channels)
        self.net_n = NetN(in_ch=channels, hidden_ch=base_ch, num_layers=netn_layers, out_ch=channels)
        self.net_r = NetR(in_ch=channels, hidden_ch=base_ch, num_layers=netr_layers, out_ch=channels)

        # learnable scalar rho_k
        self.rho = nn.Parameter(torch.tensor(float(init_rho)))

    def forward(self, r_prev, n_prev):
        # Dk = (R_{k-1} - N_{k-1}) + NetD(R_{k-1} - N_{k-1})
        x_clean = r_prev - n_prev
        d_k = x_clean + self.net_d(x_clean)

        # stage-consistent noise update
        noise_input = n_prev + r_prev - d_k
        n_k = r_prev + n_prev - d_k - self.rho * self.net_n(noise_input)

        # synthesize next raw EEG
        r_k = self.net_r(n_k + d_k)

        return r_k, n_k, d_k


# ---------------------------------------------------------
# Full unrolled LRR-UNet
# ---------------------------------------------------------
class LRRUNet(nn.Module):
    """
    Input:
        y: [B, 1, T]
    Returns:
        dict with denoised signal and intermediate states
    """
    def __init__(self, stages=3, channels=1, base_ch=32, netn_layers=4, netr_layers=4):
        super().__init__()
        self.stages = nn.ModuleList([
            LRRUNetStage(
                channels=channels,
                base_ch=base_ch,
                netn_layers=netn_layers,
                netr_layers=netr_layers,
                init_rho=0.5,
            )
            for _ in range(stages)
        ])

    def forward(self, y, return_all=False):
        r = y
        n = torch.zeros_like(y)

        all_r, all_n, all_d = [], [], []

        for stage in self.stages:
            r, n, d = stage(r, n)
            all_r.append(r)
            all_n.append(n)
            all_d.append(d)

        out = {
            "denoised": all_d[-1],   
            "R": r,                  
            "N": n,                  
        }

        if return_all:
            out["all_R"] = all_r
            out["all_N"] = all_n
            out["all_D"] = all_d

        return out
    
class LRRUNetLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()

    def forward(self, out, clean):
        d_k = out["denoised"]
        r_k = out["R"]
        n_k = out["N"]

        loss_denoise = self.mse_loss(d_k, clean)
        loss_raw = self.mse_loss(r_k, clean)
        loss_noise = self.mse_loss(n_k, clean)

        total_loss = self.alpha * loss_denoise 
        return total_loss


if __name__ == "__main__":
    x = torch.randn(8, 1, 512)  
    model = LRRUNet(stages=3, channels=1, base_ch=32, netn_layers=4, netr_layers=4)

    out = model(x, return_all=True)
    print("denoised:", out["denoised"].shape)  # [8, 1, 512]
    print("noise:", out["N"].shape)            # [8, 1, 512]
    print("R last:", out["R"].shape)           # [8, 1, 512]

    print("\nModel summary:")
    print(summary(model, input_size=(8, 1, 512), verbose=0))