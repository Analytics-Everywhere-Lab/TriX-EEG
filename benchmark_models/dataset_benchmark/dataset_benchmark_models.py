import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# References: 
# - https://github.com/ncclabsustech/EEGdenoiseNet/blob/master/code/benchmark_networks/Network_structure.py
# - https://github.com/ncclabsustech/EEGdenoiseNet/blob/master/code/Novel_CNN/Novel_CNN.py


# -----------------------------
# Fully connected network
# Input: [B, datanum]
# Output: [B, datanum]
# -----------------------------
class FcNN(nn.Module):
    def __init__(self, datanum: int, input_channels: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(datanum, datanum),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(datanum, datanum),
        )

    def forward(self, x):
        return self.net(x)


def fcNN(datanum, input_channels: int = 1):
    return FcNN(datanum, input_channels=input_channels)


# -----------------------------
# LSTM model
# Input: [B, datanum, 1]
# Output: [B, datanum]
# -----------------------------
class RNNLSTM(nn.Module):
    def __init__(self, datanum: int, input_channels: int = 1):
        super().__init__()
        self.datanum = datanum
        self.input_channels = input_channels

        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=input_channels,
            batch_first=True
        )

        self.fc1 = nn.Linear(datanum, datanum)
        self.fc2 = nn.Linear(datanum, datanum)
        self.fc3 = nn.Linear(datanum, datanum)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        legacy_btc = x.dim() == 3 and x.size(1) == self.datanum
        if x.dim() == 3 and x.size(1) != self.datanum:
            x = x.transpose(1, 2)  # [B,C,T] -> [B,T,C]

        x, _ = self.lstm(x)          # [B, datanum, 1]
        x = x.transpose(1, 2)        # [B, C, datanum]

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        if legacy_btc and x.size(1) == 1:
            x = x.squeeze(1)
        return x


def RNN_lstm(datanum, input_channels: int = 1):
    return RNNLSTM(datanum, input_channels=input_channels)


# -----------------------------
# Simple CNN
# Input: [B, datanum, 1]
# Output: [B, datanum]
# In PyTorch Conv1d expects [B, C, L]
# so we permute inside forward
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, datanum: int, input_channels: int = 1):
        super().__init__()
        self.datanum = datanum
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc = nn.Linear(64 * datanum, input_channels * datanum)

    def forward(self, x):
        legacy_btc = x.dim() == 3 and x.size(1) == self.datanum
        if legacy_btc:
            x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = x.reshape(x.size(0), self.input_channels, self.datanum)
        if legacy_btc and x.size(1) == 1:
            x = x.squeeze(1)
        return x


def simple_CNN(datanum, input_channels: int = 1):
    return SimpleCNN(datanum, input_channels=input_channels)


# -----------------------------
# ResNet Basic Block
# Input/Output channel count should match for residual add
# Your TF code assumes input channels = 32
# -----------------------------
class ResBasicBlock(nn.Module):
    def __init__(self, kernelsize, stride=1):
        super().__init__()
        padding = kernelsize // 2

        self.bblock = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=kernelsize, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 16, kernel_size=kernelsize, stride=1, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=kernelsize, stride=1, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.jump_layer = nn.Identity()

    def forward(self, x):
        out = self.bblock(x)
        identity = self.jump_layer(x)
        out = out + identity
        return out


# -----------------------------
# Multi-branch block
# Output channels = 32 * 3 = 96
# -----------------------------
class BasicBlockAll(nn.Module):
    def __init__(self, stride=1):
        super().__init__()

        self.bblock3 = nn.Sequential(
            ResBasicBlock(3),
            ResBasicBlock(3)
        )

        self.bblock5 = nn.Sequential(
            ResBasicBlock(5),
            ResBasicBlock(5)
        )

        self.bblock7 = nn.Sequential(
            ResBasicBlock(7),
            ResBasicBlock(7)
        )

    def forward(self, x):
        out3 = self.bblock3(x)
        out5 = self.bblock5(x)
        out7 = self.bblock7(x)

        out = torch.cat([out3, out5, out7], dim=1)  # concat on channel axis
        return out


# -----------------------------
# Complex CNN
# Input: [B, datanum, 1]
# Output: [B, datanum]
# -----------------------------
class ComplexCNN(nn.Module):
    def __init__(self, datanum: int, input_channels: int = 1):
        super().__init__()
        self.datanum = datanum
        self.input_channels = input_channels

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.block = BasicBlockAll()

        self.proj = nn.Sequential(
            nn.Conv1d(96, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.fc = nn.Linear(32 * datanum, input_channels * datanum)

    def forward(self, x):
        legacy_btc = x.dim() == 3 and x.size(1) == self.datanum
        if legacy_btc:
            x = x.permute(0, 2, 1)

        x = self.stem(x)
        x = self.block(x)
        x = self.proj(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = x.reshape(x.size(0), self.input_channels, self.datanum)
        if legacy_btc and x.size(1) == 1:
            x = x.squeeze(1)
        return x


def Complex_CNN(datanum, input_channels: int = 1):
    return ComplexCNN(datanum, input_channels=input_channels)




if __name__ == "__main__":
    datanum = 512
    batch_size = 4

    x_fc = torch.randn(batch_size, datanum)
    x_seq = torch.randn(batch_size, datanum, 1)

    model_fc = fcNN(datanum)
    model_lstm = RNN_lstm(datanum)
    model_simple = simple_CNN(datanum)
    model_complex = Complex_CNN(datanum)

    print("fcNN:", model_fc(x_fc).shape)           # [4, 128]
    print("RNN_lstm:", model_lstm(x_seq).shape)    # [4, 128]
    print("simple_CNN:", model_simple(x_seq).shape)# [4, 128]
    print("Complex_CNN:", model_complex(x_seq).shape)  # [4, 128]


    print("\nModel summaries:")
    summary(model_fc, input_size=(batch_size, datanum))
    summary(model_lstm, input_size=(batch_size, datanum, 1))
    summary(model_simple, input_size=(batch_size, datanum, 1))
    summary(model_complex, input_size=(batch_size, datanum, 1))

