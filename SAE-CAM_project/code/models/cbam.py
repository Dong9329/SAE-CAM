
from __future__ import annotations
import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.mlp(self.pool(x))
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size//2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        m = x.mean(dim=1, keepdim=True)  # [B,1,T]
        w = self.sig(self.conv(m))
        return x * w

class CBAM1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
