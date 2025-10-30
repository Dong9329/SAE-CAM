from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
from torch import nn


__all__ = [
"ChannelAttention",
"SpatialAttention",
"CBAM",
"CBAM1D",
"CBAM2D",
]




def _conv_nd(dim: int, in_ch: int, out_ch: int, k: int, **kw) -> nn.Module:
if dim == 1:
return nn.Conv1d(in_ch, out_ch, k, **kw)
elif dim == 2:
return nn.Conv2d(in_ch, out_ch, k, **kw)
raise ValueError("dim must be 1 or 2")




def _pool_nd(dim: int, kind: str) -> nn.Module:
if kind == "avg":
return nn.AdaptiveAvgPool1d(1) if dim == 1 else nn.AdaptiveAvgPool2d(1)
if kind == "max":
return nn.AdaptiveMaxPool1d(1) if dim == 1 else nn.AdaptiveMaxPool2d(1)
raise ValueError("unknown pool kind")




class ChannelAttention(nn.Module):
"""Channel attention with optional dual pooling and group MLP.


Args:
channels: input channels
reduction: hidden ratio
dim: 1 or 2 (1D time-series; 2D spectrogram/image)
use_max_pool: add max-pooling branch in addition to avg-pooling
groups: optional grouped bottleneck for large channel counts
"""
def __init__(self, channels: int, reduction: int = 16, dim: int = 1,
use_max_pool: bool = True, groups: int = 1) -> None:
super().__init__()
hidden = max(channels // reduction, 8)
self.dim = dim
self.pool_avg = _pool_nd(dim, "avg")
self.pool_max = _pool_nd(dim, "max") if use_max_pool else None
# lightweight bottleneck; bias=False for cleaner BN/affine
Conv = nn.Conv1d if dim == 1 else nn.Conv2d
self.mlp = nn.Sequential(
Conv(channels, hidden, 1, bias=False, groups=groups), nn.ReLU(inplace=True),
Conv(hidden, channels, 1, bias=False, groups=groups)
)
self.act = nn.Sigmoid()


def forward(self, x: torch.Tensor) -> torch.Tensor:
s = self.mlp(self.pool_avg(x))
if self.pool_max is not None:
s = s + self.mlp(self.pool_max(x))
w = self.act(s)
return x * w




class SpatialAttention(nn.Module):
"""Spatial attention as in CBAM, with depthwise conv and learnable logit scale.


For 1D: kernel along time; For 2D: kernel over HxW.
"""
def __init__(self, dim: int = 1, kernel_size: int = 7) -> None:
super().__init__()
padding = kernel_size // 2
if dim == 1:
conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
else:
conv = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
self.conv = conv
self.logit_scale = nn.Parameter(torch.tensor(1.0))
self.sig = nn.Sigmoid()
self.dim = dim


def forward(self, x: torch.Tensor) -> torch.Tensor:
# channel squeeze by mean + max for richer spatial cue
if self.dim == 1:
m = x.mean(dim=1, keepdim=True)
M = x.amax(dim=1, keepdim=True)
s = (m + M) * 0.5
else:
m = x.mean(dim=1, keepdim=True)
super().__init__(channels, dim=2, reduction=reduction, kernel_size=kernel_size)
