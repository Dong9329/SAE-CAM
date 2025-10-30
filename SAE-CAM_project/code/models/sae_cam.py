from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn
from einops import rearrange


# Local imports assume the four modules co-reside in a package
from .cbam import CBAM1D
from .rellm import ReLLM
from .losses import SupConLoss, mmd_loss, OrthogonalityLoss, StageAwareRedundancy




# ---------- Utility blocks ----------


def window_snr(x: torch.Tensor, win: int = 64, eps: float = 1e-6) -> torch.Tensor:
"""Estimate local SNR proxy over time via sliding windows (encoder feature space)."""
B, C, T = x.shape
pads = (0, (win - (T % win)) % win)
xp = nn.functional.pad(x, pads)
U = xp.unfold(dimension=2, size=win, step=win // 2) # [B,C,Nw,win]
power = U.pow(2).mean(-1).clamp_min(eps) # [B,C,Nw]
signal = U.abs().mean(-1)
snr = (signal / power.sqrt()).clamp(0, 10.0)
snr = nn.functional.interpolate(snr, size=T, mode="linear", align_corners=False)
return snr




class ConvEncoder(nn.Module):
"""Multi-scale 1D encoder with CBAM."""
def __init__(self, in_ch: int = 1, base: int = 32) -> None:
super().__init__()
self.net = nn.Sequential(
nn.Conv1d(in_ch, base, 7, padding=3), nn.BatchNorm1d(base), nn.GELU(),
nn.Conv1d(base, base * 2, 5, stride=2, padding=2), nn.BatchNorm1d(base * 2), nn.GELU(),
nn.Conv1d(base * 2, base * 4, 3, stride=2, padding=1), nn.BatchNorm1d(base * 4), nn.GELU(),
CBAM1D(base * 4),
)
self.out_ch = base * 4


def forward(self, x: torch.Tensor) -> torch.Tensor:
return self.net(x)




class BiCrossAttention(nn.Module):
"""Bidirectional cross-attention with cosine similarity logits (noise-robust)."""
def __init__(self, dim_q: int, dim_kv: int, heads: int = 4, dim_out: int | None = None) -> None:
super().__init__()
self.h = heads
d = dim_out or dim_q
self.q = nn.Linear(dim_q, d, bias=False)
self.k = nn.Linear(dim_kv, d, bias=False)
self.v = nn.Linear(dim_kv, d, bias=False)
self.o = nn.Linear(d, d, bias=False)
self.scale = (d // heads) ** -0.5


def forward(self, Q: torch.Tensor, KV: torch.Tensor) -> torch.Tensor:
q = self.q(Q); k = self.k(KV); v = self.v(KV)
B, Nq, D = q.shape; Nk = k.shape[1]; H = self.h; d = D // H
q = rearrange(q, "b n (h d) -> b h n d", h=H)
k = rearrange(k, "b n (h d) -> b h n d", h=H)
v = rearrange(v, "b n (h d) -> b h n d", h=H)
qn = nn.functional.normalize(q, dim=-1)
kn = nn.functional.normalize(k, dim=-1)
return out
