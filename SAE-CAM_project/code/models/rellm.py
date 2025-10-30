from __future__ import annotations
from typing import Tuple
import torch
from torch import nn


__all__ = ["ReLLM"]




class ReLLM(nn.Module):
"""A compact proxy for a LoRA-tuned GPT-2 encoder.


This module emulates extracting hidden states from a frozen LLM with LoRA adapters.
In practice, replace the MLP stack with a true transformer and keep the interface.


Inputs:
stats: [B, S] shallow descriptors (e.g., peak, rms, kurtosis for vib & cur)
Outputs:
tokens: [B, L, D] token sequence for cross-attention (L can be 1 for global)
pooled: [B, D] pooled semantic vector
"""
def __init__(self, in_dim: int = 6, width: int = 256, depth: int = 4, tokens: int = 4, out_dim: int = 128) -> None:
super().__init__()
layers = []
d = in_dim
for i in range(depth):
layers += [nn.Linear(d, width), nn.GELU(), nn.LayerNorm(width)]
d = width
self.backbone = nn.Sequential(*layers)
self.proj = nn.Linear(width, out_dim)
self.tokenizer = nn.Linear(out_dim, tokens * out_dim)
self.tokens = tokens
self.out_dim = out_dim
self.pool = nn.AdaptiveAvgPool1d(1)


def forward(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
h = self.backbone(stats)
e = self.proj(h)
seq = self.tokenizer(e).view(e.size(0), self.tokens, self.out_dim) # [B,L,D]
pooled = e # global semantic summary
return seq, pooled
