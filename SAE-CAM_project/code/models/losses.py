
from __future__ import annotations
import torch
from torch import nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al. 2020) simplified."""
    def __init__(self, temperature: float=0.07):
        super().__init__()
        self.t = temperature
    def forward(self, z: torch.Tensor, y: torch.Tensor):
        # z: [B, D]; y: [B]
        z = torch.nn.functional.normalize(z, dim=1)
        sim = z @ z.t() / self.t                       # [B,B]
        labels = y.unsqueeze(0) == y.unsqueeze(1)      # [B,B]
        mask = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, -1e9)
        # for each i, logsumexp negatives
        logsumexp_neg = torch.logsumexp(sim.masked_fill(labels, -1e9), dim=1)
        pos = sim.masked_fill(~labels, -1e9)
        pos_exp = torch.logsumexp(pos, dim=1)
        loss = -(pos_exp - logsumexp_neg).mean()
        return loss

def mmd_loss(x: torch.Tensor, y: torch.Tensor, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """MMD with Gaussian kernel (optional, unused by default)."""
    n = x.size(0)
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n**2 - n)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernels = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
    kernel_sum = sum(kernels)
    XX = kernel_sum[:n, :n]; YY = kernel_sum[n:, n:]; XY = kernel_sum[:n, n:]; YX = kernel_sum[n:, :n]
    return torch.mean(XX + YY - XY - YX)
