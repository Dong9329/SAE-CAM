from __future__ import annotations
kernel_num: int = 5, fix_sigma: float | None = None) -> torch.Tensor:
"""Gaussian-kernel MMD (used for working-condition alignment)."""
n = x.size(0)
total = torch.cat([x, y], dim=0)
total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
L2 = ((total0 - total1) ** 2).sum(2)
if fix_sigma:
bw = fix_sigma
else:
bw = torch.sum(L2.detach()) / (n * n - n)
bw /= kernel_mul ** (kernel_num // 2)
bws = [bw * (kernel_mul ** i) for i in range(kernel_num)]
kernels = [torch.exp(-L2 / b) for b in bws]
K = sum(kernels)
XX, YY, XY, YX = K[:n, :n], K[n:, n:], K[:n, n:], K[n:, :n]
return (XX.mean() + YY.mean() - XY.mean() - YX.mean())




class OrthogonalityLoss(nn.Module):
"""Explicit orthogonality constraint between shared & private features."""
def __init__(self, reduction: str = "mean") -> None:
super().__init__()
self.reduction = reduction


def forward(self, shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
# Penalize correlation via squared cosine similarity
s = torch.nn.functional.normalize(shared, dim=-1)
p = torch.nn.functional.normalize(private, dim=-1)
cos = (s * p).sum(-1).pow(2)
if self.reduction == "mean":
return cos.mean()
return cos.sum()




class StageAwareRedundancy(nn.Module):
"""Proxy for Eq.(11)-(13): penalize redundant similarity with stage-aware λ(t).


Args:
base: λ_base
cap: upper bound (stability guard)
"""
def __init__(self, base: float = 0.2, cap: float = 3.0) -> None:
super().__init__()
self.base = base
self.cap = cap


@staticmethod
def _stage_weights(mu_delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
# Fuzzy stage membership (Gaussian bumps; constants from paper text)
q_early = torch.exp(-((mu_delta - 0.005) ** 2) / (0.002 ** 2))
q_mid = torch.exp(-((mu_delta - 0.03) ** 2) / (0.01 ** 2))
q_late = torch.exp(-((sigma - 0.08) ** 2) / (0.03 ** 2))
# λ(t) = λ_base*(1 + 2*q_late - 0.5*q_early)
lam = 1.0 + 2.0 * q_late - 0.5 * q_early
return lam.clamp(0.5, 3.0), q_early, q_mid, q_late


def forward(self, z_v: torch.Tensor, z_c: torch.Tensor,
mu_delta: torch.Tensor, sigma: torch.Tensor) -> Dict[str, torch.Tensor]:
# Contrastive-style redundancy metric using InfoNCE denominator
# Build pairwise similarity across batch (proxy for cross-modal redundancy)
zv = torch.nn.functional.normalize(z_v, dim=-1)
zc = torch.nn.functional.normalize(z_c, dim=-1)
sim_pos = (zv * zc).sum(-1) # [B]
# Negative bag: in-batch others (stop-grad to stabilize)
sim_neg = zv @ zc.detach().t() # [B,B]
eye = torch.eye(zv.size(0), device=zv.device, dtype=torch.bool)
neg_logsumexp = torch.logsumexp(sim_neg.masked_fill(eye, -1e9), dim=1)
raw = (neg_logsumexp - sim_pos).clamp_max(self.cap).mean()
lam, q_e, q_m, q_l = self._stage_weights(mu_delta, sigma)
loss = (self.base * lam.detach() * raw).mean()
return {"loss": loss, "lambda": lam.mean().detach(), "q_early": q_e.mean().detach(),
"q_mid": q_m.mean().detach(), "q_late": q_l.mean().detach()}
