
from __future__ import annotations
import torch, math
from torch import nn
from einops import rearrange
from .cbam import CBAM1D
from .rellm import ReLLM

def window_snr(x, win=64, eps=1e-6):
    # x: [B, C, T]
    B,C,T = x.shape
    unfolded = x.unfold(dimension=2, size=win, step=win//2)  # [B,C,Nw,win]
    power = unfolded.pow(2).mean(dim=-1)      # [B,C,Nw]
    signal = unfolded.abs().mean(dim=-1)      # proxy
    snr = (signal / (power.sqrt()+eps)).clamp(0, 10.0) # simple proxy gate
    # upsample to T
    snr = torch.nn.functional.interpolate(snr, size=T, mode="linear", align_corners=False)
    return snr

class ConvEncoder(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.Conv1d(base, base*2, 5, stride=2, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.Conv1d(base*2, base*4, 3, stride=2, padding=1, dilation=1), nn.BatchNorm1d(base*4), nn.ReLU(),
            CBAM1D(base*4),
        )
        self.out_ch = base*4
    def forward(self, x):
        return self.net(x)  # [B, C, T/4]

class BiCrossAttention(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, heads: int=4, dim_out: int|None=None):
        super().__init__()
        self.h = heads
        d = dim_out or dim_q
        self.q_proj = nn.Linear(dim_q, d, bias=False)
        self.k_proj = nn.Linear(dim_kv, d, bias=False)
        self.v_proj = nn.Linear(dim_kv, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.scale = (d//heads) ** -0.5

    def forward(self, Q, KV):
        # Q: [B, Nq, Dq], KV: [B, Nk, Dk]
        q = self.q_proj(Q); k = self.k_proj(KV); v = self.v_proj(KV)
        B, Nq, D = q.shape; Nk = k.shape[1]; H = self.h; d = D//H
        q = rearrange(q, "b n (h d) -> b h n d", h=H)
        k = rearrange(k, "b n (h d) -> b h n d", h=H)
        v = rearrange(v, "b n (h d) -> b h n d", h=H)
        # cosine attention for noise robustness
        qn = torch.nn.functional.normalize(q, dim=-1)
        kn = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.einsum("bhnd,bhmd->bhnm", qn, kn) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.o_proj(out)  # [B, Nq, D]

class SAE_CAM(nn.Module):
    def __init__(self, n_classes: int, t_embed: int=128):
        super().__init__()
        # encoders for vibration & current
        self.enc_v = ConvEncoder(1, base=32)
        self.enc_c = ConvEncoder(1, base=32)
        # semantic branch (ReLLM proxy)
        self.rellm = ReLLM(in_dim=6, hidden=128, out_dim=t_embed)
        # projection to token sequences
        self.proj_v = nn.Conv1d(self.enc_v.out_ch, 128, 1)
        self.proj_c = nn.Conv1d(self.enc_c.out_ch, 128, 1)
        self.pos_v = nn.Parameter(torch.randn(1, 128, 64))
        self.pos_c = nn.Parameter(torch.randn(1, 128, 64))
        # cross attention (both directions)
        self.ca_v_t = BiCrossAttention(128, t_embed, heads=4, dim_out=128)
        self.ca_t_v = BiCrossAttention(t_embed, 128, heads=4, dim_out=128)
        # heads
        self.fc_shared = nn.Linear(128, 128)
        self.fc_private_v = nn.Linear(128, 128)
        self.fc_private_c = nn.Linear(128, 128)
        self.classifier = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, n_classes))
        # loss helpers
        self.ort_lambda = 1.0

    def forward(self, x):
        # x: [B, 2, T]
        vib = x[:,0:1,:]; cur = x[:,1:2,:]
        ev = self.enc_v(vib)   # [B, Cv, Tv]
        ec = self.enc_c(cur)   # [B, Cc, Tc]
        # SNR-gated mask (proxy)
        snr_v = window_snr(ev); snr_c = window_snr(ec)
        ev = ev * (snr_v.sigmoid())
        ec = ec * (snr_c.sigmoid())
        # project to tokens
        tv = torch.nn.functional.interpolate(self.proj_v(ev), size=self.pos_v.shape[-1], mode="linear")
        tc = torch.nn.functional.interpolate(self.proj_c(ec), size=self.pos_c.shape[-1], mode="linear")
        tv = tv + self.pos_v; tc = tc + self.pos_c  # [B, 128, L]
        Tv = tv.transpose(1,2); Tc = tc.transpose(1,2)  # [B, L, 128]
        # build lightweight "text" stats from raw x for ReLLM proxy
        stats = self._stats(x)  # [B, 6]
        Tsem = self.rellm(stats)[:, None, :]  # [B, 1, 128]

        # bidirectional cross-attention
        # (1) vib/private attends to text
        v_from_t = self.ca_v_t(Tv, Tsem).mean(dim=1)          # [B, 128]
        t_from_v = self.ca_t_v(Tsem, Tv).mean(dim=1)          # [B, 128]
        c_from_t = self.ca_v_t(Tc, Tsem).mean(dim=1)          # [B, 128]
        t_from_c = self.ca_t_v(Tsem, Tc).mean(dim=1)          # [B, 128]

        shared = self.fc_shared(0.5*(t_from_v + t_from_c))
        pv = self.fc_private_v(v_from_t)
        pc = self.fc_private_c(c_from_t)

        # orthogonality loss for shared vs private
        loss_ort = (shared*pv).mean().abs() + (shared*pc).mean().abs()

        # simple dynamic redundancy penalty proxy (L2 alignment penalty)
        loss_dyn = 0.5*((pv-pc)**2).mean()

        z = torch.cat([shared, pv, pc], dim=-1)  # [B, 384]
        logits = self.classifier(z)
        return {"logits": logits, "z": z, "loss_ort": loss_ort, "loss_dyn": loss_dyn}

    @staticmethod
    def _stats(x):
        # x: [B, 2, T]
        vib = x[:,0]; cur = x[:,1]
        def s3(t):
            peak = t.abs().amax(dim=1)
            rms = (t.pow(2).mean(dim=1).sqrt())
            kurt = ( ((t - t.mean(dim=1, keepdim=True))**4).mean(dim=1) / (t.var(dim=1)+1e-6)**2 ).clamp(0, 50.)
            return torch.stack([peak, rms, kurt], dim=1)
        sv = s3(vib); sc = s3(cur)
        return torch.cat([sv, sc], dim=1)  # [B, 6]
