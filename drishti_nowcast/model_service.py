"""Multi-task nowcasting model (v0) and checkpoint I/O.

Architecture
  1. Temporal encoder: a ConvLSTM reads the satellite-derived frames
     (IWV, IWV tendency, CTT, CTT drop rate, QPE) over the last 3 hours.
  2. Context encoder: convolutions over the IMDAA thermodynamic and DEM
     terrain channels (CAPE, CIN, convergence, shear, steering, terrain).
  3. Cross-attention: satellite-state tokens (queries) attend to
     thermodynamic/terrain context tokens (keys/values) on a coarse 8x8 token
     grid, so each location can pull in the environment around it.
  4. Shared fused representation -> three hazard heads (see heads.py).

v0 uses a ConvLSTM as the temporal encoder. The planned v1 swaps it for a
spatiotemporal transformer encoder; the cross-attention fusion and heads stay.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .features import DYN_CHANNELS, STATIC_CHANNELS
from .heads import MultiTaskHeads


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hid: int):
        super().__init__()
        self.hid = hid
        self.gates = nn.Conv2d(in_ch + hid, 4 * hid, 3, padding=1)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.chunk(self.gates(torch.cat([x, h], 1)), 4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class CrossAttentionFusion(nn.Module):
    """Queries from the satellite state, keys/values from the context."""

    def __init__(self, ch: int, heads: int = 4, pool: int = 4):
        super().__init__()
        self.pool = pool
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.norm = nn.LayerNorm(ch)

    def forward(self, q_map, kv_map):
        b, c, h, w = q_map.shape
        q = F.avg_pool2d(q_map, self.pool).flatten(2).transpose(1, 2)     # [B, N, C]
        kv = F.avg_pool2d(kv_map, self.pool).flatten(2).transpose(1, 2)
        out, weights = self.attn(self.norm(q), self.norm(kv), self.norm(kv))
        out = out.transpose(1, 2).reshape(b, c, h // self.pool, w // self.pool)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False), weights


class NowcastNet(nn.Module):
    def __init__(self, hid: int = 32):
        super().__init__()
        self.cell = ConvLSTMCell(len(DYN_CHANNELS), hid)
        self.ctx = nn.Sequential(
            nn.Conv2d(len(STATIC_CHANNELS), hid, 3, padding=1), nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1), nn.GELU(),
        )
        self.xattn = CrossAttentionFusion(hid)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * hid, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4), nn.GELU(),
        )
        self.heads = MultiTaskHeads(64)

    def forward(self, dyn, sta, return_attention: bool = False):
        # dyn [B, T, C_d, H, W]; sta [B, C_s, H, W]
        b, t, _, h, w = dyn.shape
        hs = dyn.new_zeros(b, self.cell.hid, h, w)
        cs = torch.zeros_like(hs)
        for k in range(t):
            hs, cs = self.cell(dyn[:, k], (hs, cs))
        ctx = self.ctx(sta)
        att, weights = self.xattn(hs, ctx)
        feat = self.fuse(torch.cat([hs, ctx, att], 1))
        logits = self.heads(feat)                                   # [B, 3, L, H, W]
        return (logits, weights) if return_attention else logits


def save(model: NowcastNet, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(meta or {}, indent=2))


def load(path: str | Path) -> NowcastNet:
    model = NowcastNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
