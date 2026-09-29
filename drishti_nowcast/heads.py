"""Task-specific output heads for multi-task learning.

The shared backbone produces one feature map. Each hazard gets its own small
head that maps it to a probability map per lead time (+2, +3, +4, +6 h).
"""
from __future__ import annotations

import torch
from torch import nn

from .config import HEADS, LEADS_H


class HazardHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 32, n_leads: int = len(LEADS_H)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, n_leads, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                           # logits [B, n_leads, H, W]


class MultiTaskHeads(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.heads = nn.ModuleDict({h: HazardHead(in_ch) for h in HEADS})

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.heads[h](feat) for h in HEADS], dim=1)   # [B, 3, L, H, W]
