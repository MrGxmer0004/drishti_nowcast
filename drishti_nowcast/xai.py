"""Explainable AI: which predictors drove a forecast.

Integrated Gradients (Sundararajan et al., 2017) from a neutral baseline
(all scaled features = 0, i.e. near-climatological values) to the actual
input, for one hazard / lead / location. Attributions are summed per input
channel and grouped into the predictor families shown on the dashboard.

    python -m drishti_nowcast.xai
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import grid_store, model_service
from .config import HEADS, LEADS_H
from .features import DYN_CHANNELS, FAMILIES, STATIC_CHANNELS
from .train import predict, to_tensors


def integrated_gradients(model, dyn, sta, head: int, lead: int, iy: int, ix: int,
                         steps: int = 32, radius: int = 1):
    """dyn [T, Cd, H, W], sta [Cs, H, W] -> per-channel attribution dict."""
    model.eval()
    dyn0, sta0 = torch.zeros_like(dyn), torch.zeros_like(sta)
    total_d, total_s = torch.zeros_like(dyn), torch.zeros_like(sta)
    ys = slice(max(iy - radius, 0), iy + radius + 1)
    xs = slice(max(ix - radius, 0), ix + radius + 1)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        d = (dyn0 + a * (dyn - dyn0)).unsqueeze(0).requires_grad_(True)
        s = (sta0 + a * (sta - sta0)).unsqueeze(0).requires_grad_(True)
        out = model(d, s)[0, head, lead, ys, xs].mean()
        gd, gs = torch.autograd.grad(out, (d, s))
        total_d += gd[0]; total_s += gs[0]
    attr_d = ((dyn - dyn0) * total_d / steps).sum(dim=(0, 2, 3))
    attr_s = ((sta - sta0) * total_s / steps).sum(dim=(1, 2))
    per_channel = {c: float(v) for c, v in zip(DYN_CHANNELS, attr_d)}
    per_channel.update({c: float(v) for c, v in zip(STATIC_CHANNELS, attr_s)})
    return per_channel


def family_shares(per_channel: dict) -> dict:
    mags = {fam: sum(abs(per_channel[c]) for c in chans) for fam, chans in FAMILIES.items()}
    tot = sum(mags.values()) or 1.0
    return {fam: round(v / tot, 3) for fam, v in sorted(mags.items(), key=lambda kv: -kv[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="cloudburst", choices=HEADS)
    ap.add_argument("--lead", type=int, default=3, choices=LEADS_H)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="artifacts/nowcast_v0.pt")
    args = ap.parse_args()

    ds = grid_store.get_split("test", args.n_test, 300 + args.seed, args.data_dir)
    dyn, sta, y = to_tensors(ds)
    model = model_service.load(args.model)
    h, j = HEADS.index(args.head), LEADS_H.index(args.lead)
    prob = predict(model, dyn, sta)[:, h, j]
    # the most confident correct forecast in the test set
    hit = np.where(y[:, h, j].numpy() > 0, prob, 0)
    n, iy, ix = np.unravel_index(np.argmax(hit), hit.shape)
    per_channel = integrated_gradients(model, dyn[n], sta[n], h, j, iy, ix)
    result = {
        "head": args.head, "lead_h": args.lead, "test_sample": int(n), "cell": [int(iy), int(ix)],
        "probability": float(prob[n, iy, ix]), "observed": bool(y[n, h, j, iy, ix]),
        "family_shares": family_shares(per_channel), "per_channel": per_channel,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/xai_example.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("head", "lead_h", "probability", "family_shares")}, indent=2))


if __name__ == "__main__":
    main()
