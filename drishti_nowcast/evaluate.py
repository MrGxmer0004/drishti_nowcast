"""Evaluate the trained model against two baselines on a held-out test set.

    python -m drishti_nowcast.evaluate

Baselines
  * Persistence: whatever is happening now keeps happening (a thunderstorm
    or cloudburst observed at issue time is forecast at every lead).
  * Gradient boosting: per-pixel HistGradientBoosting on the same engineered
    features at issue time (no spatial or temporal context beyond them).

Metrics per hazard and lead time: PR-AUC, Brier score, and at a threshold
chosen on the validation set (max CSI): POD, FAR, CSI, plus the Fractions
Skill Score over a 5x5-cell (20 km) neighbourhood.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from . import grid_store, model_service
from .config import CLOUDBURST_MM_PER_H, HEADS, LEADS_H, THUNDER_CTT_K, THUNDER_RAIN_MM_PER_H
from .train import predict, to_tensors


def categorical(pred, obs):
    tp = np.sum(pred & obs); fp = np.sum(pred & ~obs); fn = np.sum(~pred & obs)
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    return pod, far, csi


def fss(pred, obs, size=5):
    """Fractions Skill Score over size x size neighbourhoods (per scene, pooled)."""
    pf = np.stack([uniform_filter(p.astype(float), size, mode="constant") for p in pred])
    of = np.stack([uniform_filter(o.astype(float), size, mode="constant") for o in obs])
    mse = np.mean((pf - of) ** 2)
    ref = np.mean(pf ** 2) + np.mean(of ** 2)
    return 1 - mse / ref if ref > 0 else np.nan


def best_threshold(prob, obs):
    grid = np.linspace(0.05, 0.95, 19)
    scores = [categorical(prob >= t, obs)[2] for t in grid]
    scores = np.nan_to_num(scores, nan=-1)
    return float(grid[int(np.argmax(scores))])


def scores(prob, obs, thr):
    pred = prob >= thr
    pod, far, csi = categorical(pred, obs)
    return {
        "pr_auc": float(average_precision_score(obs.ravel(), prob.ravel())) if obs.any() else None,
        "brier": float(np.mean((prob - obs) ** 2)),
        "threshold": thr, "pod": float(pod), "far": float(far), "csi": float(csi),
        "fss_20km": float(fss(pred, obs)),
        "base_rate": float(obs.mean()),
    }


def pixel_table(dyn_f, sta_f):
    """Per-pixel features at issue time for the tabular baseline."""
    last = dyn_f[:, -1]                                   # [N, 5, H, W]
    x = np.concatenate([last, sta_f], 1)                  # [N, 14, H, W]
    return x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2400)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--model", default="artifacts/nowcast_v0.pt")
    args = ap.parse_args()

    ds = {s: grid_store.get_split(s, n, off + args.seed, args.data_dir)
          for s, n, off in (("train", args.n_train, 100), ("val", args.n_val, 200), ("test", args.n_test, 300))}
    tens = {s: to_tensors(d) for s, d in ds.items()}
    model = model_service.load(args.model)
    prob = {s: predict(model, tens[s][0], tens[s][1]) for s in ("val", "test")}
    y = {s: ds[s]["y"].astype(bool) for s in ds}

    # persistence: current observed state, identical at every lead
    def persist(d):
        qpe, ctt = d["dyn"][:, -1, 2], d["dyn"][:, -1, 1]
        ts = (qpe >= THUNDER_RAIN_MM_PER_H) & (ctt <= THUNDER_CTT_K)
        cb = qpe >= CLOUDBURST_MM_PER_H
        ff = d["y"][:, 2, 0].astype(bool) & False            # no flood observation at issue time
        return np.stack([ts, cb, ff], 1)                       # [N, 3, H, W]

    # gradient boosting on per-pixel features
    rng = np.random.default_rng(args.seed)
    xtr = pixel_table(tens["train"][0].numpy(), tens["train"][1].numpy())
    xte = pixel_table(tens["test"][0].numpy(), tens["test"][1].numpy())
    xva = pixel_table(tens["val"][0].numpy(), tens["val"][1].numpy())

    results = {"model": {}, "persistence": {}, "gradient_boosting": {}}
    n_te = len(y["test"])
    for h, name in enumerate(HEADS):
        for j, L in enumerate(LEADS_H):
            key = f"{name}@+{L}h"
            obs_te, obs_va = y["test"][:, h, j], y["val"][:, h, j]

            thr = best_threshold(prob["val"][:, h, j], obs_va)
            results["model"][key] = scores(prob["test"][:, h, j], obs_te, thr)

            pp = persist(ds["test"])[:, h].astype(float)
            results["persistence"][key] = scores(pp, obs_te, 0.5)

            ytr = y["train"][:, h, j].ravel()
            pos = np.flatnonzero(ytr)
            neg = rng.choice(np.flatnonzero(~ytr), size=min(len(ytr) - len(pos), 40 * max(len(pos), 1), 300_000), replace=False)
            idx = np.concatenate([pos, neg])
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=args.seed)
            gb.fit(xtr[idx], ytr[idx])
            # correct probabilities for negative subsampling
            w = len(neg) / max(np.sum(~ytr), 1)
            def calib(p): return (p * w) / (p * w + (1 - p) + 1e-12)
            pva = calib(gb.predict_proba(xva)[:, 1]).reshape(obs_va.shape)
            pte = calib(gb.predict_proba(xte)[:, 1]).reshape(n_te, *obs_te.shape[1:])
            results["gradient_boosting"][key] = scores(pte, obs_te, best_threshold(pva, obs_va))
            m, g = results["model"][key], results["gradient_boosting"][key]
            print(f"{key:24s} model PR-AUC {m['pr_auc']:.3f} CSI {m['csi']:.3f} | GBM PR-AUC {g['pr_auc']:.3f} CSI {g['csi']:.3f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(clean(results), indent=2, allow_nan=False))
    write_markdown(results)


def clean(obj):
    """Replace NaN (e.g. FAR with no forecasts) by null so the file is valid JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def write_markdown(results):
    def f(v): return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"
    lines = ["# Test-set metrics (synthetic data)", "",
             "Held-out synthetic test set. These numbers show the pipeline learns the physics encoded in",
             "`synthetic.py`; they are **not** estimates of skill on real observations.", "",
             "| Hazard @ lead | Base rate | Model PR-AUC | GBM PR-AUC | Persistence CSI | Model POD | Model FAR | Model CSI | GBM CSI | Model FSS (20 km) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in results["model"].items():
        g, p = results["gradient_boosting"][key], results["persistence"][key]
        lines.append(f"| {key} | {m['base_rate']*100:.2f}% | {f(m['pr_auc'])} | {f(g['pr_auc'])} | {f(p['csi'])} | "
                     f"{f(m['pod'])} | {f(m['far'])} | {f(m['csi'])} | {f(g['csi'])} | {f(m['fss_20km'])} |")
    Path("results/metrics.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
