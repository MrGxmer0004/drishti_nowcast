"""Figures for the README: one test case (inputs, truth, forecast) and training curve."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_nowcast import features, grid_store, model_service  # noqa: E402
from drishti_nowcast.config import HEADS, LEADS_H  # noqa: E402
from drishti_nowcast.train import predict, to_tensors  # noqa: E402

ds = grid_store.get_split("test", 400, 300, "data")
dyn, sta, y = to_tensors(ds)
prob = predict(model_service.load("artifacts/nowcast_v0.pt"), dyn, sta)
ex = json.loads(Path("results/xai_example.json").read_text())
n = ex["test_sample"]
cb, ff = HEADS.index("cloudburst"), HEADS.index("flash_flood")
j3, j4 = LEADS_H.index(3), LEADS_H.index(4)

fig, ax = plt.subplots(2, 4, figsize=(15, 7.4), constrained_layout=True)
raw = ds["dyn"][n]
conv = features.convergence(ds["ctx"][n][2], ds["ctx"][n][3])
panels = [
    (raw[-1, 0], "IWV at issue (mm)", "Blues", None),
    (raw[-1, 0] - raw[-4, 0], "IWV rise over 90 min (mm)", "PuBu", None),
    (raw[-1, 1], "Cloud-top temperature (K)", "magma", None),
    (conv, "850 hPa convergence (1e-5 s$^{-1}$)", "RdBu_r", (-4, 4)),
    (y[n, cb, j3].numpy(), "Observed cloudburst, +3 h", "Reds", (0, 1)),
    (prob[n, cb, j3], "Forecast P(cloudburst), +3 h", "Reds", (0, 1)),
    (y[n, ff, j4].numpy(), "Observed flash flood, +4 h", "Purples", (0, 1)),
    (prob[n, ff, j4], "Forecast P(flash flood), +4 h", "Purples", (0, 1)),
]
for a, (img, title, cmap, lim) in zip(ax.ravel(), panels):
    im = a.imshow(img, cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
    a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a, shrink=0.8)
fig.suptitle("Held-out synthetic test case · 128 km domain, 4 km grid", fontsize=12)
fig.savefig("results/example_case.png", dpi=130)

hist = json.loads(Path("results/train_history.json").read_text())
fig, a = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
e = [h["epoch"] for h in hist]
a.plot(e, [h["val_pr_auc_severe"] for h in hist], marker="o", color="#B4532A")
a.set_xlabel("epoch"); a.set_ylabel("val PR-AUC (cloudburst + flash flood)")
a.grid(alpha=0.3); a.set_ylim(0, 1)
fig.savefig("results/training_curve.png", dpi=130)
print("figures written")
