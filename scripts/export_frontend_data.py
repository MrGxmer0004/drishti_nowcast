"""Export real model inference on held-out test scenes for the dashboard.

Runs the trained checkpoint on the synthetic test split, picks a spread of
scenes (cloudburst, flash flood, thunderstorm-only, quiet), and writes one
JSON per scene plus an index to frontend/public/data/. Every probability
shown in the dashboard comes from these files.

    python scripts/export_frontend_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drishti_nowcast import features, grid_store, model_service  # noqa: E402
from drishti_nowcast.config import DX_KM, GRID, HEADS, HIST_TIMES_H, LEADS_H  # noqa: E402
from drishti_nowcast.risk_engine import flow_accumulation  # noqa: E402
from drishti_nowcast.train import predict, to_tensors  # noqa: E402
from drishti_nowcast.xai import family_shares, integrated_gradients  # noqa: E402

OUT = Path("frontend/public/data")
ZONE = 8                       # 8 x 8 cells = 32 km zones -> 4 x 4 zones
ROWS = "ABCD"


def r2(a):
    return np.round(np.asarray(a, dtype=float), 3).tolist()


def pick_scenes(y, prob):
    cb_any = y[:, 1].reshape(len(y), -1).any(1)
    ff_any = y[:, 2].reshape(len(y), -1).any(1)
    ts_any = y[:, 0].reshape(len(y), -1).any(1)
    cb_peak = prob[:, 1, :2].reshape(len(y), -1).max(1)          # confident near-term cloudburst
    order = lambda m: [int(i) for i in np.argsort(-cb_peak) if m[i]]
    picks = []
    for pool, k in ((order(cb_any & ff_any), 4), (order(cb_any & ~ff_any), 2),
                    (order(ff_any & ~cb_any), 2), (order(ts_any & ~cb_any & ~ff_any), 2),
                    ([int(i) for i in np.flatnonzero(~ts_any & ~cb_any & ~ff_any)], 1)):
        picks += [i for i in pool if i not in picks][:k]
    return picks


def main():
    ds = grid_store.get_split("test", 400, 300, "data")
    dyn, sta, yt = to_tensors(ds)
    model = model_service.load("artifacts/nowcast_v0.pt")
    prob = predict(model, dyn, sta)
    y = yt.numpy().astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)

    index = []
    for n in pick_scenes(y, prob):
        raw = ds["dyn"][n]
        dem = ds["dem"][n]
        conv = features.convergence(ds["ctx"][n][2], ds["ctx"][n][3])
        cape = ds["ctx"][n][0]
        # XAI at the strongest forecast cell of each head (at its peak lead)
        xai = {}
        for h, head in enumerate(HEADS):
            j, iy, ix = np.unravel_index(np.argmax(prob[n, h]), prob[n, h].shape)
            if prob[n, h, j, iy, ix] < 0.2:
                continue
            per = integrated_gradients(model, dyn[n], sta[n], h, int(j), int(iy), int(ix), steps=24)
            xai[head] = {"lead_h": LEADS_H[j], "cell": [int(iy), int(ix)],
                         "p": round(float(prob[n, h, j, iy, ix]), 3), "families": family_shares(per)}
        # zone summaries
        zones = []
        for zi in range(GRID // ZONE):
            for zj in range(GRID // ZONE):
                sl = (slice(zi * ZONE, (zi + 1) * ZONE), slice(zj * ZONE, (zj + 1) * ZONE))
                zones.append({
                    "id": f"{ROWS[zi]}{zj + 1}", "row": zi, "col": zj,
                    "p": {head: [round(float(prob[n, h, j][sl].max()), 3) for j in range(len(LEADS_H))]
                          for h, head in enumerate(HEADS)},
                    "obs": {head: [int(y[n, h, j][sl].any()) for j in range(len(LEADS_H))]
                            for h, head in enumerate(HEADS)},
                    "series": {
                        "iwv_mean": r2(raw[:, 0][:, sl[0], sl[1]].mean(axis=(1, 2))),
                        "iwv_max": r2(raw[:, 0][:, sl[0], sl[1]].max(axis=(1, 2))),
                        "ctt_min": r2(raw[:, 1][:, sl[0], sl[1]].min(axis=(1, 2))),
                        "qpe_max": r2(raw[:, 2][:, sl[0], sl[1]].max(axis=(1, 2))),
                    },
                    "elev_mean": round(float(dem[sl].mean())),
                    "cape_max": round(float(cape[sl].max())),
                })
        scene = {
            "id": f"T{n:03d}", "test_index": n, "grid": GRID, "dx_km": DX_KM,
            "hist_times_h": list(HIST_TIMES_H), "leads_h": list(LEADS_H), "heads": list(HEADS),
            "prob": {head: [r2(prob[n, h, j]) for j in range(len(LEADS_H))] for h, head in enumerate(HEADS)},
            "obs": {head: [y[n, h, j].tolist() for j in range(len(LEADS_H))] for h, head in enumerate(HEADS)},
            "layers": {
                "dem": r2(np.round(dem)), "channel": (flow_accumulation(dem) >= 8).astype(int).tolist(),
                "iwv": r2(raw[-1, 0]), "iwv_rise": r2(raw[-1, 0] - raw[-4, 0]),
                "ctt": r2(raw[-1, 1]), "qpe": r2(raw[-1, 2]), "convergence": r2(conv),
            },
            "zones": zones, "xai": xai,
        }
        (OUT / f"{scene['id']}.json").write_text(json.dumps(scene, separators=(",", ":")))
        peak = {head: round(float(prob[n, h].max()), 2) for h, head in enumerate(HEADS)}
        events = {head: int(y[n, h].any()) for h, head in enumerate(HEADS)}
        index.append({"id": scene["id"], "peak": peak, "observed": events})
        print(scene["id"], peak, events)

    metrics = json.loads(Path("results/metrics.json").read_text())
    meta = json.loads(Path("artifacts/nowcast_v0.json").read_text())
    (OUT / "index.json").write_text(json.dumps({"scenes": index, "model": meta, "metrics": metrics}, indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
