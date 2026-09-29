"""Train the multi-task nowcasting model.

    python -m drishti_nowcast.train --epochs 15

Loss: focal loss per hazard head (rare-event classification), with the two
severe heads (cloudburst, flash flood) weighted above thunderstorm. The
checkpoint kept is the one with the best validation PR-AUC averaged over
the cloudburst and flash-flood heads.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from . import features, grid_store, model_service
from .config import HEADS, LEADS_H

HEAD_WEIGHTS = torch.tensor([1.0, 1.5, 1.5])


def focal_loss(logits, target, alpha=0.85, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return a_t * (1 - p_t) ** gamma * ce


def to_tensors(ds):
    dyn_f, sta_f = features.build(ds)
    return (torch.from_numpy(dyn_f), torch.from_numpy(sta_f),
            torch.from_numpy(ds["y"].astype(np.float32)))


@torch.no_grad()
def predict(model, dyn, sta, batch=32):
    model.eval()
    out = [torch.sigmoid(model(dyn[i:i + batch], sta[i:i + batch])) for i in range(0, len(dyn), batch)]
    return torch.cat(out).numpy()


def pr_auc_table(prob, y):
    table = {}
    for h, name in enumerate(HEADS):
        for j, L in enumerate(LEADS_H):
            yt = y[:, h, j].ravel()
            table[f"{name}@+{L}h"] = float(average_precision_score(yt, prob[:, h, j].ravel())) if yt.any() else None
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2400)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="artifacts/nowcast_v0.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train = to_tensors(grid_store.get_split("train", args.n_train, 100 + args.seed, args.data_dir))
    val = to_tensors(grid_store.get_split("val", args.n_val, 200 + args.seed, args.data_dir))

    model = model_service.NowcastNet()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=(len(train[0]) + args.batch - 1) // args.batch)
    history, best, best_state = [], -1.0, None
    print(f"params: {n_params:,}  train: {len(train[0])}  val: {len(val[0])}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, perm, total = time.time(), torch.randperm(len(train[0])), 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            dyn, sta, y = train[0][idx], train[1][idx], train[2][idx]
            if torch.rand(1) < 0.5:                                  # flip augmentation
                dyn, sta, y = dyn.flip(-1), sta.flip(-1), y.flip(-1)
                sta = sta.clone(); sta[:, 4] = -sta[:, 4]            # steering u flips sign
            loss_map = focal_loss(model(dyn, sta), y)                # [B, 3, L, H, W]
            loss = (loss_map.mean(dim=(0, 2, 3, 4)) * HEAD_WEIGHTS).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(idx)
        prob = predict(model, val[0], val[1])
        table = pr_auc_table(prob, val[2].numpy())
        severe = np.mean([v for k, v in table.items() if not k.startswith("thunderstorm") and v is not None])
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_pr_auc_severe": severe,
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:2d}  loss {total / len(perm):.5f}  val PR-AUC (cb+ff) {severe:.3f}  {time.time() - t0:.0f}s")
        if severe > best:
            best, best_state = severe, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model_service.save(model, args.out, {
        "model": "NowcastNet v0 (ConvLSTM + cross-attention fusion + 3 heads)",
        "params": n_params, "epochs": args.epochs, "best_val_pr_auc_severe": best,
        "data": "synthetic (drishti_nowcast.synthetic)", "seed": args.seed,
        "heads": list(HEADS), "leads_h": list(LEADS_H),
    })
    Path("results").mkdir(exist_ok=True)
    Path("results/train_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved {args.out}  best val PR-AUC (cb+ff) {best:.3f}")


if __name__ == "__main__":
    main()
