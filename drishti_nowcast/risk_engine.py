"""DEM Flood Router.

Turns rainfall (or cloudburst probability) into flash-flood risk by pushing it
down the terrain's drainage network. Uses D8 flow directions: every cell
drains to its steepest-descent neighbour.

Two uses:
  * `route_runoff` sums runoff downstream (used to build flash-flood labels
    from simulated rainfall, and for physically-based flood scoring).
  * `route_probability` carries the highest upstream cloudburst probability
    downstream with distance decay (used to translate the model's cloudburst
    head into ward-level flood exposure).
"""
from __future__ import annotations

import numpy as np

from .config import DX_KM

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def d8_receivers(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (receiver, order).

    receiver[k] is the flat index of the cell that cell k drains into, or -1
    for pits and cells that drain off the grid edge. order lists cells from
    highest to lowest, which is a valid upstream-to-downstream processing order.
    """
    h, w = dem.shape
    flat = dem.ravel()
    receiver = np.full(h * w, -1, dtype=np.int64)
    for i in range(h):
        for j in range(w):
            z = dem[i, j]
            best, best_drop = -1, 0.0
            for di, dj in _NEIGHBOURS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < h and 0 <= nj < w):
                    continue
                dist = np.hypot(di, dj)
                drop = (z - dem[ni, nj]) / dist
                if drop > best_drop:
                    best, best_drop = ni * w + nj, drop
            receiver[i * w + j] = best
    order = np.argsort(-flat, kind="stable")
    return receiver, order


def flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Number of cells (including itself) draining through each cell."""
    receiver, order = d8_receivers(dem)
    acc = np.ones(dem.size)
    for k in order:
        r = receiver[k]
        if r >= 0:
            acc[r] += acc[k]
    return acc.reshape(dem.shape)


def route_runoff(runoff: np.ndarray, receiver: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Accumulate runoff (any additive quantity) downstream."""
    acc = runoff.astype(np.float64).ravel().copy()
    for k in order:
        r = receiver[k]
        if r >= 0:
            acc[r] += acc[k]
    return acc.reshape(runoff.shape)


def route_probability(p: np.ndarray, receiver: np.ndarray, order: np.ndarray,
                      decay_km: float = 40.0) -> np.ndarray:
    """Carry the highest upstream probability downstream, decaying with distance."""
    h, w = p.shape
    out = p.astype(np.float64).ravel().copy()
    for k in order:
        r = receiver[k]
        if r < 0:
            continue
        di = abs(k // w - r // w)
        dj = abs(k % w - r % w)
        step_km = DX_KM * np.hypot(di, dj)
        out[r] = max(out[r], out[k] * np.exp(-step_km / decay_km))
    return out.reshape(p.shape)


def slope(dem: np.ndarray) -> np.ndarray:
    """Terrain slope as rise over run (m/m)."""
    gy, gx = np.gradient(dem, DX_KM * 1000.0)
    return np.hypot(gx, gy)
