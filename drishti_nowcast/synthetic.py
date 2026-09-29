"""Synthetic but physically structured training data.

There is no real INSAT / IMDAA data in this repository yet. This module
generates scenes that follow the same causal story the real model has to
learn, so the full pipeline (features, model, evaluation, XAI) can be built
and tested end to end:

  * Convective cells form over low-level convergence zones.
  * Integrated water vapour (IWV) pools about 2 h before a cell peaks.
  * Cloud tops cool rapidly (CTT drop) about 1 h before the peak.
  * Peak rain rate grows with CAPE and available moisture, and weak vertical
    wind shear keeps a cell stationary, so rain piles up in one place
    (the cloudburst case).
  * Rain that falls on steep terrain runs off and is routed down the DEM
    drainage network, producing flash floods in the channels downstream.

Nothing here is a substitute for real data. Metrics on this data only show
that the pipeline learns the encoded physics; they say nothing about skill on
real Indian monsoon convection.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import (CLOUDBURST_MM_PER_H, DX_KM, FLOOD_MIN_FLOWACC, FLOOD_ROUTED_MM, GRID,
                     HIST_TIMES_H, LEADS_H, THUNDER_CTT_K, THUNDER_RAIN_MM_PER_H)
from .risk_engine import d8_receivers, route_runoff, slope

KMH_PER_MS = 3.6


def _smooth_unit(rng, n, sigma):
    f = gaussian_filter(rng.normal(size=(n, n)), sigma, mode="reflect")
    f -= f.min()
    return f / (f.max() + 1e-9)


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_dem(rng, n=GRID):
    """Mountain terrain: broad massif + ridges + a regional tilt so rivers form."""
    broad = _smooth_unit(rng, n, 6)
    ridges = 1.0 - np.abs(2 * _smooth_unit(rng, n, 2.5) - 1)
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:n, 0:n] / n
    tilt = np.cos(ang) * xx + np.sin(ang) * yy
    return 700 + 1600 * broad + 500 * ridges + 900 * tilt


def _gauss(xx, yy, cx, cy, s):
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))


def make_scene(rng, n=GRID):
    """One sample: satellite history, IMDAA-style context, DEM and 3x4 label maps."""
    yy, xx = np.mgrid[0:n, 0:n] * DX_KM           # km coordinates
    dem = make_dem(rng, n)
    slp = slope(dem)
    receiver, order = d8_receivers(dem)

    # --- thermodynamic and kinematic environment (IMDAA-like, at issue time)
    cape = 300 + 2900 * _smooth_unit(rng, n, 5) * rng.uniform(0.5, 1.0)
    cin = -(15 + 150 * _smooth_unit(rng, n, 5))
    iwv_base = 26 + 16 * _smooth_unit(rng, n, 6) * rng.uniform(0.6, 1.0)
    u850 = rng.normal(0, 4) + gaussian_filter(rng.normal(0, 1.0, (n, n)), 4)
    v850 = rng.normal(0, 4) + gaussian_filter(rng.normal(0, 1.0, (n, n)), 4)
    shear_mag = rng.uniform(2, 22)
    shear_dir = rng.uniform(0, 2 * np.pi)
    u500 = u850 + shear_mag * np.cos(shear_dir)
    v500 = v850 + shear_mag * np.sin(shear_dir)

    # --- storm cells
    n_cells = min(4, rng.poisson(2.2))
    cells = []
    for _ in range(n_cells):
        cx, cy = rng.uniform(0.1, 0.9, 2) * n * DX_KM
        tp = rng.uniform(-0.5, 7.5)                 # peak time, h after issue
        ix, iy = int(np.clip(cx / DX_KM, 0, n - 1)), int(np.clip(cy / DX_KM, 0, n - 1))
        cape_l, iwv_l = cape[iy, ix], iwv_base[iy, ix]
        oro = 1.0 + 25.0 * slp[iy, ix]              # orographic enhancement
        peak = (15 + 300 * _sig((cape_l - 1400) / 450) * np.clip((iwv_l - 26) / 12, 0.1, 1.5)
                * (0.35 + 0.65 * np.clip(1 - shear_mag / 20, 0, 1)) * min(oro, 1.8) * rng.uniform(0.7, 1.25))
        steer_u = 0.5 * (u850[iy, ix] + u500[iy, ix]) * KMH_PER_MS
        steer_v = 0.5 * (v850[iy, ix] + v500[iy, ix]) * KMH_PER_MS
        slow = np.clip(shear_mag / 20, 0.15, 1.0)   # weak shear -> slow-moving cell
        cells.append(dict(cx=cx, cy=cy, tp=tp, peak=peak, s_t=rng.uniform(0.7, 1.2),
                          r_km=rng.uniform(6, 10), vu=steer_u * slow, vv=steer_v * slow))
        # low-level convergence that seeds the cell (present from ~2.5 h before peak)
        strength = 3.0 * np.clip(peak / 120, 0.2, 1.3) * _sig((0 - (tp - 2.5)) / 0.5)
        bump = _gauss(xx, yy, cx - cells[-1]["vu"] * tp, cy - cells[-1]["vv"] * tp, 14.0)
        gy, gx = np.gradient(bump, DX_KM * 1000)
        u850 = u850 + strength * gx / (np.abs(gx).max() + 1e-9)
        v850 = v850 + strength * gy / (np.abs(gy).max() + 1e-9)
    # decoy convergence without a storm (keeps convergence from being a giveaway)
    if rng.random() < 0.5:
        bump = _gauss(xx, yy, *rng.uniform(0.1, 0.9, 2) * n * DX_KM, 14.0)
        gy, gx = np.gradient(bump, DX_KM * 1000)
        u850 = u850 + 2.0 * gx / (np.abs(gx).max() + 1e-9)
        v850 = v850 + 2.0 * gy / (np.abs(gy).max() + 1e-9)

    ctt_clear = rng.uniform(268, 285)

    def fields_at(t):
        """IWV (mm), cloud-top temperature (K) and rain rate (mm/h) at time t."""
        iwv = iwv_base.copy()
        ctt = np.full((n, n), ctt_clear) - 6 * _smooth_unit(rng_fixed, n, 3)
        rain = np.zeros((n, n))
        for c in cells:
            px, py = c["cx"] + c["vu"] * (t - c["tp"]), c["cy"] + c["vv"] * (t - c["tp"])
            dt = t - c["tp"]
            # moisture pools ~2 h before the peak, lingers ~1 h after
            iwv_amp = (4 + 9 * c["peak"] / 150) * _sig((dt + 2.0) / 0.45) * _sig((1.5 - dt) / 0.5)
            iwv += iwv_amp * _gauss(xx, yy, px, py, 12.0)
            # cloud top cools ~1 h before the peak
            depth = (ctt_clear - (200 + 25 * np.exp(-c["peak"] / 80))) * _sig((dt + 1.0) / 0.3) * _sig((2.0 - dt) / 0.5)
            ctt = np.minimum(ctt, ctt_clear - depth * _gauss(xx, yy, px, py, 9.0))
            inten = np.exp(-dt ** 2 / (2 * c["s_t"] ** 2))
            rain += c["peak"] * inten * _gauss(xx, yy, px, py, c["r_km"])
        return iwv, ctt, rain

    rng_fixed = np.random.default_rng(int(rng.integers(1 << 31)))
    state = rng_fixed.bit_generator.state

    def at(t):
        rng_fixed.bit_generator.state = state      # identical background every call
        return fields_at(t)

    # --- satellite history (observed, with sensor noise)
    dyn = np.zeros((len(HIST_TIMES_H), 3, n, n), dtype=np.float32)
    for i, t in enumerate(HIST_TIMES_H):
        iwv, ctt, rain = at(t)
        dyn[i, 0] = iwv + rng.normal(0, 0.7, (n, n))
        dyn[i, 1] = ctt + rng.normal(0, 1.2, (n, n))
        dyn[i, 2] = np.clip(rain * rng.lognormal(0, 0.15, (n, n)), 0, None)   # satellite QPE

    ctx = np.stack([cape, cin, u850, v850, u500, v500]).astype(np.float32)

    # --- labels at each lead
    runoff_coef = np.clip(0.35 + 8.0 * slp, 0.35, 0.9)
    flowacc = route_runoff(np.ones((n, n)), receiver, order)
    y = np.zeros((3, len(LEADS_H), n, n), dtype=np.uint8)
    for j, L in enumerate(LEADS_H):
        window = [at(L + d) for d in (-0.5, 0.0, 0.5)]
        rain_max = np.max([w[2] for w in window], axis=0)
        ctt_min = np.min([w[1] for w in window], axis=0)
        y[0, j] = (rain_max >= THUNDER_RAIN_MM_PER_H) & (ctt_min <= THUNDER_CTT_K)
        y[1, j] = rain_max >= CLOUDBURST_MM_PER_H
        # rain accumulated over the 2 h before the lead time, routed downstream
        accum = sum(at(L - dt)[2] for dt in np.arange(0.0, 2.0, 0.25)) * 0.25
        routed = route_runoff(accum * runoff_coef, receiver, order)
        y[2, j] = (routed >= FLOOD_ROUTED_MM) & (flowacc >= FLOOD_MIN_FLOWACC)

    return dict(dyn=dyn, ctx=ctx, dem=dem.astype(np.float32), y=y)


def make_dataset(n_samples: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scenes = [make_scene(rng) for _ in range(n_samples)]
    return {k: np.stack([s[k] for s in scenes]) for k in scenes[0]}
