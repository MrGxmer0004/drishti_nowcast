"""Feature extraction: the moisture / instability / lift / observation / terrain
predictors named in the problem statement.

Dynamic (per satellite frame):
  IWV, IWV tendency over 90 min, CTT, CTT drop rate per 30 min, satellite QPE.
Static (at issue time, from IMDAA and the DEM):
  CAPE, CIN, 850 hPa convergence, 0-6 km bulk shear, steering wind (u, v),
  elevation, slope, log flow accumulation.

Every channel is scaled with fixed physical constants rather than dataset
statistics, so the same scaling applies unchanged to real data later.
"""
from __future__ import annotations

import numpy as np

from .config import DX_KM
from .risk_engine import flow_accumulation, slope

DYN_CHANNELS = ("iwv", "iwv_tendency_90min", "ctt", "ctt_drop_rate", "qpe_log")
STATIC_CHANNELS = ("cape", "cin", "convergence_850", "bulk_shear", "steer_u", "steer_v",
                   "elevation", "slope", "log_flowacc")

FAMILIES = {
    "moisture (IWV)": ("iwv", "iwv_tendency_90min"),
    "observation (CTT, QPE)": ("ctt", "ctt_drop_rate", "qpe_log"),
    "instability (CAPE/CIN)": ("cape", "cin"),
    "lift (convergence, shear)": ("convergence_850", "bulk_shear", "steer_u", "steer_v"),
    "terrain (DEM)": ("elevation", "slope", "log_flowacc"),
}


def convergence(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Horizontal convergence, -(du/dx + dv/dy), in units of 1e-5 s^-1."""
    dx = DX_KM * 1000.0
    dudx = np.gradient(u, dx, axis=-1)
    dvdy = np.gradient(v, dx, axis=-2)
    return -(dudx + dvdy) * 1e5


def dynamic_features(dyn: np.ndarray) -> np.ndarray:
    """dyn: [T, 3, H, W] (IWV mm, CTT K, QPE mm/h) -> [T, 5, H, W]."""
    iwv, ctt, qpe = dyn[:, 0], dyn[:, 1], dyn[:, 2]
    lag = 3                                            # 3 frames = 90 min
    iwv_tend = np.zeros_like(iwv)
    iwv_tend[lag:] = iwv[lag:] - iwv[:-lag]
    iwv_tend[:lag] = iwv_tend[lag]
    ctt_drop = np.zeros_like(ctt)
    ctt_drop[1:] = ctt[:-1] - ctt[1:]                  # positive = cooling
    ctt_drop[0] = ctt_drop[1]
    out = np.stack([
        (iwv - 35.0) / 8.0,
        iwv_tend / 4.0,
        (ctt - 250.0) / 25.0,
        ctt_drop / 8.0,
        np.log1p(qpe) / 3.0,
    ], axis=1)
    return out.astype(np.float32)


def static_features(ctx: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """ctx: [6, H, W] (CAPE, CIN, u850, v850, u500, v500); dem: [H, W] metres."""
    cape, cin, u850, v850, u500, v500 = ctx
    conv = convergence(u850, v850)
    shear = np.hypot(u500 - u850, v500 - v850)
    out = np.stack([
        cape / 1500.0 - 1.0,
        cin / 80.0 + 1.0,
        conv / 3.0,
        shear / 10.0 - 1.0,
        0.5 * (u850 + u500) / 8.0,
        0.5 * (v850 + v500) / 8.0,
        (dem - 2000.0) / 800.0,
        slope(dem) * 10.0,
        np.log(flow_accumulation(dem)) / 3.0,
    ])
    return out.astype(np.float32)


def build(dataset: dict) -> tuple[np.ndarray, np.ndarray]:
    """Features for a whole dataset -> (dyn_feat [N,T,5,H,W], static_feat [N,9,H,W])."""
    dyn_f = np.stack([dynamic_features(d) for d in dataset["dyn"]])
    sta_f = np.stack([static_features(c, z) for c, z in zip(dataset["ctx"], dataset["dem"])])
    return dyn_f, sta_f
