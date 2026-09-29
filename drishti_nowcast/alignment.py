"""Data fusion and alignment: put every source on one spatiotemporal grid.

Real inputs arrive on different grids and clocks:
  * INSAT-3D/3DR imager frames: ~4 km IR pixels every 15-30 min (staggered satellites)
  * IMDAA reanalysis: ~12 km, hourly, on pressure levels
  * CartoDEM / SRTM: 30 m, static

`regrid` interpolates a field from its native lat/lon grid onto the model
grid; `time_align` interpolates a stack of frames onto the model's 30-minute
history steps. The synthetic generator already produces aligned grids, so
these functions are exercised by the tests and will be used by the real
ingestion services.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom


def model_grid(lat0: float, lon0: float, n: int, dx_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes (north to south) and longitudes (west to east)."""
    lats = lat0 - (np.arange(n) + 0.5) * dx_deg
    lons = lon0 + (np.arange(n) + 0.5) * dx_deg
    return lats, lons


def regrid(field: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray,
           dst_lats: np.ndarray, dst_lons: np.ndarray, method: str = "linear") -> np.ndarray:
    """Interpolate a 2-D field onto a new regular lat/lon grid."""
    lat_ax, f = (src_lats, field) if src_lats[0] < src_lats[-1] else (src_lats[::-1], field[::-1])
    interp = RegularGridInterpolator((lat_ax, src_lons), f, method=method,
                                     bounds_error=False, fill_value=None)
    gl, gn = np.meshgrid(dst_lats, dst_lons, indexing="ij")
    return interp(np.stack([gl.ravel(), gn.ravel()], -1)).reshape(gl.shape)


def block_average(field: np.ndarray, factor: int) -> np.ndarray:
    """Aggregate a fine grid (e.g. 30 m DEM) to a coarser one by block mean."""
    h, w = field.shape
    h2, w2 = h // factor, w // factor
    return field[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def time_align(frames: np.ndarray, times_h: np.ndarray, target_times_h: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames [T, ...] observed at times_h onto target times.

    Targets outside the observed range take the nearest frame (and should be
    flagged as stale by the caller).
    """
    frames = np.asarray(frames, dtype=np.float64)
    t = np.asarray(times_h, dtype=np.float64)
    out = np.empty((len(target_times_h),) + frames.shape[1:])
    for k, tt in enumerate(target_times_h):
        if tt <= t[0]:
            out[k] = frames[0]
        elif tt >= t[-1]:
            out[k] = frames[-1]
        else:
            i = np.searchsorted(t, tt) - 1
            f = (tt - t[i]) / (t[i + 1] - t[i])
            out[k] = (1 - f) * frames[i] + f * frames[i + 1]
    return out


def upsample(field: np.ndarray, factor: float) -> np.ndarray:
    """Bilinear upsampling, e.g. 12 km IMDAA -> 4 km model grid."""
    return zoom(field, factor, order=1)
