"""Satellite and reanalysis ingestion services (in build).

These are the entry points real data will come through. They are
deliberately not implemented yet: the team does not have MOSDAC / NCMRWF
data access set up, and the pipeline currently trains on
`drishti_nowcast.synthetic`. Each function documents the product it will
read and the array it must return, so the rest of the pipeline is written
against the final interface.
"""
from __future__ import annotations

import numpy as np


def read_insat_frame(path: str) -> dict[str, np.ndarray]:
    """Read one INSAT-3D/3DR imager L1B HDF5 file from MOSDAC.

    Returns {"wv": brightness temperature 6.7 um (K), "tir1": 10.8 um (K),
    "lat": [...], "lon": [...], "time_utc": ...}. IWV comes from the
    sounder/TPW product (or a WV-channel retrieval); QPE from the
    Hydro-Estimator (HEM) rain product.
    """
    raise NotImplementedError("MOSDAC ingestion is in build; see README 'Status'.")


def read_imdaa(path: str, levels_hpa=(850, 700, 500)) -> dict[str, np.ndarray]:
    """Read IMDAA reanalysis (NCMRWF) NetCDF on pressure levels.

    Returns temperature, specific humidity, geopotential and U/V wind per
    level, used to derive CAPE/CIN (parcel theory), 850 hPa convergence and
    0-6 km bulk shear. Operationally the same variables come from a
    near-real-time analysis, because reanalysis is not available live.
    """
    raise NotImplementedError("IMDAA ingestion is in build; see README 'Status'.")


def read_dem(path: str) -> np.ndarray:
    """Read a CartoDEM / SRTM GeoTIFF tile (metres), to be block-averaged
    onto the model grid with alignment.block_average."""
    raise NotImplementedError("DEM loader is designed; see README 'Status'.")
