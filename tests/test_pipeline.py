import numpy as np
import torch

from drishti_nowcast import alignment, features, risk_engine, synthetic
from drishti_nowcast.config import GRID, HIST_STEPS, LEADS_H
from drishti_nowcast.model_service import NowcastNet


def test_convergence_sign():
    # flow toward the centre of the grid is convergent (positive)
    n = 16
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
    u, v = -xx.astype(float), -yy.astype(float)
    assert features.convergence(u, v)[n // 2, n // 2] > 0


def test_routing_conserves_mass():
    rng = np.random.default_rng(0)
    dem = synthetic.make_dem(rng)
    rec, order = risk_engine.d8_receivers(dem)
    runoff = np.ones_like(dem)
    routed = risk_engine.route_runoff(runoff, rec, order)
    outlets = rec == -1
    assert np.isclose(routed.ravel()[outlets].sum(), runoff.sum())


def test_route_probability_never_decreases_at_source():
    rng = np.random.default_rng(1)
    dem = synthetic.make_dem(rng)
    rec, order = risk_engine.d8_receivers(dem)
    p = rng.uniform(0, 1, dem.shape)
    assert np.all(risk_engine.route_probability(p, rec, order) >= p - 1e-12)


def test_scene_shapes_and_labels():
    s = synthetic.make_scene(np.random.default_rng(2))
    assert s["dyn"].shape == (HIST_STEPS, 3, GRID, GRID)
    assert s["y"].shape == (3, len(LEADS_H), GRID, GRID)
    assert set(np.unique(s["y"])) <= {0, 1}


def test_model_output_shape():
    s = synthetic.make_scene(np.random.default_rng(3))
    d = torch.from_numpy(features.dynamic_features(s["dyn"]))[None]
    st = torch.from_numpy(features.static_features(s["ctx"], s["dem"]))[None]
    out = NowcastNet()(d, st)
    assert out.shape == (1, 3, len(LEADS_H), GRID, GRID)


def test_regrid_identity_and_time_align():
    lats, lons = alignment.model_grid(31.0, 78.5, 10, 0.04)
    f = np.add.outer(lats, lons)
    assert np.allclose(alignment.regrid(f, lats, lons, lats, lons), f)
    frames = np.stack([np.zeros((2, 2)), np.ones((2, 2))])
    mid = alignment.time_align(frames, [0.0, 1.0], [0.5])
    assert np.allclose(mid, 0.5)
