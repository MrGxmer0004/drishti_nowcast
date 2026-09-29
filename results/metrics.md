# Test-set metrics (synthetic data)

Held-out synthetic test set. These numbers show the pipeline learns the physics encoded in
`synthetic.py`; they are **not** estimates of skill on real observations.

| Hazard @ lead | Base rate | Model PR-AUC | GBM PR-AUC | Persistence CSI | Model POD | Model FAR | Model CSI | GBM CSI | Model FSS (20 km) |
|---|---|---|---|---|---|---|---|---|---|
| thunderstorm@+2h | 1.35% | 0.41 | 0.20 | 0.03 | 0.40 | 0.53 | 0.28 | 0.14 | 0.61 |
| thunderstorm@+3h | 1.49% | 0.22 | 0.10 | 0.01 | 0.24 | 0.65 | 0.17 | 0.08 | 0.43 |
| thunderstorm@+4h | 1.47% | 0.08 | 0.05 | 0.00 | 0.10 | 0.79 | 0.07 | 0.05 | 0.22 |
| thunderstorm@+6h | 1.27% | 0.02 | 0.02 | 0.00 | 0.53 | 0.98 | 0.02 | 0.00 | 0.05 |
| cloudburst@+2h | 0.25% | 0.54 | 0.15 | 0.06 | 0.57 | 0.55 | 0.33 | 0.16 | 0.67 |
| cloudburst@+3h | 0.25% | 0.28 | 0.08 | 0.01 | 0.29 | 0.62 | 0.20 | 0.08 | 0.47 |
| cloudburst@+4h | 0.24% | 0.14 | 0.03 | 0.00 | 0.15 | 0.64 | 0.12 | 0.02 | 0.37 |
| cloudburst@+6h | 0.15% | 0.01 | 0.01 | 0.01 | 0.21 | 0.99 | 0.01 | 0.00 | 0.03 |
| flash_flood@+2h | 0.18% | 0.24 | 0.29 | 0.00 | 0.63 | 0.80 | 0.18 | 0.20 | 0.45 |
| flash_flood@+3h | 0.20% | 0.23 | 0.24 | 0.00 | 0.39 | 0.76 | 0.18 | 0.18 | 0.47 |
| flash_flood@+4h | 0.20% | 0.17 | 0.16 | 0.00 | 0.39 | 0.84 | 0.13 | 0.13 | 0.38 |
| flash_flood@+6h | 0.16% | 0.08 | 0.10 | 0.00 | 0.17 | 0.90 | 0.07 | 0.10 | 0.22 |
