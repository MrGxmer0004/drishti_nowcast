#!/usr/bin/env bash
# Full pipeline: generate synthetic data, train, evaluate, explain, plot.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m drishti_nowcast.train --epochs 15
python -m drishti_nowcast.evaluate
python -m drishti_nowcast.xai --head cloudburst --lead 3
python scripts/make_figures.py
