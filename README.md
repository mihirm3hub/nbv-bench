# NBV-Bench (ray-coverage MVP)

A small, method-agnostic Next-Best-View (NBV) **benchmarking starter** with
realistic **ray-visibility coverage** (Open3D), area-uniform hemisphere
candidates, and a lean run loop that logs just what you need to compare methods.

> Goal: a clean, reproducible baseline where adding new selectors (Greedy,
> Random, PSO, SMA, …) is trivial and results are directly comparable.

---
## Project layout
``` bash 

assets/
  tsdf_mesh.ply            # test mesh/asset
experiments/
  configs/baseline.yaml    # balanced defaults
  results/                 # metrics + selected-indices CSVs

nbvbench/
  core.py                  # main loop (minimal outputs)
  data.py                  # mesh loading + hemisphere candidates (area-uniform)
  geo.py                   # RayCoverage (Open3D), AUC helpers
  selectors.py             # Greedy, Random; drop-in for new methods
  scorers.py               # interfaces to geo coverage

visualize_frustums_o3d.py  # white-bg viz: aim/frusta/both (spheres)
CHANGELOG.md               # versioned changes
README.md                  # this file

```
## What’s in this MVP (v0.2)

- **RayCoverage** (Open3D tensor raycasting) with real occlusion:
  frustum gating (fx/fy/W/H), near/far planes, optional front-facing filter,
  and `eps` hit tolerance over GT surface samples.
- **Area-uniform hemisphere** sampling over a φ-band
  (z-stratified spherical Fibonacci) to avoid “top-ring” bias.
- **Lean outputs** only:
  - `experiments/results/metrics_<tag>_<selector>.csv` (per-step coverage)
  - `experiments/results/<tag>_<selector>_selected.csv` (chosen indices)
- **Visualizer** (`visualize_frustums_o3d.py`):
  white background; gray candidates; green selected spheres; aim lines and/or
  frusta; sphere sizes ∝ scene extent; unit scaling; auto-fit.
- Balanced `baseline.yaml` with realistic defaults.

See `CHANGELOG.md` for v0.1 → v0.2 details.

---

## Requirements

- Python 3.9+ (tested)
- `open3d`, `numpy`, `pyyaml`
- A triangle mesh (e.g., `assets/tsdf_mesh.ply`)

## Install:
``` bash
pip install -r requirements.txt
```

Quickstart

Run Greedy or Random on the baseline config:
``` bash
# Greedy
python -m nbvbench.core --cfg experiments/configs/baseline.yaml --selector greedy

# Random
python -m nbvbench.core --cfg experiments/configs/baseline.yaml --selector random
```

Visualization

Visualize a run (white background, gray candidates, green selected; aim lines + frusta):
``` bash
python visualize_frustums_o3d.py \
  --cfg experiments/configs/baseline.yaml \
  --indices_csv experiments/results/baseline_greedy_selected.csv \
  --mode both --recenter --axes --auto_fit \
  --cand_glyph spheres \
  --cand_radius_factor 0.015 \
  --sel_radius_factor 0.03 \
  --frustum_scale_factor 0.8

```

## Roadmap

- TSDF coverage (raycast depth → integrate → nearest-distance coverage)
- Additional selectors: PSO, SMA baselines
- Multi-object scenes / presets