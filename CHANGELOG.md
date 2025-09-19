# Changelog

This project follows a simple, versioned changelog: concise entries per milestone, focused on benchmarking-relevant changes.

## NBB-v0.2
**Realistic ray coverage (Open3D) + area-uniform hemisphere; lean core; viz upgrade**

### Added
- **RayCoverage (Open3D tensor raycasting)** with true occlusions, frustum gating (fx/fy/W/H), near/far planes, and optional front-facing filtering.
- **Visualizer** (`visualize_frustums_o3d.py`): white background, gray candidates, green selected; modes `aim`, `frusta`, `both`; sphere glyphs (size ∝ scene), unit scaling, auto-fit.
- **Auto-safe far plane** in `core.py`: `far_m = max(cfg.far_m, 1.25 × radius)` to prevent accidental clipping when radius changes.

### Changed
- `core.py` now **uses RayCoverage** in the main loop (replaces proxy coverage).  
  Outputs remain minimal: `metrics_<tag>_<selector>.csv` and `<tag>_<selector>_selected.csv`.
- `data.py::hemisphere_candidates_auto` now returns **5 values**:  
  `(centers, dirs, radius, (fx, fy, W, H), origin)`.
- Hemisphere sampling is **area-uniform** over `[phi_min, phi_max]` (z-stratified spherical Fibonacci) to avoid top-ring clustering.
- YAML accepts **`hemisphere.z_offset_m` or `hemisphere.z_margin_m`** (treated equivalently).
- Balanced defaults in `experiments/configs/baseline.yaml` (realistic FOV/φ-band, strict coverage).
- `.gitignore` ignores `experiments/results/`, caches, and temp artifacts.

### Removed
- No default export of frusta `.ply` or candidates `.npy` in the MVP path (keeps benchmark lean).

### Migration notes
- Update any callers to unpack **five** returns from `hemisphere_candidates_auto`.
- Ray coverage **requires Open3D and a valid mesh**; errors if missing.
- If you override `coverage.far_m`, keep it ≥ camera→origin distance; otherwise the auto-guard handles it.

---

## NBB-v0.1
**Minimal repo skeleton**

### Added
- Initial package layout `nbvbench/` and `experiments/`.
- Baseline **selectors**: `greedy`, `random`.
- Minimal **coverage/metrics export**: per-step coverage CSV + selected indices CSV.
- Starter `baseline.yaml` config and simple run entrypoint.
