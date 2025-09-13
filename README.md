
# nbv-bench (minimal starter)

A tiny, method-agnostic Next-Best-View (NBV) benchmarking starter you can open directly in VS Code.

This is an **MVP skeleton** intended to grow into a full benchmark:
- Fixed baseline config (hemisphere candidates, budget K)
- Pluggable selectors (Greedy, Random)
- Ground-truth scorer **stub** (replace with ray-based coverage)
- Simple run loop + CSV logging

> Note: Geometry functions are **placeholders** so the project runs without heavy dependencies.
> Replace `geo.delta_coverage_stub` with real 3D ray-based coverage and plug in Open3D for TSDF fusion.

## Quickstart

```bash
pip install -r requirements.txt

# Run a simple experiment (Greedy)
python -m nbvbench.core --cfg experiments/configs/baseline.yaml --selector greedy

# Or Random
python -m nbvbench.core --cfg experiments/configs/baseline.yaml --selector random
```

Results (CSV + simple plot) are saved to `experiments/results/`.
