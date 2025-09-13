
import numpy as np
from typing import List, Sequence

def angular_diversity_gain(directions: np.ndarray, v: int, history: Sequence[int]) -> float:
    """Compute a simple novelty score: 1 - max cosine similarity to selected directions."""
    if len(history) == 0:
        return 1.0
    dv = directions[v] / (np.linalg.norm(directions[v]) + 1e-9)
    sims = []
    for u in history:
        du = directions[u] / (np.linalg.norm(directions[u]) + 1e-9)
        sims.append(abs(float(np.dot(dv, du))))
    return float(1.0 - max(sims))

def delta_coverage_stub(directions: np.ndarray, v: int, history: Sequence[int]) -> float:
    """Placeholder for ray-based delta coverage. Replace with real 3D visibility tests."""
    return angular_diversity_gain(directions, v, history)

def update_coverage_proxy(directions: np.ndarray, history: Sequence[int]) -> float:
    """Integrate proxy 'coverage' over history by accumulating diminishing gains (demo only)."""
    total = 0.0
    for i, v in enumerate(history):
        total += delta_coverage_stub(directions, v, history[:i])
    return float(np.clip(total / max(1, len(history)), 0.0, 1.0))

def compute_auc(cov_list: Sequence[float]) -> float:
    """AUC normalized by K (simple mean of cumulative coverage per step)."""
    if len(cov_list) == 0:
        return 0.0
    return float(np.mean(cov_list))

def fuse_tsdf_stub(*args, **kwargs):
    """Placeholder. In real code, call Open3D TSDF fusion and export mesh."""
    pass
