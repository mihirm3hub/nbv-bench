
from typing import Sequence, Dict, Tuple
import numpy as np
from .geo import delta_coverage_stub

class GTScorer:
    """Ground-truth scorer (stub)—replace delta_coverage_stub with real ray-based delta coverage."""
    def __init__(self, directions: np.ndarray, cache: bool = True):
        self.directions = directions
        self.cache: Dict[Tuple[int, Tuple[int,...]], float] = {} if cache else None

    def score(self, v: int, history: Sequence[int]) -> float:
        key = (v, tuple(history))
        if self.cache is not None and key in self.cache:
            return self.cache[key]
        val = float(delta_coverage_stub(self.directions, v, history))
        if self.cache is not None:
            self.cache[key] = val
        return val