
from typing import List, Sequence, Protocol
import numpy as np
import random

class BaseScorer(Protocol):
    def score(self, v: int, history: Sequence[int]) -> float: ...

class BaseSelector(Protocol):
    def choose_next(self, remaining: Sequence[int], history: List[int], scorer: BaseScorer) -> int: ...

class Greedy:
    def choose_next(self, remaining, history, scorer):
        scores = [scorer.score(v, history) for v in remaining]
        return remaining[int(np.argmax(scores))]

class RandomSel:
    def choose_next(self, remaining, history, scorer):
        return random.choice(list(remaining))
