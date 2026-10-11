"""Small repeatable wall-clock measurement helper for WM calls."""

import statistics
import time
import math


class CostMeter:
    def __init__(self, warmup=5, reps=30):
        if warmup < 0 or reps < 1:
            raise ValueError('warmup must be >= 0 and reps must be >= 1.')
        self.warmup, self.reps = warmup, reps

    def measure(self, fn):
        """Run ``fn()`` and return ``(median_s, p95_s, all_s)``."""
        for _ in range(self.warmup):
            fn()
        xs = []
        for _ in range(self.reps):
            started = time.perf_counter()
            fn()
            xs.append(time.perf_counter() - started)
        xs.sort()
        p95_index = max(0, math.ceil(0.95 * len(xs)) - 1)
        return statistics.median(xs), xs[p95_index], xs
