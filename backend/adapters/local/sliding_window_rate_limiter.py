"""SlidingWindowRateLimiter — in-memory sliding window rate limiter."""

import threading
import time
from collections import defaultdict
from typing import Dict, List

from ports.rate_limiter import RateLimiterPort


class SlidingWindowRateLimiter(RateLimiterPort):
    def __init__(self, max_requests: int = 10, window_seconds: float = 60):
        self._max = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _prune(self, api_key: str, now: float) -> None:
        cutoff = now - self._window
        timestamps = self._requests[api_key]
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

    def check(self, api_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune(api_key, now)
            if len(self._requests[api_key]) >= self._max:
                return False
            self._requests[api_key].append(now)
            return True

    def remaining(self, api_key: str) -> int:
        now = time.monotonic()
        with self._lock:
            self._prune(api_key, now)
            return max(0, self._max - len(self._requests[api_key]))

    def reset_time(self, api_key: str) -> float:
        now_mono = time.monotonic()
        now_wall = time.time()
        with self._lock:
            self._prune(api_key, now_mono)
            timestamps = self._requests[api_key]
            if not timestamps:
                return 0.0
            oldest_age = now_mono - timestamps[0]
            return now_wall + (self._window - oldest_age)
