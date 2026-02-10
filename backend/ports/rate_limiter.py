"""RateLimiterPort — abstract interface for request rate limiting."""

from abc import ABC, abstractmethod


class RateLimiterPort(ABC):
    @abstractmethod
    def check(self, api_key: str) -> bool:
        """Return True if request is allowed, False if rate-limited."""

    @abstractmethod
    def remaining(self, api_key: str) -> int:
        """Return number of remaining requests in current window."""

    @abstractmethod
    def reset_time(self, api_key: str) -> float:
        """Return wall-clock epoch when the oldest request expires from the window."""
