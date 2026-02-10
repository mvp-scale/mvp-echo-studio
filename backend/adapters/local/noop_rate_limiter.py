"""NoOpRateLimiter — allows all requests (current behavior)."""

from ports.rate_limiter import RateLimiterPort


class NoOpRateLimiter(RateLimiterPort):
    def check(self, api_key: str) -> bool:
        return True

    def remaining(self, api_key: str) -> int:
        return 999999

    def reset_time(self, api_key: str) -> float:
        return 0.0
