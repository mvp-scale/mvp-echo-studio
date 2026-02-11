"""UsagePort — abstract interface for logging API usage."""

from abc import ABC, abstractmethod
from typing import Any


class UsagePort(ABC):
    @abstractmethod
    def log(self, api_key: str, metadata: dict[str, Any]) -> None:
        """Log a usage event (file size, duration, engine, etc.)."""

    @abstractmethod
    def get(self, api_key: str) -> dict[str, Any]:
        """Return accumulated usage stats for a given API key."""
