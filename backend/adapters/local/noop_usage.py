"""NoOpUsageAdapter — discards usage data (current behavior)."""

from typing import Any

from ports.usage import UsagePort


class NoOpUsageAdapter(UsagePort):
    def log(self, api_key: str, metadata: dict[str, Any]) -> None:
        pass

    def get(self, api_key: str) -> dict[str, Any]:
        return {}
