"""JsonFileUsageTracker — thread-safe, JSON-backed per-key usage tracker."""

import json
import logging
import os
import tempfile
import threading
from typing import Any

from ports.usage import UsagePort

logger = logging.getLogger(__name__)


class JsonFileUsageTracker(UsagePort):
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path) as f:
                self._data = json.load(f)
            logger.info(f"Usage data loaded from {self._path} ({len(self._data)} keys)")
        except FileNotFoundError:
            logger.info(f"No usage file at {self._path}, starting fresh")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load usage file: {e}, starting fresh")

    def _flush(self) -> None:
        """Atomically write data to disk via temp-file + os.replace."""
        dir_name = os.path.dirname(self._path)
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except OSError as e:
            logger.error(f"Failed to flush usage data: {e}")

    def log(self, api_key: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            entry = self._data.setdefault(api_key, {
                "request_count": 0,
                "total_audio_minutes": 0.0,
                "total_bytes_uploaded": 0,
                "total_gpu_seconds": 0.0,
                "last_used": None,
            })
            entry["request_count"] += 1
            entry["total_audio_minutes"] += metadata.get("audio_minutes", 0.0)
            entry["total_bytes_uploaded"] += metadata.get("bytes_uploaded", 0)
            entry["total_gpu_seconds"] += metadata.get("gpu_seconds", 0.0)

            import time as _time
            entry["last_used"] = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())

            self._flush()

    def get(self, api_key: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._data.get(api_key, {}))
