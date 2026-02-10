"""
API key auth middleware for MVP-Echo Scribe.
All /v1/audio/* requests require a Bearer token from api-keys.json.
"""

import json
import logging
import math
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ports.rate_limiter import RateLimiterPort

logger = logging.getLogger(__name__)

# Paths that never require auth (health, model list, static frontend assets)
OPEN_PATHS = {"/health", "/v1/models"}

# Only these path prefixes require API key auth.
# Everything else (static files, SPA routes) is served without auth.
AUTH_PREFIXES = ("/v1/audio/",)

KEYS_FILE = "/data/api-keys.json"


def _load_keys() -> dict:
    try:
        with open(KEYS_FILE) as f:
            data = json.load(f)
        return {k["key"]: k for k in data.get("keys", []) if k.get("active", True)}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not load keys file: {e}")
        return {}


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter: RateLimiterPort | None = None):
        super().__init__(app)
        self._rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next):
        # Only enforce auth on API endpoints that process data
        path = request.url.path
        if path in OPEN_PATHS or not any(path.startswith(p) for p in AUTH_PREFIXES):
            return await call_next(request)

        # Allow OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Client IP for logging only
        client_ip = request.client.host if request.client else "unknown"

        # Check Bearer token
        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing API key"})

        token = auth[7:].strip()
        keys = _load_keys()
        if token not in keys:
            logger.warning(f"Invalid API key from {client_ip}")
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})

        # Rate limiting
        if self._rate_limiter is not None:
            if not self._rate_limiter.check(token):
                remaining = self._rate_limiter.remaining(token)
                reset_epoch = self._rate_limiter.reset_time(token)
                retry_after = max(1, math.ceil(reset_epoch - time.time()))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={
                        "X-RateLimit-Remaining": str(remaining),
                        "X-RateLimit-Reset": str(int(reset_epoch)),
                        "Retry-After": str(retry_after),
                    },
                )

        response = await call_next(request)

        # Attach rate limit headers to successful responses
        if self._rate_limiter is not None:
            remaining = self._rate_limiter.remaining(token)
            reset_epoch = self._rate_limiter.reset_time(token)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(reset_epoch))

        return response
