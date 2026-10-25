"""In-process rate limiting for the research endpoints.

S.P.I.D.E.Y runs locally for one user, so a per-client sliding window in
memory is enough and avoids another dependency. Swap this module for a real
limiter if the API is ever exposed beyond localhost.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from . import config

_lock = threading.Lock()
_hits: dict[str, deque[float]] = defaultdict(deque)


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def enforce(request: Request) -> None:
    """Raise 429 when the caller exceeded its window; otherwise record the hit."""
    limit = config.RATE_LIMIT_REQUESTS
    window = float(config.RATE_LIMIT_WINDOW_SECONDS)
    now = time.monotonic()
    key = client_key(request)

    with _lock:
        bucket = _hits[key]
        while bucket and now - bucket[0] >= window:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(window - (now - bucket[0])) + 1)
            raise HTTPException(
                429,
                f"Too many research requests. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


def reset() -> None:
    """Clear all windows; used by tests."""
    with _lock:
        _hits.clear()
