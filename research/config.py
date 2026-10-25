"""Limits for the research engine, overrideable through the backend .env.

Every cap used by the pipeline lives here so the individual modules stay free
of magic numbers and one place documents the safety envelope.
"""

from __future__ import annotations

import os


def _int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value >= minimum else default


def _float(name: str, default: float, minimum: float = 1.0) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value >= minimum else default


# Search stage.
MAX_SOURCES = 8  # hard cap per request, also the citation ceiling
MAX_SEARCHES = 6  # provider calls per research request
SEARCH_DEPTH = os.environ.get("RESEARCH_SEARCH_DEPTH", "basic").strip() or "basic"
SEARCH_TIMEOUT_SECONDS = _float("RESEARCH_SEARCH_TIMEOUT", 20.0)

# Page reading stage.
MAX_PAGES = 8
MAX_PAGE_BYTES = 2_000_000
MAX_PAGE_CHARS = 12_000
PAGE_TIMEOUT_SECONDS = _float("RESEARCH_PAGE_TIMEOUT", 15.0)

# Prompt and answer budget.
MAX_EVIDENCE_CHARS = 9_000
MAX_PROMPT_CHARS = 12_000
MAX_ANSWER_TOKENS = _int("RESEARCH_MAX_ANSWER_TOKENS", 320)
MAX_QUERIES = 4
MAX_SEARCH_ROUNDS = 3

# Input limits.
MIN_QUERY_CHARS = 2
MAX_QUERY_CHARS = 500

# Endpoint rate limiting (per client address, sliding window).
RATE_LIMIT_REQUESTS = _int("RESEARCH_RATE_LIMIT", 12)
RATE_LIMIT_WINDOW_SECONDS = _int("RESEARCH_RATE_WINDOW", 60)
