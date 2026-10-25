"""Tavily-backed web search.

This is the only module that talks to the search provider. Everything else in
the pipeline consumes the normalized ``WebSource`` dicts produced here, so
swapping providers later means touching this file only.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlsplit

from . import config

LOGGER = logging.getLogger("spidey.research.search")

MAX_TITLE_CHARS = 200
MAX_SNIPPET_CHARS = 1_200
_WHITESPACE = re.compile(r"\s+")


class WebSource(TypedDict):
    """One normalized search hit. ``id`` is the citation number."""

    id: int
    title: str
    url: str
    domain: str
    content: str
    score: float
    published_date: NotRequired[str | None]


class SearchNotConfigured(RuntimeError):
    """The search provider is unusable: missing key or missing package."""


class SearchFailed(RuntimeError):
    """The search provider rejected the request or could not be reached."""


def search(
    query: str,
    max_results: int = config.MAX_SOURCES,
    *,
    search_depth: str | None = None,
) -> list[WebSource]:
    """Run one web search and return at most ``max_results`` usable sources.

    Raises ``SearchNotConfigured`` when the API key or client package is
    missing, and ``SearchFailed`` for any provider-side error. Messages are
    scrubbed so the API key never reaches a log line or an HTTP response.
    """
    api_key = _api_key()
    client = _build_client(api_key)
    depth = search_depth or config.SEARCH_DEPTH

    try:
        payload = _call_provider(client, query, max_results, depth)
    except Exception as error:  # Provider-specific errors vary by version.
        LOGGER.warning(
            "Tavily search failed for %r: %s", query, _scrub(str(error), api_key)
        )
        raise SearchFailed("The web search provider could not complete the search.") from error

    sources = _normalize(payload, max_results)
    LOGGER.info("Tavily returned %d usable sources for %r", len(sources), query)
    return sources


def _api_key() -> str:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise SearchNotConfigured(
            "Web research is not configured. Add TAVILY_API_KEY to the backend .env file."
        )
    return api_key


def _build_client(api_key: str):
    try:
        from tavily import TavilyClient
    except ImportError as error:
        raise SearchNotConfigured(
            "Web research needs the tavily-python package. Install it with "
            "pip install tavily-python."
        ) from error
    return TavilyClient(api_key=api_key)


def _call_provider(client, query: str, max_results: int, search_depth: str) -> dict[str, Any]:
    """Call Tavily, passing a timeout only on versions that support it."""
    kwargs: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    if "timeout" in inspect.signature(client.search).parameters:
        kwargs["timeout"] = config.SEARCH_TIMEOUT_SECONDS
    return client.search(**kwargs)


def _normalize(payload: Any, max_results: int) -> list[WebSource]:
    """Keep usable hits, drop duplicates, and number them for citation."""
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        return []

    sources: list[WebSource] = []
    seen: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        url = _clean_text(item.get("url"), 2_000)
        if not url.startswith(("http://", "https://")):
            continue
        key = _url_key(url)
        if key in seen:
            continue
        seen.add(key)

        domain = _domain(url)
        source: WebSource = {
            "id": len(sources) + 1,
            "title": _clean_text(item.get("title"), MAX_TITLE_CHARS) or domain,
            "url": url,
            "domain": domain,
            "content": _clean_text(item.get("content"), MAX_SNIPPET_CHARS),
            "score": _score(item.get("score")),
        }
        published = _clean_text(item.get("published_date"), 40)
        if published:
            source["published_date"] = published
        sources.append(source)
        if len(sources) >= max_results:
            break
    return sources


def _clean_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()[:limit]


def _url_key(url: str) -> str:
    """Comparison key: same page on http/https or with a fragment is one hit."""
    parts = urlsplit(url.strip().lower())
    host = parts.netloc.removeprefix("www.")
    path = parts.path.rstrip("/")
    return f"{host}{path}?{parts.query}" if parts.query else f"{host}{path}"


def _domain(url: str) -> str:
    return urlsplit(url).netloc.removeprefix("www.").lower()


def _score(value: Any) -> float:
    try:
        return round(min(max(float(value), 0.0), 1.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _scrub(message: str, secret: str) -> str:
    return message.replace(secret, "***") if secret else message
