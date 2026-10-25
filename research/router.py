"""HTTP surface of the research engine.

Mounted by ``app.py`` with ``app.include_router``. Errors follow the existing
convention of the rest of the API: a JSON body of the shape ``{"error": "..."}``
produced by the global HTTPException handler.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from . import config, rate_limit, search_engine

LOGGER = logging.getLogger("spidey.research.api")

router = APIRouter(prefix="/api/research", tags=["research"])


def _throttle(request: Request) -> None:
    rate_limit.enforce(request)


def parse_question(value: Any, field: str = "query") -> str:
    """Validate free-text input the same way for every research endpoint."""
    if not isinstance(value, str):
        raise HTTPException(400, f"'{field}' must be a string.")
    question = " ".join(value.split())
    if len(question) < config.MIN_QUERY_CHARS:
        raise HTTPException(400, f"'{field}' must be at least {config.MIN_QUERY_CHARS} characters.")
    if len(question) > config.MAX_QUERY_CHARS:
        raise HTTPException(400, f"'{field}' is limited to {config.MAX_QUERY_CHARS} characters.")
    return question


def _parse_search_body(payload: Any) -> tuple[str, int]:
    if not isinstance(payload, dict):
        raise HTTPException(400, "Request body must be a JSON object.")

    question = parse_question(payload.get("query"))

    max_results = payload.get("max_results", config.MAX_SOURCES)
    if isinstance(max_results, bool) or not isinstance(max_results, int):
        raise HTTPException(400, "'max_results' must be an integer.")
    if not 1 <= max_results <= config.MAX_SOURCES:
        raise HTTPException(
            400, f"'max_results' must be between 1 and {config.MAX_SOURCES}."
        )
    return question, max_results


@router.post("/search", dependencies=[Depends(_throttle)])
def search_sources(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Return raw, numbered sources for one query. No LLM is involved yet."""
    question, max_results = _parse_search_body(payload)

    try:
        sources = search_engine.search(question, max_results)
    except search_engine.SearchNotConfigured as error:
        raise HTTPException(503, str(error)) from error
    except search_engine.SearchFailed as error:
        raise HTTPException(502, str(error)) from error
    except Exception as error:  # Never leak a stack trace to the client.
        LOGGER.exception("Unexpected research search failure")
        raise HTTPException(502, "The web search failed unexpectedly.") from error

    return {"query": question, "count": len(sources), "sources": sources}
