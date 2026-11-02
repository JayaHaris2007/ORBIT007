"""Dependency injection contract for the agent package.

Defines the dependencies injected from app.py to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, tzinfo
from typing import Any, Callable


@dataclass
class AgentDeps:
    """Dependencies injected from app.py into the agent tools and loop."""

    # MemoryStore instance
    memory_store: Any

    # Reminder unified operations
    create_reminder_unified: Callable[..., dict[str, Any]]
    cancel_reminder_unified: Callable[[str], bool]
    parse_reminder_time: Callable[[str], datetime | None]
    validate_recurrence: Callable[[str | None], None]
    live_reminders: Callable[[], list[dict[str, Any]]]

    # Web search engine (Tavily or similar)
    web_search: Callable[..., list[dict[str, Any]]]

    # Learning module hooks
    learning_schema_initialized: Callable[[], bool]
    learning_overall_progress: Callable[[], dict[str, Any]]
    learning_next_topics: Callable[[int], list[dict[str, Any]]]
    run_learning_hook: Callable[[Callable[[], Any]], None]
    suggest_learning_for_task: Callable[[int], None]

    # Timezone and page routing
    local_timezone: tzinfo
    page_aliases: dict[str, tuple[str, ...]]

    # Fallback LLM generation without tools
    generate_response: Callable[..., str]
    token_budget: Callable[[str], int]

