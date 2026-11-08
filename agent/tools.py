"""Native tool registry, JSON schemas, argument validation, and executors for S.P.I.D.E.Y.

Every tool interacts directly with the existing data layer through AgentDeps.
Schemas are strictly constrained (<40 words per description) to preserve token budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import rapidfuzz.fuzz

from agent.confirm import ConfirmationManager
from agent.deps import AgentDeps

log = logging.getLogger("orbit007.tools")


@dataclass
class ToolContext:
    """Turn-scoped execution context passed to tool executors."""

    deps: AgentDeps
    chat_id: str
    channel: str = "web"
    confirm_mgr: ConfirmationManager = field(default_factory=ConfirmationManager)
    ui_actions: list[dict[str, Any]] = field(default_factory=list)
    has_called_web_search: bool = False
    tools_called: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Tool metadata and execution specification."""

    name: str
    description: str
    parameters: dict[str, Any]
    destructive: bool
    executor: Callable[[dict[str, Any], ToolContext], dict[str, Any]]

    def to_ollama_tool(self) -> dict[str, Any]:
        """Convert to the OpenAI / Ollama tool definition JSON format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --- Entity Resolution Helper ---

def resolve_entity(
    query: str | int,
    candidates: list[dict[str, Any]],
    title_key: str = "title",
    entity_name: str = "item",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve an entity reference (id or title) using rapidfuzz against live candidates.

    Returns:
        (resolved_item, error_dict)
        If resolved_item is present, resolution succeeded.
        If error_dict is present, resolution failed (not_found or ambiguous).
    """
    if not candidates:
        return None, {"status": "not_found", "message": f"No {entity_name}s exist."}

    # 1. Exact ID check (integer or string UUID)
    raw_str = str(query).strip().lstrip("#")
    for item in candidates:
        item_id = item.get("id")
        if item_id is not None:
            if str(item_id) == raw_str or (raw_str.isdigit() and item_id == int(raw_str)):
                return item, None

    # 2. String query check against title / name / text / content
    query_str = str(query).strip().lower()
    if not query_str:
        return None, {"status": "not_found", "message": f"Empty reference for {entity_name}."}

    scored: list[tuple[float, dict[str, Any]]] = []
    for item in candidates:
        candidate_text = str(
            item.get(title_key)
            or item.get("title")
            or item.get("name")
            or item.get("text")
            or item.get("content")
            or ""
        ).strip()
        if not candidate_text:
            continue
        # Check exact case-insensitive match
        if candidate_text.lower() == query_str:
            return item, None
        score = float(rapidfuzz.fuzz.token_set_ratio(query_str, candidate_text.lower()))
        scored.append((score, item))

    if not scored:
        return None, {"status": "not_found", "message": f"No {entity_name} found matching '{query}'."}

    scored.sort(key=lambda x: x[0], reverse=True)
    top_score, top_item = scored[0]

    # Weak match refusal
    if top_score < 70.0:
        return None, {"status": "not_found", "message": f"No {entity_name} found matching '{query}' (best match {top_score:.0f}% < 70%)."}

    # Check ambiguity: second candidate >= 70 and gap < 10
    if len(scored) > 1:
        second_score, second_item = scored[1]
        if second_score >= 70.0 and (top_score - second_score) < 10.0:
            cands = []
            for s, itm in scored[:3]:
                if s >= 70.0:
                    cands.append({
                        "id": itm.get("id"),
                        "title": itm.get(title_key) or itm.get("name") or itm.get("text") or itm.get("content"),
                        "score": round(s, 1),
                    })
            return None, {
                "status": "ambiguous",
                "message": f"Multiple {entity_name}s matched '{query}'. Please specify which one.",
                "candidates": cands,
            }

    return top_item, None


# --- Tool Executors ---

# 1. Reminders
def _exec_create_reminder(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    title = str(args.get("title", "")).strip()
    when = str(args.get("when", "")).strip()
    recurrence = args.get("recurrence")
    if not title or not when:
        return {"status": "error", "message": "'title' and 'when' are required."}

    try:
        reminder = ctx.deps.create_reminder_unified(
            title=title, when=when, recurrence=recurrence, chat_id=ctx.chat_id
        )
        ctx.ui_actions.append({"type": "refresh", "sections": ["reminders"]})
        return {
            "status": "ok",
            "reminder": {
                "id": reminder["id"],
                "title": reminder["title"],
                "remind_at": reminder["remind_at"],
                "recurrence": reminder.get("recurrence"),
            },
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _exec_list_reminders(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    rows = ctx.deps.live_reminders()[:15]
    return {
        "status": "ok",
        "reminders": [
            {
                "id": r["id"],
                "title": r["title"],
                "remind_at": r["remind_at"],
                "recurrence": r.get("recurrence"),
            }
            for r in rows
        ],
    }


def _exec_update_reminder(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    rem_ref = args.get("reminder")
    new_when = args.get("when")
    new_title = args.get("title")
    if not new_when and not new_title:
        return {"status": "error", "message": "At least one of 'when' or 'title' must be provided."}

    live = ctx.deps.live_reminders()
    item, err = resolve_entity(rem_ref, live, title_key="title", entity_name="reminder")
    if err:
        return err

    rem_id = str(item["id"])
    target_iso = None
    if new_when:
        parsed = ctx.deps.parse_reminder_time(str(new_when))
        if parsed is None:
            return {"status": "error", "message": f"Could not parse reminder time: '{new_when}'."}
        target_iso = parsed.isoformat()

    ctx.deps.cancel_reminder_unified(rem_id)
    title_to_use = new_title.strip() if new_title else item["title"]
    updated = ctx.deps.memory_store.create_reminder(
        title_to_use,
        target_iso or item["remind_at"],
        item.get("recurrence"),
        item.get("chat_id") or ctx.chat_id,
    )
    from app import _schedule_reminder
    _schedule_reminder(updated)

    ctx.ui_actions.append({"type": "refresh", "sections": ["reminders"]})
    return {
        "status": "ok",
        "reminder": {
            "id": updated["id"],
            "title": updated["text"],
            "remind_at": updated["run_at"],
            "recurrence": updated.get("recurrence"),
        },
    }


def _exec_delete_reminder(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if ctx.has_called_web_search:
        return {"status": "error", "message": "Destructive operations are forbidden in the same turn as web search."}

    rem_ref = args.get("reminder")
    live = ctx.deps.live_reminders()
    item, err = resolve_entity(rem_ref, live, title_key="title", entity_name="reminder")
    if err:
        return err

    summary = f"Delete the reminder '{item['title']}' (scheduled for {item['remind_at']})?"
    return ctx.confirm_mgr.request_confirmation(
        chat_id=ctx.chat_id,
        tool_name="delete_reminder",
        target=item,
        summary=summary,
    )


# 2. Tasks
def _exec_create_task(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    title = str(args.get("title", "")).strip()
    if not title:
        return {"status": "error", "message": "'title' is required."}
    priority = str(args.get("priority", "medium")).lower()
    if priority not in ("low", "medium", "high"):
        priority = "medium"

    due = args.get("due")
    due_iso = None
    if due:
        parsed = ctx.deps.parse_reminder_time(str(due))
        if parsed:
            due_iso = parsed.isoformat()

    task = ctx.deps.memory_store.create_task(title, priority=priority, due_at=due_iso)
    ctx.deps.run_learning_hook(lambda: ctx.deps.suggest_learning_for_task(task["id"]))

    ctx.ui_actions.append({"type": "refresh", "sections": ["tasks", "gamification"]})
    return {
        "status": "ok",
        "task": {
            "id": task["id"],
            "title": task["title"],
            "priority": task["priority"],
            "due_at": task.get("due_at"),
            "completed": bool(task.get("completed")),
        },
    }


def _exec_list_tasks(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    status_filter = str(args.get("status", "pending")).lower()
    include_completed = status_filter in ("completed", "all")
    all_tasks = ctx.deps.memory_store.list_tasks(include_completed=include_completed)

    if status_filter == "completed":
        filtered = [t for t in all_tasks if bool(t.get("completed"))]
    elif status_filter == "pending":
        filtered = [t for t in all_tasks if not bool(t.get("completed"))]
    else:
        filtered = all_tasks

    capped = filtered[:15]
    return {
        "status": "ok",
        "tasks": [
            {
                "id": t["id"],
                "title": t["title"],
                "priority": t["priority"],
                "due_at": t.get("due_at"),
                "completed": bool(t.get("completed")),
            }
            for t in capped
        ],
    }


def _exec_update_task(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    task_ref = args.get("task")
    tasks = ctx.deps.memory_store.list_tasks(include_completed=True)
    item, err = resolve_entity(task_ref, tasks, title_key="title", entity_name="task")
    if err:
        return err

    title = args.get("title")
    priority = args.get("priority")
    completed = args.get("completed")

    updated = ctx.deps.memory_store.update_task(
        task_id=item["id"],
        title=str(title).strip() if title else None,
        priority=str(priority).lower() if priority else None,
        completed=bool(completed) if completed is not None else None,
    )

    ctx.ui_actions.append({"type": "refresh", "sections": ["tasks", "gamification"]})
    return {
        "status": "ok",
        "task": {
            "id": updated["id"],
            "title": updated["title"],
            "priority": updated["priority"],
            "completed": bool(updated.get("completed")),
        },
    }


def _exec_delete_task(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if ctx.has_called_web_search:
        return {"status": "error", "message": "Destructive operations are forbidden in the same turn as web search."}

    task_ref = args.get("task")
    tasks = ctx.deps.memory_store.list_tasks(include_completed=True)
    item, err = resolve_entity(task_ref, tasks, title_key="title", entity_name="task")
    if err:
        return err

    summary = f"Delete the task '{item['title']}'?"
    return ctx.confirm_mgr.request_confirmation(
        chat_id=ctx.chat_id,
        tool_name="delete_task",
        target=item,
        summary=summary,
    )


# 3. Habits
def _exec_create_habit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    name = str(args.get("name", "")).strip()
    if not name:
        return {"status": "error", "message": "'name' is required."}
    daily_goal = int(args.get("daily_goal", 1))

    habit = ctx.deps.memory_store.create_habit(name=name, daily_goal=daily_goal)
    ctx.ui_actions.append({"type": "refresh", "sections": ["habits", "gamification"]})
    return {
        "status": "ok",
        "habit": {
            "id": habit["id"],
            "name": habit["name"],
            "streak": habit.get("streak", 0),
            "daily_goal": habit.get("daily_goal", 1),
        },
    }


def _exec_list_habits(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    habits = ctx.deps.memory_store.list_habits()[:15]
    today = datetime.now(ctx.deps.local_timezone).date().isoformat()
    return {
        "status": "ok",
        "habits": [
            {
                "id": h["id"],
                "name": h["name"],
                "streak": h.get("streak", 0),
                "completed_today": today in h.get("completed_dates", []),
            }
            for h in habits
        ],
    }


def _exec_complete_habit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    habit_ref = args.get("habit")
    habits = ctx.deps.memory_store.list_habits()
    item, err = resolve_entity(habit_ref, habits, title_key="name", entity_name="habit")
    if err:
        return err

    today = datetime.now(ctx.deps.local_timezone).date().isoformat()
    date_str = str(args.get("date", today))
    completed = bool(args.get("completed", True))

    updated = ctx.deps.memory_store.set_habit_completion(item["id"], date_str, completed)
    ctx.ui_actions.append({"type": "refresh", "sections": ["habits", "gamification"]})
    return {
        "status": "ok",
        "habit": {
            "id": updated["id"],
            "name": updated["name"],
            "streak": updated.get("streak", 0),
            "completed": completed,
            "date": date_str,
        },
    }


def _exec_delete_habit(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if ctx.has_called_web_search:
        return {"status": "error", "message": "Destructive operations are forbidden in the same turn as web search."}

    habit_ref = args.get("habit")
    habits = ctx.deps.memory_store.list_habits()
    item, err = resolve_entity(habit_ref, habits, title_key="name", entity_name="habit")
    if err:
        return err

    summary = f"Delete the habit '{item['name']}'?"
    return ctx.confirm_mgr.request_confirmation(
        chat_id=ctx.chat_id,
        tool_name="delete_habit",
        target=item,
        summary=summary,
    )


# 4. Memory and Knowledge
def _exec_save_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    content = str(args.get("content", "")).strip()
    if not content:
        return {"status": "error", "message": "'content' is required."}

    mem = ctx.deps.memory_store.add_memory(content)
    ctx.ui_actions.append({"type": "refresh", "sections": ["memories"]})
    return {
        "status": "ok",
        "memory": {
            "id": mem["id"],
            "content": mem["content"],
        },
    }


def _exec_search_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"status": "error", "message": "'query' is required."}

    results = ctx.deps.memory_store.search_memories(query, limit=5)
    return {
        "status": "ok",
        "results": [
            {"id": m["id"], "content": m["content"], "score": round(m.get("score", 0.0), 3)}
            for m in results
        ],
    }


def _exec_delete_memory(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if ctx.has_called_web_search:
        return {"status": "error", "message": "Destructive operations are forbidden in the same turn as web search."}

    mem_ref = args.get("memory")
    memories = ctx.deps.memory_store.list_memories()
    item, err = resolve_entity(mem_ref, memories, title_key="content", entity_name="memory")
    if err:
        return err

    summary = f"Delete saved memory: '{item['content'][:60]}'?"
    return ctx.confirm_mgr.request_confirmation(
        chat_id=ctx.chat_id,
        tool_name="delete_memory",
        target=item,
        summary=summary,
    )


# 5. Insight
def _exec_get_overview(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    tasks = ctx.deps.memory_store.list_tasks(include_completed=False)[:10]
    habits = ctx.deps.memory_store.list_habits()[:10]
    reminders = ctx.deps.live_reminders()[:5]
    gamification = ctx.deps.memory_store.get_gamification()

    today = datetime.now(ctx.deps.local_timezone).date().isoformat()
    learning = {}
    if ctx.deps.learning_schema_initialized():
        try:
            overview = ctx.deps.learning_overall_progress()
            next_t = ctx.deps.learning_next_topics(3)
            learning = {
                "progress": f"{overview.get('completed_topics', 0)}/{overview.get('topic_count', 0)} topics ({overview.get('completion_percent', 0)}%)",
                "next_topics": [t.get("name") for t in next_t],
            }
        except Exception:
            pass

    return {
        "status": "ok",
        "pending_tasks": [{"id": t["id"], "title": t["title"], "priority": t["priority"]} for t in tasks],
        "habits": [{"name": h["name"], "streak": h.get("streak", 0), "done_today": today in h.get("completed_dates", [])} for h in habits],
        "active_reminders": [{"id": r["id"], "title": r["title"], "remind_at": r["remind_at"]} for r in reminders],
        "gamification": {"xp": gamification.get("xp", 0), "global_streak": gamification.get("global_streak", 0)},
        "learning": learning,
    }


# 6. Web and UI
def _exec_web_search(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"status": "error", "message": "'query' is required."}

    ctx.has_called_web_search = True

    try:
        raw_results = ctx.deps.web_search(query, max_results=5)
        formatted = []
        for r in raw_results[:5]:
            snippet = r.get("content") or ""
            formatted.append({
                "title": r.get("title", ""),
                "domain": r.get("domain", ""),
                "snippet": snippet[:200],
                "url": r.get("url", ""),
            })
        return {"status": "ok", "results": formatted}
    except Exception as exc:
        err_msg = str(exc)
        if "not configured" in err_msg.lower() or "missing" in err_msg.lower() or "searchnotconfigured" in type(exc).__name__.lower():
            return {"status": "error", "message": "Web search is not configured."}
        return {"status": "error", "message": f"Search failed: {err_msg}"}


def _exec_navigate(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    if ctx.channel not in ("web", "agent"):
        return {"status": "error", "message": "Navigation is only available in the web app."}

    page = str(args.get("page", "")).strip().lower()
    valid_pages = ("dashboard", "habits", "todos", "reminders", "learning", "analytics", "settings")

    # Map aliases like "tasks" -> "todos"
    if page in ("task", "tasks", "todo"):
        page = "todos"
    elif page in ("habit",):
        page = "habits"
    elif page in ("reminder", "notes", "memory", "memories"):
        page = "reminders"
    elif page in ("home", "overview"):
        page = "dashboard"

    if page not in valid_pages:
        return {"status": "error", "message": f"Unknown page '{page}'. Valid pages: {', '.join(valid_pages)}."}

    route = f"/{page}"
    ctx.ui_actions.append({"type": "navigate", "page": page, "route": route})
    return {"status": "ok", "page": page, "route": route}


# --- Tool Registry Definition ---

TOOLS: list[ToolDefinition] = [
    # Reminders
    ToolDefinition(
        name="create_reminder",
        description="Schedule a reminder using natural language time (e.g. in 10 minutes, tomorrow at 9am).",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "What to be reminded about."},
                "when": {"type": "string", "description": "Natural language time (e.g. 'in 5 minutes', 'tomorrow 3pm')."},
                "recurrence": {"type": "string", "description": "Optional recurring pattern (e.g. 'every day', 'every 2 hours')."},
            },
            "required": ["title", "when"],
        },
        destructive=False,
        executor=_exec_create_reminder,
    ),
    ToolDefinition(
        name="list_reminders",
        description="List active scheduled reminders with their times.",
        parameters={"type": "object", "properties": {}},
        destructive=False,
        executor=_exec_list_reminders,
    ),
    ToolDefinition(
        name="update_reminder",
        description="Reschedule or rename an existing reminder by ID or title.",
        parameters={
            "type": "object",
            "properties": {
                "reminder": {"type": "string", "description": "Reminder ID or title."},
                "when": {"type": "string", "description": "New natural language time."},
                "title": {"type": "string", "description": "New title."},
            },
            "required": ["reminder"],
        },
        destructive=False,
        executor=_exec_update_reminder,
    ),
    ToolDefinition(
        name="delete_reminder",
        description="Delete a reminder by ID or title (requires user confirmation).",
        parameters={
            "type": "object",
            "properties": {
                "reminder": {"type": "string", "description": "Reminder ID or title to delete."},
            },
            "required": ["reminder"],
        },
        destructive=True,
        executor=_exec_delete_reminder,
    ),
    # Tasks
    ToolDefinition(
        name="create_task",
        description="Create a new task with optional priority and due time.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task description or title."},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Priority level."},
                "due": {"type": "string", "description": "Optional natural language due date/time."},
            },
            "required": ["title"],
        },
        destructive=False,
        executor=_exec_create_task,
    ),
    ToolDefinition(
        name="list_tasks",
        description="List tasks filtered by status (pending, completed, or all).",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "completed", "all"], "description": "Task status filter."},
            },
        },
        destructive=False,
        executor=_exec_list_tasks,
    ),
    ToolDefinition(
        name="update_task",
        description="Update a task's title, priority, or completion status.",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task ID or title."},
                "title": {"type": "string", "description": "New task title."},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "New priority."},
                "completed": {"type": "boolean", "description": "Set task completed status."},
            },
            "required": ["task"],
        },
        destructive=False,
        executor=_exec_update_task,
    ),
    ToolDefinition(
        name="delete_task",
        description="Delete a task by ID or title (requires user confirmation).",
        parameters={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task ID or title to delete."},
            },
            "required": ["task"],
        },
        destructive=True,
        executor=_exec_delete_task,
    ),
    # Habits
    ToolDefinition(
        name="create_habit",
        description="Create a new daily habit to track.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the habit."},
                "daily_goal": {"type": "integer", "description": "Daily target count."},
            },
            "required": ["name"],
        },
        destructive=False,
        executor=_exec_create_habit,
    ),
    ToolDefinition(
        name="list_habits",
        description="List all habits with their streaks and today's completion status.",
        parameters={"type": "object", "properties": {}},
        destructive=False,
        executor=_exec_list_habits,
    ),
    ToolDefinition(
        name="complete_habit",
        description="Mark a habit completed or uncompleted for today or a specific date.",
        parameters={
            "type": "object",
            "properties": {
                "habit": {"type": "string", "description": "Habit ID or name."},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format (defaults to today)."},
                "completed": {"type": "boolean", "description": "True to mark complete, false to unmark."},
            },
            "required": ["habit"],
        },
        destructive=False,
        executor=_exec_complete_habit,
    ),
    ToolDefinition(
        name="delete_habit",
        description="Delete a habit by ID or name (requires user confirmation).",
        parameters={
            "type": "object",
            "properties": {
                "habit": {"type": "string", "description": "Habit ID or name to delete."},
            },
            "required": ["habit"],
        },
        destructive=True,
        executor=_exec_delete_habit,
    ),
    # Memory and Knowledge
    ToolDefinition(
        name="save_memory",
        description="Save durable facts or preferences the user states about themselves.",
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact or preference to remember."},
            },
            "required": ["content"],
        },
        destructive=False,
        executor=_exec_save_memory,
    ),
    ToolDefinition(
        name="search_memory",
        description="Search saved memories. ONLY source for facts about the user.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or question to search memory."},
            },
            "required": ["query"],
        },
        destructive=False,
        executor=_exec_search_memory,
    ),
    ToolDefinition(
        name="delete_memory",
        description="Delete a saved memory by ID or content excerpt (requires confirmation).",
        parameters={
            "type": "object",
            "properties": {
                "memory": {"type": "string", "description": "Memory ID or text excerpt."},
            },
            "required": ["memory"],
        },
        destructive=True,
        executor=_exec_delete_memory,
    ),
    # Insight
    ToolDefinition(
        name="get_overview",
        description="Get a compact snapshot of pending tasks, habits, reminders, XP, and learning.",
        parameters={"type": "object", "properties": {}},
        destructive=False,
        executor=_exec_get_overview,
    ),
    # Web and UI
    ToolDefinition(
        name="web_search",
        description="Search the web for up-to-date information, news, or articles.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
            },
            "required": ["query"],
        },
        destructive=False,
        executor=_exec_web_search,
    ),
    ToolDefinition(
        name="navigate",
        description="Open a section in the web app (dashboard, habits, todos, reminders, learning, analytics, settings).",
        parameters={
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "enum": ["dashboard", "habits", "todos", "reminders", "learning", "analytics", "settings"],
                    "description": "Page to open.",
                },
            },
            "required": ["page"],
        },
        destructive=False,
        executor=_exec_navigate,
    ),
]

TOOL_REGISTRY: dict[str, ToolDefinition] = {t.name: t for t in TOOLS}


# --- Argument Validation and Execution Dispatch ---

def validate_and_execute_tool(
    tool_name: str,
    raw_args: Any,
    ctx: ToolContext,
) -> dict[str, Any]:
    """Validate tool arguments against JSON schema and execute the tool.

    Returns a small result dict:
        {"status": "ok" | "error" | "not_found" | "ambiguous" | "needs_confirmation", ...}
    """
    tool_def = TOOL_REGISTRY.get(tool_name)
    if not tool_def:
        return {"status": "error", "message": f"Unknown tool '{tool_name}'."}

    if not isinstance(raw_args, dict):
        return {"status": "error", "message": "Tool arguments must be a JSON object."}

    schema = tool_def.parameters
    props = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required keys
    for req_key in required:
        if req_key not in raw_args or raw_args[req_key] is None or (isinstance(raw_args[req_key], str) and not raw_args[req_key].strip()):
            return {"status": "error", "message": f"Missing required parameter '{req_key}' for tool '{tool_name}'."}

    # Strip unknown keys and validate types / enums
    cleaned_args: dict[str, Any] = {}
    for key, val in raw_args.items():
        if key not in props:
            continue  # Strip unknown keys silently

        expected_type = props[key].get("type")
        if expected_type == "string" and not isinstance(val, (str, int, float)):
            return {"status": "error", "message": f"Parameter '{key}' must be a string."}
        elif expected_type == "integer" and not (isinstance(val, int) and not isinstance(val, bool)):
            if isinstance(val, str) and val.isdigit():
                val = int(val)
            else:
                return {"status": "error", "message": f"Parameter '{key}' must be an integer."}
        elif expected_type == "boolean" and not isinstance(val, bool):
            if isinstance(val, str):
                val = val.lower() in ("true", "1", "yes")
            else:
                return {"status": "error", "message": f"Parameter '{key}' must be a boolean."}

        # Check enums
        enums = props[key].get("enum")
        if enums and str(val).lower() not in [e.lower() for e in enums]:
            return {"status": "error", "message": f"Parameter '{key}' must be one of: {', '.join(enums)}."}

        cleaned_args[key] = val

    try:
        return tool_def.executor(cleaned_args, ctx)
    except Exception as exc:
        log.exception("Tool %s execution failed", tool_name)
        return {"status": "error", "message": f"Execution error in {tool_name}: {exc}"}

