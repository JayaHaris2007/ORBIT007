"""Weekly learning nudges: compose a plain-spoken recap and send it once.

The nudge is delivered through the same notification bus reminders use, so
the web panel surfaces it without any extra frontend plumbing. Everything
here is deterministic and cheap — no model calls, no new tables.

Guards:

- ``build_weekly_nudge`` returns ``None`` when nudges are disabled, when
  there is no progress to report, or when there is nothing actionable left.
- ``deliver_weekly_nudge`` is idempotent per week: each local chat gets at
  most one nudge every seven days, keyed by the notification's reminder id.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import progress, schema

NUDGE_NOTIFICATION_KEY = "weekly-learning-nudge"
IDLE_TOPICS_IN_MESSAGE = 2
NEXT_TOPICS_IN_MESSAGE = 2
NUDGE_REPEAT_WINDOW_DAYS = 6


def build_weekly_nudge() -> str | None:
    """Plain-sentence weekly recap, or None when there is nothing to say."""
    if not progress.nudges_enabled():
        return None
    overview = progress.overall_progress()
    total = overview["topic_count"]
    done = overview["completed_topics"]
    if total == 0:
        return None

    parts: list[str] = []
    percent = overview["completion_percent"]
    if done >= total:
        parts.append(
            f"You've finished all {total} of your learning topics — "
            "that's 100% complete. Nice work."
        )
    else:
        in_progress = overview["in_progress_topics"]
        middle = (
            f", {in_progress} in progress" if in_progress else ""
        )
        parts.append(
            f"You've completed {done} of {total} learning topics "
            f"({percent}%{middle})."
        )

    candidates = progress.next_topics(limit=20)
    idle = [
        topic for topic in candidates
        if topic["idle_days"] >= progress.IDLE_DAYS_FOR_REASON
    ][:IDLE_TOPICS_IN_MESSAGE]
    for topic in idle:
        parts.append(
            f"You haven't touched {topic['name']} in "
            f"{topic['roadmap_name']} for {topic['idle_days']} days — "
            "pick it back up."
        )

    if not idle and not parts[0].startswith("You've finished"):
        fresh = candidates[:NEXT_TOPICS_IN_MESSAGE]
        if fresh:
            first = fresh[0]
            lead = (
                f"{first['name']} ({first['roadmap_name']}) is ready for you."
            )
            if len(fresh) > 1:
                names = " and ".join(topic["name"] for topic in fresh)
                lead = f"Up next: {names}."
            parts.append(lead)

    return " ".join(parts).strip() or None


def _recently_sent(connection, chat_id: str, now: datetime) -> bool:
    cutoff = (now - timedelta(days=NUDGE_REPEAT_WINDOW_DAYS)).isoformat()
    row = connection.execute(
        "SELECT 1 FROM notifications WHERE chat_id = ? "
        "AND reminder_id LIKE ? AND created_at > ? LIMIT 1",
        (chat_id, f"{NUDGE_NOTIFICATION_KEY}-%", cutoff),
    ).fetchone()
    return row is not None


def deliver_weekly_nudge(
    chat_ids: list[str],
    add_notification,
    recently_sent=None,
    *,
    now: datetime | None = None,
) -> int:
    """Send the weekly nudge to each chat that has not had one this week.

    ``add_notification(chat_id, key, content)`` is the MemoryStore method;
    it is injected so this module never imports app or memory. The same
    goes for ``recently_sent(chat_id, now)``: notifications live in the
    memory database, not the learning database, so the caller supplies the
    lookup too.
    """
    if not progress.nudges_enabled():
        return 0
    message = build_weekly_nudge()
    if message is None:
        return 0
    now = now or datetime.now(timezone.utc)
    sent = 0
    with schema.connect() as connection:
        for chat_id in chat_ids:
            if recently_sent is not None and recently_sent(chat_id, now):
                continue
            add_notification(
                chat_id, f"{NUDGE_NOTIFICATION_KEY}-{now.date().isoformat()}",
                message,
            )
            sent += 1
    return sent
