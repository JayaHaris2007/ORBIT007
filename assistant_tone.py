"""Human-sounding copy for the assistant's deterministic replies.

``app.py`` decides *what* happened; this module decides how S.P.I.D.E.Y says
it. Wording varies a little on purpose so a repeated command doesn't read like
a template, but every sentence still reports the real result — nothing here
invents an action or a time.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Sequence

from memory import LOCAL_TIMEZONE


_last_pick: dict[tuple[str, ...], str] = {}


def _pick(options: Sequence[str]) -> str:
    """Choose a phrasing, never repeating the previous one for the same set."""
    key = tuple(options)
    choices = [option for option in options if option != _last_pick.get(key)]
    choice = random.choice(choices or list(options))
    _last_pick[key] = choice
    return choice


def _as_local(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=LOCAL_TIMEZONE)
    return moment.astimezone(LOCAL_TIMEZONE)


def _clock(moment: datetime) -> str:
    """'11:11 PM', or '9 PM' when the minutes are on the hour."""
    pattern = "%I %p" if moment.minute == 0 else "%I:%M %p"
    return moment.strftime(pattern).lstrip("0")


def _day_phrase(offset: int, scheduled: datetime, reference: datetime) -> str:
    if offset == 0:
        if scheduled.hour >= 20:
            return "tonight"
        if scheduled.hour >= 17:
            return "this evening"
        return "this afternoon" if scheduled.hour >= 12 else "today"
    if offset == 1:
        return "tomorrow"
    if offset == -1:
        return "yesterday"
    if 1 < offset <= 6:
        return f"on {scheduled.strftime('%A')}"
    day = f"{scheduled.day} {scheduled.strftime('%b')}"
    if scheduled.year != reference.year:
        return f"on {day} {scheduled.year}"
    return f"on {day}"


def humanize_when(value: str | datetime | None, *, now: datetime | None = None) -> str:
    """Say a timestamp the way a person would, not like a log line."""
    scheduled = _as_local(value)
    if scheduled is None:
        return str(value or "")
    reference = _as_local(now) or datetime.now(LOCAL_TIMEZONE)
    seconds = (scheduled - reference).total_seconds()

    if 0 <= seconds < 45:
        return "in a moment"
    if 0 <= seconds < 90:
        return "in about a minute"
    if 0 <= seconds < 3_600:
        return f"in {max(2, round(seconds / 60))} minutes"

    offset = (scheduled.date() - reference.date()).days
    return f"{_day_phrase(offset, scheduled, reference)} at {_clock(scheduled)}"


def _open_tasks_line(count: int) -> str:
    if count <= 0:
        return "Your list is clear now."
    if count == 1:
        return "It's the only thing on your list right now."
    return f"That's {count} open tasks on your plate."


def _personal_opener(name: str) -> str:
    return _pick([f"{name}, ", f"Okay {name}, ", f"Hey {name}, "]) if name else ""


def reminder_scheduled(
    text: str,
    run_at: str | datetime | None,
    *,
    recurrence: str | None = None,
    name: str = "",
) -> str:
    when = humanize_when(run_at)
    if recurrence:
        return _pick([
            f"Done — I'll remind you about **{text}** {when}, and keep it going "
            f"{recurrence}.",
            f"Got it. **{text}** {when}, repeating {recurrence}.",
        ])
    options = [
        f"Done — I'll remind you about **{text}** {when}.",
        f"Okay, that's set: **{text}**, {when}.",
        f"Got it. **{text}** — I'll bring it up {when}.",
    ]
    if name:
        options.append(f"You're all set, {name} — I'll remind you about **{text}** {when}.")
    return _pick(options)


def reminder_moved(text: str, run_at: str | datetime | None, *, name: str = "") -> str:
    when = humanize_when(run_at)
    return _pick([
        f"Moved **{text}** to {when}.",
        f"Done — **{text}** now comes up {when}.",
        f"Okay, **{text}** is pushed to {when}.",
    ])


def reminder_cancelled(text: str, *, name: str = "") -> str:
    return _pick([
        f"Cancelled — I won't bring up **{text}** anymore.",
        f"Done, that reminder about **{text}** is gone.",
        f"Okay, **{text}** is off my list.",
    ])


def reminder_fired(text: str, *, now: datetime | None = None, name: str = "") -> str:
    """The notification that arrives when a reminder is due."""
    clock = _clock(_as_local(now) or datetime.now(LOCAL_TIMEZONE))
    opener = f"Hey {name}, " if name else "Hey, "
    return _pick([
        f"{opener}quick reminder — **{text}**.",
        f"It's {clock}. Just a heads-up: **{text}**.",
        f"Just a nudge: **{text}**. It's {clock} right now.",
    ])


def task_created(
    title: str, priority: str, open_count: int, *, name: str = ""
) -> str:
    options = [
        f"Added **{title}** ({priority} priority). {_open_tasks_line(open_count)}",
        f"**{title}** is on your list — {priority} priority. "
        f"{_open_tasks_line(open_count)}",
        f"Done, **{title}** is in. {_open_tasks_line(open_count)}",
    ]
    if name:
        options.append(
            f"Nice, {name} — **{title}** is on the list ({priority} priority). "
            f"{_open_tasks_line(open_count)}"
        )
    return _pick(options)


def task_completed(
    title: str,
    remaining: int,
    *,
    next_title: str = "",
    next_priority: str = "",
    name: str = "",
) -> str:
    if remaining <= 0:
        return _pick([
            f"Nice — **{title}** is done, and that clears your list. Take a breath.",
            f"**{title}** is finished and nothing else is waiting. Good place to be.",
        ])
    if next_title:
        priority = " (high priority)" if next_priority == "high" else ""
        options = [
            f"Nice — **{title}** is done. **{next_title}**{priority} is next when "
            "you're ready.",
            f"**{title}** is off the list. {remaining} to go — **{next_title}** is "
            "the one I'd pick next.",
            f"That's **{title}** finished. **{next_title}** is the next one worth "
            "your time.",
        ]
        if name:
            options.append(
                f"Nice one, {name} — **{title}** is done. **{next_title}**{priority} "
                "is waiting if you want to keep going."
            )
        return _pick(options)
    options = [
        f"Nice work — **{title}** is done. {remaining} to go.",
        f"**{title}** is off the list. {remaining} left.",
        f"That's **{title}** finished — {remaining} still open.",
    ]
    if name:
        options.append(f"Nice one, {name} — **{title}** is done. {remaining} to go.")
    return _pick(options)


def task_reopened(title: str, *, name: str = "") -> str:
    return _pick([
        f"Okay, **{title}** is back on your list.",
        f"Reopened **{title}** — it's waiting for you again.",
    ])


def task_deleted(title: str, remaining: int) -> str:
    return _pick([
        f"Dropped **{title}**. {_open_tasks_line(remaining)}",
        f"**{title}** is gone. {_open_tasks_line(remaining)}",
    ])


def task_priority(title: str, priority: str) -> str:
    if priority == "high":
        return _pick([
            f"**{title}** is high priority now — I'll treat it as the important one.",
            f"Bumped **{title}** up to high priority.",
        ])
    return _pick([
        f"**{title}** is {priority} priority now.",
        f"Changed — **{title}** sits at {priority} priority.",
    ])


def task_due(title: str, due_at: str | datetime | None) -> str:
    return _pick([
        f"Got it — **{title}** is due {humanize_when(due_at)}.",
        f"**{title}** now falls due {humanize_when(due_at)}.",
    ])


def task_renamed(title: str) -> str:
    return _pick([
        f"Renamed it to **{title}**.",
        f"Done — the task is called **{title}** now.",
    ])


def habit_created(name: str, total: int) -> str:
    if total <= 1:
        return _pick([
            f"**{name}** is on your habit list — first one, and I'll start the "
            "streak today.",
            f"Got it, **{name}** is in. Check it off once today and the streak begins.",
        ])
    return _pick([
        f"**{name}** is on your habit list. That's {total} habits I'm tracking for you.",
        f"Got it — **{name}** is added. You've got {total} habits going now.",
        f"**{name}** is in. {total} habits on the list, I'll keep the streak for you.",
    ])


def habit_completed(
    name: str, streak: int, *, pending: Sequence[str] = (), person: str = ""
) -> str:
    if streak <= 1:
        base = _pick([
            f"**{name}** is checked off — day one of the streak.",
            f"First check-off for **{name}**. The streak starts now.",
        ])
    elif streak % 7 == 0:
        base = _pick([
            f"**{name}** is done — that's a {streak}-day streak. A full week, nicely held.",
            f"{streak} days straight on **{name}**. That's the habit sticking.",
        ])
    else:
        options = [
            f"**{name}** is checked off for today. {streak} days in a row now.",
            f"Done — **{name}** is on the board. Streak's at {streak}.",
            f"**{name}** is done. {streak} days running.",
        ]
        if person:
            options.append(
                f"Nice one, {person} — **{name}** is done. {streak} days in a row."
            )
        base = _pick(options)

    if not pending:
        return f"{base} {_pick(['That clears every habit for today.', 'And that is the whole habit list done for today.'])}"
    if len(pending) == 1:
        return (
            f"{base} "
            + _pick([
                f"**{pending[0]}** is still open if you want to keep going.",
                f"Only **{pending[0]}** left today — want to finish the set?",
            ])
        )
    if len(pending) == 2:
        return (
            f"{base} "
            + _pick([
                f"**{pending[0]}** and **{pending[1]}** are still open today.",
                f"Two habits left: **{pending[0]}** and **{pending[1]}**.",
            ])
        )
    return (
        f"{base} "
        + _pick([
            f"You've still got **{pending[0]}** and {len(pending) - 1} more today.",
            f"**{pending[0]}** is next if you're on a roll — {len(pending)} habits left.",
        ])
    )


def habit_renamed(name: str) -> str:
    return _pick([
        f"Renamed the habit to **{name}**.",
        f"Done — it's **{name}** now.",
    ])


def habit_deleted(name: str) -> str:
    return _pick([
        f"Removed **{name}** from your habits.",
        f"Okay, **{name}** is off the habit list.",
    ])
