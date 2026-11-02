"""Pending confirmation manager for destructive actions.

The model can never execute destructive actions on its own. Instead, destructive
tools store a pending action and request user confirmation. At the start of the
next turn, user affirmation deterministically executes the pending action, while
negation or any unrelated message cancels it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from agent.deps import AgentDeps

# Affirmative patterns
_AFFIRMATIVE_PATTERN = re.compile(
    r"^(?:yes|yeah|yep|yup|confirm|do\s+it|ok\s+delete\s+it|okay\s+delete\s+it|sure|proceed|please\s+do|affirmative)(?:[\s.!,]+.*)?$",
    re.IGNORECASE,
)

# Negative patterns
_NEGATIVE_PATTERN = re.compile(
    r"^(?:no|nah|nope|cancel|don\'?t|stop|never\s*mind|abort|negative)(?:[\s.!,]+.*)?$",
    re.IGNORECASE,
)


@dataclass
class PendingAction:
    chat_id: str
    tool_name: str
    target: dict[str, Any]
    summary: str
    expires_at: float


class ConfirmationManager:
    """Manages pending destructive confirmations per chat session."""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._pending: dict[str, PendingAction] = {}

    def request_confirmation(
        self, chat_id: str, tool_name: str, target: dict[str, Any], summary: str
    ) -> dict[str, Any]:
        """Store pending action and return status for the tool executor."""
        action = PendingAction(
            chat_id=chat_id,
            tool_name=tool_name,
            target=target,
            summary=summary,
            expires_at=time.time() + self.timeout_seconds,
        )
        self._pending[chat_id] = action
        return {
            "status": "needs_confirmation",
            "summary": summary,
            "tool": tool_name,
            "target_id": target.get("id"),
        }

    def get_pending(self, chat_id: str) -> PendingAction | None:
        """Get valid pending action if not expired."""
        action = self._pending.get(chat_id)
        if action is None:
            return None
        if time.time() > action.expires_at:
            self._pending.pop(chat_id, None)
            return None
        return action

    def clear(self, chat_id: str) -> None:
        """Clear any pending action for a chat."""
        self._pending.pop(chat_id, None)

    def check_and_handle(
        self, chat_id: str, prompt: str, deps: AgentDeps
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Check at turn start if this prompt confirms or denies a pending action.

        Returns:
            (handled, reply_text, ui_actions)
            If handled is True, the turn is completed deterministically.
            If handled is False, proceed to rules and the normal turn.
        """
        pending = self.get_pending(chat_id)
        if pending is None:
            return False, "", []

        prompt_clean = prompt.strip().lower()

        # Check affirmative
        if _AFFIRMATIVE_PATTERN.match(prompt_clean):
            self._pending.pop(chat_id, None)
            reply, actions = self._execute_action(pending, deps)
            return True, reply, actions

        # Check negative
        if _NEGATIVE_PATTERN.match(prompt_clean):
            self._pending.pop(chat_id, None)
            target_name = (
                pending.target.get("title")
                or pending.target.get("name")
                or pending.target.get("text")
                or pending.target.get("content")
                or f"ID {pending.target.get('id')}"
            )
            return (
                True,
                f"Action cancelled. The item '{target_name}' was not deleted.",
                [],
            )

        # Unrelated message: cancel pending action and let normal turn handle the message
        self._pending.pop(chat_id, None)
        return False, "", []

    def _execute_action(
        self, action: PendingAction, deps: AgentDeps
    ) -> tuple[str, list[dict[str, Any]]]:
        """Execute the confirmed action deterministically."""
        target = action.target
        target_id = target.get("id")
        target_name = (
            target.get("title")
            or target.get("name")
            or target.get("text")
            or target.get("content")
            or f"ID {target_id}"
        )

        tool = action.tool_name

        if tool == "delete_task":
            if target_id is not None:
                deps.memory_store.delete_task(int(target_id))
            return (
                f"Task '{target_name}' has been deleted.",
                [{"type": "refresh", "sections": ["tasks", "gamification"]}],
            )

        elif tool == "delete_habit":
            if target_id is not None:
                deps.memory_store.delete_habit(int(target_id))
            return (
                f"Habit '{target_name}' has been deleted.",
                [{"type": "refresh", "sections": ["habits", "gamification"]}],
            )

        elif tool == "delete_reminder":
            if target_id is not None:
                deps.cancel_reminder_unified(str(target_id))
            return (
                f"Reminder '{target_name}' has been deleted.",
                [{"type": "refresh", "sections": ["reminders"]}],
            )

        elif tool == "delete_memory":
            if target_id is not None:
                deps.memory_store.delete_memory(int(target_id))
            return (
                f"Memory fact '{target_name}' has been deleted.",
                [{"type": "refresh", "sections": ["memories"]}],
            )

        return f"Completed {action.summary}.", []

