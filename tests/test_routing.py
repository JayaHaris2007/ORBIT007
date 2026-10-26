"""Routing regression tests for the intent detectors in app.py.

Advice questions about habits/tasks/reminders must fall through to the LLM
instead of being trapped by the deterministic list handlers.
"""

import app


def test_habit_list_detector_true_cases():
    assert app._is_habit_list_request("what are habits i am having") is True
    assert app._is_habit_list_request("show my pending habits") is True


def test_habit_list_detector_advice_false_cases():
    assert (
        app._is_habit_list_request(
            "what new habits do i need to add to improve myself"
        )
        is False
    )
    assert (
        app._is_habit_list_request(
            "my habits and what should i do to improve"
        )
        is False
    )
    assert (
        app._is_habit_list_request("what do you think about my habits") is False
    )


def test_task_list_detector_cases():
    assert app._is_task_list_request("show my tasks") is True
    assert app._is_task_list_request("give me task ideas") is False
    assert app._is_task_list_request("how can i manage my tasks better") is False


def test_reminder_list_detector_cases():
    assert app._is_reminder_list_request("show my reminders") is True
    assert app._is_reminder_list_request("suggest some reminders i should set") is False


def test_entity_parsers_still_work():
    assert app._parse_chat_habit("add habit drink water") is not None
    assert app._parse_task_create("add task finish report") is not None
