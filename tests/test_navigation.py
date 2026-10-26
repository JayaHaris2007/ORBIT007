"""Page-navigation command parsing tests for app.py.

'Go to tasks' style chat commands are answered locally with a
local-navigate:<page> provider so the frontend can switch pages.
"""

import pytest

import app

CASES = [
    ("go to tasks", "tasks"),
    ("move to the habit section", "habits"),
    ("can you take me to the reminders page", "reminders"),
    ("open dashboard", "dashboard"),
    ("switch to to-do page", "tasks"),
    ("go to sleep", None),
    ("open task 3", None),
    ("show my habits", None),
]


@pytest.mark.parametrize("prompt, expected", CASES)
def test_parse_navigation(prompt, expected):
    assert app._parse_navigation(prompt) == expected
