import shutil
import tempfile
from pathlib import Path

# Use a COPY of assistant.db so real data isn't touched
_temp_dir = tempfile.TemporaryDirectory()
_test_db = Path(_temp_dir.name) / "assistant.db"
_src_db = Path(__file__).resolve().parent / "assistant.db"
if _src_db.is_file():
    shutil.copy2(_src_db, _test_db)

import app

app.DATABASE_PATH = _test_db
app.memory_store = app.MemoryStore(_test_db)

app.init_learning_module("discord")
chat = app.memory_store.get_or_create_external_chat("discord", "test-user")["id"]

for phrase in [
    "Create a new habits exercise",
    "Set new habits exercise",
    "Mark the explore AI basics tasks completed",
    "what should I learn next",
    "I completed HTML Fundamentals",
    "make a habit",
    "create a banana",
]:
    try:
        print(f"{phrase!r} -> {app._handle_local_command(phrase, chat)}")
    except Exception as error:
        print(f"{phrase!r} -> ERROR {error}")
