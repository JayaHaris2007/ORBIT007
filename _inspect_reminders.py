import sqlite3
from datetime import datetime, timezone

connection = sqlite3.connect("assistant.db")
rows = connection.execute(
    "SELECT id, text, run_at, recurrence, active, chat_id IS NOT NULL FROM reminders ORDER BY run_at"
).fetchall()
print("total reminders:", len(rows))
for row in rows:
    print(row[4], row[3], row[2], "|", row[1][:40])
print("active:", connection.execute("SELECT COUNT(*) FROM reminders WHERE active=1").fetchone()[0])
