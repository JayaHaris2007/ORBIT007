from db import DB_PATH, init_db, list_reminders

init_db()
print(f"Inspecting database at: {DB_PATH}")
reminders = list_reminders(include_completed=True)
print("total reminders:", len(reminders))
for r in reminders:
    print(r.get("id"), r.get("completed"), r.get("remind_at"), "|", r.get("title"))
active = list_reminders(include_completed=False)
print("active:", len(active))
