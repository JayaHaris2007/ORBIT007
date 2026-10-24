import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const MAX_NEW_TOKENS = 128;

function apiUrl(path) {
  const rootPath = path === "/health" || path === "/generate";
  return `${API_URL}${rootPath ? path : `/api${path}`}`;
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(apiUrl(path), {
      headers: options.body ? { "Content-Type": "application/json" } : {},
      ...options,
    });
  } catch {
    throw new Error(
      "Cannot connect to FastAPI at http://127.0.0.1:8000. Start it with: python app.py --server",
    );
  }
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : { error: `Backend returned HTTP ${response.status} instead of JSON.` };
  if (!response.ok) {
    const error = new Error(data.error || `Request failed for ${path}.`);
    error.status = response.status;
    error.path = path;
    throw error;
  }
  return data;
}

function App() {
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState("");
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState(64);
  const [memories, setMemories] = useState([]);
  const [memory, setMemory] = useState("");
  const [tasks, setTasks] = useState([]);
  const [reminders, setReminders] = useState([]);
  const [taskTitle, setTaskTitle] = useState("");
  const [taskPriority, setTaskPriority] = useState("medium");
  const [reminderText, setReminderText] = useState("");
  const [reminderAt, setReminderAt] = useState("");
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState("");
  const didLoad = useRef(false);
  const lastNotificationId = useRef(0);
  const notificationPollInFlight = useRef(false);

  async function refreshWorkspace() {
    const [healthData, chatData, memoryData, taskData, reminderData] = await Promise.all([
      api("/health"), api("/chats"), api("/memories"),
      api("/tasks?include_completed=1"), api("/reminders"),
    ]);
    if (healthData.api_version !== "jarvis-local-2026-11") {
      throw new Error(
        "The FastAPI backend is outdated. Stop it and restart: python app.py --server",
      );
    }
    setChats(chatData.chats);
    setMemories(memoryData.memories);
    setTasks(taskData.tasks);
    setReminders(reminderData.reminders);
    setOnline(healthData.status === "ok");
    let activeChatId;
    if (chatData.chats.length) {
      await selectChat(chatData.chats[0].id);
      activeChatId = chatData.chats[0].id;
    } else {
      const chat = await createChat();
      activeChatId = chat.id;
    }
    if (taskData.tasks.filter((t) => !t.completed).length > 0 && activeChatId) {
      api("/tasks/nudge", {
        method: "POST",
        body: JSON.stringify({ chat_id: activeChatId }),
      }).catch(() => {});
    }
  }

  useEffect(() => {
    if (didLoad.current) return;
    didLoad.current = true;
    refreshWorkspace().catch((requestError) => {
      setOnline(false);
      setError(requestError.message);
    });
  }, []);

  useEffect(() => {
    if (!chatId) return undefined;
    let active = true;
    const checkNotifications = async () => {
      if (notificationPollInFlight.current) return;
      notificationPollInFlight.current = true;
      try {
        const data = await api(
          `/notifications/${chatId}?after_id=${lastNotificationId.current}`,
        );
        if (!active || !data.notifications.length) return;
        lastNotificationId.current = data.notifications[data.notifications.length - 1].id;
        setMessages((current) => {
          const existingNotificationIds = new Set(
            current
              .filter((message) => message.notification)
              .map((message) => message.notificationId),
          );
          const existingContent = new Set(
            current
              .filter((message) => message.role === "assistant")
              .map((message) => message.content),
          );
          const newNotifications = data.notifications
            .filter(
              (notification) =>
                !existingNotificationIds.has(notification.id) &&
                !existingContent.has(notification.content),
            )
            .map((notification) => ({
              role: "assistant",
              content: notification.content,
              notification: true,
              notificationId: notification.id,
            }));
          return [...current, ...newNotifications];
        });
      } catch (requestError) {
        if (active && requestError.status !== 404) {
          setError(requestError.message);
        }
      } finally {
        notificationPollInFlight.current = false;
      }
    };
    checkNotifications();
    const interval = window.setInterval(checkNotifications, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [chatId]);

  async function createChat() {
    const data = await api("/chats", { method: "POST" });
    setChats((current) => [data, ...current]);
    setChatId(data.id);
    setMessages([]);
    lastNotificationId.current = 0;
    return data;
  }

  async function selectChat(id) {
    const [historyData, notificationData] = await Promise.all([
      api(`/history/${id}`),
      api(`/notifications/${id}?after_id=0`).catch((requestError) => {
        if (requestError.status !== 404) setError(requestError.message);
        return { notifications: [] };
      }),
    ]);
    const notifications = notificationData.notifications || [];
    lastNotificationId.current = notifications.length
      ? notifications[notifications.length - 1].id
      : 0;
    setMessages(historyData.messages);
    setChatId(id);
  }

  async function deleteChat(id) {
    if (!window.confirm("Delete this conversation?")) return;
    await api(`/chats/${id}`, { method: "DELETE" });
    const remaining = chats.filter((chat) => chat.id !== id);
    setChats(remaining);
    if (id === chatId) remaining.length ? selectChat(remaining[0].id) : createChat();
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const value = prompt.trim();
    if (!value || loading) return;
    setLoading(true);
    setError("");
    setPrompt("");
    try {
      let activeChatId = chatId;
      if (!activeChatId || !chats.some((chat) => chat.id === activeChatId)) {
        const chat = await createChat();
        activeChatId = chat.id;
      }
      setMessages((current) => [...current, { role: "user", content: value }]);
      const data = await api("/generate", {
        method: "POST",
        body: JSON.stringify({ prompt: value, max_new_tokens: maxNewTokens, chat_id: activeChatId }),
      });
      setMessages(data.messages);
      setChats((current) => current.map((chat) => (
        chat.id === data.chat_id && chat.title === "New conversation"
          ? { ...chat, title: value.slice(0, 80) } : chat
      )));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  async function addTask(event) {
    event.preventDefault();
    if (!taskTitle.trim()) return;
    try {
      const task = await api("/tasks", {
        method: "POST",
        body: JSON.stringify({ title: taskTitle, priority: taskPriority }),
      });
      setTasks((current) => [...current, task]);
      setTaskTitle("");
    } catch (requestError) { setError(requestError.message); }
  }

  async function toggleTask(task) {
    try {
      const updated = await api(`/tasks/${task.id}`, {
        method: "PATCH",
        body: JSON.stringify({ completed: !task.completed, chat_id: chatId }),
      });
      setTasks((current) => current.map((item) => item.id === updated.id ? updated : item));
    } catch (requestError) { setError(requestError.message); }
  }

  async function removeTask(id) {
    try {
      await api(`/tasks/${id}`, { method: "DELETE" });
      setTasks((current) => current.filter((task) => task.id !== id));
    } catch (requestError) { setError(requestError.message); }
  }

  async function addReminder(event) {
    event.preventDefault();
    if (!reminderText.trim() || !reminderAt) return;
    try {
      const reminder = await api("/reminders", {
        method: "POST",
        body: JSON.stringify({
          text: reminderText,
          run_at: new Date(reminderAt).toISOString(),
          chat_id: chatId,
        }),
      });
      setReminders((current) => [...current, reminder]);
      setReminderText("");
      setReminderAt("");
    } catch (requestError) { setError(requestError.message); }
  }

  async function cancelReminder(id) {
    try {
      await api(`/reminders/${id}`, { method: "DELETE" });
      setReminders((current) => current.filter((reminder) => reminder.id !== id));
    } catch (requestError) { setError(requestError.message); }
  }

  async function saveMemory(event) {
    event.preventDefault();
    if (!memory.trim()) return;
    try {
      const item = await api("/memories", {
        method: "POST", body: JSON.stringify({ content: memory }),
      });
      setMemories((current) => [item, ...current]);
      setMemory("");
    } catch (requestError) { setError(requestError.message); }
  }

  async function removeMemory(id) {
    await api(`/memories/${id}`, { method: "DELETE" });
    setMemories((current) => current.filter((item) => item.id !== id));
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">J</span><div><strong>JARVIS</strong><small>PERSONAL OPERATIONS</small></div></div>
        <button className="new-chat" onClick={createChat}>+ NEW CHAT</button>
        <nav className="chat-list" aria-label="Conversations">
          <p className="section-label">CONVERSATIONS</p>
          {chats.map((chat) => (
            <div className={`chat-item ${chat.id === chatId ? "selected" : ""}`} key={chat.id}>
              <button onClick={() => selectChat(chat.id)}>{chat.title}</button>
              <button className="icon-button" onClick={() => deleteChat(chat.id)} aria-label="Delete conversation">×</button>
            </div>
          ))}
        </nav>
        <section className="side-memory">
          <p className="section-label">LONG-TERM MEMORY</p>
          <form onSubmit={saveMemory} className="inline-form">
            <input value={memory} onChange={(event) => setMemory(event.target.value)} placeholder="Save a fact..." maxLength={500} />
            <button type="submit">+</button>
          </form>
          {memories.slice(0, 4).map((item) => (
            <div className="memory-item" key={item.id}><span>{item.content}</span><button className="icon-button" onClick={() => removeMemory(item.id)}>×</button></div>
          ))}
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div><p className="eyebrow">LOCAL INTELLIGENCE / CONTROL ROOM</p><h1>Good evening.</h1><p className="subtitle">Your private assistant is ready to help you focus.</p></div>
          <div className="health-badge"><span className={`status-dot ${online ? "" : "offline"}`} /> {online ? "SYSTEM ONLINE" : "OFFLINE"}<small>QWEN 1.7B · LOCAL</small></div>
        </header>

        <div className="dashboard-grid">
          <section className="panel chat-panel">
            <div className="panel-heading"><div><span className="panel-kicker">ACTIVE SESSION</span><h2>Conversation</h2></div><span className="live-pill">● PRIVATE</span></div>
            <div className="conversation" aria-live="polite">
              {!messages.length && <div className="empty-state"><span className="empty-mark">✦</span><h3>How can I help?</h3><p>Ask about your plans, tasks, or anything on your mind.</p></div>}
              {messages.map((message, index) => (
                <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  <span className="message-label">{message.role === "user" ? "YOU" : "JARVIS"}</span>
                  {message.role === "assistant" ? <ReactMarkdown>{message.content}</ReactMarkdown> : <p>{message.content}</p>}
                </article>
              ))}
              {loading && <article className="message assistant"><span className="message-label">JARVIS</span><p className="thinking">Thinking<span>.</span><span>.</span><span>.</span></p></article>}
            </div>
            {error && <div className="error" role="alert">{error}</div>}
            <form className="composer" onSubmit={handleSubmit}>
              <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Message JARVIS..." maxLength={2000} rows={2} disabled={loading} />
              <div className="composer-footer"><span>{prompt.length} / 2000</span><label>MAX TOKENS <input type="number" min="1" max={MAX_NEW_TOKENS} value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} /></label><button type="submit" disabled={loading || !prompt.trim()}>{loading ? "WORKING..." : "SEND ↗"}</button></div>
            </form>
          </section>

          <aside className="tools-column">
            <section className="panel tool-panel"><div className="panel-heading"><div><span className="panel-kicker">PRODUCTIVITY</span><h2>Tasks</h2></div><span className="count-badge">{tasks.filter((task) => !task.completed).length}</span></div>
              <form className="stack-form" onSubmit={addTask}><input value={taskTitle} onChange={(event) => setTaskTitle(event.target.value)} placeholder="Add a task..." maxLength={500} /><div><select value={taskPriority} onChange={(event) => setTaskPriority(event.target.value)}><option value="high">High priority</option><option value="medium">Medium priority</option><option value="low">Low priority</option></select><button type="submit">ADD</button></div></form>
              <div className="item-list">{tasks.slice(0, 8).map((task) => <div className={`task-item ${task.completed ? "completed" : ""}`} key={task.id}><button className="check-button" onClick={() => toggleTask(task)} aria-label="Complete task">{task.completed ? "✓" : ""}</button><span>{task.title}<small className={`priority-${task.priority}`}>{task.priority.toUpperCase()}</small></span><button className="icon-button" onClick={() => removeTask(task.id)}>×</button></div>)}</div>
            </section>
            <section className="panel tool-panel"><div className="panel-heading"><div><span className="panel-kicker">UP NEXT</span><h2>Reminders</h2></div><span className="count-badge">{reminders.length}</span></div>
              <form className="stack-form" onSubmit={addReminder}><input value={reminderText} onChange={(event) => setReminderText(event.target.value)} placeholder="Reminder description..." maxLength={500} /><div><input type="datetime-local" value={reminderAt} onChange={(event) => setReminderAt(event.target.value)} required /><button type="submit">SET</button></div></form>
              <div className="item-list">{reminders.slice(0, 5).map((reminder) => <div className="reminder-item" key={reminder.id}><span className="reminder-dot" /><span>{reminder.text}<small>{new Date(reminder.run_at).toLocaleString()}</small></span><button className="icon-button" onClick={() => cancelReminder(reminder.id)}>×</button></div>)}</div>
            </section>
          </aside>
        </div>
      </section>
    </main>
  );
}

export default App;
