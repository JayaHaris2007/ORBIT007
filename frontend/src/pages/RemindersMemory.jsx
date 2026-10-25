import { useEffect, useState } from 'react';
import { format } from 'date-fns';
import { Bell, Brain, Plus, Trash2 } from 'lucide-react';
import { apiRequest } from '../services/api';
import { useAppData } from '../context/AppDataContext';

const ACTIVE_CHAT_KEY = 'spidey_active_chat_id';
const LEGACY_ACTIVE_CHAT_KEY = 'jarvis_active_chat_id';

const byRunAt = (a, b) => new Date(a.run_at) - new Date(b.run_at);

const RemindersMemory = () => {
    const { revision } = useAppData();
    const [reminders, setReminders] = useState([]);
    const [memories, setMemories] = useState([]);
    const [reminderText, setReminderText] = useState('');
    const [reminderAt, setReminderAt] = useState('');
    const [memoryText, setMemoryText] = useState('');
    const [chatId, setChatId] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        let cancelled = false;
        const load = async () => {
            try {
                const [chatData, reminderData, memoryData] = await Promise.all([
                    apiRequest('/api/chats'),
                    apiRequest('/api/reminders'),
                    apiRequest('/api/memories')
                ]);
                if (cancelled) return;
                const savedChatId = localStorage.getItem(ACTIVE_CHAT_KEY) || localStorage.getItem(LEGACY_ACTIVE_CHAT_KEY);
                const chat = chatData.chats.find((item) => item.id === savedChatId) || chatData.chats[0];
                setChatId(chat?.id || '');
                setReminders([...(reminderData.reminders || [])].sort(byRunAt));
                setMemories(memoryData.memories || []);
            } catch (requestError) {
                if (!cancelled) setError(requestError.message);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };
        load();
        return () => { cancelled = true; };
    }, [revision]);

    const addReminder = async (event) => {
        event.preventDefault();
        if (!reminderText.trim() || !reminderAt) return;
        try {
            const reminder = await apiRequest('/api/reminders', {
                method: 'POST',
                body: JSON.stringify({
                    text: reminderText.trim(),
                    run_at: new Date(reminderAt).toISOString(),
                    chat_id: chatId || undefined
                })
            });
            setReminders((current) => [...current, reminder].sort(byRunAt));
            setReminderText('');
            setReminderAt('');
            setError('');
        } catch (requestError) {
            setError(requestError.message);
        }
    };

    const cancelReminder = async (id) => {
        try {
            await apiRequest(`/api/reminders/${id}`, { method: 'DELETE' });
            setReminders((current) => current.filter((item) => item.id !== id));
        } catch (requestError) {
            setError(requestError.message);
        }
    };

    const addMemory = async (event) => {
        event.preventDefault();
        if (!memoryText.trim()) return;
        try {
            const item = await apiRequest('/api/memories', {
                method: 'POST',
                body: JSON.stringify({ content: memoryText.trim() })
            });
            setMemories((current) => [item, ...current]);
            setMemoryText('');
            setError('');
        } catch (requestError) {
            setError(requestError.message);
        }
    };

    const removeMemory = async (id) => {
        try {
            await apiRequest(`/api/memories/${id}`, { method: 'DELETE' });
            setMemories((current) => current.filter((item) => item.id !== id));
        } catch (requestError) {
            setError(requestError.message);
        }
    };

    if (loading) return <div className="p-10 text-center animate-pulse text-textMuted">Loading reminders and memories...</div>;

    return (
        <div className="space-y-6 pb-20">
            <header>
                <p className="text-xs uppercase tracking-[0.2em] text-textMuted">Remembered for you</p>
                <h1 className="mt-2 text-3xl font-bold uppercase tracking-widest text-white glow-white">Reminders &amp; memory</h1>
                <p className="mt-1 text-sm font-mono text-textMuted">Everything S.P.I.D.E.Y should not forget.</p>
            </header>

            {error && <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">{error}</div>}

            <section className="glass-card">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="flex items-center gap-2 text-lg font-bold text-white">
                        <Bell size={18} className="text-orange-400" /> Reminders
                    </h2>
                    <span className="font-mono text-[10px] uppercase tracking-wider text-textMuted">{reminders.length} scheduled</span>
                </div>

                <form onSubmit={addReminder} className="mt-5 flex flex-col gap-3 md:flex-row">
                    <input value={reminderText} onChange={(event) => setReminderText(event.target.value)} maxLength={500} placeholder="What should I remind you about?" className="min-w-0 flex-1 rounded-lg border border-white/10 bg-surfaceHighlight px-3 py-2.5 text-sm text-white placeholder:text-neutral-600 focus:border-orange-400/40 focus:outline-none" />
                    <input type="datetime-local" value={reminderAt} onChange={(event) => setReminderAt(event.target.value)} className="rounded-lg border border-white/10 bg-surfaceHighlight px-3 py-2.5 text-sm text-white [color-scheme:dark] focus:border-orange-400/40 focus:outline-none" />
                    <button type="submit" disabled={!reminderText.trim() || !reminderAt} className="flex items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 text-xs font-bold uppercase tracking-wider text-black transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-40">
                        <Plus size={14} /> Add
                    </button>
                </form>

                <div className="mt-4 divide-y divide-white/5">
                    {reminders.map((reminder) => (
                        <div key={reminder.id} className="flex items-center gap-3 py-3">
                            <div className="min-w-0 flex-1">
                                <p className="truncate text-sm text-white">{reminder.text}</p>
                                <p className="mt-0.5 text-xs font-mono text-textMuted">
                                    {format(new Date(reminder.run_at), 'MMM d, h:mm a')}
                                    {reminder.recurrence ? ` · ${reminder.recurrence}` : ''}
                                </p>
                            </div>
                            <button type="button" onClick={() => cancelReminder(reminder.id)} className="text-textMuted transition-colors hover:text-red-300 p-2" aria-label={`Cancel reminder ${reminder.text}`}>
                                <Trash2 size={15} />
                            </button>
                        </div>
                    ))}
                    {!reminders.length && <p className="py-3 text-sm italic text-textMuted">Nothing scheduled. Ask for a reminder or add one above.</p>}
                </div>
            </section>

            <section className="glass-card">
                <div className="flex items-center justify-between gap-3">
                    <h2 className="flex items-center gap-2 text-lg font-bold text-white">
                        <Brain size={18} className="text-orange-400" /> Saved memories
                    </h2>
                    <span className="font-mono text-[10px] uppercase tracking-wider text-textMuted">{memories.length} saved</span>
                </div>

                <form onSubmit={addMemory} className="mt-5 flex gap-3">
                    <input value={memoryText} onChange={(event) => setMemoryText(event.target.value)} maxLength={500} placeholder="Save a fact worth remembering..." className="min-w-0 flex-1 rounded-lg border border-white/10 bg-surfaceHighlight px-3 py-2.5 text-sm text-white placeholder:text-neutral-600 focus:border-orange-400/40 focus:outline-none" />
                    <button type="submit" disabled={!memoryText.trim()} className="flex items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 text-xs font-bold uppercase tracking-wider text-black transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-40">
                        <Plus size={14} /> Save
                    </button>
                </form>

                <div className="mt-4 divide-y divide-white/5">
                    {memories.map((item) => (
                        <div key={item.id} className="flex items-start gap-3 py-3">
                            <p className="min-w-0 flex-1 text-sm text-text">{item.content}</p>
                            <button type="button" onClick={() => removeMemory(item.id)} className="text-textMuted transition-colors hover:text-red-300 p-2" aria-label={`Delete memory ${item.content}`}>
                                <Trash2 size={15} />
                            </button>
                        </div>
                    ))}
                    {!memories.length && <p className="py-3 text-sm italic text-textMuted">No memories yet. Anything you save here stays on this machine.</p>}
                </div>
            </section>
        </div>
    );
};

export default RemindersMemory;
