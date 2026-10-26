import { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { format } from 'date-fns';
import { Bot, Check, History, MessageSquarePlus, PanelRightClose, Send, Trash2, X } from 'lucide-react';
import { apiRequest } from '../services/api';
import { useAppData } from '../context/AppDataContext';
import { cleanAssistantText } from '../utils/textUtils';

const ACTIVE_CHAT_KEY = 'spidey_active_chat_id';
const MINIMIZED_KEY = 'spidey_panel_minimized';
// Older builds stored the panel state under the previous name.
const LEGACY_ACTIVE_CHAT_KEY = 'jarvis_active_chat_id';
const LEGACY_MINIMIZED_KEY = 'jarvis_panel_minimized';

const readStored = (key, legacyKey) => localStorage.getItem(key) ?? localStorage.getItem(legacyKey);

// One-tap openers for an empty conversation.
const STARTER_PROMPTS = [
    'What do you know about me?',
    "What's pending today?",
    'Check my habits'
];

const formatWhen = (value) => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '' : format(date, 'MMM d, h:mm a');
};

const AssistantSidebar = () => {
    const { revision, refresh } = useAppData();
    const navigate = useNavigate();
    const location = useLocation();
    // The panel is docked on wide screens; the toggle only matters below xl.
    const [open, setOpen] = useState(false);
    const [online, setOnline] = useState(false);
    const [chats, setChats] = useState([]);
    const [historyOpen, setHistoryOpen] = useState(false);
    const [confirmingId, setConfirmingId] = useState('');
    const [minimized, setMinimized] = useState(() => readStored(MINIMIZED_KEY, LEGACY_MINIMIZED_KEY) === '1');
    const [chatId, setChatId] = useState('');
    const [messages, setMessages] = useState([]);
    const [prompt, setPrompt] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const lastNotificationId = useRef(0);
    const conversationEndRef = useRef(null);
    const activeChatRef = useRef('');
    const requestSeq = useRef(0);

    // Map a page key from the backend to a real app route; unknown pages do nothing.
    const goToPage = useCallback((page) => {
        if (!page) return;
        if (page === 'chat') {
            setOpen(true);
            return;
        }
        const routes = {
            dashboard: '/dashboard',
            habits: '/habits',
            tasks: '/todos',
            reminders: '/reminders',
            memories: '/reminders',
            settings: '/settings',
        };
        const route = routes[page];
        if (route && location.pathname !== route) navigate(route);
    }, [navigate, location.pathname]);

    // Load one conversation and make it the active one.
    const openChat = useCallback(async (id) => {
        const seq = requestSeq.current + 1;
        requestSeq.current = seq;
        activeChatRef.current = id;
        localStorage.setItem(ACTIVE_CHAT_KEY, id);
        setChatId(id);
        setMessages([]);
        setError('');
        lastNotificationId.current = 0;
        try {
            const [history, notificationData] = await Promise.all([
                apiRequest(`/api/history/${id}`),
                apiRequest(`/api/notifications/${id}?after_id=0`)
            ]);
            if (requestSeq.current !== seq) return;
            setMessages(history.messages || []);
            const notifications = notificationData.notifications || [];
            lastNotificationId.current = notifications.length
                ? notifications[notifications.length - 1].id
                : 0;
        } catch (requestError) {
            if (requestSeq.current === seq) setError(requestError.message);
        }
    }, []);

    useEffect(() => {
        let cancelled = false;
        const loadWorkspace = async () => {
            try {
                const [health, chatData] = await Promise.all([
                    apiRequest('/api/health'),
                    apiRequest('/api/chats')
                ]);
                if (cancelled) return;
                setOnline(health.status === 'ok');

                const savedChatId = readStored(ACTIVE_CHAT_KEY, LEGACY_ACTIVE_CHAT_KEY);
                const savedChats = chatData.chats || [];
                const chat = savedChats.find((item) => item.id === savedChatId) || savedChats[0];
                if (!chat) {
                    const created = await apiRequest('/api/chats', { method: 'POST' });
                    if (cancelled) return;
                    setChats([created]);
                    openChat(created.id);
                    return;
                }
                setChats(savedChats);
                openChat(chat.id);
            } catch (requestError) {
                if (!cancelled) {
                    setOnline(false);
                    setError(requestError.message);
                }
            }
        };
        loadWorkspace();
        return () => { cancelled = true; };
    }, [openChat]);

    useEffect(() => {
        conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    // The collapsed state lives on <html> so the page layout can drop the reserved width.
    useEffect(() => {
        document.documentElement.classList.toggle('spidey-minimized', minimized);
        localStorage.setItem(MINIMIZED_KEY, minimized ? '1' : '0');
    }, [minimized]);

    useEffect(() => {
        if (!chatId) return undefined;
        const interval = window.setInterval(async () => {
            try {
                const data = await apiRequest(`/api/notifications/${chatId}?after_id=${lastNotificationId.current}`);
                if (activeChatRef.current !== chatId) return;
                if (!data.notifications?.length) return;
                lastNotificationId.current = data.notifications[data.notifications.length - 1].id;
                setMessages((current) => [...current, ...data.notifications.map((item) => ({
                    role: 'assistant', content: item.content, notification: true
                }))]);
                // A fired reminder may have changed reminders or tasks on the server.
                refresh();
            } catch {
                // Keep the shared panel quiet during a temporary polling failure.
            }
        }, 5000);
        return () => window.clearInterval(interval);
    }, [chatId, refresh]);

    // Pick up chats created or renamed elsewhere (another tab, Discord bot).
    useEffect(() => {
        if (!revision) return;
        apiRequest('/api/chats')
            .then((data) => setChats(data.chats || []))
            .catch(() => { /* keep the current list */ });
    }, [revision]);

    const refreshChats = async () => {
        try {
            const data = await apiRequest('/api/chats');
            setChats(data.chats || []);
        } catch {
            // The conversation still works; the list catches up on the next poll.
        }
    };

    const startChat = async () => {
        try {
            const chat = await apiRequest('/api/chats', { method: 'POST' });
            setChats((current) => [chat, ...current]);
            setHistoryOpen(false);
            setPrompt('');
            await openChat(chat.id);
        } catch (requestError) {
            setError(requestError.message);
        }
    };

    const selectChat = (id) => {
        setHistoryOpen(false);
        setConfirmingId('');
        if (id !== chatId) openChat(id);
    };

    const deleteChat = async (id) => {
        try {
            await apiRequest(`/api/chats/${id}`, { method: 'DELETE' });
            setConfirmingId('');
            const remaining = chats.filter((chat) => chat.id !== id);
            setChats(remaining);
            if (id !== chatId) return;
            if (remaining.length) {
                await openChat(remaining[0].id);
                return;
            }
            const created = await apiRequest('/api/chats', { method: 'POST' });
            setChats([created]);
            await openChat(created.id);
        } catch (requestError) {
            setError(requestError.message);
        }
    };

    // Shared by the composer and the starter prompts in the empty state.
    const submitPrompt = async (rawValue) => {
        const value = rawValue.trim();
        if (!value || loading || !chatId) return;
        // Show the message immediately instead of waiting for the reply.
        const optimisticId = `pending-${Date.now()}`;
        setMessages((current) => [...current, {
            id: optimisticId,
            role: 'user',
            content: value,
            pending: true
        }]);
        setLoading(true);
        setPrompt('');
        setError('');
        try {
            const data = await apiRequest('/api/generate', {
                method: 'POST',
                body: JSON.stringify({ prompt: value, chat_id: chatId })
            });
            // Ignore a late reply if another conversation was opened meanwhile.
            if (activeChatRef.current !== chatId) return;
            setMessages(data.messages || []);
            if (data.navigate) goToPage(data.navigate);
            refreshChats();
            // The assistant may have created or completed tasks, habits, reminders or memories.
            refresh();
        } catch (requestError) {
            if (activeChatRef.current !== chatId) return;
            setError(requestError.message);
            setMessages((current) => current.map((item) => (
                item.id === optimisticId ? { ...item, pending: false, failed: true } : item
            )));
        } finally {
            setLoading(false);
        }
    };

    const sendMessage = (event) => {
        event.preventDefault();
        submitPrompt(prompt);
    };

    // Enter sends, Shift+Enter writes a new line.
    const handleKeyDown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
        }
    };

    return (
        <>
            <button
                type="button"
                onClick={() => {
                    // Below xl the panel is an overlay, so the pill only opens/closes it.
                    if (minimized && window.matchMedia('(min-width: 1280px)').matches) {
                        setMinimized(false);
                        return;
                    }
                    setOpen((current) => !current);
                }}
                className="workspace-toggle fixed bottom-20 right-4 z-40 flex items-center gap-2 rounded-full border border-orange-400/30 bg-[#14100e] px-4 py-3 text-sm font-semibold text-white shadow-xl xl:hidden"
                aria-expanded={open}
                aria-label={minimized ? 'Restore S.P.I.D.E.Y panel' : open ? 'Close S.P.I.D.E.Y panel' : 'Open S.P.I.D.E.Y panel'}
            >
                <Bot size={18} className="text-orange-400" /> S.P.I.D.E.Y
            </button>
            {open && <button type="button" className="workspace-backdrop fixed inset-0 z-40 bg-black/60 xl:hidden" aria-label="Close S.P.I.D.E.Y panel" onClick={() => setOpen(false)} />}

            <aside className={`workspace-sidebar fixed right-0 top-0 z-50 flex h-[calc(100dvh-4rem)] w-full max-w-[390px] flex-col overflow-hidden border-l border-orange-400/15 bg-[#0d0d0d] shadow-2xl transition-transform duration-300 md:h-screen ${open ? 'workspace-sidebar-open' : ''}`}>
                <header className="flex shrink-0 items-center gap-3 border-b border-white/10 px-5 py-4">
                    <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl border border-orange-400/20 bg-orange-500/10 text-orange-300"><Bot size={20} /></div>
                    <div className="min-w-0 flex-1">
                        <h2 className="truncate font-bold text-white">S.P.I.D.E.Y</h2>
                        <p className="mt-0.5 flex items-center gap-1.5 text-[10px] uppercase tracking-[0.16em] text-textMuted">
                            <span className={`h-1.5 w-1.5 rounded-full ${online ? 'bg-orange-400' : 'bg-neutral-600'}`} />
                            {online ? 'Ready when you are' : 'Connecting'}
                        </p>
                    </div>
                    <button type="button" onClick={() => setHistoryOpen((current) => !current)} className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg border !p-0 transition-colors ${historyOpen ? 'border-orange-400/40 bg-orange-500/10 text-orange-300' : 'border-white/10 bg-white/5 text-textMuted hover:border-orange-400/30 hover:!bg-white/10 hover:text-orange-300'}`} title="Chat history" aria-label="Toggle chat history" aria-expanded={historyOpen}><History size={16} /></button>
                    <button type="button" onClick={startChat} className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-white/10 bg-white/5 !p-0 text-textMuted transition-colors hover:border-orange-400/30 hover:!bg-white/10 hover:text-orange-300" title="Start a new chat" aria-label="Start a new chat"><MessageSquarePlus size={16} /></button>
                    <button type="button" onClick={() => setMinimized(true)} className="hidden h-9 w-9 shrink-0 place-items-center rounded-lg border border-white/10 bg-white/5 !p-0 text-textMuted transition-colors hover:border-orange-400/30 hover:!bg-white/10 hover:text-orange-300 xl:grid" title="Minimize panel" aria-label="Minimize S.P.I.D.E.Y panel"><PanelRightClose size={16} /></button>
                    <button type="button" onClick={() => setOpen(false)} className="workspace-close grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-white/10 bg-white/5 !p-0 text-textMuted transition-colors hover:!bg-white/10 hover:text-white xl:hidden" aria-label="Close S.P.I.D.E.Y panel"><X size={16} /></button>
                </header>

                <div className="flex shrink-0 items-center justify-between px-5 pb-2 pt-4">
                    <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-textMuted">Conversation</span>
                    <span className="text-[10px] font-medium uppercase tracking-[0.12em] text-neutral-600">
                        {chats.length ? `${chats.length} saved` : 'No history'}
                    </span>
                </div>

                {historyOpen && (
                    <div className="mx-4 mb-2 shrink-0">
                        <div className="max-h-64 space-y-1 overflow-y-auto rounded-xl border border-white/10 bg-white/[0.03] p-1.5">
                            {chats.map((chat) => {
                                const active = chat.id === chatId;
                                const confirming = confirmingId === chat.id;
                                return (
                                    <div key={chat.id} className={`group flex items-center gap-1 rounded-lg transition-colors ${active ? 'bg-orange-500/15' : 'hover:bg-white/5'}`}>
                                        <button
                                            type="button"
                                            onClick={() => selectChat(chat.id)}
                                            className={`min-w-0 flex-1 rounded-lg !border-0 !bg-transparent px-2.5 py-2 text-left ${active ? 'text-orange-100' : 'text-neutral-300'}`}
                                        >
                                            <span className="flex items-center gap-1.5">
                                                <span className="block truncate text-xs font-medium">{chat.title || 'New conversation'}</span>
                                                {active && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-orange-400" aria-label="Active conversation" />}
                                            </span>
                                            <span className="mt-0.5 block text-[10px] text-neutral-500">{formatWhen(chat.updated_at)}</span>
                                        </button>
                                        {confirming ? (
                                            <div className="flex shrink-0 items-center gap-1 pr-1">
                                                <button type="button" onClick={() => deleteChat(chat.id)} className="grid h-7 w-7 place-items-center rounded-md border border-red-500/30 bg-red-500/15 !p-0 text-red-300 hover:bg-red-500/25" title="Delete this chat" aria-label="Confirm delete chat"><Check size={13} /></button>
                                                <button type="button" onClick={() => setConfirmingId('')} className="grid h-7 w-7 place-items-center rounded-md border border-white/10 bg-white/5 !p-0 text-neutral-400 hover:text-white" title="Cancel" aria-label="Cancel delete"><X size={13} /></button>
                                            </div>
                                        ) : (
                                            <button type="button" onClick={() => setConfirmingId(chat.id)} className="mr-1 grid h-7 w-7 shrink-0 place-items-center rounded-md !border-0 !bg-transparent !p-0 text-neutral-500 transition-opacity hover:text-red-300 focus-visible:opacity-100 xl:opacity-0 xl:group-hover:opacity-100" title="Delete chat" aria-label={`Delete ${chat.title || 'conversation'}`}><Trash2 size={13} /></button>
                                        )}
                                    </div>
                                );
                            })}
                            {!chats.length && <p className="px-2 py-3 text-center text-[11px] text-neutral-500">No conversations yet.</p>}
                        </div>
                    </div>
                )}

                <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 pb-4">
                    {!messages.length && !loading && (
                        <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
                            <div className="grid h-12 w-12 place-items-center rounded-2xl border border-orange-400/20 bg-orange-500/10 text-orange-300"><Bot size={22} /></div>
                            <p className="mt-1 text-sm font-semibold text-white">S.P.I.D.E.Y is standing by</p>
                            <p className="text-xs leading-relaxed text-textMuted">Ask about your day, your tasks — or what I remember about you.</p>
                            <div className="mt-2 flex flex-wrap justify-center gap-1.5">
                                {STARTER_PROMPTS.map((starter) => (
                                    <button
                                        key={starter}
                                        type="button"
                                        onClick={() => submitPrompt(starter)}
                                        disabled={loading || !chatId}
                                        className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[11px] font-medium text-neutral-300 transition-colors hover:border-orange-400/30 hover:bg-orange-500/10 hover:text-orange-200 disabled:opacity-40"
                                    >
                                        {starter}
                                    </button>
                                ))}
                            </div>
                            {chats.length > 1 && (
                                <button type="button" onClick={() => setHistoryOpen(true)} className="mt-2 flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-textMuted !p-0 hover:text-orange-300">
                                    <History size={12} /> Open a previous chat
                                </button>
                            )}
                        </div>
                    )}

                    {messages.slice(-30).map((item, index) => {
                        const isUser = item.role === 'user';
                        const surface = item.failed
                            ? 'border border-red-500/40 bg-red-500/10 text-red-200'
                            : isUser
                                ? 'bg-orange-500/15 text-orange-50'
                                : 'border border-white/5 bg-white/5 text-neutral-200';
                        return (
                            <div
                                key={`${item.role}-${item.id || index}`}
                                className={`w-fit max-w-[85%] rounded-2xl px-3.5 py-2.5 text-xs leading-relaxed ${isUser ? 'ml-auto rounded-br-md' : 'mr-auto rounded-bl-md'} ${surface} ${item.pending ? 'opacity-70' : ''}`}
                            >
                                {item.role === 'assistant'
                                    ? <div className="assistant-message"><ReactMarkdown>{cleanAssistantText(item.content)}</ReactMarkdown></div>
                                    : item.content}
                                {item.failed && <p className="mt-1 text-[10px] text-red-300">Not sent — try again</p>}
                            </div>
                        );
                    })}

                    {loading && (
                        <div className="mr-auto flex w-fit items-center gap-1.5 rounded-2xl rounded-bl-md border border-white/5 bg-white/5 px-3.5 py-2.5 text-xs text-orange-300">
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-400" />
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-400 [animation-delay:150ms]" />
                            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-orange-400 [animation-delay:300ms]" />
                        </div>
                    )}

                    <div ref={conversationEndRef} />
                </div>

                <form onSubmit={sendMessage} className="shrink-0 border-t border-white/10 bg-black/30 px-4 py-3">
                    <div className="flex items-end gap-2 rounded-2xl border border-white/10 bg-white/5 p-2 transition-colors focus-within:border-orange-400/40">
                        <textarea
                            value={prompt}
                            onChange={(event) => setPrompt(event.target.value)}
                            onKeyDown={handleKeyDown}
                            rows={1}
                            maxLength={2000}
                            placeholder="Ask S.P.I.D.E.Y…"
                            className="max-h-28 min-h-[38px] flex-1 resize-none bg-transparent px-2 py-2 text-xs leading-relaxed text-white outline-none placeholder:text-neutral-500"
                            disabled={!chatId}
                        />
                        <button
                            type="submit"
                            disabled={loading || !prompt.trim() || !chatId}
                            className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-white text-black shadow-[0_0_15px_rgba(255,106,26,0.25)] disabled:shadow-none disabled:opacity-40"
                            aria-label="Send message"
                        >
                            <Send size={15} />
                        </button>
                    </div>
                    <p className="mt-2 px-1 text-[10px] text-neutral-600">Enter to send · Shift + Enter for a new line</p>
                    {error && <p className="mt-2 px-1 text-xs text-red-300">{error}</p>}
                </form>
            </aside>
        </>
    );
};

export default AssistantSidebar;
