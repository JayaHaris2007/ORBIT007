import React, { useState } from 'react';
import { useOrbitAgent } from '../useOrbitAgent';
import { Mic, MicOff, Send, Volume2, CheckCircle2, Circle, Trash2, Plus, Wifi, WifiOff, Loader2 } from 'lucide-react';

export default function OrbitAgentView() {
  const {
    messages,
    reminders,
    connected,
    isLoading,
    isRecording,
    sendMessage,
    startVoiceCapture,
    stopVoiceCapture,
    addReminder,
    toggleReminder,
    deleteReminder,
  } = useOrbitAgent();

  const [inputText, setInputText] = useState('');
  const [newTitle, setNewTitle] = useState('');
  const [newRemindAt, setNewRemindAt] = useState('');
  const [showReminderForm, setShowReminderForm] = useState(false);

  const handleSend = (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    sendMessage(inputText.trim());
    setInputText('');
  };

  const handleCreateReminder = (e) => {
    e.preventDefault();
    if (!newTitle.trim() || !newRemindAt.trim()) return;
    addReminder(newTitle.trim(), newRemindAt.trim());
    setNewTitle('');
    setNewRemindAt('');
    setShowReminderForm(false);
  };

  return (
    <div className="flex flex-col h-full bg-[#0d0f12] text-white">
      {/* Top Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/10 bg-white/[0.02]">
        <div className="flex items-center gap-2">
          {connected ? (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 font-mono">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <Wifi size={14} /> LIVE WS
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-xs text-amber-400 font-mono">
              <WifiOff size={14} /> RECONNECTING
            </span>
          )}
        </div>
        <div className="text-[11px] font-mono text-neutral-400">
          ORBIT007 Agent
        </div>
      </div>

      {/* Main Content Area: Split Reminders & Chat */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Active Reminders Accordion / Strip */}
        <div className="border-b border-white/10 bg-white/[0.01] px-4 py-2">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-bold uppercase tracking-wider text-orange-400">
              Live Reminders ({reminders.length})
            </span>
            <button
              type="button"
              onClick={() => setShowReminderForm(!showReminderForm)}
              className="text-xs text-neutral-300 hover:text-white flex items-center gap-1 bg-white/5 hover:bg-white/10 px-2 py-0.5 rounded border border-white/10"
            >
              <Plus size={12} /> {showReminderForm ? 'Cancel' : 'Add'}
            </button>
          </div>

          {showReminderForm && (
            <form onSubmit={handleCreateReminder} className="my-2 p-2 rounded bg-black/40 border border-orange-500/30 flex flex-col gap-2">
              <input
                type="text"
                placeholder="Reminder title..."
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                className="w-full bg-neutral-900 border border-white/10 rounded px-2 py-1 text-xs text-white outline-none focus:border-orange-400"
              />
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="When (e.g. 5:00 PM or tomorrow)"
                  value={newRemindAt}
                  onChange={(e) => setNewRemindAt(e.target.value)}
                  className="flex-1 bg-neutral-900 border border-white/10 rounded px-2 py-1 text-xs text-white outline-none focus:border-orange-400"
                />
                <button
                  type="submit"
                  disabled={!newTitle.trim() || !newRemindAt.trim()}
                  className="bg-orange-500 hover:bg-orange-600 disabled:opacity-40 text-black font-bold px-3 py-1 rounded text-xs"
                >
                  Save
                </button>
              </div>
            </form>
          )}

          <div className="max-h-28 overflow-y-auto space-y-1 pr-1">
            {reminders.length === 0 ? (
              <p className="text-[11px] text-neutral-500 italic py-1">No active reminders in database.</p>
            ) : (
              reminders.map((r) => (
                <div
                  key={r.id}
                  className={`flex items-center justify-between gap-2 p-1.5 rounded text-xs transition-colors ${
                    r.completed ? 'bg-white/[0.02] text-neutral-500 line-through' : 'bg-white/[0.04] text-neutral-200'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => toggleReminder(r.id, !r.completed)}
                    className="flex items-center gap-1.5 min-w-0 text-left flex-1"
                  >
                    {r.completed ? (
                      <CheckCircle2 size={13} className="text-emerald-500 shrink-0" />
                    ) : (
                      <Circle size={13} className="text-orange-400 shrink-0" />
                    )}
                    <span className="truncate">{r.title || r.text}</span>
                    <span className="text-[10px] text-neutral-500 shrink-0">
                      ({r.remind_at || r.run_at})
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteReminder(r.id)}
                    className="text-neutral-500 hover:text-red-400 p-0.5"
                    title="Delete reminder"
                  >
                    <Trash2 size={12} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Message History */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center text-neutral-500">
              <p className="text-xs">No messages yet in this session.</p>
              <p className="text-[11px] mt-1 text-neutral-600">
                Say &ldquo;remind me to check server at 6pm&rdquo; or start speaking.
              </p>
            </div>
          ) : (
            messages.map((m, idx) => {
              const isUser = m.role === 'user';
              return (
                <div
                  key={idx}
                  className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
                >
                  <div
                    className={`max-w-[85%] rounded-xl px-3 py-2 text-xs leading-relaxed ${
                      isUser
                        ? 'bg-orange-500/20 text-orange-100 border border-orange-500/30'
                        : 'bg-white/[0.06] text-neutral-200 border border-white/10'
                    }`}
                  >
                    <div className="text-[9px] font-mono opacity-50 mb-0.5 uppercase tracking-wide">
                      {m.role}
                    </div>
                    <div>{m.content}</div>
                  </div>
                </div>
              );
            })
          )}
          {isLoading && (
            <div className="flex items-center gap-2 text-xs text-orange-400 animate-pulse">
              <Loader2 size={13} className="animate-spin" />
              <span>ORBIT007 is thinking...</span>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Bar: AudioWorklet Mic + Input */}
      <div className="p-3 border-t border-white/10 bg-black/40">
        <form onSubmit={handleSend} className="flex items-center gap-2">
          {/* AudioWorklet Microphone Capture Button */}
          <button
            type="button"
            onClick={isRecording ? stopVoiceCapture : startVoiceCapture}
            className={`p-2.5 rounded-xl border transition-all ${
              isRecording
                ? 'bg-red-500/20 border-red-500 text-red-400 animate-pulse ring-2 ring-red-500/50'
                : 'bg-white/5 border-white/10 text-neutral-400 hover:text-white hover:bg-white/10'
            }`}
            title={isRecording ? 'Stop voice capture' : 'Start voice capture (AudioWorklet PCM)'}
          >
            {isRecording ? <MicOff size={16} /> : <Mic size={16} />}
          </button>

          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={isRecording ? 'Streaming audio...' : 'Type a message or reminder command...'}
            disabled={isLoading}
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder:text-neutral-500 outline-none focus:border-orange-500/50"
          />

          <button
            type="submit"
            disabled={isLoading || !inputText.trim()}
            className="p-2.5 rounded-xl bg-orange-500 text-black font-bold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-orange-400 transition-colors"
            title="Send"
          >
            <Send size={15} />
          </button>
        </form>
      </div>
    </div>
  );
}

