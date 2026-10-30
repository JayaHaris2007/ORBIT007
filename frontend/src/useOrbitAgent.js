import { useState, useEffect, useRef, useCallback } from 'react';

const ORBIT_TOKEN = (import.meta.env.VITE_ORBIT_TOKEN || '').trim();
const getAuthHeaders = (extra = {}) => ({
  ...extra,
  ...(ORBIT_TOKEN ? { 'x-orbit-token': ORBIT_TOKEN } : {}),
});

export function useOrbitAgent() {
  const [messages, setMessages] = useState([]);
  const [reminders, setReminders] = useState([]);
  const [connected, setConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const wsRef = useRef(null);
  const audioRef = useRef(null);
  const audioCtxRef = useRef(null);
  const mediaStreamRef = useRef(null);

  const isSpeakEnabled = () => {
    try {
      return localStorage.getItem('spidey_speak_enabled') === '1';
    } catch {
      return false;
    }
  };

  const playAudio = useCallback((audioUrl) => {
    if (!audioUrl) return;
    if (!isSpeakEnabled()) return;

    // Discard any currently active speech globally
    window.dispatchEvent(new CustomEvent('spidey:audio_stop'));

    if (!audioRef.current) {
      audioRef.current = new Audio();
    }
    audioRef.current.pause();
    audioRef.current.currentTime = 0;
    audioRef.current.src = audioUrl;
    audioRef.current.play().catch((err) => {
      console.warn('Audio playback blocked or failed:', err);
    });
  }, []);

  // Listen for global stop requests
  useEffect(() => {
    const onStop = () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    };
    window.addEventListener('spidey:audio_stop', onStop);
    return () => window.removeEventListener('spidey:audio_stop', onStop);
  }, []);

  // Initial DB hydration via REST
  const fetchInitialData = useCallback(async () => {
    try {
      const [remRes, histRes] = await Promise.all([
        fetch('/api/reminders', { headers: getAuthHeaders() }),
        fetch('/api/history', { headers: getAuthHeaders() }),
      ]);
      if (remRes.ok) {
        const remData = await remRes.json();
        setReminders(remData.reminders || []);
      }
      if (histRes.ok) {
        const histData = await histRes.json();
        setMessages(histData.messages || []);
      }
    } catch (err) {
      console.error('Failed to fetch initial DB state:', err);
    }
  }, []);

  // WebSocket connection with auto-reconnect (leak-free)
  useEffect(() => {
    fetchInitialData();

    let reconnectTimer = null;
    let isTornDown = false;

    const connectWs = () => {
      if (isTornDown) return;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const tokenQuery = ORBIT_TOKEN ? `?token=${encodeURIComponent(ORBIT_TOKEN)}` : '';
      const wsUrl = `${protocol}//${window.location.host}/ws${tokenQuery}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      let pingTimer = null;
      ws.onopen = () => {
        if (isTornDown) {
          ws.close();
          return;
        }
        setConnected(true);
        pingTimer = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, 30000);
      };

      ws.onmessage = (event) => {
        if (isTornDown) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'connected') {
            if (data.reminders) setReminders(data.reminders);
            if (data.history) setMessages(data.history);
          } else if (data.type === 'status') {
            if (data.state === 'thinking') setIsLoading(true);
          } else if (data.type === 'pong') {
            // Keep-alive pong
          } else if (data.type === 'transcript') {
            setMessages((prev) => [...prev, { role: 'user', content: data.text }]);
          } else if (data.type === 'reminder_alert') {
            if (data.text) {
              setMessages((prev) => [...prev, { role: 'assistant', content: `⏰ Reminder: ${data.text}` }]);
            }
            if (data.reminders) setReminders(data.reminders);
            if (data.audio_url && isSpeakEnabled()) playAudio(data.audio_url);
            window.dispatchEvent(new CustomEvent('orbit:reminder_alert', { detail: data }));
          } else if (data.type === 'agent_response') {
            setIsLoading(false);
            if (data.history) {
              setMessages(data.history);
            } else if (data.text) {
              setMessages((prev) => [...prev, { role: 'assistant', content: data.text }]);
            }
            if (data.reminders) setReminders(data.reminders);
            if (data.audio_url && isSpeakEnabled()) playAudio(data.audio_url);
          } else if (data.type === 'error') {
            setIsLoading(false);
            console.error('Agent error:', data.error);
          }
        } catch (e) {
          console.error('Invalid WS JSON:', e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (pingTimer) clearInterval(pingTimer);
        if (!isTornDown) {
          reconnectTimer = setTimeout(connectWs, 3000);
        }
      };
    };

    connectWs();
    return () => {
      isTornDown = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    };
  }, [fetchInitialData, playAudio]);

  // Send chat message (WS primary, REST fallback)
  const sendMessage = useCallback(async (text) => {
    if (!text.trim()) return;
    setIsLoading(true);
    setMessages((prev) => [...prev, { role: 'user', content: text }]);

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ message: text }));
      return;
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      if (data.history) setMessages(data.history);
      if (data.reminders) setReminders(data.reminders);
      if (data.audio_url) playAudio(data.audio_url);
    } catch (err) {
      console.error('Chat request failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, [playAudio]);

  // Start/Stop Microphone PCM Capture via public/audio-worklet/pcm-capture.js
  const startVoiceCapture = useCallback(async () => {
    if (isRecording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;

      await audioCtx.audioWorklet.addModule('/audio-worklet/pcm-capture.js');
      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture-processor');

      workletNode.port.onmessage = (event) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(event.data);
        }
      };

      source.connect(workletNode);
      workletNode.connect(audioCtx.destination);
      setIsRecording(true);
    } catch (err) {
      console.error('Microphone capture error:', err);
    }
  }, [isRecording]);

  const stopVoiceCapture = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close();
      audioCtxRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'end_audio' }));
      } catch {}
    }
    setIsRecording(false);
  }, []);

  // Reminder CRUD helpers
  const addReminder = useCallback(async (title, remind_at) => {
    const res = await fetch('/api/reminders', {
      method: 'POST',
      headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ title, remind_at }),
    });
    if (res.ok) {
      const data = await res.json();
      setReminders(data.reminders || []);
    }
  }, []);

  const toggleReminder = useCallback(async (id, completed) => {
    const res = await fetch(`/api/reminders/${id}`, {
      method: 'PATCH',
      headers: getAuthHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ completed }),
    });
    if (res.ok) {
      const data = await res.json();
      setReminders(data.reminders || []);
    }
  }, []);

  const deleteReminder = useCallback(async (id) => {
    const res = await fetch(`/api/reminders/${id}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    if (res.ok) {
      const data = await res.json();
      setReminders(data.reminders || []);
    }
  }, []);

  return {
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
  };
}

export default useOrbitAgent;

