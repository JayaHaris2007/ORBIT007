import { API_BASE_URL, apiRequest } from './api';

export class AudioQueue {
    constructor() {
        this.ctx = null;
        this.queue = [];
        this.playing = null;
        this.epoch = 0;
        this.onIdle = null;
        this.onPlay = null;
    }

    async push(arrayBuffer, epoch) {
        if (epoch !== this.epoch) return; // Stale audio from a stopped turn
        this.ctx ||= new (window.AudioContext || window.webkitAudioContext)();
        if (this.ctx.state === 'suspended') {
            await this.ctx.resume().catch(() => {});
        }
        try {
            const buf = await this.ctx.decodeAudioData(arrayBuffer.slice(0));
            if (epoch !== this.epoch) return;
            this.queue.push(buf);
            if (!this.playing) this._next();
        } catch (err) {
            console.error('Audio decode error:', err);
        }
    }

    _next() {
        const buf = this.queue.shift();
        if (!buf) {
            this.playing = null;
            this.onIdle?.();
            return;
        }
        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.ctx.destination);
        src.onended = () => this._next();
        this.playing = src;
        this.onPlay?.();
        src.start();
    }

    setCallbacks(onPlay, onIdle) {
        this.onPlay = onPlay;
        this.onIdle = onIdle;
    }

    stop() {
        this.epoch++;
        this.queue = [];
        try {
            this.playing?.stop();
        } catch { /* ignore */ }
        this.playing = null;
        this.onIdle?.();
    }
}

let _sharedAudioQueue = null;
export function getSharedAudioQueue() {
    if (typeof window === 'undefined') return null;
    if (!_sharedAudioQueue) {
        _sharedAudioQueue = new AudioQueue();
    }
    return _sharedAudioQueue;
}

export async function fetchVoiceSettings() {
    return apiRequest('/api/voice/settings');
}

export async function updateVoiceSettings(payload) {
    return apiRequest('/api/voice/settings', {
        method: 'PUT',
        body: JSON.stringify(payload),
    });
}

export async function fetchTTSAudio(text, signal = null) {
    const token = (import.meta.env.VITE_ORBIT_TOKEN || '').trim();
    const headers = { 'Content-Type': 'application/json' };
    if (token) headers['x-orbit-token'] = token;

    const res = await fetch(`${API_BASE_URL}/api/voice/tts`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ text }),
        signal,
    });
    if (!res.ok) {
        const errorText = await res.text().catch(() => 'Unknown error');
        throw new Error(`TTS failed (${res.status}): ${errorText}`);
    }
    return res.arrayBuffer();
}

export async function stopVoiceTurn(sessionId = 'default') {
    return apiRequest('/api/voice/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId }),
    });
}

export function splitSentencesClient(text, maxChars = 200) {
    if (!text || !text.trim()) return [];
    const normalized = text.replace(/\s+/g, ' ').trim();
    const rawChunks = normalized.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [normalized];
    const results = [];
    for (const chunk of rawChunks) {
        const trimmed = chunk.trim();
        if (!trimmed) continue;
        if (trimmed.length <= maxChars) {
            results.push(trimmed);
        } else {
            const words = trimmed.split(' ');
            let current = '';
            for (const word of words) {
                if ((current + ' ' + word).trim().length <= maxChars) {
                    current = (current + ' ' + word).trim();
                } else {
                    if (current) results.push(current);
                    current = word;
                }
            }
            if (current) results.push(current);
        }
    }
    return results;
}
