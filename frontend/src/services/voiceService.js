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

    stop() {
        this.epoch++;
        this.queue = [];
        try {
            this.playing?.stop();
        } catch {}
        this.playing = null;
        this.onIdle?.();
    }
}

export const fetchVoiceSettings = async () => {
    return await apiRequest('/api/voice/settings');
};

export const updateVoiceSettings = async (settings) => {
    return await apiRequest('/api/voice/settings', {
        method: 'PUT',
        body: JSON.stringify(settings)
    });
};

export const stopVoiceTurn = async (sessionId = 'default') => {
    return await apiRequest('/api/voice/stop', {
        method: 'POST',
        body: JSON.stringify({ session_id: sessionId })
    });
};

export const fetchTTSAudio = async (text, signal) => {
    const res = await fetch(`${API_BASE_URL}/api/voice/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `TTS request failed (${res.status})`);
    }
    return await res.arrayBuffer();
};

export const splitSentencesClient = (text, minChars = 25, maxChars = 220) => {
    if (!text || !text.trim()) return [];
    // Split on terminal punctuation followed by space, or newlines
    const rawParts = text.split(/(?<=[.!?…])\s+|\n+/).map((p) => p.trim()).filter(Boolean);
    const out = [];
    let buf = '';
    for (const p of rawParts) {
        buf = `${buf} ${p}`.trim();
        if (buf.length >= minChars) {
            out.push(buf);
            buf = '';
        }
    }
    if (buf) {
        if (out.length > 0) out[out.length - 1] += ' ' + buf;
        else out.push(buf);
    }
    const final = [];
    for (let s of out) {
        while (s.length > maxChars) {
            let cut = s.lastIndexOf(',', maxChars);
            if (cut < 40) cut = s.lastIndexOf(' ', maxChars);
            if (cut <= 0) cut = maxChars;
            final.push(s.slice(0, cut + 1).trim());
            s = s.slice(cut + 1).trim();
        }
        if (s) final.push(s);
    }
    return final.filter(Boolean);
};

