import React from 'react';
import { Square } from 'lucide-react';

export default function StopButton({ state = 'idle', onStop }) {
    if (state === 'idle') return null;

    return (
        <button
            type="button"
            onClick={onStop}
            aria-label="Stop current voice turn"
            title="Stop voice turn (discard transient audio)"
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-red-500/40 bg-red-500/10 text-red-400 shadow-[0_0_12px_rgba(239,68,68,0.2)] transition-all hover:border-red-400 hover:bg-red-500/20 hover:text-white"
        >
            <Square size={14} className="fill-current" />
        </button>
    );
}

