import React from 'react';
import { Mic, MicOff, Loader2 } from 'lucide-react';

export default function ListenButton({
    state = 'idle',
    cloudStt = false,
    onClick,
    disabled = false
}) {
    const isListening = state === 'listening';
    const isBusy = state === 'transcribing' || state === 'thinking';

    return (
        <div className="relative inline-flex items-center">
            <button
                type="button"
                onClick={onClick}
                disabled={disabled || isBusy}
                aria-label={
                    isListening
                        ? 'Listening to speech...'
                        : isBusy
                        ? `${state}...`
                        : 'Start voice input (Listen)'
                }
                title={
                    isListening
                        ? 'Listening (click to stop)'
                        : isBusy
                        ? `${state.charAt(0).toUpperCase() + state.slice(1)}...`
                        : 'Listen (Voice Input)'
                }
                className={`relative flex h-9 w-9 items-center justify-center rounded-xl border transition-all ${
                    isListening
                        ? 'border-red-500/50 bg-red-500/20 text-red-300 shadow-[0_0_15px_rgba(239,68,68,0.3)] animate-pulse'
                        : isBusy
                        ? 'border-orange-500/40 bg-orange-500/10 text-orange-300'
                        : 'border-white/10 bg-white/5 text-neutral-300 hover:border-orange-400/40 hover:bg-orange-500/10 hover:text-white'
                } disabled:opacity-40`}
            >
                {isBusy ? (
                    <Loader2 size={16} className="animate-spin text-orange-400" />
                ) : isListening ? (
                    <Mic size={16} className="text-red-400" />
                ) : (
                    <Mic size={16} />
                )}

                {/* Cloud STT badge */}
                {cloudStt && (
                    <span
                        className="absolute -top-1 -right-1 flex h-3.5 items-center rounded-full bg-blue-500/80 px-1 text-[8px] font-bold text-white uppercase tracking-tighter"
                        title="Cloud STT active"
                    >
                        cloud
                    </span>
                )}
            </button>
        </div>
    );
}
