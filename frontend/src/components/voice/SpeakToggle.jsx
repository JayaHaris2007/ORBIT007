import React from 'react';
import { Volume2, VolumeX, AlertCircle } from 'lucide-react';

export default function SpeakToggle({ speakEnabled, engineReady, onToggle, disabled = false, error = null }) {
    const isUnavailable = !engineReady && Boolean(error);

    return (
        <div className="relative inline-flex items-center">
            <button
                type="button"
                onClick={onToggle}
                disabled={disabled || isUnavailable}
                aria-pressed={speakEnabled}
                aria-label={
                    isUnavailable
                        ? 'Voice unavailable (TTS engine offline)'
                        : speakEnabled
                        ? 'Mute speech replies'
                        : 'Enable speech replies'
                }
                title={
                    isUnavailable
                        ? 'Voice unavailable'
                        : speakEnabled
                        ? 'Speak replies: ON'
                        : 'Speak replies: OFF'
                }
                className={`group relative flex h-7 items-center gap-1.5 rounded-lg border px-2 py-1 text-[11px] font-medium transition-all ${
                    isUnavailable
                        ? 'border-white/5 bg-white/[0.02] text-neutral-500 cursor-not-allowed opacity-60'
                        : speakEnabled
                        ? 'border-orange-500/40 bg-orange-500/10 text-orange-300 shadow-[0_0_12px_rgba(255,106,26,0.15)] hover:border-orange-400 hover:bg-orange-500/20'
                        : 'border-white/10 bg-white/5 text-neutral-400 hover:border-white/20 hover:text-white'
                }`}
            >
                {isUnavailable ? (
                    <>
                        <VolumeX size={13} className="text-neutral-500" />
                        <span className="hidden sm:inline text-[10px] text-neutral-500">Voice unavailable</span>
                    </>
                ) : speakEnabled ? (
                    <>
                        <Volume2 size={13} className="text-orange-400 animate-pulse" />
                        <span className="hidden sm:inline text-[10px] tracking-wide">Speak ON</span>
                    </>
                ) : (
                    <>
                        <VolumeX size={13} className="text-neutral-400 group-hover:text-white" />
                        <span className="hidden sm:inline text-[10px] text-neutral-400 group-hover:text-white">Speak OFF</span>
                    </>
                )}
            </button>
        </div>
    );
}

