import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Headphones, X, Square, Clock, ChevronDown, ChevronUp } from 'lucide-react';

export default function LiveOverlay({
    isOpen,
    state = 'idle',
    transcript = '',
    reply = '',
    timing = {},
    error = null,
    onStopTurn,
    onClose
}) {
    const [showDebug, setShowDebug] = useState(false);

    if (!isOpen) return null;

    const stateColors = {
        listening: 'from-orange-500/30 to-red-500/20 border-orange-500/50 text-orange-300',
        transcribing: 'from-amber-500/30 to-orange-500/20 border-amber-500/50 text-amber-300',
        thinking: 'from-blue-500/30 to-indigo-500/20 border-blue-500/50 text-blue-300',
        speaking: 'from-emerald-500/30 to-teal-500/20 border-emerald-500/50 text-emerald-300',
        idle: 'from-neutral-700/30 to-neutral-800/20 border-neutral-600/40 text-neutral-400'
    };

    const orbScale = {
        listening: [1, 1.15, 1],
        transcribing: [1, 1.05, 1],
        thinking: [1, 1.25, 0.95, 1],
        speaking: [1, 1.2, 1.05, 1.15, 1],
        idle: [1]
    };

    const orbDuration = {
        listening: 1.8,
        transcribing: 1.2,
        thinking: 2.0,
        speaking: 1.0,
        idle: 0
    };

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-50 flex flex-col items-center justify-between bg-black/90 p-6 backdrop-blur-xl"
            >
                {/* Header */}
                <div className="flex w-full max-w-xl items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="flex h-2.5 w-2.5 rounded-full bg-orange-500 animate-ping" />
                        <span className="text-xs font-semibold uppercase tracking-wider text-neutral-300">
                            S.P.I.D.E.Y. Live
                        </span>
                    </div>

                    <div className="flex items-center gap-3">
                        <button
                            type="button"
                            onClick={() => setShowDebug(!showDebug)}
                            className="flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2 py-1 text-[11px] text-neutral-400 hover:text-white"
                            title="Toggle latency diagnostics"
                        >
                            <Clock size={12} />
                            <span>Stats</span>
                            {showDebug ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                        </button>

                        <button
                            type="button"
                            onClick={onClose}
                            className="grid h-8 w-8 place-items-center rounded-lg border border-white/10 bg-white/5 text-neutral-400 transition-colors hover:border-white/20 hover:text-white"
                            aria-label="End live session"
                        >
                            <X size={16} />
                        </button>
                    </div>
                </div>

                {/* Center Animated Orb */}
                <div className="flex flex-col items-center justify-center my-auto">
                    <div className="relative flex items-center justify-center">
                        {/* Outer Glow */}
                        <motion.div
                            animate={{
                                scale: orbScale[state] || [1],
                                opacity: [0.3, 0.6, 0.3]
                            }}
                            transition={{
                                repeat: Infinity,
                                duration: orbDuration[state] || 2,
                                ease: 'easeInOut'
                            }}
                            className={`absolute h-48 w-48 rounded-full bg-gradient-to-tr ${
                                state === 'speaking'
                                    ? 'from-emerald-500/40 to-teal-500/30'
                                    : state === 'thinking'
                                    ? 'from-blue-500/40 to-indigo-500/30'
                                    : 'from-orange-500/40 to-red-500/30'
                            } blur-3xl`}
                        />

                        {/* Interactive Orb */}
                        <motion.div
                            animate={{
                                scale: orbScale[state] || [1]
                            }}
                            transition={{
                                repeat: Infinity,
                                duration: orbDuration[state] || 2,
                                ease: 'easeInOut'
                            }}
                            className={`relative flex h-36 w-36 items-center justify-center rounded-full border-2 bg-gradient-to-b shadow-[0_0_40px_rgba(255,106,26,0.2)] ${
                                stateColors[state] || stateColors.idle
                            }`}
                        >
                            <span className="text-xs font-bold uppercase tracking-widest">
                                {state}
                            </span>
                        </motion.div>
                    </div>

                    {/* Speech Display */}
                    <div className="mt-8 max-w-md text-center">
                        {error && (
                            <p className="mb-3 rounded-lg border border-red-500/30 bg-red-950/40 px-3 py-1.5 text-xs font-medium text-red-300">
                                {error}
                            </p>
                        )}
                        {transcript && (
                            <p className="text-xs text-neutral-400 italic">
                                &ldquo;{transcript}&rdquo;
                            </p>
                        )}
                        {reply && (
                            <p className="mt-2 text-sm leading-relaxed text-white font-medium">
                                {reply}
                            </p>
                        )}
                    </div>
                </div>

                {/* Debug Panel (Off by default) */}
                {showDebug && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-4 w-full max-w-md rounded-xl border border-white/10 bg-neutral-900/90 p-3 text-[11px] font-mono text-neutral-300"
                    >
                        <div className="grid grid-cols-2 gap-2">
                            <div>Speech &rarr; Transcript: <span className="text-orange-400">{timing.transcript || 0} ms</span></div>
                            <div>Transcript &rarr; Reply: <span className="text-orange-400">{timing.firstSentence || 0} ms</span></div>
                            <div>Sentence &rarr; Audio: <span className="text-orange-400">{timing.firstAudio || 0} ms</span></div>
                            <div className="font-bold">Total Turn Latency: <span className="text-emerald-400">{timing.total || 0} ms</span></div>
                        </div>
                    </motion.div>
                )}

                {/* Footer Controls */}
                <div className="flex w-full max-w-xl flex-col items-center gap-3">
                    <div className="flex items-center gap-1.5 text-[11px] text-neutral-500">
                        <Headphones size={13} />
                        <span>Tip: Wear headphones for barge-in echo suppression.</span>
                    </div>

                    <div className="flex items-center gap-4">
                        <button
                            type="button"
                            onClick={onStopTurn}
                            className="flex items-center gap-1.5 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-2 text-xs font-semibold text-red-300 transition-colors hover:border-red-400 hover:bg-red-500/20"
                        >
                            <Square size={13} className="fill-current" />
                            <span>Stop Turn</span>
                        </button>

                        <button
                            type="button"
                            onClick={onClose}
                            className="rounded-xl border border-white/20 bg-white/10 px-5 py-2 text-xs font-semibold text-white transition-colors hover:bg-white/20"
                        >
                            End Conversation
                        </button>
                    </div>
                </div>
            </motion.div>
        </AnimatePresence>
    );
}

