import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Bot,
    Sparkles,
    CheckCircle2,
    ArrowRight,
    Volume2,
    VolumeX,
    Compass,
    ListTodo,
    Flame,
    GraduationCap,
    Play,
    RotateCcw,
    MessageSquare,
    Clock,
    Zap,
} from 'lucide-react';
import { buildAssistantBriefing } from '../utils/briefingUtils';
import {
    fetchTTSAudio,
    AudioQueue,
    splitSentencesClient,
    fetchVoiceSettings,
    stopVoiceTurn,
} from '../services/voiceService';

const ENTRY_DISMISSED_KEY = 'spidey_entry_briefing_seen_v1';

const STEP_ICONS = {
    task: ListTodo,
    habit: Flame,
    learning: GraduationCap,
    plan: Compass,
};

/**
 * Helper to trigger S.P.I.D.E.Y. sidebar chat with a specific prompt.
 */
export function triggerAssistantPrompt(promptText) {
    if (!promptText) return;
    window.dispatchEvent(
        new CustomEvent('spidey:prompt', {
            detail: { prompt: promptText },
        })
    );
}

/**
 * Speaks text using browser SpeechSynthesis as a responsive local fallback.
 */
function useBriefingSpeech(textToSpeak) {
    const [speaking, setSpeaking] = useState(false);
    const audioQueueRef = useRef(null);
    const abortControllerRef = useRef(null);

    if (!audioQueueRef.current && typeof window !== 'undefined') {
        audioQueueRef.current = new AudioQueue();
    }

    const fallbackBrowserSpeech = useCallback((text) => {
        if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
            setSpeaking(false);
            return;
        }
        try {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 0.95;
            utterance.onend = () => setSpeaking(false);
            utterance.onerror = () => setSpeaking(false);
            setSpeaking(true);
            window.speechSynthesis.speak(utterance);
        } catch {
            setSpeaking(false);
        }
    }, []);

    const stopSpeech = useCallback(() => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
        if (audioQueueRef.current) {
            audioQueueRef.current.stop();
        }
        stopVoiceTurn().catch(() => {});
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        setSpeaking(false);
    }, []);

    const toggleSpeech = useCallback(async () => {
        if (speaking) {
            stopSpeech();
            return;
        }
        if (!textToSpeak || !textToSpeak.trim()) return;

        let engineReady = false;
        try {
            const settings = await fetchVoiceSettings();
            if (settings && (settings.engine_ready || settings.available)) {
                engineReady = true;
            }
        } catch {
            engineReady = false;
        }

        if (!engineReady) {
            fallbackBrowserSpeech(textToSpeak);
            return;
        }

        const controller = new AbortController();
        abortControllerRef.current = controller;
        const queue = audioQueueRef.current;
        const currentEpoch = queue.epoch;

        queue.onPlay = () => setSpeaking(true);
        queue.onIdle = () => setSpeaking(false);
        setSpeaking(true);

        try {
            const sentences = splitSentencesClient(textToSpeak);
            if (!sentences.length) {
                setSpeaking(false);
                return;
            }

            for (const sentence of sentences) {
                if (controller.signal.aborted) break;
                const wavBuffer = await fetchTTSAudio(sentence, controller.signal);
                if (controller.signal.aborted) break;
                await queue.push(wavBuffer, currentEpoch);
            }
        } catch (err) {
            if (controller.signal.aborted) return;
            console.warn('OmniVoice TTS playback error, falling back to Web Speech:', err);
            fallbackBrowserSpeech(textToSpeak);
        }
    }, [speaking, textToSpeak, stopSpeech, fallbackBrowserSpeech]);

    useEffect(() => () => stopSpeech(), [stopSpeech]);

    return { speaking, toggleSpeech, stopSpeech };
}

/**
 * Interactive 3D parallax card wrapper that tilts subtly toward the pointer.
 */
const ParallaxTiltSurface = ({ children, className = '', intensity = 6 }) => {
    const cardRef = useRef(null);
    const frameRef = useRef(0);

    const handlePointerMove = (e) => {
        const el = cardRef.current;
        if (!el) return;
        if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
        const rect = el.getBoundingClientRect();
        const x = (e.clientX - rect.left) / Math.max(rect.width, 1);
        const y = (e.clientY - rect.top) / Math.max(rect.height, 1);
        const rotateY = (x - 0.5) * intensity;
        const rotateX = (0.5 - y) * intensity;

        cancelAnimationFrame(frameRef.current);
        frameRef.current = requestAnimationFrame(() => {
            el.style.setProperty('--tilt-rx', `${rotateX.toFixed(2)}deg`);
            el.style.setProperty('--tilt-ry', `${rotateY.toFixed(2)}deg`);
            el.style.setProperty('--spot-x', `${(x * 100).toFixed(1)}%`);
            el.style.setProperty('--spot-y', `${(y * 100).toFixed(1)}%`);
        });
    };

    const handlePointerLeave = () => {
        const el = cardRef.current;
        if (!el) return;
        cancelAnimationFrame(frameRef.current);
        el.style.setProperty('--tilt-rx', '0deg');
        el.style.setProperty('--tilt-ry', '0deg');
        el.style.setProperty('--spot-x', '50%');
        el.style.setProperty('--spot-y', '50%');
    };

    useEffect(() => () => cancelAnimationFrame(frameRef.current), []);

    return (
        <div
            ref={cardRef}
            onPointerMove={handlePointerMove}
            onPointerLeave={handlePointerLeave}
            className={`spidey-tilt-card ${className}`}
        >
            {children}
        </div>
    );
};

/**
 * Full-screen opening greeting experience + Docked Dashboard Command Deck.
 */
const AssistantEntryBriefing = ({
    tasks = [],
    habits = [],
    nextTopics = [],
    progress = null,
    userName = 'Shadow Hunter',
    onQuickCompleteTask,
    onQuickToggleHabit,
    onQuickStartTopic,
}) => {
    const navigate = useNavigate();
    const [now, setNow] = useState(() => new Date());
    const [busyStepId, setBusyStepId] = useState('');
    const [actionFeedback, setActionFeedback] = useState('');

    // Check if ?autostart=1 is present or if user already saw the full entry screen this session
    const [showEntryOverlay, setShowEntryOverlay] = useState(() => {
        if (typeof window === 'undefined') return false;
        const params = new URLSearchParams(window.location.search);
        if (params.get('autostart') === '1') return false;
        return sessionStorage.getItem(ENTRY_DISMISSED_KEY) !== '1';
    });

    const [typedText, setTypedText] = useState('');
    const overlayStageRef = useRef(null);

    useEffect(() => {
        const timer = window.setInterval(() => setNow(new Date()), 60000);
        return () => window.clearInterval(timer);
    }, []);

    const briefing = useMemo(
        () =>
            buildAssistantBriefing({
                tasks,
                habits,
                nextTopics,
                progress,
                now,
                userName,
            }),
        [tasks, habits, nextTopics, progress, now, userName]
    );

    const { speaking, toggleSpeech, stopSpeech } = useBriefingSpeech(briefing.spokenSummary);

    // Typewriter effect for the assistant's spoken summary
    useEffect(() => {
        const fullText = briefing.spokenSummary;
        if (
            typeof window !== 'undefined' &&
            window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
        ) {
            setTypedText(fullText);
            return undefined;
        }
        let index = 0;
        setTypedText('');
        const interval = window.setInterval(() => {
            index += 2;
            if (index >= fullText.length) {
                setTypedText(fullText);
                window.clearInterval(interval);
            } else {
                setTypedText(fullText.slice(0, index));
            }
        }, 16);
        return () => window.clearInterval(interval);
    }, [briefing.spokenSummary]);

    const dismissOverlay = useCallback(() => {
        sessionStorage.setItem(ENTRY_DISMISSED_KEY, '1');
        stopSpeech();
        setShowEntryOverlay(false);
    }, [stopSpeech]);

    const reopenOverlay = useCallback(() => {
        setShowEntryOverlay(true);
    }, []);

    // Pointer parallax handler for the full-screen entry overlay
    const handleOverlayPointerMove = (e) => {
        const el = overlayStageRef.current;
        if (!el) return;
        if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
        const x = (e.clientX / Math.max(window.innerWidth, 1) - 0.5) * 2;
        const y = (e.clientY / Math.max(window.innerHeight, 1) - 0.5) * 2;
        el.style.setProperty('--entry-px', `${(x * 28).toFixed(1)}px`);
        el.style.setProperty('--entry-py', `${(y * 28).toFixed(1)}px`);
    };

    const handleStepPrimaryAction = async (step) => {
        if (!step) return;
        setBusyStepId(step.id);
        setActionFeedback('');
        try {
            if (step.kind === 'task' && onQuickCompleteTask && step.entityId) {
                await onQuickCompleteTask(step.entityId);
                setActionFeedback(`Completed "${step.title}" — recalculating next priority.`);
            } else if (step.kind === 'habit' && onQuickToggleHabit && step.habitObj) {
                await onQuickToggleHabit(step.habitObj);
                setActionFeedback(`Checked in "${step.title}" for today!`);
            } else if (step.kind === 'learning' && onQuickStartTopic && step.topicObj) {
                await onQuickStartTopic(step.topicObj);
                setActionFeedback(`Marked "${step.title}" as in progress.`);
            } else if (step.prompt) {
                triggerAssistantPrompt(step.prompt);
                dismissOverlay();
            }
        } finally {
            setBusyStepId('');
        }
    };

    const handleNavigateStep = (step) => {
        if (!step) return;
        dismissOverlay();
        if (step.route) {
            navigate(step.route);
        }
    };

    const handleAskSpideyStep = (step) => {
        const prompt = step?.prompt || "What's my best starting plan for today?";
        dismissOverlay();
        triggerAssistantPrompt(prompt);
    };

    const PrimaryIcon = STEP_ICONS[briefing.primaryStart.kind] || Compass;

    return (
        <>
            {/* Docked Assistant Daily Briefing & Start-Here Command Deck */}
            <ParallaxTiltSurface
                intensity={4}
                className="relative overflow-hidden rounded-3xl border border-orange-400/25 bg-gradient-to-br from-[#14100d]/95 via-[#0d0d0d]/95 to-[#17110c]/95 p-6 md:p-7 shadow-[0_20px_60px_rgba(0,0,0,0.6),0_0_40px_rgba(255,106,26,0.08)]"
            >
                {/* Parallax Ambient Layers inside the Command Deck */}
                <div
                    aria-hidden="true"
                    className="spidey-parallax-orb pointer-events-none absolute -right-16 -top-16 h-64 w-64 rounded-full bg-orange-500/15 blur-3xl"
                />
                <div
                    aria-hidden="true"
                    className="spidey-parallax-grid pointer-events-none absolute inset-0 opacity-25"
                />

                <div className="relative z-10 flex flex-col gap-6">
                    {/* Top Row: Assistant Status + Controls */}
                    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
                        <div className="flex items-center gap-3">
                            <div className="relative grid h-11 w-11 shrink-0 place-items-center rounded-2xl border border-orange-400/35 bg-orange-500/15 text-orange-300 shadow-[0_0_24px_rgba(255,106,26,0.25)]">
                                <Bot size={22} />
                                <span className="absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-[#0d0d0d] bg-emerald-400" />
                            </div>
                            <div>
                                <div className="flex items-center gap-2">
                                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-orange-300">
                                        S.P.I.D.E.Y. {briefing.greeting.period} briefing
                                    </span>
                                    <span className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-0.5 font-mono text-[10px] text-textMuted tabular-nums">
                                        <Clock size={10} className="text-orange-400" />
                                        {now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                    </span>
                                </div>
                                <h2
                                    data-testid="assistant-greeting-headline"
                                    className="mt-0.5 text-2xl md:text-3xl font-extrabold tracking-tight text-white"
                                >
                                    {briefing.greeting.headline}
                                </h2>
                            </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                            <button
                                type="button"
                                onClick={toggleSpeech}
                                className={`inline-flex items-center gap-1.5 rounded-xl border px-3 py-2 text-xs font-semibold transition-all ${
                                    speaking
                                        ? 'border-orange-400 bg-orange-500/20 text-orange-200'
                                        : 'border-white/10 bg-white/5 text-textMuted hover:border-orange-400/30 hover:text-white'
                                }`}
                                aria-label={speaking ? 'Stop voice greeting' : 'Speak assistant greeting'}
                            >
                                {speaking ? <VolumeX size={14} className="text-orange-400" /> : <Volume2 size={14} className="text-orange-400" />}
                                <span>{speaking ? 'Stop Voice' : 'Voice Greeting'}</span>
                            </button>

                            <button
                                type="button"
                                onClick={() =>
                                    triggerAssistantPrompt(
                                        "Give me my daily briefing and tell me where I should start today."
                                    )
                                }
                                className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-textMuted transition-all hover:border-orange-400/30 hover:text-white"
                            >
                                <MessageSquare size={14} className="text-orange-400" />
                                <span>Discuss in Chat</span>
                            </button>

                            <button
                                type="button"
                                onClick={reopenOverlay}
                                className="inline-flex items-center gap-1.5 rounded-xl border border-orange-400/25 bg-orange-500/10 px-3 py-2 text-xs font-semibold text-orange-200 transition-all hover:border-orange-400/50 hover:bg-orange-500/20"
                            >
                                <RotateCcw size={13} />
                                <span>Entry Sequence</span>
                            </button>
                        </div>
                    </div>

                    {/* Middle Row: Spoken Assistant Message + Remaining Counters */}
                    <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-center">
                        <div className="lg:col-span-7 space-y-3">
                            <p className="text-xs font-mono uppercase tracking-[0.16em] text-orange-300/80">
                                {briefing.greeting.atmosphere}
                            </p>
                            <div
                                data-testid="assistant-spoken-summary"
                                className="rounded-2xl border border-white/10 bg-black/45 p-4 text-sm md:text-base leading-relaxed text-neutral-200 shadow-inner"
                            >
                                <span className="mr-2 inline-block font-mono text-xs font-bold uppercase tracking-wider text-orange-400">
                                    [S.P.I.D.E.Y.]
                                </span>
                                <span>{typedText || briefing.spokenSummary}</span>
                            </div>

                            {actionFeedback && (
                                <motion.p
                                    initial={{ opacity: 0, y: -4 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex items-center gap-2 text-xs font-medium text-emerald-300"
                                >
                                    <CheckCircle2 size={14} />
                                    <span>{actionFeedback}</span>
                                </motion.p>
                            )}
                        </div>

                        {/* Tabular-nums Telemetry Strip */}
                        <div className="lg:col-span-5 grid grid-cols-3 gap-3">
                            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 text-center">
                                <p className="font-mono text-2xl md:text-3xl font-extrabold text-white tabular-nums" data-testid="briefing-remaining-tasks">
                                    {briefing.stats.remainingTasksCount}
                                </p>
                                <p className="mt-1 text-[11px] font-semibold uppercase tracking-wider text-textMuted">
                                    Tasks Left
                                </p>
                                {briefing.stats.highPriorityCount > 0 && (
                                    <p className="mt-0.5 font-mono text-[10px] text-orange-300 tabular-nums">
                                        {briefing.stats.highPriorityCount} high prio
                                    </p>
                                )}
                            </div>

                            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 text-center">
                                <p className="font-mono text-2xl md:text-3xl font-extrabold text-white tabular-nums" data-testid="briefing-remaining-habits">
                                    {briefing.stats.remainingHabitsCount}
                                </p>
                                <p className="mt-1 text-[11px] font-semibold uppercase tracking-wider text-textMuted">
                                    Habits Left
                                </p>
                                <p className="mt-0.5 font-mono text-[10px] text-emerald-300/80 tabular-nums">
                                    {briefing.stats.completedHabitsCount}/{briefing.stats.totalHabitsCount} done
                                </p>
                            </div>

                            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3.5 text-center">
                                <p className="font-mono text-2xl md:text-3xl font-extrabold text-orange-300 tabular-nums" data-testid="briefing-learning-percent">
                                    {briefing.stats.learningPercent}%
                                </p>
                                <p className="mt-1 text-[11px] font-semibold uppercase tracking-wider text-textMuted">
                                    Learning
                                </p>
                                <p className="mt-0.5 font-mono text-[10px] text-textMuted tabular-nums">
                                    {briefing.stats.inProgressTopics} active
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Bottom Row: Where We Should Start — Ordered Launch Sequence */}
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <span className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-orange-300">
                                <Compass size={14} className="text-orange-400" />
                                Recommended Starting Sequence
                            </span>
                            <span className="text-[11px] text-textMuted hidden sm:inline">
                                Ranked by urgency, daily streak impact, and unlocked prerequisites
                            </span>
                        </div>

                        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                            {briefing.startingSteps.map((step, index) => {
                                const StepIcon = STEP_ICONS[step.kind] || Compass;
                                const isPrimary = index === 0;
                                const isBusy = busyStepId === step.id;

                                return (
                                    <motion.div
                                        key={step.id}
                                        initial={{ opacity: 0, y: 12 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: index * 0.07, duration: 0.35 }}
                                        data-testid={isPrimary ? 'primary-start-recommendation' : `start-step-${index + 1}`}
                                        className={`flex flex-col justify-between rounded-2xl border p-4 transition-all ${
                                            isPrimary
                                                ? 'border-orange-400/45 bg-gradient-to-br from-orange-500/15 via-orange-500/5 to-transparent shadow-[0_0_28px_rgba(255,106,26,0.12)]'
                                                : 'border-white/10 bg-white/[0.025] hover:border-white/20'
                                        }`}
                                    >
                                        <div>
                                            <div className="flex items-start justify-between gap-2">
                                                <span className="inline-flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.15em] text-orange-300">
                                                    <StepIcon size={13} className="shrink-0 text-orange-400" />
                                                    <span>Step 0{index + 1} · {step.badge}</span>
                                                </span>
                                                {isPrimary && (
                                                    <span className="shrink-0 whitespace-nowrap rounded-md bg-orange-500/20 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-orange-200">
                                                        Start Here
                                                    </span>
                                                )}
                                            </div>

                                            <h3 className="mt-2 font-bold text-white text-base leading-snug line-clamp-1">
                                                {step.title}
                                            </h3>
                                            <p className="mt-1 text-xs text-textMuted leading-relaxed line-clamp-2">
                                                {step.reason}
                                            </p>
                                        </div>

                                        <div className="mt-4 flex flex-col gap-2 pt-3 border-t border-white/10">
                                            {(step.kind === 'task' || step.kind === 'habit' || step.kind === 'learning') && (
                                                <button
                                                    type="button"
                                                    disabled={isBusy}
                                                    onClick={() => handleStepPrimaryAction(step)}
                                                    className={`w-full inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-xl px-3 py-2 text-xs font-bold transition-all ${
                                                        isPrimary
                                                            ? 'bg-white text-black shadow-[0_0_16px_rgba(255,106,26,0.3)] hover:opacity-95'
                                                            : 'border border-orange-400/30 bg-orange-500/10 text-orange-200 hover:bg-orange-500/20'
                                                    }`}
                                                >
                                                    <CheckCircle2 size={13} className="shrink-0" />
                                                    <span>{isBusy ? 'Updating…' : step.completeLabel}</span>
                                                </button>
                                            )}

                                            <div className="flex items-center gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => handleNavigateStep(step)}
                                                    className="flex-1 inline-flex items-center justify-center gap-1 whitespace-nowrap rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold text-neutral-200 hover:border-white/25 hover:text-white"
                                                    title={`Open ${step.title}`}
                                                >
                                                    <span>Open</span>
                                                    <ArrowRight size={12} />
                                                </button>

                                                <button
                                                    type="button"
                                                    onClick={() => handleAskSpideyStep(step)}
                                                    className="inline-flex items-center justify-center gap-1 whitespace-nowrap rounded-xl border border-white/10 bg-white/5 px-2.5 py-1.5 text-xs font-medium text-textMuted hover:border-orange-400/30 hover:text-orange-300"
                                                    title="Ask S.P.I.D.E.Y. to help with this step"
                                                    aria-label={`Ask S.P.I.D.E.Y. about ${step.title}`}
                                                >
                                                    <Sparkles size={12} />
                                                    <span>Ask AI</span>
                                                </button>
                                            </div>
                                        </div>
                                    </motion.div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            </ParallaxTiltSurface>

            {/* Full-Screen Opening Entry Briefing Overlay Portaled to document.body */}
            {typeof document !== 'undefined' &&
                createPortal(
                    <AnimatePresence>
                        {showEntryOverlay && (
                            <motion.div
                                key="spidey-entry-overlay"
                                ref={overlayStageRef}
                                onPointerMove={handleOverlayPointerMove}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0, scale: 1.03, filter: 'blur(8px)' }}
                                transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
                                role="dialog"
                                aria-modal="true"
                                aria-labelledby="entry-briefing-title"
                                data-testid="entry-briefing-overlay"
                                className="fixed inset-0 z-[120] flex items-center justify-center overflow-y-auto bg-[#060608]/92 p-4 md:p-8 backdrop-blur-2xl"
                            >
                                {/* Layer 1: Deep Parallax Tactical Grid */}
                                <div
                                    aria-hidden="true"
                                    className="spidey-entry-layer-deep pointer-events-none fixed inset-0 opacity-30"
                                />

                                {/* Layer 2: Mid Parallax Ambient Glowing Orbs */}
                                <div
                                    aria-hidden="true"
                                    className="spidey-entry-layer-mid pointer-events-none fixed inset-0 flex items-center justify-center"
                                >
                                    <div className="h-[460px] w-[460px] rounded-full bg-orange-500/15 blur-[130px]" />
                                    <div className="h-[320px] w-[320px] -ml-32 mt-28 rounded-full bg-amber-500/10 blur-[110px]" />
                                </div>

                                {/* Layer 3: Interactive 3D Foreground Briefing Card */}
                                <motion.div
                                    initial={{ opacity: 0, y: 36, scale: 0.94, rotateX: 8 }}
                                    animate={{ opacity: 1, y: 0, scale: 1, rotateX: 0 }}
                                    exit={{ opacity: 0, y: -20, scale: 0.97 }}
                                    transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
                                    className="spidey-entry-layer-front relative z-10 w-full max-w-3xl rounded-3xl border border-orange-400/35 bg-gradient-to-b from-[#16120f] via-[#0e0d0c] to-[#0a0a0a] p-6 md:p-9 shadow-[0_32px_90px_rgba(0,0,0,0.85),0_0_60px_rgba(255,106,26,0.16)]"
                                >
                                    {/* Top Assistant Identity Bar */}
                                    <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 pb-5">
                                        <div className="flex items-center gap-3.5">
                                            <div className="relative grid h-14 w-14 place-items-center rounded-2xl border border-orange-400/40 bg-orange-500/15 text-orange-300 shadow-[0_0_30px_rgba(255,106,26,0.3)]">
                                                <Bot size={28} />
                                                <span className="absolute -right-1 -top-1 flex h-3.5 w-3.5">
                                                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-orange-400 opacity-75" />
                                                    <span className="relative inline-flex h-3.5 w-3.5 rounded-full bg-orange-500" />
                                                </span>
                                            </div>
                                            <div>
                                                <p className="text-[11px] font-mono font-bold uppercase tracking-[0.24em] text-orange-300">
                                                    S.P.I.D.E.Y. · Personal Intelligence Online
                                                </p>
                                                <h1
                                                    id="entry-briefing-title"
                                                    className="mt-1 text-3xl md:text-4xl font-black tracking-tight text-white"
                                                >
                                                    {briefing.greeting.headline}
                                                </h1>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2">
                                            <button
                                                type="button"
                                                onClick={toggleSpeech}
                                                className="inline-flex items-center gap-2 rounded-xl border border-orange-400/30 bg-orange-500/10 px-3.5 py-2 text-xs font-semibold text-orange-200 hover:bg-orange-500/20"
                                            >
                                                {speaking ? <VolumeX size={15} /> : <Volume2 size={15} />}
                                                <span>{speaking ? 'Mute Briefing' : 'Listen to Briefing'}</span>
                                            </button>
                                        </div>
                                    </div>

                                    {/* Assistant Spoken Greeting & Workload Summary */}
                                    <div className="mt-6 rounded-2xl border border-white/10 bg-black/50 p-5">
                                        <p className="text-xs font-mono uppercase tracking-[0.18em] text-orange-300/90 mb-2">
                                            {briefing.greeting.atmosphere}
                                        </p>
                                        <p className="text-base md:text-lg leading-relaxed text-neutral-100">
                                            &ldquo;{briefing.spokenSummary}&rdquo;
                                        </p>
                                    </div>

                                    {/* Remaining Workload Metrics */}
                                    <div className="mt-6 grid grid-cols-3 gap-3 md:gap-4">
                                        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                                            <span className="text-[11px] font-bold uppercase tracking-wider text-textMuted">
                                                Remaining Tasks
                                            </span>
                                            <p className="mt-1 font-mono text-3xl font-extrabold text-white tabular-nums">
                                                {briefing.stats.remainingTasksCount}
                                            </p>
                                            <p className="mt-1 text-xs text-orange-300 font-mono tabular-nums">
                                                {briefing.stats.highPriorityCount > 0
                                                    ? `${briefing.stats.highPriorityCount} high priority`
                                                    : `${briefing.stats.completedTasksCount} completed`}
                                            </p>
                                        </div>

                                        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                                            <span className="text-[11px] font-bold uppercase tracking-wider text-textMuted">
                                                Habits Today
                                            </span>
                                            <p className="mt-1 font-mono text-3xl font-extrabold text-white tabular-nums">
                                                {briefing.stats.remainingHabitsCount}
                                            </p>
                                            <p className="mt-1 text-xs text-emerald-300 font-mono tabular-nums">
                                                {briefing.stats.completedHabitsCount} of {briefing.stats.totalHabitsCount} checked in
                                            </p>
                                        </div>

                                        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                                            <span className="text-[11px] font-bold uppercase tracking-wider text-textMuted">
                                                Learning Mastery
                                            </span>
                                            <p className="mt-1 font-mono text-3xl font-extrabold text-orange-300 tabular-nums">
                                                {briefing.stats.learningPercent}%
                                            </p>
                                            <p className="mt-1 text-xs text-textMuted font-mono tabular-nums">
                                                {briefing.stats.inProgressTopics} active topics
                                            </p>
                                        </div>
                                    </div>

                                    {/* Highlighted "Where We Should Start" Card */}
                                    <div className="mt-6 rounded-2xl border border-orange-400/40 bg-gradient-to-r from-orange-500/15 via-orange-500/5 to-transparent p-5">
                                        <div className="flex items-center justify-between gap-2">
                                            <span className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-[0.18em] text-orange-300">
                                                <Zap size={14} className="text-orange-400" />
                                                Where We Should Start First
                                            </span>
                                            <span className="rounded-md border border-orange-400/30 bg-orange-500/20 px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-orange-200">
                                                {briefing.primaryStart.badge}
                                            </span>
                                        </div>

                                        <div className="mt-3 flex flex-col md:flex-row md:items-center justify-between gap-4">
                                            <div className="flex items-start gap-3">
                                                <div className="mt-0.5 grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-orange-500/20 text-orange-300">
                                                    <PrimaryIcon size={20} />
                                                </div>
                                                <div>
                                                    <h3 className="text-xl font-bold text-white">
                                                        {briefing.primaryStart.title}
                                                    </h3>
                                                    <p className="mt-1 text-sm text-neutral-300">
                                                        {briefing.primaryStart.reason}
                                                    </p>
                                                </div>
                                            </div>

                                            <div className="flex flex-wrap items-center gap-2.5 shrink-0">
                                                <button
                                                    type="button"
                                                    onClick={() => handleNavigateStep(briefing.primaryStart)}
                                                    className="inline-flex items-center gap-2 rounded-xl bg-white px-4 py-2.5 text-xs font-bold uppercase tracking-wider text-black shadow-[0_0_24px_rgba(255,106,26,0.35)] hover:opacity-95"
                                                >
                                                    <Play size={14} />
                                                    <span>{briefing.primaryStart.ctaLabel}</span>
                                                </button>

                                                <button
                                                    type="button"
                                                    onClick={() => handleAskSpideyStep(briefing.primaryStart)}
                                                    className="inline-flex items-center gap-1.5 rounded-xl border border-orange-400/30 bg-orange-500/10 px-3.5 py-2.5 text-xs font-semibold text-orange-200 hover:bg-orange-500/20"
                                                >
                                                    <Sparkles size={14} />
                                                    <span>Ask S.P.I.D.E.Y.</span>
                                                </button>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Bottom Entry Controls */}
                                    <div className="mt-7 flex flex-col sm:flex-row items-center justify-between gap-3 pt-4 border-t border-white/10">
                                        <p className="text-xs text-textMuted">
                                            Move your cursor to experience 3D depth parallax · Click below to enter workspace
                                        </p>
                                        <button
                                            type="button"
                                            onClick={dismissOverlay}
                                            data-testid="enter-workspace-button"
                                            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 rounded-xl border border-white/20 bg-white/10 px-5 py-2.5 text-xs font-bold uppercase tracking-wider text-white hover:bg-white/15"
                                        >
                                            <span>Enter Dashboard</span>
                                            <ArrowRight size={14} />
                                        </button>
                                    </div>
                                </motion.div>
                            </motion.div>
                        )}
                    </AnimatePresence>,
                    document.body
                )}
        </>
    );
};

export default AssistantEntryBriefing;
