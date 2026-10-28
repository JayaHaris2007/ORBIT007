import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    Link2, Check, X, RefreshCw, Loader2, ChevronDown,
} from 'lucide-react';
import { learningService } from '../../services/learningService';

const sourceLabel = (source) => (
    source === 'embedding' ? 'semantic' : source === 'manual' ? 'manual' : 'rule'
);

/**
 * Review queue for task→topic links the classifier suggested. Confirming a
 * link means finishing that task will count as learning progress; rejecting
 * it hides the suggestion.
 */
const SuggestedLinks = ({ onChanged }) => {
    const [links, setLinks] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [busyId, setBusyId] = useState(null);
    const [changingId, setChangingId] = useState(null);
    const [topicOptions, setTopicOptions] = useState([]);
    const [pickingId, setPickingId] = useState(null);

    const load = useCallback(async () => {
        try {
            setError('');
            setLinks(await learningService.listSuggestedLinks());
        } catch (loadError) {
            setError(loadError.message || 'Could not load suggested links.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    const handleAction = async (link, patch) => {
        setBusyId(link.id);
        setError('');
        try {
            await learningService.updateLink(link.id, patch);
            setLinks((current) => current.filter((item) => item.id !== link.id));
            onChanged?.();
        } catch (actionError) {
            setError(actionError.message);
        } finally {
            setBusyId(null);
        }
    };

    const openTopicPicker = async (link) => {
        setChangingId(link.id);
        setPickingId(null);
        setError('');
        try {
            const roadmap = await learningService.getRoadmap(link.roadmapId);
            setTopicOptions(roadmap.topics.filter((topic) => topic.id !== link.topicId));
        } catch (pickError) {
            setError(pickError.message);
            setChangingId(null);
        }
    };

    return (
        <section className="space-y-4">
            <div className="flex items-center gap-2 text-white/80">
                <Link2 size={18} className="text-orange-400" />
                <h2 className="text-lg font-semibold uppercase tracking-widest">Suggested task links</h2>
                {!loading && links.length > 0 && (
                    <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded-full bg-orange-500/15 text-orange-300 border border-orange-400/20">
                        {links.length} to review
                    </span>
                )}
            </div>

            {error && (
                <div className="glass-card p-4 border-red-500/30 text-red-300 text-sm flex items-center justify-between gap-3">
                    <span>{error}</span>
                    <button onClick={load} className="p-1.5 rounded-lg text-textMuted hover:text-white hover:bg-white/5 transition-colors" title="Retry">
                        <RefreshCw size={14} />
                    </button>
                </div>
            )}

            {loading ? (
                <div className="glass-card p-6 flex items-center justify-center gap-2 text-textMuted text-sm">
                    <Loader2 size={16} className="animate-spin" /> Checking for suggested links…
                </div>
            ) : links.length === 0 ? (
                <div className="glass-card p-8 text-center space-y-2">
                    <div className="p-3 bg-white/5 rounded-full w-fit mx-auto text-textMuted"><Link2 size={20} /></div>
                    <p className="text-sm text-textMuted">
                        Nothing to review. When you create a task that matches a topic,
                        I will suggest the link here.
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <AnimatePresence>
                        {links.map((link) => (
                            <motion.div
                                key={link.id}
                                layout
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                className="glass-card p-4 space-y-3 hover:border-white/20 transition-colors"
                            >
                                <div className="flex items-start justify-between gap-3">
                                    <div className="min-w-0">
                                        <p className="font-medium text-white truncate">{link.taskTitle}</p>
                                        <p className="text-xs text-textMuted mt-0.5 truncate">
                                            → <span className="text-orange-300">{link.topicName}</span>
                                            <span className="text-textMuted"> · {link.roadmapName}</span>
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-1.5 shrink-0">
                                        <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full border border-white/10 text-textMuted">
                                            {sourceLabel(link.source)}
                                        </span>
                                        <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-textMuted">
                                            {Math.round((link.confidence ?? 0) * 100)}%
                                        </span>
                                    </div>
                                </div>

                                {changingId === link.id ? (
                                    <div className="flex items-center gap-2">
                                        <select
                                            value={pickingId ?? ''}
                                            onChange={(e) => setPickingId(Number(e.target.value))}
                                            className="flex-1 bg-surfaceHighlight border border-white/10 rounded-lg px-2 py-2 text-xs text-white focus:border-orange-400/50 outline-none"
                                        >
                                            <option value="" disabled>Pick a topic…</option>
                                            {topicOptions.map((topic) => (
                                                <option key={topic.id} value={topic.id}>{topic.name}</option>
                                            ))}
                                        </select>
                                        <button
                                            onClick={() => handleAction(link, { state: 'confirmed', topicId: pickingId })}
                                            disabled={!pickingId || busyId === link.id}
                                            className="px-3 py-2 rounded-lg bg-primary hover:bg-orange-500 text-white text-xs font-semibold transition-colors disabled:opacity-50"
                                        >
                                            Save
                                        </button>
                                        <button
                                            onClick={() => setChangingId(null)}
                                            className="p-2 rounded-lg text-textMuted hover:text-white hover:bg-white/5 transition-colors"
                                            title="Cancel"
                                        >
                                            <X size={14} />
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => handleAction(link, { state: 'confirmed' })}
                                            disabled={busyId === link.id}
                                            className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg bg-emerald-500/15 text-emerald-300 border border-emerald-400/20 hover:bg-emerald-500/25 transition-colors text-xs font-semibold disabled:opacity-50"
                                        >
                                            <Check size={14} /> Confirm
                                        </button>
                                        <button
                                            onClick={() => openTopicPicker(link)}
                                            disabled={busyId === link.id}
                                            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-white/10 text-textMuted hover:text-white hover:bg-white/5 transition-colors text-xs font-semibold disabled:opacity-50"
                                            title="Link to a different topic"
                                        >
                                            <ChevronDown size={13} /> Change
                                        </button>
                                        <button
                                            onClick={() => handleAction(link, { state: 'rejected' })}
                                            disabled={busyId === link.id}
                                            className="flex items-center gap-1 px-3 py-2 rounded-lg border border-white/10 text-textMuted hover:text-red-400 hover:bg-red-500/10 transition-colors text-xs font-semibold disabled:opacity-50"
                                            title="Not related"
                                        >
                                            <X size={13} /> Reject
                                        </button>
                                    </div>
                                )}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                </div>
            )}
        </section>
    );
};

export default SuggestedLinks;
