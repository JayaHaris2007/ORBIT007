import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X, Lock, Play, CheckCircle2, PauseCircle, Trash2, Plus,
    Pencil, BookOpen, Sparkles,
} from 'lucide-react';
import { learningService } from '../../services/learningService';

const STATUS_LABELS = {
    not_started: 'Not started',
    learning: 'Learning',
    completed: 'Completed',
    on_hold: 'On hold',
};

const STATUS_CHIP = {
    not_started: 'bg-white/5 text-textMuted border-white/10',
    learning: 'bg-amber-500/15 text-amber-300 border-amber-400/20',
    completed: 'bg-emerald-500/15 text-emerald-300 border-emerald-400/20',
    on_hold: 'bg-red-500/15 text-red-300 border-red-400/20',
};

const RoadmapDetail = ({ roadmap, onClose, onChanged }) => {
    const [detail, setDetail] = useState(roadmap);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [busy, setBusy] = useState(false);

    // Edit roadmap name
    const [isRenaming, setIsRenaming] = useState(false);
    const [draftName, setDraftName] = useState('');

    // Add topic form
    const [isAdding, setIsAdding] = useState(false);
    const [newName, setNewName] = useState('');
    const [newPriority, setNewPriority] = useState(2);

    useEffect(() => {
        let cancelled = false;
        setLoading(true);
        learningService.getRoadmap(roadmap.id)
            .then((data) => {
                if (!cancelled) {
                    setDetail(data);
                    setLoading(false);
                }
            })
            .catch((loadError) => {
                if (!cancelled) {
                    setError(loadError.message || 'Could not load this roadmap.');
                    setLoading(false);
                }
            });
        return () => { cancelled = true; };
    }, [roadmap.id]);

    const reload = async () => {
        const data = await learningService.getRoadmap(roadmap.id);
        setDetail(data);
        onChanged?.();
    };

    const handleRename = async () => {
        const name = draftName.trim();
        if (!name || name === detail.name || busy) return;
        setBusy(true);
        try {
            const updated = await learningService.updateRoadmap(detail.id, { name });
            setDetail((current) => ({ ...current, ...updated }));
            setIsRenaming(false);
            onChanged?.();
        } catch (renameError) {
            setError(renameError.message);
        } finally {
            setBusy(false);
        }
    };

    const handleStatus = async (topic, status) => {
        setBusy(true);
        setError('');
        try {
            await learningService.updateTopic(topic.id, { status });
            await reload();
        } catch (statusError) {
            setError(statusError.message);
        } finally {
            setBusy(false);
        }
    };

    const handleDeleteTopic = async (topic) => {
        setBusy(true);
        setError('');
        try {
            await learningService.deleteTopic(topic.id);
            await reload();
        } catch (deleteError) {
            setError(deleteError.message);
        } finally {
            setBusy(false);
        }
    };

    const handleAddTopic = async (e) => {
        e.preventDefault();
        const name = newName.trim();
        if (!name || busy) return;
        setBusy(true);
        setError('');
        try {
            await learningService.createTopic({
                roadmap_id: detail.id,
                roadmapId: detail.id,
                name,
                priority: newPriority,
            });
            setNewName('');
            setNewPriority(2);
            setIsAdding(false);
            await reload();
        } catch (addError) {
            setError(addError.message);
        } finally {
            setBusy(false);
        }
    };

    const topics = detail.topics ?? [];
    const completed = topics.filter((topic) => topic.status === 'completed').length;
    const percent = topics.length ? Math.round((completed / topics.length) * 100) : 0;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[90] bg-black/70 backdrop-blur-sm" onClick={onClose} />
            <motion.aside
                initial={{ x: '100%' }}
                animate={{ x: 0 }}
                exit={{ x: '100%' }}
                transition={{ type: 'tween', duration: 0.25 }}
                className="fixed top-0 right-0 bottom-0 z-[95] w-full max-w-lg bg-surface border-l border-white/10 shadow-2xl flex flex-col"
            >
                {/* Header */}
                <div className="p-6 border-b border-white/10 flex items-start justify-between gap-3">
                    <div className="min-w-0">
                        {isRenaming ? (
                            <div className="flex items-center gap-2">
                                <input
                                    type="text"
                                    value={draftName}
                                    onChange={(e) => setDraftName(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleRename()}
                                    className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-white text-lg font-bold focus:border-orange-400/50 outline-none"
                                    autoFocus
                                    maxLength={120}
                                />
                                <button
                                    onClick={handleRename}
                                    className="p-2 rounded-lg bg-primary text-white shrink-0 hover:bg-orange-500 transition-colors"
                                    title="Save name"
                                >
                                    <CheckCircle2 size={18} />
                                </button>
                            </div>
                        ) : (
                            <div className="flex items-center gap-2">
                                <h2 className="text-xl font-bold text-white truncate">{detail.name}</h2>
                                <button
                                    onClick={() => { setDraftName(detail.name); setIsRenaming(true); }}
                                    className="p-1.5 rounded-lg text-textMuted hover:text-white hover:bg-white/5 transition-colors shrink-0"
                                    title="Rename roadmap"
                                >
                                    <Pencil size={14} />
                                </button>
                            </div>
                        )}
                        <p className="text-xs text-textMuted mt-1 line-clamp-2">{detail.description}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg text-textMuted hover:text-white hover:bg-white/5 transition-colors shrink-0"
                        title="Close"
                    >
                        <X size={18} />
                    </button>
                </div>

                {/* Progress strip */}
                <div className="px-6 py-4 border-b border-white/10 space-y-2">
                    <div className="flex justify-between text-xs text-textMuted">
                        <span className="uppercase tracking-widest font-bold">Progress</span>
                        <span>{completed} / {topics.length} topics · {percent}%</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                        <div
                            className="h-full rounded-full bg-gradient-to-r from-orange-500 to-amber-400 transition-all duration-500"
                            style={{ width: `${percent}%` }}
                        />
                    </div>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {error && (
                        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">
                            {error}
                        </div>
                    )}

                    {loading ? (
                        <p className="text-sm text-textMuted text-center py-10">Loading topics…</p>
                    ) : topics.length === 0 ? (
                        <div className="glass-card p-8 text-center space-y-3">
                            <div className="p-3 bg-white/5 rounded-full w-fit mx-auto text-textMuted"><BookOpen size={22} /></div>
                            <p className="text-sm text-textMuted">No topics yet — add the first one below.</p>
                        </div>
                    ) : (
                        topics.map((topic) => {
                            const locked = !topic.available;
                            return (
                                <motion.div
                                    key={topic.id}
                                    layout
                                    initial={{ opacity: 0, y: 8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className={`glass-card p-4 space-y-3 ${locked ? 'opacity-75' : ''}`}
                                >
                                    <div className="flex items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <div className="flex items-center gap-2">
                                                <p className="font-medium text-white truncate">{topic.name}</p>
                                                {locked && <Lock size={13} className="text-textMuted shrink-0" />}
                                            </div>
                                            <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                                                <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full border ${STATUS_CHIP[topic.status] || STATUS_CHIP.not_started}`}>
                                                    {STATUS_LABELS[topic.status] || topic.status}
                                                </span>
                                                <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full border border-white/10 text-textMuted">
                                                    P{topic.priority || 2}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-1.5 shrink-0">
                                            {!locked && topic.status !== 'completed' && topic.status !== 'learning' && (
                                                <button
                                                    onClick={() => handleStatus(topic, 'learning')}
                                                    disabled={busy}
                                                    className="p-2 rounded-lg bg-amber-500/15 text-amber-300 hover:bg-amber-500/25 transition-colors disabled:opacity-50"
                                                    title="Start learning"
                                                >
                                                    <Play size={15} />
                                                </button>
                                            )}
                                            {!locked && topic.status !== 'completed' && (
                                                <button
                                                    onClick={() => handleStatus(topic, 'completed')}
                                                    disabled={busy}
                                                    className="p-2 rounded-lg bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25 transition-colors disabled:opacity-50"
                                                    title="Mark completed"
                                                >
                                                    <CheckCircle2 size={15} />
                                                </button>
                                            )}
                                            {topic.status === 'learning' && (
                                                <button
                                                    onClick={() => handleStatus(topic, 'on_hold')}
                                                    disabled={busy}
                                                    className="p-2 rounded-lg bg-red-500/15 text-red-300 hover:bg-red-500/25 transition-colors disabled:opacity-50"
                                                    title="Put on hold"
                                                >
                                                    <PauseCircle size={15} />
                                                </button>
                                            )}
                                            <button
                                                onClick={() => handleDeleteTopic(topic)}
                                                disabled={busy}
                                                className="p-2 rounded-lg text-textMuted hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
                                                title="Delete topic"
                                            >
                                                <Trash2 size={15} />
                                            </button>
                                        </div>
                                    </div>

                                    {locked && topic.unmetPrerequisites?.length > 0 && (
                                        <div className="flex items-center gap-2 text-xs text-textMuted bg-white/5 rounded-lg px-3 py-2 border border-white/5">
                                            <Sparkles size={13} className="text-orange-400 shrink-0" />
                                            <span className="truncate">
                                                Complete first: {topic.unmetPrerequisites.join(', ')}
                                            </span>
                                        </div>
                                    )}
                                </motion.div>
                            );
                        })
                    )}
                </div>

                {/* Add topic footer */}
                <div className="p-6 border-t border-white/10">
                    {isAdding ? (
                        <form onSubmit={handleAddTopic} className="space-y-3">
                            <div className="flex gap-2">
                                <input
                                    type="text"
                                    value={newName}
                                    onChange={(e) => setNewName(e.target.value)}
                                    placeholder="Topic name"
                                    className="flex-1 bg-black/30 border border-white/10 rounded-lg px-3 py-2.5 text-white text-sm focus:border-orange-400/50 outline-none"
                                    autoFocus
                                    maxLength={120}
                                />
                                <select
                                    value={newPriority}
                                    onChange={(e) => setNewPriority(Number(e.target.value))}
                                    className="w-24 bg-surfaceHighlight border border-white/10 rounded-lg px-2 py-2.5 text-sm text-white focus:border-orange-400/50 outline-none"
                                    title="Priority"
                                >
                                    <option value={1}>P1</option>
                                    <option value={2}>P2</option>
                                    <option value={3}>P3</option>
                                </select>
                            </div>
                            <div className="flex gap-2">
                                <button
                                    type="button"
                                    onClick={() => setIsAdding(false)}
                                    className="flex-1 py-2.5 rounded-lg border border-white/10 text-textMuted hover:text-white hover:bg-white/5 transition-all uppercase text-xs font-bold tracking-wider"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={!newName.trim() || busy}
                                    className="flex-1 py-2.5 rounded-lg bg-primary hover:bg-orange-500 text-white transition-all uppercase text-xs font-bold tracking-wider disabled:opacity-50"
                                >
                                    Add topic
                                </button>
                            </div>
                        </form>
                    ) : (
                        <button
                            onClick={() => setIsAdding(true)}
                            className="w-full flex items-center justify-center gap-2 py-3 rounded-xl border border-dashed border-white/15 text-textMuted hover:text-white hover:border-orange-400/40 hover:bg-orange-500/5 transition-colors text-sm font-semibold"
                        >
                            <Plus size={16} /> Add topic
                        </button>
                    )}
                </div>
            </motion.aside>
        </AnimatePresence>
    );
};

export default RoadmapDetail;
