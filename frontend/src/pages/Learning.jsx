import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useAppData } from '../context/AppDataContext';
import { learningService } from '../services/learningService';
import {
    Plus, GraduationCap, Map, Target, Sparkles, Trash2,
    BookOpen, TrendingUp, CheckCircle2,
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import CircularProgress from '../components/CircularProgress';
import ConfirmationModal from '../components/ConfirmationModal';
import RoadmapDetail from '../components/learning/RoadmapDetail';
import SuggestedLinks from '../components/learning/SuggestedLinks';

const CATEGORIES = ['academic', 'technical', 'personal'];

const categoryStyle = (category) => {
    if (category === 'academic') return 'bg-sky-500/15 text-sky-300 border-sky-400/20';
    if (category === 'personal') return 'bg-emerald-500/15 text-emerald-300 border-emerald-400/20';
    return 'bg-orange-500/15 text-orange-300 border-orange-400/20';
};

const statusDot = {
    learning: 'bg-amber-400',
    completed: 'bg-emerald-400',
    not_started: 'bg-white/25',
    on_hold: 'bg-red-400',
};

const Learning = () => {
    const { currentUser } = useAuth();
    const { revision } = useAppData();

    const [progress, setProgress] = useState(null);
    const [roadmaps, setRoadmaps] = useState([]);
    const [nextTopics, setNextTopics] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    // Roadmap creation + deletion modals
    const [isCreateOpen, setIsCreateOpen] = useState(false);
    const [newName, setNewName] = useState('');
    const [newCategory, setNewCategory] = useState('personal');
    const [roadmapToDelete, setRoadmapToDelete] = useState(null);
    const [selectedRoadmap, setSelectedRoadmap] = useState(null);
    const [busy, setBusy] = useState(false);

    const loadAll = useCallback(async () => {
        try {
            setError('');
            const [progressData, roadmapList, nextList] = await Promise.all([
                learningService.getProgress(),
                learningService.listRoadmaps(),
                learningService.getNext(),
            ]);
            setProgress(progressData);
            setRoadmaps(roadmapList);
            setNextTopics(nextList);
        } catch (loadError) {
            console.error('Error loading learning data:', loadError);
            setError(loadError.message || 'Could not load the learning tracker.');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (currentUser) loadAll();
    }, [currentUser, revision, loadAll]);

    const handleCreateRoadmap = async (e) => {
        e.preventDefault();
        if (!newName.trim() || busy) return;
        setBusy(true);
        try {
            await learningService.createRoadmap({ name: newName.trim(), category: newCategory });
            setNewName('');
            setNewCategory('personal');
            setIsCreateOpen(false);
            await loadAll();
        } catch (createError) {
            setError(createError.message);
        } finally {
            setBusy(false);
        }
    };

    const handleDeleteRoadmap = async () => {
        if (!roadmapToDelete || busy) return;
        setBusy(true);
        try {
            await learningService.deleteRoadmap(roadmapToDelete.id);
            setRoadmapToDelete(null);
            await loadAll();
        } catch (deleteError) {
            setError(deleteError.message);
        } finally {
            setBusy(false);
        }
    };

    const handleCompleteTopic = async (topic) => {
        try {
            await learningService.updateTopic(topic.topicId, { status: 'completed' });
            await loadAll();
        } catch (completeError) {
            setError(completeError.message);
        }
    };

    if (loading) {
        return (
            <div className="space-y-8 pb-20">
                <h1 className="text-3xl font-bold text-white uppercase tracking-widest glow-white">Learning</h1>
                <div className="glass-card p-12 text-center text-textMuted">Loading your roadmaps…</div>
            </div>
        );
    }

    return (
        <div className="space-y-8 pb-20">
            <div className="flex items-center justify-between gap-4 flex-wrap">
                <div className="flex items-center gap-3">
                    <h1 className="text-3xl font-bold text-white uppercase tracking-widest glow-white">
                        Learning
                    </h1>
                    {learningService.isMock && (
                        <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-1 rounded-full bg-white/5 border border-white/10 text-textMuted">
                            mock mode
                        </span>
                    )}
                </div>
                <button
                    onClick={() => setIsCreateOpen(true)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-orange-500/90 text-white text-sm font-semibold hover:bg-orange-500 transition-colors shadow-[0_0_20px_rgba(255,106,26,0.25)]"
                >
                    <Plus size={16} /> New roadmap
                </button>
            </div>

            {error && (
                <div className="glass-card p-4 border-red-500/30 text-red-300 text-sm">
                    {error}
                </div>
            )}

            {/* Progress overview */}
            {progress && (
                <div className="glass-card p-6 flex flex-col md:flex-row items-center gap-8">
                    <CircularProgress value={progress.completionPercent} max={100} size={132} />
                    <div className="flex-1 grid grid-cols-2 sm:grid-cols-4 gap-6 text-center md:text-left">
                        <div>
                            <p className="text-3xl font-bold text-white">{progress.roadmapCount}</p>
                            <p className="text-xs uppercase tracking-widest text-textMuted mt-1">Roadmaps</p>
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-white">{progress.topicCount}</p>
                            <p className="text-xs uppercase tracking-widest text-textMuted mt-1">Topics</p>
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-emerald-400">{progress.completedTopics}</p>
                            <p className="text-xs uppercase tracking-widest text-textMuted mt-1">Completed</p>
                        </div>
                        <div>
                            <p className="text-3xl font-bold text-amber-400">{progress.inProgressTopics}</p>
                            <p className="text-xs uppercase tracking-widest text-textMuted mt-1">In progress</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Roadmaps grid */}
            <section className="space-y-4">
                <div className="flex items-center gap-2 text-white/80">
                    <Map size={18} className="text-orange-400" />
                    <h2 className="text-lg font-semibold uppercase tracking-widest">Roadmaps</h2>
                </div>
                {roadmaps.length === 0 ? (
                    <div className="glass-card p-10 text-center space-y-3">
                        <div className="p-3 bg-white/5 rounded-full w-fit mx-auto text-textMuted"><GraduationCap size={24} /></div>
                        <p className="text-textMuted">No roadmaps yet — create your first learning path.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                        <AnimatePresence>
                            {roadmaps.map((roadmap) => (
                                <motion.div
                                    key={roadmap.id}
                                    layout
                                    initial={{ opacity: 0, y: 12 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, scale: 0.96 }}
                                    className="glass-card p-6 flex flex-col gap-4 group hover:border-white/20 transition-colors cursor-pointer"
                                    onClick={() => setSelectedRoadmap(roadmap)}
                                >
                                    <div className="flex items-start justify-between gap-3">
                                        <div className="flex items-center gap-3 min-w-0">
                                            <div className="p-2.5 bg-orange-500/15 text-orange-300 rounded-xl shrink-0">
                                                <BookOpen size={18} />
                                            </div>
                                            <div className="min-w-0">
                                                <h3 className="font-semibold text-white truncate">{roadmap.name}</h3>
                                                <span className={`text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full border ${categoryStyle(roadmap.category)}`}>
                                                    {roadmap.category}
                                                </span>
                                            </div>
                                        </div>
                                        <button
                                            onClick={(e) => { e.stopPropagation(); setRoadmapToDelete(roadmap); }}
                                            className="text-textMuted hover:text-red-400 transition-colors shrink-0"
                                            title="Delete roadmap"
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>

                                    <p className="text-sm text-textMuted line-clamp-2 flex-1">{roadmap.description}</p>

                                    <div className="space-y-2">
                                        <div className="flex justify-between text-xs text-textMuted">
                                            <span>{roadmap.completedCount} / {roadmap.topicCount} topics</span>
                                            <span>{roadmap.completionPercent}%</span>
                                        </div>
                                        <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                                            <div
                                                className="h-full rounded-full bg-gradient-to-r from-orange-500 to-amber-400 transition-all duration-500"
                                                style={{ width: `${roadmap.completionPercent}%` }}
                                            />
                                        </div>
                                    </div>

                                    {roadmap.nextTopic && (
                                        <div className="flex items-center gap-2 text-xs text-textMuted bg-white/5 rounded-lg px-3 py-2 border border-white/5">
                                            <Sparkles size={14} className="text-orange-400 shrink-0" />
                                            <span className="truncate">Next: <span className="text-white">{roadmap.nextTopic}</span></span>
                                        </div>
                                    )}
                                </motion.div>
                            ))}
                        </AnimatePresence>
                    </div>
                )}
            </section>

            {/* Suggested task links review queue */}
            <SuggestedLinks onChanged={loadAll} />

            {/* Next topics */}
            <section className="space-y-4">
                <div className="flex items-center gap-2 text-white/80">
                    <Target size={18} className="text-orange-400" />
                    <h2 className="text-lg font-semibold uppercase tracking-widest">Ready to learn next</h2>
                </div>
                {nextTopics.length === 0 ? (
                    <div className="glass-card p-10 text-center space-y-3">
                        <div className="p-3 bg-white/5 rounded-full w-fit mx-auto text-textMuted"><TrendingUp size={24} /></div>
                        <p className="text-textMuted">Nothing queued — complete a topic to unlock the next one.</p>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {nextTopics.map((topic) => (
                            <motion.div
                                key={topic.topicId}
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card p-4 flex items-center justify-between gap-4 hover:border-white/20 transition-colors"
                            >
                                <div className="flex items-center gap-3 min-w-0">
                                    <span className={`w-2 h-2 rounded-full shrink-0 ${statusDot[topic.status] || statusDot.not_started}`} />
                                    <div className="min-w-0">
                                        <p className="font-medium text-white truncate">{topic.name}</p>
                                        <p className="text-xs text-textMuted truncate">{topic.roadmapName} · {topic.reason}</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => handleCompleteTopic(topic)}
                                    className="shrink-0 p-2 rounded-lg bg-white/5 text-textMuted hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
                                    title="Mark as completed"
                                >
                                    <CheckCircle2 size={18} />
                                </button>
                            </motion.div>
                        ))}
                    </div>
                )}
            </section>

            {/* Create roadmap modal */}
            <AnimatePresence>
                {isCreateOpen && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="glass-card w-full max-w-md border border-white/10 p-6 relative"
                        >
                            <div className="absolute top-0 left-0 w-full h-1 bg-primary" />
                            <h3 className="text-xl font-bold text-white uppercase tracking-wide mb-4">New roadmap</h3>
                            <form onSubmit={handleCreateRoadmap} className="space-y-4">
                                <div>
                                    <label className="text-[10px] uppercase font-bold text-textMuted mb-2 block">Name</label>
                                    <input
                                        type="text"
                                        value={newName}
                                        onChange={(e) => setNewName(e.target.value)}
                                        placeholder="e.g. Machine Learning"
                                        className="w-full bg-black/30 border border-white/10 rounded-lg px-3 py-2.5 text-white text-sm focus:border-orange-400/50 outline-none transition-colors"
                                        autoFocus
                                    />
                                </div>
                                <div>
                                    <label className="text-[10px] uppercase font-bold text-textMuted mb-2 block">Category</label>
                                    <div className="flex gap-2">
                                        {CATEGORIES.map((category) => (
                                            <button
                                                key={category}
                                                type="button"
                                                onClick={() => setNewCategory(category)}
                                                className={`flex-1 py-2 rounded-lg border text-xs uppercase font-bold tracking-wider transition-colors ${newCategory === category
                                                    ? 'border-orange-400/60 bg-orange-500/15 text-orange-300'
                                                    : 'border-white/10 text-textMuted hover:text-white hover:bg-white/5'
                                                    }`}
                                            >
                                                {category}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="flex gap-3 pt-2">
                                    <button
                                        type="button"
                                        onClick={() => setIsCreateOpen(false)}
                                        className="flex-1 py-3 rounded-lg border border-white/10 text-textMuted hover:text-white hover:bg-white/5 transition-all uppercase text-xs font-bold tracking-wider"
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="submit"
                                        disabled={!newName.trim() || busy}
                                        className="flex-1 py-3 rounded-lg bg-primary hover:bg-orange-500 text-white transition-all uppercase text-xs font-bold tracking-wider disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {busy ? 'Creating…' : 'Create'}
                                    </button>
                                </div>
                            </form>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>

            {/* Delete roadmap confirmation */}
            <ConfirmationModal
                isOpen={Boolean(roadmapToDelete)}
                onClose={() => setRoadmapToDelete(null)}
                onConfirm={handleDeleteRoadmap}
                title="Delete roadmap"
                message={`Delete "${roadmapToDelete?.name ?? ''}"? Its topics and links will be removed too.`}
                type="danger"
                confirmString={roadmapToDelete?.name}
            />

            {/* Roadmap detail panel */}
            {selectedRoadmap && (
                <RoadmapDetail
                    roadmap={selectedRoadmap}
                    onClose={() => setSelectedRoadmap(null)}
                    onChanged={loadAll}
                />
            )}
        </div>
    );
};

export default Learning;
