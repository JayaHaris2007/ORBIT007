import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BookOpen, GraduationCap } from 'lucide-react';
import { learningService } from '../services/learningService';
import CircularProgress from './CircularProgress';

// Compact learning snapshot for the dashboard: overall progress ring plus
// the recommended next topics. Self-contained — loads through the same
// learning service as the Learning page, so mock mode still applies.
const LearningWidget = () => {
    const navigate = useNavigate();
    const [progress, setProgress] = useState(null);
    const [next, setNext] = useState([]);

    useEffect(() => {
        Promise.all([learningService.getProgress(), learningService.getNext()])
            .then(([progressData, nextData]) => {
                setProgress(progressData);
                setNext(nextData.slice(0, 3));
            })
            .catch((error) => {
                console.error('Error loading learning widget:', error);
            });
    }, []);

    const percent = progress?.completionPercent ?? 0;

    return (
        <div className="w-full h-full min-h-[160px] bg-black/40 backdrop-blur-xl border border-white/10 rounded-3xl p-5 flex flex-col relative overflow-hidden group hover:border-white/20 transition-all">
            {/* Background Gradient */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-orange-500/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none" />

            <div className="flex justify-between items-center mb-4 z-10">
                <h3 className="text-sm font-bold text-white/90 uppercase tracking-widest flex items-center gap-2">
                    <GraduationCap size={14} className="text-orange-400" />
                    Learning
                </h3>
                <button
                    onClick={() => navigate('/learning')}
                    className="text-xs font-mono text-orange-200/60 hover:text-orange-300 transition-colors"
                >
                    open tracker →
                </button>
            </div>

            <div className="flex-1 flex items-center gap-4 z-10">
                <div className="shrink-0">
                    <CircularProgress value={percent} max={100} size={64} strokeWidth={6} />
                </div>
                <div className="flex-1 min-w-0 space-y-2">
                    {next.length > 0 ? (
                        next.map((topic) => (
                            <button
                                key={topic.topicId}
                                onClick={() => navigate('/learning')}
                                className="w-full flex items-center gap-2 text-left group/item"
                                title={topic.reason}
                            >
                                <BookOpen size={12} className="shrink-0 text-orange-400/70 group-hover/item:text-orange-400 transition-colors" />
                                <span className="text-sm text-white/80 font-medium truncate group-hover/item:text-white transition-colors">
                                    {topic.name}
                                </span>
                                <span className="ml-auto shrink-0 text-[10px] text-white/30 font-mono">
                                    {topic.roadmapName}
                                </span>
                            </button>
                        ))
                    ) : (
                        <p className="text-xs text-white/40 italic">
                            {progress?.topicCount ? 'Nothing queued — nice work.' : 'No roadmaps yet.'}
                        </p>
                    )}
                </div>
            </div>

            {progress && (
                <div className="mt-3 pt-3 border-t border-white/5 flex justify-between items-center z-10">
                    <span className="text-xs text-white/40 font-mono">
                        {progress.completedTopics}/{progress.topicCount} topics
                    </span>
                    <span className="text-xs text-white/40 font-mono">
                        {progress.inProgressTopics} in progress
                    </span>
                </div>
            )}
        </div>
    );
};

export default LearningWidget;
