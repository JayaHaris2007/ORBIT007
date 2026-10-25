import { motion } from 'framer-motion';
import { format } from 'date-fns';
import { Check, CheckCircle, Target, Pencil } from 'lucide-react';

const TodoWidget = ({ tasks, onToggle, onEdit }) => {
    // Filter pending and sort by priority
    const pendingTasks = tasks.filter(t => !t.completed);
    const topTasks = pendingTasks
        .sort((a, b) => {
            const p = { high: 3, medium: 2, low: 1 };
            return p[b.priority] - p[a.priority];
        })
        .slice(0, 3); // Show top 3

    const allDone = pendingTasks.length === 0 && tasks.length > 0;

    return (
        <div className="w-full h-full min-h-[160px] bg-black/40 backdrop-blur-xl border border-white/10 rounded-3xl p-5 flex flex-col relative overflow-hidden group hover:border-white/20 transition-all">
            {/* Background Gradient */}
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-orange-500/10 rounded-full blur-3xl -ml-16 -mb-16 pointer-events-none" />

            <div className="flex justify-between items-center mb-4 z-10">
                <h3 className="text-sm font-bold text-white/90 uppercase tracking-widest flex items-center gap-2">
                    <Target size={14} className="text-orange-400" />
                    To-Do
                </h3>
                <span className="text-xs font-mono text-orange-200/60">
                    {pendingTasks.length} left
                </span>
            </div>

            <div className="flex-1 flex flex-col gap-3 z-10">
                {allDone ? (
                    <div className="flex-1 flex items-center justify-center text-center">
                        <div>
                            <div className="bg-orange-500/20 p-3 rounded-full inline-flex mb-2 text-orange-400">
                                <CheckCircle size={24} />
                            </div>
                            <p className="text-xs text-white/60 font-medium">You&apos;re all caught up</p>
                        </div>
                    </div>
                ) : topTasks.length > 0 ? (
                    topTasks.map(task => (
                        <div key={task.id} className="flex items-center justify-between group/item">
                            <div className="flex items-center gap-3 overflow-hidden">
                                <button
                                    onClick={() => onToggle(task.id, true)}
                                className="w-5 h-5 rounded-md border border-white/20 flex-shrink-0 flex items-center justify-center hover:bg-orange-500 hover:border-orange-500 transition-all group-hover/item:border-orange-500/50"
                                >
                                    <div className="w-full h-full opacity-0 hover:opacity-100 flex items-center justify-center text-white">
                                        <Check size={12} strokeWidth={3} />
                                    </div>
                                </button>
                                <div className="flex flex-col min-w-0">
                                    <span className="text-sm text-white/90 font-medium truncate">{task.title}</span>
                                    {task.dueDate && (
                                        <span className="text-[10px] text-white/40 font-mono">
                                            {format(new Date(task.dueDate), 'MMM d')}
                                        </span>
                                    )}
                                </div>
                            </div>
                            <div className="flex items-center gap-1 shrink-0">
                                {onEdit && (
                                    <button
                                        onClick={() => onEdit(task)}
                                        className="text-white/30 hover:text-white opacity-100 md:opacity-0 md:group-hover/item:opacity-100 transition-all p-1"
                                        aria-label={`Edit ${task.title}`}
                                    >
                                        <Pencil size={12} />
                                    </button>
                                )}
                                <div className={`w-1.5 h-1.5 rounded-full ${task.priority === 'high' ? 'bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]' :
                                        task.priority === 'medium' ? 'bg-yellow-500' : 'bg-orange-500'
                                    }`} />
                            </div>
                        </div>
                    ))
                ) : (
                    <div className="flex-1 flex items-center justify-center text-white/30 text-xs italic">
                        No pending tasks
                    </div>
                )}
            </div>
            {pendingTasks.length > 3 && (
                <div className="text-xs text-center mt-2 text-white/30 font-medium">
                    + {pendingTasks.length - 3} more
                </div>
            )}
        </div>
    );
};

export default TodoWidget;
