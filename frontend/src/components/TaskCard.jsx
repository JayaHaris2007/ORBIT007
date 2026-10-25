import { motion } from 'framer-motion';
import { Check, Trash2, Calendar, AlertCircle, Pencil } from 'lucide-react';
import { format } from 'date-fns';

const priorityStyles = {
    low: 'border-white/20 text-textMuted bg-white/5',
    medium: 'border-white/40 text-text bg-white/10',
    high: 'border-white text-white bg-white/20 shadow-[0_0_10px_rgba(255,255,255,0.1)]'
};

const TaskCard = ({ task, onToggle, onDelete, onEdit }) => {
    return (
        <motion.div
            layout
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className={`glass-card group flex items-start justify-between transition-opacity duration-300 ${task.completed ? 'opacity-40 grayscale' : ''
                }`}
        >
            <div className="flex items-start space-x-4 w-full">
                <button
                    onClick={() => onToggle(task.id, !task.completed)}
                    className={`mt-1 w-6 h-6 rounded-full border flex items-center justify-center transition-all duration-300 ${task.completed
                        ? 'bg-white border-white text-black'
                        : 'bg-transparent border-textMuted hover:border-white'
                        }`}
                >
                    {task.completed && <Check size={14} strokeWidth={3} />}
                </button>

                <div className="flex-1">
                    <h3 className={`font-semibold tracking-wide ${task.completed ? 'line-through text-textMuted' : 'text-white'}`}>
                        {task.title}
                    </h3>
                    {task.description && (
                        <p className="text-sm text-textMuted mt-1">{task.description}</p>
                    )}

                    <div className="flex items-center space-x-3 mt-3 text-xs uppercase tracking-wider">
                        {task.dueDate && (
                            <span className="flex items-center text-textMuted bg-surfaceHighlight border border-white/5 px-3 py-1 rounded-full">
                                <Calendar size={12} className="mr-1.5" />
                                {format(new Date(task.dueDate), 'MMM d')}
                            </span>
                        )}

                        <span className={`flex items-center px-3 py-1 rounded-full border ${priorityStyles[task.priority]}`}>
                            <AlertCircle size={12} className="mr-1.5" />
                            {task.priority}
                        </span>
                    </div>
                </div>
            </div>

            <div className="flex items-center">
                {onEdit && (
                    <button
                        onClick={() => onEdit(task)}
                        className="text-textMuted hover:text-white opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity p-2"
                        aria-label={`Edit ${task.title}`}
                    >
                        <Pencil size={16} />
                    </button>
                )}
                <button
                    onClick={() => onDelete(task.id)}
                    className="text-textMuted hover:text-red-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity p-2"
                    aria-label={`Delete ${task.title}`}
                >
                    <Trash2 size={16} />
                </button>
            </div>
        </motion.div>
    );
};

export default TaskCard;
