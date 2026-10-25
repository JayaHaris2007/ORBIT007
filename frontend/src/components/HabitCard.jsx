import { motion } from 'framer-motion';
import { Check, Flame, Trash2, Pencil } from 'lucide-react';
import { format } from 'date-fns';
import * as Icons from 'lucide-react';

const HabitCard = ({ habit, onToggle, onDelete, onEdit }) => {
    const today = format(new Date(), 'yyyy-MM-dd');
    const isCompletedToday = habit.completedDates?.includes(today);

    // Dynamic Icon
    const IconComponent = Icons[habit.icon] || Icons.Activity;

    return (
        <motion.div
            layout
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className={`glass-card relative overflow-hidden group transition-all duration-500 ${isCompletedToday
                ? 'border-white/20 bg-white/5'
                : 'border-white/5'
                }`}
        >
            <div className="flex items-center justify-between relative z-10">
                <div className="flex items-center space-x-4">
                    <div className={`p-3 rounded-full transition-all duration-500 ${isCompletedToday ? 'bg-white text-black' : 'bg-surfaceHighlight text-textMuted'
                        }`}>
                        <IconComponent size={24} strokeWidth={1.5} />
                    </div>

                    <div>
                        <h3 className={`font-semibold text-lg tracking-wide transition-colors duration-300 ${isCompletedToday ? 'text-white' : 'text-text'}`}>
                            {habit.name}
                        </h3>
                        <div className="flex items-center text-xs text-textMuted space-x-2 mt-0.5">
                            <Flame size={12} className={habit.streak > 0 ? 'text-white' : 'text-textMuted'} />
                            <span className={habit.streak > 0 ? 'text-white font-medium' : ''}>{habit.streak} day streak</span>
                        </div>
                    </div>
                </div>

                <div className="flex items-center space-x-2">
                    <button
                        onClick={() => onEdit(habit)}
                        className="p-2 text-textMuted hover:text-white opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
                        aria-label={`Edit ${habit.name}`}
                    >
                        <Pencil size={16} />
                    </button>
                    <button
                        onClick={() => onToggle(habit)}
                        className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 transform active:scale-95 border ${isCompletedToday
                            ? 'bg-white border-white text-black shadow-[0_0_15px_rgba(255,255,255,0.3)]'
                            : 'bg-transparent border-white/10 text-white/10 hover:border-white/30 hover:text-white/50'
                            }`}
                    >
                        <Check size={20} strokeWidth={3} />
                    </button>

                    <button
                        onClick={() => onDelete(habit.id)}
                        className="p-2 text-textMuted hover:text-red-400 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity"
                    >
                        <Trash2 size={18} />
                    </button>
                </div>
            </div>

            {/* Glow Effect */}
            {isCompletedToday && (
                <motion.div
                    layoutId={`highlight-${habit.id}`}
                    className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent pointer-events-none"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5 }}
                />
            )}
        </motion.div>
    );
};

export default HabitCard;
