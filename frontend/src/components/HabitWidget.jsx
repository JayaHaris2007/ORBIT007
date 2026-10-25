import { motion } from 'framer-motion';
import { format } from 'date-fns';
import * as Icons from 'lucide-react';
import { Check } from 'lucide-react';

const HabitWidget = ({ habits, onToggle }) => {
    const today = format(new Date(), 'yyyy-MM-dd');
    const todaysHabits = habits; // Assuming all habits are for today
    const uncompletedHabits = todaysHabits.filter(h => !h.completedDates?.includes(today));

    // Show top 3 uncompleted habits, or if all done, show a success message
    const displayHabits = uncompletedHabits.slice(0, 3);
    const allDone = uncompletedHabits.length === 0 && todaysHabits.length > 0;

    return (
        <div className="w-full h-full min-h-[160px] bg-black/40 backdrop-blur-xl border border-white/10 rounded-3xl p-5 flex flex-col relative overflow-hidden group hover:border-white/20 transition-all">
            {/* Background Gradient */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-orange-500/10 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none" />

            <div className="flex justify-between items-center mb-4 z-10">
                <h3 className="text-sm font-bold text-white/90 uppercase tracking-widest flex items-center gap-2">
                    <Icons.Zap size={14} className="text-orange-400" />
                    Habits
                </h3>
                <span className="text-xs font-mono text-orange-200/60">
                    {todaysHabits.length - uncompletedHabits.length}/{todaysHabits.length}
                </span>
            </div>

            <div className="flex-1 flex flex-col gap-3 z-10">
                {allDone ? (
                    <div className="flex-1 flex items-center justify-center text-center">
                        <div>
                            <div className="bg-orange-500/20 p-3 rounded-full inline-flex mb-2 text-orange-400">
                                <Check size={24} />
                            </div>
                            <p className="text-xs text-white/60 font-medium">All habits done for today</p>
                        </div>
                    </div>
                ) : displayHabits.length > 0 ? (
                    displayHabits.map(habit => {
                        const IconComponent = Icons[habit.icon] || Icons.Activity;
                        return (
                            <div key={habit.id} className="flex items-center justify-between group/item">
                                <div className="flex items-center gap-3">
                                    <div className="p-1.5 rounded-lg bg-white/5 text-white/70 group-hover/item:text-orange-400 group-hover/item:bg-orange-500/10 transition-colors">
                                        <IconComponent size={14} />
                                    </div>
                                    <span className="text-sm text-white/80 font-medium truncate max-w-[120px]">{habit.name}</span>
                                </div>
                                <button
                                    onClick={() => onToggle(habit)}
                                    className="w-6 h-6 rounded-full border border-white/10 flex items-center justify-center hover:bg-orange-500 hover:border-orange-500 transition-all text-transparent hover:text-white"
                                >
                                    <Check size={12} strokeWidth={3} />
                                </button>
                            </div>
                        );
                    })
                ) : (
                    <div className="flex-1 flex items-center justify-center text-white/30 text-xs italic">
                        No active habits
                    </div>
                )}
            </div>

            {uncompletedHabits.length > 3 && (
                <div className="text-xs text-center mt-2 text-white/30 font-medium">
                    + {uncompletedHabits.length - 3} more
                </div>
            )}
        </div>
    );
};

export default HabitWidget;
