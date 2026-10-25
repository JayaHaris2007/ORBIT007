import { motion } from 'framer-motion';

const WeekdayChart = ({ data }) => {
    return (
        <div className="h-48 flex items-end justify-between gap-4">
            {data.map((day, index) => (
                <div key={day.day} className="flex flex-col items-center flex-1 group">
                    <div className="relative w-full flex justify-center h-full items-end">
                        <motion.div
                            initial={{ height: 0 }}
                            animate={{ height: `${day.normalizedHeight || 1}%` }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                            className="w-full max-w-[40px] bg-orange-500/20 hover:bg-orange-500/40 rounded-t-sm transition-colors relative min-h-[4px] border-t border-orange-500/50"
                        >
                            <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black border border-white/10 px-2 py-1 rounded text-[10px] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                                {day.count} total
                            </div>
                        </motion.div>
                    </div>
                    <span className="text-[10px] uppercase text-textMuted mt-2 font-mono">{day.day}</span>
                </div>
            ))}
        </div>
    );
};

export default WeekdayChart;
