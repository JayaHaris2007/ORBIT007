import { motion } from 'framer-motion';

const MonthlyChart = ({ data }) => {
    return (
        <div className="h-64 flex items-end justify-between gap-2 pt-10">
            {data.map((month) => (
                <div key={month.index} className="flex flex-col items-center flex-1 group">
                    <div className="relative w-full flex justify-center h-full items-end">
                        <motion.div
                            initial={{ height: 0 }}
                            animate={{ height: `${month.normalizedHeight || 1}%` }}
                            transition={{ duration: 0.5, delay: month.index * 0.05 }}
                            className="w-full max-w-[30px] bg-white/10 hover:bg-white/30 rounded-t-sm transition-colors relative min-h-[4px]"
                        >
                            <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-black border border-white/10 px-2 py-1 rounded text-[10px] opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                                {month.totalCompleted} completions
                            </div>
                        </motion.div>
                    </div>
                    <span className="text-[10px] uppercase text-textMuted mt-2 font-mono">{month.name}</span>
                </div>
            ))}
        </div>
    );
};

export default MonthlyChart;
