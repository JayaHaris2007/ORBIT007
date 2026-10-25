import { eachDayOfInterval, format, getDay, startOfYear, endOfYear } from 'date-fns';
import { motion } from 'framer-motion';

const YearlyHeatmap = ({ dailyData, year = new Date().getFullYear() }) => {
    const startDate = startOfYear(new Date(year, 0, 1));
    const endDate = endOfYear(new Date(year, 0, 1));
    const days = eachDayOfInterval({ start: startDate, end: endDate });

    // Group days by week for the grid layout
    // We need to pad the first week if it doesn't start on Sunday
    const startDayOfWeek = getDay(startDate);
    const paddedDays = Array(startDayOfWeek).fill(null).concat(days);

    const weeks = [];
    let currentWeek = [];

    paddedDays.forEach((day, index) => {
        if (day) {
            currentWeek.push(day);
        } else {
            currentWeek.push(null);
        }

        if ((index + 1) % 7 === 0 || index === paddedDays.length - 1) {
            weeks.push(currentWeek);
            currentWeek = [];
        }
    });

    // Determine color intensity based on count
    const getColor = (count) => {
        if (!count) return 'bg-white/5';
        if (count === 1) return 'bg-green-900/40';
        if (count === 2) return 'bg-green-700/60';
        if (count === 3) return 'bg-green-500/80';
        return 'bg-green-400';
    };

    return (
        <div className="overflow-x-auto pb-4">
            <div className="min-w-[800px]">
                <div className="flex text-xs text-textMuted mb-2">
                    {/* Simplified month labels logic could be added here */}
                    <span className="w-8"></span>
                    {/* Placeholder for y-axis labels if needed, or just let headers float */}
                </div>

                <div className="flex gap-1">
                    {/* Column for Day Labels (Mon, Wed, Fri) */}
                    <div className="flex flex-col gap-1 text-[10px] text-textMuted justify-between py-1 pr-2">
                        <span>Sun</span>
                        <span>&nbsp;</span>
                        <span>Tue</span>
                        <span>&nbsp;</span>
                        <span>Thu</span>
                        <span>&nbsp;</span>
                        <span>Sat</span>
                    </div>

                    {/* Heatmap Grid - Rendered Column by Column (Weeks) */}
                    {/* Actually, github renders columns as weeks. My `weeks` array is row-based if I mapped it standardly.
                        Correction: GitHub graph is columns = weeks, rows = days (Sun-Sat).
                        My `paddedDays` is a flat list. 
                        Let's render a Grid with 7 rows and 53 columns.
                    */}

                    <div className="grid grid-cols-[repeat(53,minmax(0,1fr))] gap-1 flex-1">
                        {Array.from({ length: 53 }).map((_, weekIndex) => (
                            <div key={weekIndex} className="flex flex-col gap-1">
                                {Array.from({ length: 7 }).map((_, dayIndex) => {
                                    // Calculate the date index in the flattened array
                                    // But mapped differently: week-major traversal?
                                    // date-fns `eachDay` is sequential.
                                    // We need to figure out which date corresponds to weekIndex, dayIndex.
                                    // Ideally, iterate days and place them in columns.

                                    const dayOffset = (weekIndex * 7) + dayIndex - startDayOfWeek;
                                    const day = days[dayOffset];

                                    if (!day || dayOffset < 0 || dayOffset >= days.length) {
                                        return <div key={dayIndex} className="w-3 h-3" />;
                                    }

                                    const dateStr = format(day, 'yyyy-MM-dd');
                                    const count = dailyData[dateStr] || 0;

                                    return (
                                        <motion.div
                                            key={dateStr}
                                            whileHover={{ scale: 1.2 }}
                                            className={`w-3 h-3 rounded-sm ${getColor(count)} relative group cursor-pointer`}
                                            title={`${dateStr}: ${count} completions`}
                                        >
                                            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block bg-black text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10 border border-white/10">
                                                {format(day, 'MMM d, yyyy')}: {count}
                                            </div>
                                        </motion.div>
                                    );
                                })}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default YearlyHeatmap;
