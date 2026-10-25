import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useAppData } from '../context/AppDataContext';
import { habitService } from '../services/habitService';
import { format, subDays } from 'date-fns';
import { motion } from 'framer-motion';
import { Flame, Calendar, Trophy, BarChart3, Grid, ChevronLeft, ChevronRight } from 'lucide-react';
import YearlyHeatmap from '../components/analytics/YearlyHeatmap';
import MonthlyChart from '../components/analytics/MonthlyChart';
import WeekdayChart from '../components/analytics/WeekdayChart';
import { calculateDailyCompletions, calculateMonthlyStats, calculateWeekdayStats, calculateYearlyTotal } from '../utils/analyticsUtils';

const Analytics = () => {
    const { currentUser } = useAuth();
    const { revision } = useAppData();
    const [habits, setHabits] = useState([]);
    const [loading, setLoading] = useState(true);
    const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());

    // Generate last 7 days for weekly view
    const last7Days = Array.from({ length: 7 }, (_, i) => {
        const date = subDays(new Date(), 6 - i); // 6 days ago to today
        return {
            date,
            label: format(date, 'EEE'), // Mon, Tue...
            fullDate: format(date, 'yyyy-MM-dd')
        };
    });

    useEffect(() => {
        if (currentUser) {
            habitService.getHabits().then((data) => {
                setHabits(data);
                setLoading(false);
            }).catch((error) => {
                console.error('Error loading habits:', error);
                setLoading(false);
            });
        }
    }, [currentUser, revision]);

    // Calculate overall completion rate for the week
    const getWeeklyCompletionRate = () => {
        if (habits.length === 0) return 0;

        let totalOpportunities = habits.length * 7;
        let totalCompleted = 0;

        habits.forEach(habit => {
            last7Days.forEach(day => {
                if (habit.completedDates?.includes(day.fullDate)) {
                    totalCompleted++;
                }
            });
        });

        return Math.round((totalCompleted / totalOpportunities) * 100);
    };

    const getLongestStreak = () => {
        if (habits.length === 0) return 0;
        return Math.max(...habits.map(h => h.streak || 0));
    };

    const getTopHabit = () => {
        if (habits.length === 0) return null;
        return habits.reduce((prev, current) => (prev.streak > current.streak) ? prev : current);
    }

    if (loading) return <div className="p-10 text-center animate-pulse text-textMuted">Analyzing data...</div>;

    const topHabit = getTopHabit();
    const dailyCompletions = calculateDailyCompletions(habits, selectedYear);
    const monthlyStats = calculateMonthlyStats(habits, selectedYear);
    const weekdayStats = calculateWeekdayStats(habits);
    const yearlyTotal = calculateYearlyTotal(habits, selectedYear);

    return (
        <div className="space-y-8 pb-20">
            <h1 className="text-3xl font-bold text-white uppercase tracking-widest glow-white">
                Analytics
            </h1>

            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-4 gap-6">
                <div className="glass-card p-6 flex flex-col items-center justify-center text-center space-y-2 group hover:border-white/20 transition-colors">
                    <div className="p-3 bg-white/10 text-white rounded-full mb-2 group-hover:bg-white group-hover:text-black transition-colors">
                        <Trophy size={20} />
                    </div>
                    <h3 className="text-textMuted text-xs uppercase tracking-widest">Weekly Sync</h3>
                    <p className="text-2xl font-bold text-white font-mono">{getWeeklyCompletionRate()}%</p>
                </div>

                <div className="glass-card p-6 flex flex-col items-center justify-center text-center space-y-2 group hover:border-white/20 transition-colors">
                    <div className="p-3 bg-white/10 text-white rounded-full mb-2 group-hover:bg-white group-hover:text-black transition-colors">
                        <Flame size={20} />
                    </div>
                    <h3 className="text-textMuted text-xs uppercase tracking-widest">Max Streak</h3>
                    <p className="text-2xl font-bold text-white font-mono">{getLongestStreak()} days</p>
                </div>

                <div className="glass-card p-6 flex flex-col items-center justify-center text-center space-y-2 group hover:border-white/20 transition-colors">
                    <div className="p-3 bg-white/10 text-white rounded-full mb-2 group-hover:bg-white group-hover:text-black transition-colors">
                        <Calendar size={20} />
                    </div>
                    <h3 className="text-textMuted text-xs uppercase tracking-widest">Yearly Activity</h3>
                    <p className="text-2xl font-bold text-white font-mono">{yearlyTotal}</p>
                </div>

                <div className="glass-card p-6 flex flex-col items-center justify-center text-center space-y-2 group hover:border-white/20 transition-colors">
                    <div className="p-3 bg-white/10 text-white rounded-full mb-2 group-hover:bg-white group-hover:text-black transition-colors">
                        <BarChart3 size={20} />
                    </div>
                    <h3 className="text-textMuted text-xs uppercase tracking-widest">Habit you&apos;re strongest with</h3>
                    <p className="text-lg font-bold text-white truncate w-full px-2">{topHabit ? topHabit.name : 'N/A'}</p>
                </div>
            </div>

            {/* Yearly Heatmap */}
            <div className="glass-card">
                <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xs font-bold uppercase tracking-widest px-2 text-textMuted flex items-center gap-2">
                        <Grid size={14} />
                        Yearly Overview
                    </h3>
                    <div className="flex items-center gap-3 text-xs text-textMuted font-mono">
                        <button type="button" aria-label="Previous year" onClick={() => setSelectedYear((year) => year - 1)} className="rounded-lg border border-white/10 p-2 text-textMuted hover:border-orange-400/30 hover:text-orange-300"><ChevronLeft size={14} /></button>
                        <span>{selectedYear}</span>
                        <button type="button" aria-label="Next year" onClick={() => setSelectedYear((year) => Math.min(new Date().getFullYear(), year + 1))} disabled={selectedYear >= new Date().getFullYear()} className="rounded-lg border border-white/10 p-2 text-textMuted hover:border-orange-400/30 hover:text-orange-300 disabled:opacity-30"><ChevronRight size={14} /></button>
                    </div>
                </div>
                <YearlyHeatmap dailyData={dailyCompletions} year={selectedYear} />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Monthly Performance */}
                <div className="glass-card">
                    <h3 className="text-xs font-bold uppercase tracking-widest mb-2 px-2 text-textMuted">Monthly Volume</h3>
                    <MonthlyChart data={monthlyStats} />
                </div>

                {/* Weekday Performance */}
                <div className="glass-card">
                    <h3 className="text-xs font-bold uppercase tracking-widest mb-2 px-2 text-textMuted">Day Consistency</h3>
                    <WeekdayChart data={weekdayStats} />
                </div>
            </div>

            {/* Weekly Heatmap Table */}
            <div className="glass-card overflow-x-auto">
                <h3 className="text-xs font-bold uppercase tracking-widest mb-6 px-2 text-textMuted">Weekly Progression</h3>

                <table className="w-full min-w-[600px]">
                    <thead>
                        <tr>
                            <th className="text-left py-3 px-4 text-textMuted font-medium w-1/3 text-xs uppercase tracking-wider">Habit</th>
                            {last7Days.map(day => (
                                <th key={day.fullDate} className="text-center py-3 px-2 text-textMuted font-medium">
                                    <div className="flex flex-col items-center">
                                        <span className="text-[10px] uppercase tracking-wider">{day.label}</span>
                                        <span className={`text-xs mt-1 px-1.5 rounded font-mono ${day.fullDate === format(new Date(), 'yyyy-MM-dd') ? 'bg-white text-black' : ''
                                            }`}>
                                            {format(day.date, 'd')}
                                        </span>
                                    </div>
                                </th>
                            ))}
                            <th className="text-center py-3 px-4 text-textMuted font-medium text-xs uppercase tracking-wider">Streak</th>
                        </tr>
                    </thead>
                    <tbody>
                        {habits.map(habit => (
                            <tr key={habit.id} className="border-t border-white/5 hover:bg-white/5 transition-colors">
                                <td className="py-4 px-4 font-medium text-white">{habit.name}</td>

                                {last7Days.map(day => {
                                    const isCompleted = habit.completedDates?.includes(day.fullDate);
                                    return (
                                        <td key={day.fullDate} className="py-4 px-2 text-center">
                                            <div className="flex justify-center">
                                                <motion.div
                                                    initial={{ scale: 0.8, opacity: 0 }}
                                                    animate={{ scale: 1, opacity: 1 }}
                                                    className={`w-6 h-6 rounded flex items-center justify-center transition-all ${isCompleted
                                                        ? 'bg-white shadow-[0_0_10px_rgba(255,255,255,0.3)]'
                                                        : 'bg-white/5 border border-white/5'
                                                        }`}
                                                >
                                                    {isCompleted && <div className="w-2 h-2 bg-black rounded-full" />}
                                                </motion.div>
                                            </div>
                                        </td>
                                    );
                                })}
                                <td className="py-4 px-4 text-center">
                                    <span className="inline-flex items-center px-2 py-1 rounded bg-white/5 border border-white/5 text-xs text-white uppercase tracking-wider font-mono">
                                        <Flame size={10} className="text-white mr-1.5" />
                                        {habit.streak}
                                    </span>
                                </td>
                            </tr>
                        ))}
                        {habits.length === 0 && (
                            <tr>
                                <td colSpan={9} className="py-8 text-center text-textMuted text-sm italic">
                                    No habits initialized.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Analytics;
