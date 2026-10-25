import { format, getDay, startOfYear, endOfYear, eachDayOfInterval, parseISO } from 'date-fns';

/**
 * Calculates the number of habits completed for each day in the given year.
 * @param {Array} habits - List of habit objects
 * @param {number} year - The year to analyze
 * @returns {Object} Map of date string (yyyy-MM-dd) to completion count
 */
export const calculateDailyCompletions = (habits, year) => {
    const dailyCounts = {};
    const startDate = startOfYear(new Date(year, 0, 1));
    const endDate = endOfYear(new Date(year, 0, 1));
    const daysInYear = eachDayOfInterval({ start: startDate, end: endDate });

    // Initialize all days with 0
    daysInYear.forEach(day => {
        dailyCounts[format(day, 'yyyy-MM-dd')] = 0;
    });

    habits.forEach(habit => {
        if (habit.completedDates) {
            habit.completedDates.forEach(dateStr => {
                const date = parseISO(dateStr);
                if (date.getFullYear() === year) {
                    if (dailyCounts[dateStr] !== undefined) {
                        dailyCounts[dateStr]++;
                    }
                }
            });
        }
    });

    return dailyCounts;
};

/**
 * Calculates completion rates for each month of the year.
 * @param {Array} habits - List of habit objects
 * @param {number} year - The year to analyze
 * @returns {Array} Array of objects { month: string, completionRate: number, totalCompleted: number }
 */
export const calculateMonthlyStats = (habits, year) => {
    const months = Array.from({ length: 12 }, (_, i) => ({
        index: i,
        name: format(new Date(year, i, 1), 'MMM'),
        totalCompleted: 0,
        totalOpportunities: 0
    }));

    // Calculate total opportunities per month (habits * days in month)
    // This is valid if the habit existed for the whole month. 
    // For simplicity, we assume habits existed all year or check creation date if available (but simpler is all habits * days).
    // A more precise way is to just count raw completions per month for now, as "rate" is tricky with varying start dates.
    // Let's do raw completions first, maybe normalize by number of habits?
    // User asked for "monthly performance", raw completions is a good start.

    habits.forEach(habit => {
        if (habit.completedDates) {
            habit.completedDates.forEach(dateStr => {
                const date = parseISO(dateStr);
                if (date.getFullYear() === year) {
                    months[date.getMonth()].totalCompleted++;
                }
            });
        }
    });

    // Find max to normalize for chart height
    const maxCompletions = Math.max(...months.map(m => m.totalCompleted), 1);

    return months.map(m => ({
        ...m,
        normalizedHeight: (m.totalCompleted / maxCompletions) * 100
    }));
};

/**
 * Calculates average completion counts by day of the week.
 * @param {Array} habits - List of habit objects
 * @returns {Array} Array of objects { day: string, count: number, normalizedHeight: number }
 */
export const calculateWeekdayStats = (habits) => {
    const weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    const counts = Array(7).fill(0);

    habits.forEach(habit => {
        if (habit.completedDates) {
            habit.completedDates.forEach(dateStr => {
                const dayIndex = getDay(parseISO(dateStr));
                counts[dayIndex]++;
            });
        }
    });

    const maxCount = Math.max(...counts, 1);

    return weekdays.map((day, index) => ({
        day,
        count: counts[index],
        normalizedHeight: (counts[index] / maxCount) * 100
    }));
};

/**
 * Calculates total completions for the entire year
 * @param {Array} habits 
 * @param {number} year 
 */
export const calculateYearlyTotal = (habits, year) => {
    let total = 0;
    habits.forEach(habit => {
        if (habit.completedDates) {
            habit.completedDates.forEach(dateStr => {
                if (parseISO(dateStr).getFullYear() === year) {
                    total++;
                }
            });
        }
    });
    return total;
}
