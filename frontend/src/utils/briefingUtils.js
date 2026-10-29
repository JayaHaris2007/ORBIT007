// Pure, deterministic helper functions for S.P.I.D.E.Y.'s time-based entry
// greeting, remaining workload summary, and "Where to Start" recommendation engine.

const PRIORITY_WEIGHT = {
    high: 3,
    medium: 2,
    low: 1,
};

/**
 * Returns time-of-day metadata based on a Date instance (or hour 0-23).
 * @param {Date|number} [nowInput]
 * @param {string} [userName]
 */
export function getTimeOfDayGreeting(nowInput = new Date(), userName = '') {
    const hour = typeof nowInput === 'number'
        ? nowInput
        : (nowInput instanceof Date && !Number.isNaN(nowInput.getTime()) ? nowInput.getHours() : 9);

    const cleanFirst = String(userName || '')
        .trim()
        .split(/\s+/)[0] || 'Commander';

    let period = 'morning';
    let salutation = 'Good morning';
    let atmosphere = 'Morning systems calibrated. Prime focus window is open.';
    let accentTone = 'amber';

    if (hour >= 5 && hour < 12) {
        period = 'morning';
        salutation = 'Good morning';
        atmosphere = 'Morning systems online. Tackle your highest-leverage work first.';
        accentTone = 'amber';
    } else if (hour >= 12 && hour < 17) {
        period = 'afternoon';
        salutation = 'Good afternoon';
        atmosphere = 'Midday momentum check. Keep your priority queue moving steadily.';
        accentTone = 'orange';
    } else if (hour >= 17 && hour < 22) {
        period = 'evening';
        salutation = 'Good evening';
        atmosphere = 'Evening cycle active. Wrap up key remaining tasks and lock in your streaks.';
        accentTone = 'rose';
    } else {
        period = 'night';
        salutation = 'Good evening';
        atmosphere = 'Late-hour session active. Focus on one quick win before winding down.';
        accentTone = 'indigo';
    }

    return {
        hour,
        period,
        salutation,
        firstName: cleanFirst,
        headline: `${salutation}, ${cleanFirst}`,
        atmosphere,
        accentTone,
    };
}

/**
 * Formats a Date into local YYYY-MM-DD key.
 * @param {Date} [date]
 */
export function formatDateKey(date = new Date()) {
    const d = date instanceof Date && !Number.isNaN(date.getTime()) ? date : new Date();
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * Sorts pending tasks by urgency:
 * 1) Overdue or due today first
 * 2) Priority (high > medium > low)
 * 3) Earliest due date
 */
export function rankPendingTasks(tasks = [], todayKey = formatDateKey()) {
    const pending = (Array.isArray(tasks) ? tasks : []).filter((t) => t && !t.completed);

    return [...pending].sort((a, b) => {
        const aDue = a.dueDate ? String(a.dueDate).slice(0, 10) : '';
        const bDue = b.dueDate ? String(b.dueDate).slice(0, 10) : '';

        const aUrgent = aDue && aDue <= todayKey ? 1 : 0;
        const bUrgent = bDue && bDue <= todayKey ? 1 : 0;
        if (aUrgent !== bUrgent) return bUrgent - aUrgent;

        const aPrio = PRIORITY_WEIGHT[String(a.priority || 'medium').toLowerCase()] || 2;
        const bPrio = PRIORITY_WEIGHT[String(b.priority || 'medium').toLowerCase()] || 2;
        if (aPrio !== bPrio) return bPrio - aPrio;

        if (aDue && bDue && aDue !== bDue) {
            return aDue.localeCompare(bDue);
        }
        if (aDue && !bDue) return -1;
        if (!aDue && bDue) return 1;

        return (a.id || 0) - (b.id || 0);
    });
}

/**
 * Builds the complete Assistant Entry Briefing & "Where to Start" recommendation.
 */
export function buildAssistantBriefing({
    tasks = [],
    habits = [],
    nextTopics = [],
    progress = null,
    now = new Date(),
    userName = 'Shadow Hunter',
} = {}) {
    const todayKey = formatDateKey(now);
    const greeting = getTimeOfDayGreeting(now, userName);

    const safeTasks = Array.isArray(tasks) ? tasks : [];
    const safeHabits = Array.isArray(habits) ? habits : [];
    const safeNextTopics = Array.isArray(nextTopics) ? nextTopics : [];

    const rankedTasks = rankPendingTasks(safeTasks, todayKey);
    const completedTasksCount = safeTasks.filter((t) => t && t.completed).length;
    const remainingTasksCount = rankedTasks.length;
    const highPriorityCount = rankedTasks.filter(
        (t) => String(t.priority || '').toLowerCase() === 'high'
    ).length;

    const pendingHabits = safeHabits.filter(
        (h) => h && !(Array.isArray(h.completedDates) && h.completedDates.includes(todayKey))
    );
    const completedHabitsCount = safeHabits.length - pendingHabits.length;
    const remainingHabitsCount = pendingHabits.length;

    const topTask = rankedTasks[0] || null;
    const topHabit = pendingHabits[0] || null;
    const topLearning = safeNextTopics[0] || null;

    // Build ordered starting steps (up to 3 concrete recommendations)
    const startingSteps = [];

    if (topTask) {
        const isHigh = String(topTask.priority || '').toLowerCase() === 'high';
        const dueKey = topTask.dueDate ? String(topTask.dueDate).slice(0, 10) : '';
        const isDueSoon = Boolean(dueKey && dueKey <= todayKey);
        const reason = isDueSoon
            ? `Due today (${dueKey}) — clear this first so nothing slips.`
            : isHigh
                ? 'Marked High Priority in your mission log.'
                : remainingTasksCount === 1
                    ? 'Your only remaining open task for today.'
                    : `Top of your ${remainingTasksCount} remaining tasks.`;

        startingSteps.push({
            id: `task-${topTask.id}`,
            kind: 'task',
            badge: isHigh ? 'Priority #1 Task' : 'Recommended Task',
            title: topTask.title,
            subtitle: topTask.description || reason,
            reason,
            route: '/todos',
            ctaLabel: 'Start with this task',
            completeLabel: 'Mark task done',
            entityId: topTask.id,
            priority: topTask.priority || 'medium',
            prompt: `Help me break down and start my task "${topTask.title}"`,
        });
    }

    if (topHabit) {
        const reason = remainingHabitsCount === 1
            ? 'Last unchecked daily habit to lock in a 100% habit day.'
            : `${remainingHabitsCount} daily habits still open today — quick win to build momentum.`;

        startingSteps.push({
            id: `habit-${topHabit.id}`,
            kind: 'habit',
            badge: 'Daily Habit Check-in',
            title: topHabit.name,
            subtitle: reason,
            reason,
            route: '/habits',
            ctaLabel: 'Check in habit',
            completeLabel: 'Complete habit today',
            entityId: topHabit.id,
            habitObj: topHabit,
            prompt: `Check my habits and help me stay consistent with "${topHabit.name}"`,
        });
    }

    if (topLearning) {
        const isInProgress = topLearning.status === 'learning';
        const reason = isInProgress
            ? `Already in progress in ${topLearning.roadmapName || 'your roadmap'} — keep the streak warm.`
            : topLearning.reason || `Unlocked next topic in ${topLearning.roadmapName || 'your roadmap'}.`;

        startingSteps.push({
            id: `learning-${topLearning.topicId}`,
            kind: 'learning',
            badge: isInProgress ? 'Continue Learning' : 'Next Unlocked Topic',
            title: topLearning.name,
            subtitle: `${topLearning.roadmapName || 'Learning'} · ${reason}`,
            reason,
            route: '/learning',
            ctaLabel: isInProgress ? 'Continue topic' : 'Start learning topic',
            completeLabel: 'Start topic now',
            entityId: topLearning.topicId,
            topicObj: topLearning,
            prompt: `I am starting ${topLearning.name}`,
        });
    }

    if (startingSteps.length === 0) {
        startingSteps.push({
            id: 'plan-fresh',
            kind: 'plan',
            badge: 'All Clear',
            title: 'Plan your next high-impact goal',
            subtitle: 'All tasks and daily habits are caught up. Ready to set a new target or explore a learning roadmap.',
            reason: 'Your queue is clean — a great moment to plan ahead.',
            route: '/dashboard',
            ctaLabel: 'Plan with S.P.I.D.E.Y.',
            completeLabel: 'Ask S.P.I.D.E.Y.',
            entityId: null,
            prompt: "What should I focus on next today?",
        });
    }

    const primaryStart = startingSteps[0];

    // Natural spoken / conversational summary line from S.P.I.D.E.Y.
    const statusParts = [];
    if (remainingTasksCount > 0) {
        statusParts.push(
            `${remainingTasksCount} remaining ${remainingTasksCount === 1 ? 'task' : 'tasks'}${
                highPriorityCount > 0 ? ` (${highPriorityCount} high-priority)` : ''
            }`
        );
    } else {
        statusParts.push('no pending tasks');
    }

    if (safeHabits.length > 0) {
        if (remainingHabitsCount > 0) {
            statusParts.push(
                `${remainingHabitsCount} ${remainingHabitsCount === 1 ? 'habit' : 'habits'} left for today`
            );
        } else {
            statusParts.push('all daily habits completed');
        }
    }

    const workloadClause = statusParts.join(' and ');

    let recommendationClause = '';
    if (primaryStart.kind === 'task') {
        recommendationClause = `I suggest we start with "${primaryStart.title}" (${primaryStart.reason.replace(/\.$/, '')}).`;
    } else if (primaryStart.kind === 'habit') {
        recommendationClause = `I suggest we kick off by completing your "${primaryStart.title}" habit.`;
    } else if (primaryStart.kind === 'learning') {
        recommendationClause = `Your tasks and habits are clear — let's jump into "${primaryStart.title}" in ${topLearning?.roadmapName || 'Learning'}.`;
    } else {
        recommendationClause = `Everything on your board is complete. Let's map out your next objective whenever you're ready.`;
    }

    const spokenSummary = `${greeting.salutation}, ${greeting.firstName}. You have ${workloadClause}. ${recommendationClause}`;

    return {
        greeting,
        stats: {
            remainingTasksCount,
            completedTasksCount,
            highPriorityCount,
            remainingHabitsCount,
            completedHabitsCount,
            totalHabitsCount: safeHabits.length,
            learningPercent: progress?.completionPercent ?? 0,
            inProgressTopics: progress?.inProgressTopics ?? 0,
            completedTopics: progress?.completedTopics ?? 0,
        },
        primaryStart,
        startingSteps,
        spokenSummary,
    };
}
