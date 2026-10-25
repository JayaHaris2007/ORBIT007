import { apiRequest } from './api';

const normalizeHabit = (habit) => ({
    ...habit,
    completedDates: habit.completed_dates || [],
    dailyGoal: habit.daily_goal || 1
});

// The assistant panel answers in whichever conversation is active.
const activeChatId = () => (
    localStorage.getItem('spidey_active_chat_id')
    || localStorage.getItem('jarvis_active_chat_id')
    || undefined
);

export const habitService = {
    getHabits: async () => {
        const data = await apiRequest('/api/habits');
        return data.habits.map(normalizeHabit);
    },
    addHabit: async (_userId, habitData) => normalizeHabit(await apiRequest('/api/habits', {
        method: 'POST',
        body: JSON.stringify({
            name: habitData.name,
            icon: habitData.icon || 'Activity',
            daily_goal: habitData.dailyGoal || 1
        })
    })),
    updateHabit: async (habitId, habitData) => normalizeHabit(await apiRequest(`/api/habits/${habitId}`, {
        method: 'PATCH',
        body: JSON.stringify({
            name: habitData.name,
            icon: habitData.icon,
            daily_goal: habitData.dailyGoal
        })
    })),
    toggleHabitCompletion: async (habitId, _userId, dateString, currentCompletedDates = []) => {
        const completed = !currentCompletedDates.includes(dateString);
        return normalizeHabit(await apiRequest(`/api/habits/${habitId}/completions/${dateString}`, {
            method: 'PUT',
            body: JSON.stringify({ completed, chat_id: activeChatId() })
        }));
    },
    deleteHabit: async (habitId) => apiRequest(`/api/habits/${habitId}`, { method: 'DELETE' }),
    getGamification: async () => {
        const data = await apiRequest('/api/gamification');
        return { ...data, globalStreak: data.global_streak || 0 };
    }
};
