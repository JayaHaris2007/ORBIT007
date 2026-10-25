import { apiRequest } from './api';

const normalizeTask = (task) => ({
    ...task,
    completed: Boolean(task.completed),
    dueDate: task.due_at || null,
    description: task.description || ''
});

// The assistant panel answers in whichever conversation is active.
const activeChatId = () => (
    localStorage.getItem('spidey_active_chat_id')
    || localStorage.getItem('jarvis_active_chat_id')
    || undefined
);

export const taskService = {
    getTasks: async (includeCompleted = true) => {
        const data = await apiRequest(`/api/tasks?include_completed=${includeCompleted ? 1 : 0}`);
        return data.tasks.map(normalizeTask);
    },
    addTask: async (_userId, taskData) => normalizeTask(await apiRequest('/api/tasks', {
        method: 'POST',
        body: JSON.stringify({
            title: taskData.title,
            description: taskData.description || '',
            priority: taskData.priority || 'medium',
            due_at: taskData.dueDate || null
        })
    })),
    updateTask: async (taskId, taskData) => normalizeTask(await apiRequest(`/api/tasks/${taskId}`, {
        method: 'PATCH',
        body: JSON.stringify({
            title: taskData.title,
            description: taskData.description || '',
            priority: taskData.priority || 'medium',
            due_at: taskData.dueDate || null
        })
    })),
    toggleTaskCompletion: async (taskId, completed) => normalizeTask(await apiRequest(`/api/tasks/${taskId}`, {
        method: 'PATCH',
        body: JSON.stringify({ completed: Boolean(completed), chat_id: activeChatId() })
    })),
    deleteTask: async (taskId) => apiRequest(`/api/tasks/${taskId}`, { method: 'DELETE' })
};
