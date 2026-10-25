import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useAppData } from '../context/AppDataContext';
import { taskService } from '../services/taskService';
import { Plus, Search } from 'lucide-react';
import TaskCard from '../components/TaskCard';
import TaskEditor from '../components/TaskEditor';
import { AnimatePresence } from 'framer-motion';

import ConfirmationModal from '../components/ConfirmationModal';

const Todos = () => {
    const { currentUser } = useAuth();
    const { revision } = useAppData();
    const [tasks, setTasks] = useState([]);
    const [editingTask, setEditingTask] = useState(null);
    const [filter, setFilter] = useState('all'); // all, pending, completed
    const [search, setSearch] = useState('');

    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [taskToDelete, setTaskToDelete] = useState(null);

    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!currentUser) return;
        taskService.getTasks(true).then((data) => {
            setTasks(data);
            setLoading(false);
        }).catch((error) => {
            console.error('Error loading tasks:', error);
            setLoading(false);
        });
    }, [currentUser, revision]);

    const filteredTasks = tasks.filter(task => {
        const matchesFilter =
            filter === 'all' ? true :
                filter === 'completed' ? task.completed :
                    !task.completed;

        const matchesSearch = task.title.toLowerCase().includes(search.toLowerCase());

        return matchesFilter && matchesSearch;
    });

    // Sort by completed (bottom) then priority (high > medium > low)
    const sortedTasks = filteredTasks.sort((a, b) => {
        if (a.completed === b.completed) {
            const p = { high: 3, medium: 2, low: 1 };
            return p[b.priority] - p[a.priority];
        }
        return a.completed ? 1 : -1;
    });

    const handleToggleTask = async (taskId, status) => {
        try {
            await taskService.toggleTaskCompletion(taskId, status);
            setTasks(await taskService.getTasks(true));
        } catch (error) {
            console.error("Error toggling task:", error);
        }
    };

    const handleSaveTask = async (values) => {
        if (editingTask) {
            const updated = await taskService.updateTask(editingTask.id, values);
            setTasks((current) => current.map((task) => (task.id === updated.id ? updated : task)));
            setEditingTask(updated);
        } else {
            const created = await taskService.addTask(currentUser?.uid, values);
            setTasks((current) => [created, ...current]);
        }
    };

    const confirmDeleteTask = async () => {
        if (taskToDelete) {
            try {
                await taskService.deleteTask(taskToDelete);
                setTasks(await taskService.getTasks(true));
                if (editingTask?.id === taskToDelete) setEditingTask(null);
                setTaskToDelete(null);
            } catch (error) {
                console.error(error);
            }
        }
    }

    if (loading) return <div className="p-10 text-center animate-pulse text-textMuted">Loading your tasks...</div>;

    return (
        <div className="pb-20">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-white uppercase tracking-widest glow-white">Your tasks</h1>
                    <p className="text-textMuted text-xs font-mono mt-1">Keep the important things moving.</p>
                </div>

                <button
                    onClick={() => setEditingTask(null)}
                    className="bg-white hover:bg-white/90 text-black px-4 py-2 rounded-lg font-bold shadow-[0_0_15px_rgba(255,255,255,0.3)] transition-all flex items-center space-x-2 uppercase tracking-wide text-sm"
                >
                    <Plus size={18} />
                    <span>New task</span>
                </button>
            </div>

            <div className="grid grid-cols-1 items-start gap-6 2xl:grid-cols-[minmax(0,1fr)_340px]">
                <div className="min-w-0">
                    {/* Controls */}
                    <div className="flex flex-col md:flex-row gap-4 mb-6">
                        <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-textMuted" size={18} />
                            <input
                                type="text"
                                placeholder="Search your tasks..."
                                className="w-full bg-surfaceHighlight border border-white/5 rounded-lg py-2 pl-10 pr-4 text-white focus:outline-none focus:border-white/30 transition-all placeholder:text-neutral-700"
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                            />
                        </div>

                        <div className="flex bg-surfaceHighlight p-1 rounded-lg border border-white/5">
                            {['all', 'pending', 'completed'].map((f) => (
                                <button
                                    key={f}
                                    onClick={() => setFilter(f)}
                                    className={`px-4 py-1.5 rounded-md text-xs font-bold uppercase tracking-wider transition-all ${filter === f
                                        ? 'bg-white text-black shadow-sm'
                                        : 'text-textMuted hover:text-white'
                                        }`}
                                >
                                    {f}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Task List */}
                    <div className="space-y-3">
                        <AnimatePresence>
                            {sortedTasks.map(task => (
                                <TaskCard
                                    key={task.id}
                                    task={task}
                                    onToggle={handleToggleTask}
                                    onEdit={setEditingTask}
                                    onDelete={(id) => {
                                        setTaskToDelete(id);
                                        setIsDeleteModalOpen(true);
                                    }}
                                />
                            ))}
                        </AnimatePresence>

                        {sortedTasks.length === 0 && (
                            <div className="text-center py-20 text-textMuted italic">
                                Nothing here yet. You&apos;re doing great.
                            </div>
                        )}
                    </div>
                </div>

                {/* Task editor, docked on the right of the page */}
                <div className="2xl:sticky 2xl:top-6">
                    <TaskEditor
                        task={editingTask}
                        onSave={handleSaveTask}
                        onCancel={() => setEditingTask(null)}
                    />
                </div>
            </div>

            {/* Confirmation Modal */}
            <ConfirmationModal
                isOpen={isDeleteModalOpen}
                onClose={() => setIsDeleteModalOpen(false)}
                onConfirm={confirmDeleteTask}
                title="Delete this task?"
                message="This task will be removed from your list."
            />
        </div>
    );
};

export default Todos;
