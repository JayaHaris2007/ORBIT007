import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';
import { useAppData } from '../context/AppDataContext';
import { habitService } from '../services/habitService';
import { taskService } from '../services/taskService';
import { learningService } from '../services/learningService';
import { format } from 'date-fns';
import { Zap, CheckCircle, Target, Focus, Trophy } from 'lucide-react';
import { motion } from 'framer-motion';
import HabitCard from '../components/HabitCard';
import TaskCard from '../components/TaskCard';
import CircularProgress from '../components/CircularProgress';
import AnimeQuote from '../components/AnimeQuote';
import ConfirmationModal from '../components/ConfirmationModal';
import HabitWidget from '../components/HabitWidget';
import TodoWidget from '../components/TodoWidget';
import LearningWidget from '../components/LearningWidget';
import TaskEditor from '../components/TaskEditor';
import AssistantEntryBriefing from '../components/AssistantEntryBriefing';

const STAGGER_CONTAINER = {
    hidden: { opacity: 0 },
    show: {
        opacity: 1,
        transition: {
            staggerChildren: 0.08,
        },
    },
};

const STAGGER_ITEM = {
    hidden: { opacity: 0, y: 16 },
    show: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.42, ease: [0.16, 1, 0.3, 1] },
    },
};

const Dashboard = () => {
    const { currentUser } = useAuth();
    const { revision } = useAppData();
    const [habits, setHabits] = useState([]);
    const [tasks, setTasks] = useState([]);
    const [learningProgress, setLearningProgress] = useState(null);
    const [nextTopics, setNextTopics] = useState([]);
    const [editingTask, setEditingTask] = useState(null);
    const [focusMode, setFocusMode] = useState(false);
    const [loading, setLoading] = useState(true);

    const [userData, setUserData] = useState(null);

    // Confirmation State
    const [deleteConfig, setDeleteConfig] = useState({
        isOpen: false,
        type: null, // 'habit' | 'task'
        id: null,
        title: '',
        message: ''
    });

    const loadDashboardData = useCallback(async () => {
        try {
            const [habitData, taskData, gamification, progressData, nextData] = await Promise.all([
                habitService.getHabits(),
                taskService.getTasks(true),
                habitService.getGamification(),
                learningService.getProgress().catch(() => null),
                learningService.getNext().catch(() => []),
            ]);
            setHabits(habitData);
            setTasks(taskData);
            setUserData(gamification);
            setLearningProgress(progressData);
            setNextTopics(nextData || []);
        } catch (error) {
            console.error('Error loading dashboard:', error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (currentUser) {
            loadDashboardData();
        }
    }, [currentUser, revision, loadDashboardData]);

    // Derived Stats
    const today = format(new Date(), 'yyyy-MM-dd');

    // Habits Stats
    const todaysHabits = habits; // All habits are for today typically, daily goal style
    const completedHabitsCount = habits.filter(h => h.completedDates?.includes(today)).length;
    const totalHabitsCount = habits.length;
    // Tasks Stats
    const pendingTasks = tasks.filter(t => !t.completed);
    const completedTasks = tasks.filter(t => t.completed); // Just for stats if needed

    // Focus Mode Filter: Show only incomplete habits and top 3 high priority pending tasks
    const focusHabits = habits.filter(h => !h.completedDates?.includes(today));
    const focusTasks = pendingTasks
        .sort((a, b) => {
            const bg = { high: 3, medium: 2, low: 1 };
            return bg[b.priority] - bg[a.priority];
        })
        .slice(0, 3);

    const handleToggleHabit = async (habit) => {
        try {
            await habitService.toggleHabitCompletion(habit.id, currentUser.uid, today, habit.completedDates || []);
            setHabits(await habitService.getHabits());
            setUserData(await habitService.getGamification());
        } catch (error) {
            console.error(error);
        }
    };

    const handleToggleTask = async (taskId, status) => {
        try {
            await taskService.toggleTaskCompletion(taskId, status);
            setTasks(await taskService.getTasks(true));
        } catch (error) {
            console.error(error);
        }
    };

    const handleQuickStartTopic = async (topic) => {
        if (!topic?.topicId) return;
        try {
            await learningService.updateTopic(topic.topicId, { status: 'learning' });
            const [progressData, nextData] = await Promise.all([
                learningService.getProgress().catch(() => null),
                learningService.getNext().catch(() => []),
            ]);
            setLearningProgress(progressData);
            setNextTopics(nextData || []);
        } catch (error) {
            console.error(error);
        }
    };

    const confirmDelete = async () => {
        const { type, id } = deleteConfig;
        if (!id) return;

        try {
            if (type === 'task') {
                await taskService.deleteTask(id);
                setTasks(await taskService.getTasks(true));
                if (editingTask?.id === id) setEditingTask(null);
            } else if (type === 'habit') {
                await habitService.deleteHabit(id);
                setHabits(await habitService.getHabits());
                setUserData(await habitService.getGamification());
            }
        } catch (error) {
            console.error(error);
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

    if (loading) return <div className="p-10 text-center text-textMuted animate-pulse">Getting everything ready...</div>;

    return (
        <motion.div
            variants={STAGGER_CONTAINER}
            initial="hidden"
            animate="show"
            className="space-y-8 pb-20"
        >
            <motion.header variants={STAGGER_ITEM} className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                    <h1 className="text-3xl font-bold text-white tracking-widest uppercase glow-white">
                        {focusMode ? 'Focus time' : 'Command Center'}
                    </h1>
                    <p className="text-textMuted text-sm font-mono tracking-wide mt-1">
                        {focusMode ? 'Silence the noise.' : `${format(new Date(), 'EEEE, MMMM do')} · ${pendingTasks.length} ${pendingTasks.length === 1 ? 'task' : 'tasks'} remaining`}
                    </p>
                </div>

                <button
                    onClick={() => setFocusMode(!focusMode)}
                    className={`px-6 py-2 rounded-xl flex items-center space-x-2 transition-all border ${focusMode
                        ? 'bg-white text-black border-white shadow-[0_0_15px_rgba(255,255,255,0.2)]'
                        : 'bg-transparent border-white/10 text-textMuted hover:text-white hover:border-white/30'
                        }`}
                >
                    <Focus size={18} />
                    <span className="uppercase text-xs font-bold tracking-wider">{focusMode ? 'Back to dashboard' : 'Focus time'}</span>
                </button>
            </motion.header>

            {/* S.P.I.D.E.Y. Time-Based Entry Greeting, Remaining Workload & Where-To-Start Command Deck */}
            <motion.div variants={STAGGER_ITEM}>
                <AssistantEntryBriefing
                    tasks={tasks}
                    habits={habits}
                    nextTopics={nextTopics}
                    progress={learningProgress}
                    userName={currentUser?.displayName || 'Shadow Hunter'}
                    onQuickCompleteTask={(taskId) => handleToggleTask(taskId, true)}
                    onQuickToggleHabit={handleToggleHabit}
                    onQuickStartTopic={handleQuickStartTopic}
                />
            </motion.div>

            {!focusMode && (
                <motion.div variants={STAGGER_ITEM}>
                    <AnimeQuote />
                </motion.div>
            )}

            {!focusMode && (
                <motion.div variants={STAGGER_ITEM} className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-4 gap-6">
                    {/* Progress Card */}
                    <div className="glass-card flex flex-col items-center justify-center p-6 space-y-4 relative overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
                        <h3 className="text-xs font-bold uppercase tracking-widest text-textMuted self-start w-full mb-2">Today&apos;s progress</h3>
                        <CircularProgress value={completedHabitsCount} max={totalHabitsCount || 1} />
                        <div className="text-xs text-textMuted text-center font-mono tabular-nums">
                            {completedHabitsCount} / {totalHabitsCount} habits completed
                        </div>
                    </div>

                    {/* Quick Stats */}
                    <div className="glass-card flex flex-col justify-between">
                        <h3 className="text-xs font-bold uppercase tracking-widest text-textMuted mb-4">Overview</h3>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-3 bg-white/5 border border-white/5 rounded-full px-5 group hover:border-white/20 transition-colors">
                                <div className="flex items-center space-x-3">
                                    <div className="text-orange-400"><CheckCircle size={18} /></div>
                                    <span className="text-text text-sm">Tasks Done</span>
                                </div>
                                <span className="font-bold text-lg text-white font-mono">{completedTasks.length}</span>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-white/5 border border-white/5 rounded-full px-5 group hover:border-white/20 transition-colors">
                                <div className="flex items-center space-x-3">
                                    <div className="text-textMuted group-hover:text-white transition-colors"><Target size={18} /></div>
                                    <span className="text-text text-sm">Pending</span>
                                </div>
                                <span className="font-bold text-lg text-white font-mono">{pendingTasks.length}</span>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-white/5 border border-white/5 rounded-full px-5 group hover:border-white/20 transition-colors">
                                <div className="flex items-center space-x-3">
                                    <div className="text-orange-400"><Zap size={18} /></div>
                                    <span className="text-text text-sm">Global streak</span>
                                </div>
                                <span className="font-bold text-lg text-white font-mono">
                                    {userData?.globalStreak || 0}
                                </span>
                            </div>
                            <div className="flex items-center justify-between p-3 bg-orange-500/10 border border-orange-400/15 rounded-full px-5">
                                <span className="text-text text-sm">Level {Math.floor((userData?.xp || 0) / 100) + 1} · XP</span>
                                <span className="font-bold text-lg text-orange-300 font-mono">{userData?.xp || 0}</span>
                            </div>
                        </div>
                    </div>

                    {/* Next Priority Task */}
                    <div className="glass-card relative overflow-hidden flex flex-col">
                        <div className="absolute top-0 right-0 p-3 opacity-[0.03]">
                            <Target size={120} />
                        </div>
                        <h3 className="text-xs font-bold uppercase tracking-widest text-textMuted mb-4">Top Priority</h3>
                        {focusTasks.length > 0 ? (
                            <div className="mt-2 flex-1 flex flex-col justify-between">
                                <div>
                                    <h4 className="text-lg font-bold text-white truncate">{focusTasks[0].title}</h4>
                                    <p className="text-textMuted text-sm mt-1 line-clamp-2">{focusTasks[0].description || 'No description'}</p>
                                </div>
                                <div className="mt-4 flex justify-between items-center border-t border-white/5 pt-4">
                                    <span className={`px-3 py-1 rounded-full text-[10px] uppercase font-bold tracking-wider bg-white/10 text-white border border-white/20`}>
                                        {focusTasks[0].priority}
                                    </span>
                                    <span className="text-xs text-textMuted font-mono">
                                        Due: {focusTasks[0].dueDate ? format(new Date(focusTasks[0].dueDate), 'MMM d') : 'N/A'}
                                    </span>
                                </div>
                            </div>
                        ) : (
                            <div className="h-full flex items-center justify-center text-textMuted">
                                <p className="font-mono text-xs">No pending tasks</p>
                            </div>
                        )}
                    </div>

                    <div className="glass-card relative overflow-hidden flex flex-col">
                        <div className="absolute -right-8 -top-8 h-28 w-28 rounded-full bg-orange-500/10 blur-2xl" />
                        <h3 className="relative text-xs font-bold uppercase tracking-widest text-textMuted mb-4 flex items-center gap-2"><Trophy size={15} className="text-orange-400" /> Growth</h3>
                        <div className="relative flex items-end justify-between">
                            <div><p className="text-3xl font-black text-white">LVL {Math.floor((userData?.xp || 0) / 100) + 1}</p><p className="mt-1 text-xs text-textMuted">{userData?.xp || 0} total XP</p></div>
                            <span className="text-sm font-bold text-orange-300">{(userData?.xp || 0) % 100}<span className="text-textMuted"> / 100</span></span>
                        </div>
                        <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-orange-600 to-orange-300 transition-all" style={{ width: `${(userData?.xp || 0) % 100}%` }} /></div>
                        <div className="relative mt-5 flex flex-wrap gap-2">
                            {(userData?.achievements || []).length ? userData.achievements.map((achievement) => (
                                <span key={achievement} className="rounded-full border border-orange-400/20 bg-orange-500/10 px-2.5 py-1 text-[10px] font-semibold text-orange-200">{achievement.replaceAll('-', ' ')}</span>
                            )) : <span className="text-xs text-textMuted">Your first achievement is waiting.</span>}
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Phone Widgets Section - New */}
            <motion.section variants={STAGGER_ITEM}>
                <h2 className="text-lg font-bold flex items-center space-x-2 text-white/90 uppercase tracking-widest mb-4">
                    <span>How you&apos;re doing</span>
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6">
                    <HabitWidget habits={habits} onToggle={handleToggleHabit} />
                    <TodoWidget
                        tasks={tasks}
                        onToggle={(id) => handleToggleTask(id, true)}
                        onEdit={setEditingTask}
                    />
                    <LearningWidget />
                </div>
            </motion.section>

            {/* Main Content Area */}
            <motion.div variants={STAGGER_ITEM} className="grid grid-cols-1 items-start gap-8 md:grid-cols-2 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_320px]">

                {/* Habits Section */}
                <section className="space-y-4">
                    <h2 className="text-lg font-bold flex items-center space-x-2 text-white/90 uppercase tracking-widest">
                        <Zap size={18} />
                        <span>{focusMode ? 'Pending Habits' : 'Habits'}</span>
                    </h2>
                    <div className="space-y-3">
                        {(focusMode ? focusHabits : todaysHabits).map(habit => (
                            <HabitCard
                                key={habit.id}
                                habit={habit}
                                onToggle={handleToggleHabit}
                                onDelete={(id) => setDeleteConfig({
                                    isOpen: true,
                                    type: 'habit',
                                    id,
                                    title: 'Remove this habit?',
                                    message: 'Remove this habit from your dashboard?'
                                })}
                            />
                        ))}
                        {(focusMode ? focusHabits : todaysHabits).length === 0 && (
                            <p className="text-textMuted italic text-sm border-l-2 border-white/10 pl-4 py-2">You&apos;re all caught up with your habits.</p>
                        )}
                    </div>
                </section>

                {/* Tasks Section */}
                <section className="space-y-4">
                    <h2 className="text-lg font-bold flex items-center space-x-2 text-white/90 uppercase tracking-widest">
                        <CheckCircle size={18} />
                        <span>{focusMode ? 'What matters most' : 'Your tasks'}</span>
                    </h2>
                    <div className="space-y-3">
                        {(focusMode ? focusTasks : pendingTasks.slice(0, 5)).map(task => (
                            <TaskCard
                                key={task.id}
                                task={task}
                                onToggle={handleToggleTask}
                                onEdit={setEditingTask}
                                onDelete={(id) => setDeleteConfig({
                                    isOpen: true,
                                    type: 'task',
                                    id,
                                    title: 'Abort Objective',
                                    message: 'Remove this task from your mission log?'
                                })}
                            />
                        ))}
                        {(focusMode ? focusTasks : pendingTasks).length === 0 && (
                            <p className="text-textMuted italic text-sm border-l-2 border-white/10 pl-4 py-2">Log clear.</p>
                        )}
                    </div>
                </section>

                {/* Task editor, docked on the right of the page */}
                <div className="2xl:sticky 2xl:top-6">
                    <TaskEditor
                        task={editingTask}
                        onSave={handleSaveTask}
                        onCancel={() => setEditingTask(null)}
                    />
                </div>
            </motion.div>

            {/* Confirmation Modal */}
            <ConfirmationModal
                isOpen={deleteConfig.isOpen}
                onClose={() => setDeleteConfig({ ...deleteConfig, isOpen: false })}
                onConfirm={confirmDelete}
                title={deleteConfig.title}
                message={deleteConfig.message}
            />
        </motion.div>
    );
};

export default Dashboard;
