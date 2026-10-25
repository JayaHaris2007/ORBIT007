import { useEffect, useRef, useState } from 'react';
import { Check, Plus, X } from 'lucide-react';

const emptyForm = { title: '', description: '', priority: 'medium', dueDate: '' };

const toDateInput = (value) => (
    typeof value === 'string' && /^\d{4}-\d{2}-\d{2}/.test(value) ? value.slice(0, 10) : ''
);

const TaskEditor = ({ task, onSave, onCancel }) => {
    const [form, setForm] = useState(emptyForm);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState('');
    const panelRef = useRef(null);
    const isEditing = Boolean(task);

    useEffect(() => {
        setForm(task ? {
            title: task.title || '',
            description: task.description || '',
            priority: task.priority || 'medium',
            dueDate: toDateInput(task.dueDate)
        } : emptyForm);
        setSaving(false);
        setError('');
        if (task) panelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, [task]);

    const updateField = (field) => (event) => {
        const { value } = event.target;
        setForm((current) => ({ ...current, [field]: value }));
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        const title = form.title.trim();
        if (!title || saving) return;
        // Keep any stored time-of-day when the date itself did not change.
        const originalDue = task?.dueDate;
        const dueDate = form.dueDate && typeof originalDue === 'string'
            && originalDue.slice(0, 10) === form.dueDate
            ? originalDue
            : form.dueDate;
        setSaving(true);
        setError('');
        try {
            await onSave({ ...form, title, dueDate });
            if (!isEditing) setForm(emptyForm);
        } catch (requestError) {
            setError(requestError.message || 'Could not save this task.');
        } finally {
            setSaving(false);
        }
    };

    return (
        <section ref={panelRef} className="glass-card">
            <div className="flex items-start justify-between gap-3">
                <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-textMuted">
                        {isEditing ? 'Editing task' : 'New task'}
                    </p>
                    <h2 className="mt-1 text-lg font-bold text-white">
                        {isEditing ? 'Task details' : 'Add a task'}
                    </h2>
                </div>
                {isEditing && (
                    <button
                        type="button"
                        onClick={onCancel}
                        className="text-textMuted hover:text-white transition-colors p-1"
                        aria-label="Stop editing this task"
                    >
                        <X size={16} />
                    </button>
                )}
            </div>

            {isEditing && (
                <p className="mt-3 truncate rounded-lg border border-orange-400/20 bg-orange-500/10 px-3 py-2 text-xs text-orange-200">
                    #{task.id} · {task.title}
                </p>
            )}

            <form onSubmit={handleSubmit} className="mt-5 space-y-4">
                <div>
                    <label className="mb-2 block text-[10px] font-bold uppercase tracking-wider text-textMuted" htmlFor="task-title">Title</label>
                    <input
                        id="task-title"
                        type="text"
                        required
                        value={form.title}
                        onChange={updateField('title')}
                        maxLength={500}
                        placeholder="What needs doing?"
                        className="w-full rounded-lg border border-white/10 bg-surfaceHighlight px-3 py-2.5 text-sm text-white placeholder:text-neutral-600 focus:border-orange-400/40 focus:outline-none"
                    />
                </div>

                <div>
                    <label className="mb-2 block text-[10px] font-bold uppercase tracking-wider text-textMuted" htmlFor="task-desc">Description</label>
                    <textarea
                        id="task-desc"
                        value={form.description}
                        onChange={updateField('description')}
                        maxLength={2000}
                        rows={4}
                        placeholder="Add a note, if it helps..."
                        className="w-full resize-y rounded-lg border border-white/10 bg-surfaceHighlight px-3 py-2.5 text-sm text-white placeholder:text-neutral-600 focus:border-orange-400/40 focus:outline-none"
                    />
                </div>

                <div className="grid grid-cols-2 gap-3">
                    <div>
                        <label className="mb-2 block text-[10px] font-bold uppercase tracking-wider text-textMuted" htmlFor="task-priority">Priority</label>
                        <select
                            id="task-priority"
                            value={form.priority}
                            onChange={updateField('priority')}
                            className="w-full rounded-lg border border-white/10 bg-surfaceHighlight px-2 py-2.5 text-xs text-white focus:border-orange-400/40 focus:outline-none"
                        >
                            <option value="low">LOW</option>
                            <option value="medium">MEDIUM</option>
                            <option value="high">HIGH</option>
                        </select>
                    </div>
                    <div>
                        <label className="mb-2 block text-[10px] font-bold uppercase tracking-wider text-textMuted" htmlFor="task-due">Due date</label>
                        <input
                            id="task-due"
                            type="date"
                            value={form.dueDate}
                            onChange={updateField('dueDate')}
                            className="w-full rounded-lg border border-white/10 bg-surfaceHighlight px-2 py-2.5 text-xs text-white focus:border-orange-400/40 focus:outline-none [color-scheme:dark]"
                        />
                    </div>
                </div>

                {error && (
                    <p className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-300">{error}</p>
                )}

                <button
                    type="submit"
                    disabled={!form.title.trim() || saving}
                    className="flex w-full items-center justify-center gap-2 rounded-lg bg-white py-3 text-xs font-bold uppercase tracking-wider text-black transition-colors hover:bg-gray-200 disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {isEditing ? <Check size={14} /> : <Plus size={14} />}
                    {saving ? 'Saving...' : isEditing ? 'Save changes' : 'Create task'}
                </button>
            </form>
        </section>
    );
};

export default TaskEditor;
