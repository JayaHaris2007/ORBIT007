import test from 'node:test';
import assert from 'node:assert/strict';
import {
    getTimeOfDayGreeting,
    rankPendingTasks,
    buildAssistantBriefing,
} from './briefingUtils.js';

test('getTimeOfDayGreeting returns morning, afternoon, evening based on hour', () => {
    const morning = getTimeOfDayGreeting(new Date('2026-10-29T08:30:00'), 'Alex Mercer');
    assert.equal(morning.salutation, 'Good morning');
    assert.equal(morning.period, 'morning');
    assert.equal(morning.headline, 'Good morning, Alex');

    const afternoon = getTimeOfDayGreeting(new Date('2026-10-29T14:15:00'), 'Shadow Hunter');
    assert.equal(afternoon.salutation, 'Good afternoon');
    assert.equal(afternoon.period, 'afternoon');
    assert.equal(afternoon.headline, 'Good afternoon, Shadow');

    const evening = getTimeOfDayGreeting(new Date('2026-10-29T19:45:00'), 'Peter');
    assert.equal(evening.salutation, 'Good evening');
    assert.equal(evening.period, 'evening');
    assert.equal(evening.headline, 'Good evening, Peter');

    const lateNight = getTimeOfDayGreeting(new Date('2026-10-29T23:30:00'), '');
    assert.equal(lateNight.salutation, 'Good evening');
    assert.equal(lateNight.period, 'night');
    assert.equal(lateNight.firstName, 'Commander');
});

test('rankPendingTasks prioritizes due-today and high-priority open tasks first', () => {
    const tasks = [
        { id: 1, title: 'Low task', priority: 'low', completed: false, dueDate: null },
        { id: 2, title: 'Done high task', priority: 'high', completed: true, dueDate: '2026-10-29' },
        { id: 3, title: 'High task later', priority: 'high', completed: false, dueDate: '2026-11-05' },
        { id: 4, title: 'Medium task due today', priority: 'medium', completed: false, dueDate: '2026-10-29' },
    ];

    const ranked = rankPendingTasks(tasks, '2026-10-29');
    assert.equal(ranked.length, 3);
    assert.equal(ranked[0].title, 'Medium task due today');
    assert.equal(ranked[1].title, 'High task later');
    assert.equal(ranked[2].title, 'Low task');
});

test('buildAssistantBriefing computes remaining tasks, habits, and where to start recommendation', () => {
    const now = new Date('2026-10-29T09:00:00');
    const briefing = buildAssistantBriefing({
        now,
        userName: 'Shadow Hunter',
        tasks: [
            { id: 10, title: 'Deploy API Gateway', priority: 'high', completed: false, description: 'Ship v2 routes' },
            { id: 11, title: 'Write unit tests', priority: 'medium', completed: false },
            { id: 12, title: 'Setup repo', priority: 'low', completed: true },
        ],
        habits: [
            { id: 1, name: 'Morning Run', completedDates: [] },
            { id: 2, name: 'Read 10 Pages', completedDates: ['2026-10-29'] },
        ],
        nextTopics: [
            { topicId: 101, name: 'Recursion', roadmapName: 'Java & DSA', status: 'learning', reason: 'In progress' },
        ],
        progress: { completionPercent: 42, inProgressTopics: 1, completedTopics: 5 },
    });

    assert.equal(briefing.greeting.salutation, 'Good morning');
    assert.equal(briefing.stats.remainingTasksCount, 2);
    assert.equal(briefing.stats.completedTasksCount, 1);
    assert.equal(briefing.stats.highPriorityCount, 1);
    assert.equal(briefing.stats.remainingHabitsCount, 1);
    assert.equal(briefing.stats.completedHabitsCount, 1);
    assert.equal(briefing.primaryStart.kind, 'task');
    assert.equal(briefing.primaryStart.title, 'Deploy API Gateway');
    assert.equal(briefing.startingSteps.length, 3);
    assert.ok(briefing.spokenSummary.includes('Good morning, Shadow.'));
    assert.ok(briefing.spokenSummary.includes('2 remaining tasks'));
    assert.ok(briefing.spokenSummary.includes('Deploy API Gateway'));
});

test('buildAssistantBriefing falls back to learning or planning when tasks and habits are complete', () => {
    const now = new Date('2026-10-29T18:30:00');
    const briefing = buildAssistantBriefing({
        now,
        userName: 'Aria',
        tasks: [{ id: 1, title: 'Finished task', completed: true }],
        habits: [{ id: 2, name: 'Hydrate', completedDates: ['2026-10-29'] }],
        nextTopics: [{ topicId: 202, name: 'Trees', roadmapName: 'Java & DSA', status: 'not_started', reason: 'Ready to start' }],
        progress: { completionPercent: 60, inProgressTopics: 0, completedTopics: 6 },
    });

    assert.equal(briefing.greeting.salutation, 'Good evening');
    assert.equal(briefing.stats.remainingTasksCount, 0);
    assert.equal(briefing.stats.remainingHabitsCount, 0);
    assert.equal(briefing.primaryStart.kind, 'learning');
    assert.equal(briefing.primaryStart.title, 'Trees');
});
