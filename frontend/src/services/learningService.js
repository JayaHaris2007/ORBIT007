import { apiRequest } from './api';

// Mock mode is opt-in for isolated UI demos. By default every change goes to
// the backend and is persisted in assistant.db across reloads and restarts.
const MOCK_ENABLED = import.meta.env.VITE_LEARNING_MOCK === 'true';

const normalizeRoadmap = (roadmap) => ({
    ...roadmap,
    topicCount: roadmap.topic_count ?? 0,
    completedCount: roadmap.completed_count ?? 0,
    inProgressCount: roadmap.in_progress_count ?? 0,
    completionPercent: roadmap.completion_percent ?? 0,
    nextTopic: roadmap.next_topic ?? null,
});

const normalizeTopic = (topic) => ({
    ...topic,
    roadmapId: topic.roadmap_id,
    unmetPrerequisites: topic.unmet_prerequisites ?? [],
    requiresTopicIds: topic.requires_topic_ids ?? [],
});

const normalizeLink = (link) => ({
    ...link,
    taskId: link.task_id,
    topicId: link.topic_id,
    roadmapId: link.roadmap_id,
    taskTitle: link.task_title ?? '',
    topicName: link.topic_name ?? '',
    roadmapName: link.roadmap_name ?? '',
    taskCompleted: link.task_completed ?? false,
});

const normalizeProgress = (progress) => ({
    roadmapCount: progress.roadmap_count,
    topicCount: progress.topic_count,
    completedTopics: progress.completed_topics,
    inProgressTopics: progress.in_progress_topics,
    completionPercent: progress.completion_percent,
    perRoadmap: (progress.per_roadmap ?? []).map((item) => ({
        roadmapId: item.roadmap_id,
        roadmapName: item.roadmap_name,
        completed: item.completed,
        total: item.total,
        percent: item.percent,
    })),
});

const normalizeNext = (item) => ({
    ...item,
    topicId: item.topic_id,
    roadmapId: item.roadmap_id,
    roadmapName: item.roadmap_name,
    idleDays: item.idle_days ?? 0,
});

// ---------------------------------------------------------------------------
// In-memory mock store. Seeded with the same starter roadmaps the backend
// ships, and kept stateful so PATCH/DELETE behave like the real API.
// ---------------------------------------------------------------------------

const mockTopics = (roadmapId, names) => names.map((name, index) => ({
    id: roadmapId * 100 + index + 1,
    roadmap_id: roadmapId,
    roadmapId: roadmapId,
    name,
    aliases: [],
    description: `Learn ${name.toLowerCase()} the practical way.`,
    position: index,
    priority: 2,
    status: index === 0 && roadmapId === 1 ? 'learning' : 'not_started',
    available: index <= 1,
    unmet_prerequisites: index <= 1 ? [] : [names[0]],
    created_at: '2026-10-20T10:00:00+00:00',
    updated_at: '2026-10-20T10:00:00+00:00',
}));

const mockRoadmaps = [
    {
        id: 1,
        name: 'Java & DSA',
        category: 'technical',
        description: 'Core Java plus the data structures and algorithms that interviews test.',
        created_at: '2026-10-20T10:00:00+00:00',
    },
    {
        id: 2,
        name: 'Web Development',
        category: 'technical',
        description: 'From HTML to full-stack JavaScript and deployment.',
        created_at: '2026-10-20T10:00:00+00:00',
    },
    {
        id: 3,
        name: 'DBMS',
        category: 'academic',
        description: 'Relational databases, SQL, and transaction design.',
        created_at: '2026-10-20T10:00:00+00:00',
    },
];

const mockStore = {
    roadmaps: mockRoadmaps.map((roadmap) => ({ ...roadmap })),
    topics: [
        ...mockTopics(1, ['Java Basics', 'Object-Oriented Programming', 'Recursion', 'Arrays & Strings', 'Linked Lists', 'Trees', 'Graphs']),
        ...mockTopics(2, ['HTML & CSS', 'JavaScript Fundamentals', 'DOM & Events', 'React Basics', 'REST APIs', 'Node & Express']),
        ...mockTopics(3, ['Relational Model', 'SQL Basics', 'Normalization', 'Joins & Indexes', 'Transactions', 'Concurrency Control']),
    ],
    links: [
        {
            id: 1,
            taskId: 901,
            topicId: 105,
            confidence: 0.85,
            source: 'rule',
            state: 'suggested',
            createdAt: '2026-10-24T09:00:00+00:00',
            taskTitle: 'Solve linked list problems',
            topicName: 'Linked Lists',
            roadmapName: 'Java & DSA',
            roadmapId: 1,
        },
        {
            id: 2,
            taskId: 902,
            topicId: 106,
            confidence: 0.72,
            source: 'embedding',
            state: 'suggested',
            createdAt: '2026-10-24T09:05:00+00:00',
            taskTitle: 'Practice tree traversals',
            topicName: 'Trees',
            roadmapName: 'Java & DSA',
            roadmapId: 1,
        },
    ],
    nextLinkId: 1,
    settings: { nudges_enabled: true },
};

const delay = (ms = 120) => new Promise((resolve) => setTimeout(resolve, ms));

const mockSummary = (roadmap) => {
    const topics = mockStore.topics.filter((topic) => topic.roadmap_id === roadmap.id);
    const completed = topics.filter((topic) => topic.status === 'completed').length;
    const active = topics.filter((topic) => topic.status === 'learning').length;
    const available = topics.find((topic) => topic.status !== 'completed' && topic.available);
    return {
        id: roadmap.id,
        name: roadmap.name,
        category: roadmap.category,
        description: roadmap.description,
        created_at: roadmap.created_at,
        topic_count: topics.length,
        completed_count: completed,
        in_progress_count: active,
        completion_percent: topics.length ? Math.round((completed / topics.length) * 100) : 0,
        next_topic: available?.name ?? null,
    };
};

const mockListRoadmaps = async () => {
    await delay();
    return { roadmaps: mockStore.roadmaps.map(mockSummary) };
};

const mockGetRoadmap = async (id) => {
    await delay();
    const roadmap = mockStore.roadmaps.find((item) => item.id === id);
    if (!roadmap) throw new Error('Roadmap not found.');
    const topics = mockStore.topics
        .filter((topic) => topic.roadmap_id === id)
        .sort((a, b) => a.position - b.position)
        .map((topic) => ({
            ...topic,
            available: topic.unmet_prerequisites.length === 0,
            unmet_prerequisites: topic.unmet_prerequisites,
        }));
    return { ...roadmap, topics };
};

const mockCreateRoadmap = async (data) => {
    await delay();
    const existing = mockStore.roadmaps.find((item) => item.name === data.name);
    if (existing) throw new Error('A roadmap with that name already exists.');
    const roadmap = {
        id: Math.max(0, ...mockStore.roadmaps.map((item) => item.id)) + 1,
        name: data.name,
        category: data.category || 'personal',
        description: data.description || '',
        created_at: new Date().toISOString(),
    };
    mockStore.roadmaps.push(roadmap);
    return { ...roadmap };
};

const mockPatchRoadmap = async (id, patch) => {
    await delay();
    const roadmap = mockStore.roadmaps.find((item) => item.id === id);
    if (!roadmap) throw new Error('Roadmap not found.');
    if (patch.name && mockStore.roadmaps.some((item) => item.id !== id && item.name === patch.name)) {
        throw new Error('A roadmap with that name already exists.');
    }
    Object.assign(roadmap, patch);
    return { ...roadmap };
};

const mockDeleteRoadmap = async (id) => {
    await delay();
    const index = mockStore.roadmaps.findIndex((item) => item.id === id);
    if (index === -1) throw new Error('Roadmap not found.');
    mockStore.roadmaps.splice(index, 1);
    mockStore.topics = mockStore.topics.filter((topic) => topic.roadmap_id !== id);
    return { status: 'deleted' };
};

const mockCreateTopic = async (data) => {
    await delay();
    const existing = mockStore.topics.find(
        (topic) => topic.roadmap_id === data.roadmapId && topic.name === data.name
    );
    if (existing) throw new Error('A topic with that name already exists.');
    const positions = mockStore.topics
        .filter((topic) => topic.roadmap_id === data.roadmapId)
        .map((topic) => topic.position);
    const topic = {
        id: Math.max(0, ...mockStore.topics.map((item) => item.id)) + 1,
        roadmap_id: data.roadmapId,
        roadmapId: data.roadmapId,
        name: data.name,
        aliases: data.aliases || [],
        description: data.description || '',
        position: positions.length ? Math.max(...positions) + 1 : 0,
        priority: data.priority || 2,
        status: data.status || 'not_started',
        available: true,
        unmet_prerequisites: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
    };
    mockStore.topics.push(topic);
    return { ...topic };
};

const mockDeleteTopic = async (id) => {
    await delay();
    const index = mockStore.topics.findIndex((item) => item.id === id);
    if (index === -1) throw new Error('Topic not found.');
    mockStore.topics.splice(index, 1);
    mockStore.links = mockStore.links.filter((link) => link.topicId !== id);
    return { status: 'deleted' };
};

const mockPatchTopic = async (id, patch) => {
    await delay();
    const topic = mockStore.topics.find((item) => item.id === id);
    if (!topic) throw new Error('Topic not found.');
    if (patch.status === 'completed') {
        mockStore.topics.forEach((item) => {
            if (item.unmet_prerequisites.includes(topic.name)) {
                item.unmet_prerequisites = item.unmet_prerequisites.filter(
                    (name) => name !== topic.name
                );
            }
        });
    }
    Object.assign(topic, patch);
    if (patch.status) topic.updated_at = new Date().toISOString();
    return { ...topic };
};

const mockProgress = async () => {
    await delay();
    const topics = mockStore.topics;
    const completed = topics.filter((topic) => topic.status === 'completed').length;
    const active = topics.filter((topic) => topic.status === 'learning').length;
    return {
        roadmap_count: mockStore.roadmaps.length,
        topic_count: topics.length,
        completed_topics: completed,
        in_progress_topics: active,
        completion_percent: topics.length ? Math.round((completed / topics.length) * 100) : 0,
        per_roadmap: mockStore.roadmaps.map((roadmap) => {
            const summary = mockSummary(roadmap);
            return {
                roadmap_id: roadmap.id,
                roadmap_name: roadmap.name,
                completed: summary.completed_count,
                total: summary.topic_count,
                percent: summary.completion_percent,
            };
        }),
    };
};

const mockNext = async () => {
    await delay();
    const candidates = mockStore.topics
        .filter((topic) => topic.status !== 'completed' && topic.unmet_prerequisites.length === 0)
        .map((topic) => ({
            topic_id: topic.id,
            name: topic.name,
            roadmap_id: topic.roadmap_id,
            roadmap_name: mockStore.roadmaps.find((item) => item.id === topic.roadmap_id)?.name,
            status: topic.status,
            priority: topic.priority,
            idle_days: topic.status === 'learning' ? 6 : 0,
            reason: topic.priority >= 3 ? 'High priority.' : 'Prerequisites complete — ready to start.',
        }))
        .sort((a, b) => b.priority - a.priority)
        .slice(0, 5);
    return { next: candidates };
};

const mockListSuggestedLinks = async () => {
    await delay();
    return {
        links: mockStore.links
            .filter((link) => link.state === 'suggested')
            .map((link) => ({
                id: link.id,
                task_id: link.taskId,
                topic_id: link.topicId,
                roadmap_id: link.roadmapId,
                confidence: link.confidence,
                source: link.source,
                created_at: link.createdAt,
                task_title: link.taskTitle,
                topic_name: link.topicName,
                roadmap_name: link.roadmapName,
            })),
    };
};

const mockUpdateLink = async (id, patch) => {
    await delay();
    const link = mockStore.links.find((item) => item.id === id);
    if (!link) throw new Error('Link not found.');
    if (patch.topicId !== undefined && patch.topicId !== link.topicId) {
        link.topicId = patch.topicId;
        const topic = mockStore.topics.find((item) => item.id === patch.topicId);
        link.topicName = topic?.name ?? link.topicName;
        link.roadmapId = topic?.roadmap_id ?? link.roadmapId;
        link.roadmapName = mockStore.roadmaps.find((item) => item.id === link.roadmapId)?.name ?? link.roadmapName;
        link.source = 'manual';
    }
    link.state = patch.state;
    return { ...link };
};

const mockSettings = async () => {
    await delay();
    return { nudges_enabled: mockStore.settings.nudges_enabled };
};

const mockPatchSettings = async (patch) => {
    await delay();
    mockStore.settings.nudges_enabled = patch.nudges_enabled;
    return { nudges_enabled: mockStore.settings.nudges_enabled };
};

export const learningService = {
    isMock: MOCK_ENABLED,

    listRoadmaps: async () => {
        if (MOCK_ENABLED) return (await mockListRoadmaps()).roadmaps.map(normalizeRoadmap);
        return (await apiRequest('/api/learning/roadmaps')).roadmaps.map(normalizeRoadmap);
    },

    getRoadmap: async (id) => {
        if (MOCK_ENABLED) {
            const roadmap = await mockGetRoadmap(id);
            return { ...roadmap, topics: roadmap.topics.map(normalizeTopic) };
        }
        const roadmap = await apiRequest(`/api/learning/roadmaps/${id}`);
        return { ...roadmap, topics: roadmap.topics.map(normalizeTopic) };
    },

    createRoadmap: async (data) => {
        if (MOCK_ENABLED) return mockCreateRoadmap(data);
        return apiRequest('/api/learning/roadmaps', {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },

    updateRoadmap: async (id, patch) => {
        if (MOCK_ENABLED) return mockPatchRoadmap(id, patch);
        return apiRequest(`/api/learning/roadmaps/${id}`, {
            method: 'PATCH',
            body: JSON.stringify(patch),
        });
    },

    deleteRoadmap: async (id) => {
        if (MOCK_ENABLED) return mockDeleteRoadmap(id);
        return apiRequest(`/api/learning/roadmaps/${id}`, { method: 'DELETE' });
    },

    createTopic: async (data) => {
        if (MOCK_ENABLED) return mockCreateTopic(data);
        return normalizeTopic(await apiRequest('/api/learning/topics', {
            method: 'POST',
            body: JSON.stringify(data),
        }));
    },

    updateTopic: async (id, patch) => {
        if (MOCK_ENABLED) return mockPatchTopic(id, patch);
        return normalizeTopic(await apiRequest(`/api/learning/topics/${id}`, {
            method: 'PATCH',
            body: JSON.stringify(patch),
        }));
    },

    deleteTopic: async (id) => {
        if (MOCK_ENABLED) return mockDeleteTopic(id);
        return apiRequest(`/api/learning/topics/${id}`, { method: 'DELETE' });
    },

    getProgress: async () => {
        if (MOCK_ENABLED) return normalizeProgress(await mockProgress());
        return normalizeProgress(await apiRequest('/api/learning/progress'));
    },

    getNext: async () => {
        if (MOCK_ENABLED) return (await mockNext()).next.map(normalizeNext);
        return (await apiRequest('/api/learning/next')).next.map(normalizeNext);
    },

    listSuggestedLinks: async () => {
        if (MOCK_ENABLED) return (await mockListSuggestedLinks()).links.map(normalizeLink);
        return (await apiRequest('/api/learning/links?state=suggested')).links.map(normalizeLink);
    },

    updateLink: async (id, patch) => {
        if (MOCK_ENABLED) {
            const link = await mockUpdateLink(id, patch);
            return normalizeLink({ ...link, task_id: link.taskId, topic_id: link.topicId });
        }
        const payload = { state: patch.state };
        if (patch.topicId !== undefined) payload.topic_id = patch.topicId;
        return normalizeLink(await apiRequest(`/api/learning/links/${id}`, {
            method: 'PATCH',
            body: JSON.stringify(payload),
        }));
    },

    getTopicLinks: async (topicId) => {
        if (MOCK_ENABLED) return { links: [] };
        return (await apiRequest(`/api/learning/topics/${topicId}/links`)).links.map(normalizeLink);
    },

    getSettings: async () => {
        if (MOCK_ENABLED) return mockSettings();
        return apiRequest('/api/learning/settings');
    },

    updateSettings: async (patch) => {
        if (MOCK_ENABLED) return mockPatchSettings(patch);
        return apiRequest('/api/learning/settings', {
            method: 'PATCH',
            body: JSON.stringify(patch),
        });
    },
};
