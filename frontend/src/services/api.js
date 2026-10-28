const defaultBase = typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1:8000';
const API_BASE_URL = (import.meta.env.VITE_SPIDEY_API_BASE_URL || defaultBase).replace(/\/$/, '');

export const apiRequest = async (path, options = {}) => {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...(options.headers || {})
        }
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.error || `S.P.I.D.E.Y request failed (${response.status}).`);
    }
    return payload;
};

export { API_BASE_URL };
