import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

const AppDataContext = createContext(null);

// eslint-disable-next-line react-refresh/only-export-components
export const useAppData = () => useContext(AppDataContext);

// Bumps a revision counter whenever server data may have changed, so every
// mounted page refetches without a manual reload.
export const AppDataProvider = ({ children }) => {
    const [revision, setRevision] = useState(0);
    const refresh = useCallback(() => setRevision((value) => value + 1), []);

    useEffect(() => {
        const onWake = () => {
            if (!document.hidden) refresh();
        };
        window.addEventListener('focus', onWake);
        document.addEventListener('visibilitychange', onWake);
        // Light polling keeps the UI honest when the backend changes elsewhere
        // (Discord bot, reminders firing, another browser tab).
        const interval = window.setInterval(onWake, 30000);
        return () => {
            window.removeEventListener('focus', onWake);
            document.removeEventListener('visibilitychange', onWake);
            window.clearInterval(interval);
        };
    }, [refresh]);

    const value = useMemo(() => ({ revision, refresh }), [revision, refresh]);
    return <AppDataContext.Provider value={value}>{children}</AppDataContext.Provider>;
};
