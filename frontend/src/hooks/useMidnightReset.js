import { useEffect } from 'react';
import { differenceInMilliseconds, startOfTomorrow } from 'date-fns';

export const useMidnightReset = () => {
    useEffect(() => {
        const now = new Date();
        const tomorrow = startOfTomorrow();
        const timeUntilMidnight = differenceInMilliseconds(tomorrow, now);

        // Set timeout to reload the page at midnight
        // This ensures all "today" calculations (new Date()) are fresh
        const timeoutId = setTimeout(() => {
            window.location.reload();
        }, timeUntilMidnight + 1000); // +1 second buffer

        return () => clearTimeout(timeoutId);
    }, []);
};
