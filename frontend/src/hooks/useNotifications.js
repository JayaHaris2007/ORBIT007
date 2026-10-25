import { useState } from 'react';

export const useNotifications = () => {
    const [permission, setPermission] = useState(
        typeof Notification === 'undefined' ? 'denied' : Notification.permission
    );

    const requestPermission = async () => {
        if (!('Notification' in window)) return 'denied';
        const result = await Notification.requestPermission();
        setPermission(result);
        return result;
    };

    const sendNotification = (title, body) => {
        if (permission === 'granted') new Notification(title, { body, icon: '/logo.png' });
    };

    return { permission, requestPermission, sendNotification, checkAndNotify: async () => {} };
};
