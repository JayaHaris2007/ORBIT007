import { createContext, useContext, useMemo, useState } from 'react';

const AuthContext = createContext(null);
const ThemeContext = createContext(null);

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
    return useContext(AuthContext);
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
    return useContext(ThemeContext);
}

export function AuthProvider({ children }) {
    const [theme, setTheme] = useState(
        () => localStorage.getItem('spidey_theme') || localStorage.getItem('kage_theme') || 'dark'
    );
    const value = useMemo(() => ({
        currentUser: { uid: 'local-user', displayName: 'Shadow Hunter' },
        loading: false,
        logout: () => {}
    }), []);
    const themeValue = useMemo(() => ({
        theme,
        setTheme: (nextTheme) => {
            localStorage.setItem('spidey_theme', nextTheme);
            setTheme(nextTheme);
        }
    }), [theme]);

    return (
        <AuthContext.Provider value={value}>
            <ThemeContext.Provider value={themeValue}>{children}</ThemeContext.Provider>
        </AuthContext.Provider>
    );
}
