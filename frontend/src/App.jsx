import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useTheme } from './context/AuthContext';
import { AppDataProvider } from './context/AppDataContext';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Habits from './pages/Habits';
import Todos from './pages/Todos';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import RemindersMemory from './pages/RemindersMemory';
import Learning from './pages/Learning';
import ErrorBoundary from './components/ErrorBoundary';
import AssistantSidebar from './components/AssistantSidebar';

const AppLayout = ({ children }) => {
    const { theme } = useTheme();
    return <div className={`app-theme-${theme} min-h-screen bg-background text-text font-sans flex flex-col md:flex-row app-frame`}>
        <Navbar />
        <div className="app-main-shell flex-1 md:ml-64 min-w-0 transition-all duration-300 flex flex-col">
            <main className="flex-1 p-4 md:p-8 max-w-7xl mx-auto w-full mb-16 md:mb-0">
                <div className="mt-4 md:mt-0">{children}</div>
            </main>
        </div>
        {/* S.P.I.D.E.Y is docked on the right on every page */}
        <AssistantSidebar />
    </div>;
};

function App() {
    return (
        <AuthProvider>
            <AppDataProvider>
                <Router>
                    <Routes>
                        <Route path="/" element={<Navigate to="/dashboard" replace />} />
                        <Route path="/dashboard" element={<AppLayout><Dashboard /></AppLayout>} />
                        <Route path="/habits" element={<AppLayout><Habits /></AppLayout>} />
                        <Route path="/todos" element={<AppLayout><Todos /></AppLayout>} />
                        <Route path="/reminders" element={<AppLayout><RemindersMemory /></AppLayout>} />
                        <Route path="/learning" element={<AppLayout><ErrorBoundary><Learning /></ErrorBoundary></AppLayout>} />
                        <Route path="/analytics" element={<AppLayout><Analytics /></AppLayout>} />
                        <Route path="/settings" element={<AppLayout><Settings /></AppLayout>} />
                        <Route path="*" element={<Navigate to="/dashboard" replace />} />
                    </Routes>
                </Router>
            </AppDataProvider>
        </AuthProvider>
    );
}

export default App;
