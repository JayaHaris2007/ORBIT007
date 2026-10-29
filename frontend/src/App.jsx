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
import ParallaxStage from './components/ParallaxStage';

const AppLayout = ({ children }) => {
    const { theme } = useTheme();
    return <div className={`app-theme-${theme} min-h-screen bg-background text-text font-sans flex flex-col md:flex-row app-frame`}>
        <Navbar />
        <div className="app-main-shell flex-1 md:ml-64 min-w-0 transition-all duration-300 flex flex-col">
            <ParallaxStage>
                {children}
            </ParallaxStage>
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
