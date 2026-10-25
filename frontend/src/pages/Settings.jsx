import { useState, useEffect } from 'react';
import { useAuth, useTheme } from '../context/AuthContext';
import { useNotifications } from '../hooks/useNotifications';
import { User, Bell, Shield, LogOut, ChevronRight, X, AlertTriangle, Save, Loader } from 'lucide-react';
import { format } from 'date-fns';
import { motion, AnimatePresence } from 'framer-motion';
import ConfirmationModal from '../components/ConfirmationModal';

const Settings = () => {
    const { currentUser, logout } = useAuth();
    const { theme, setTheme } = useTheme();
    const { requestPermission } = useNotifications(currentUser);
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState({ type: '', text: '' });

    // Modal Config
    const [modalConfig, setModalConfig] = useState({
        isOpen: false,
        title: '',
        message: '',
        type: 'danger',
        onConfirm: () => { },
        confirmString: null
    });

    // Profile State
    const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);
    const [displayName, setDisplayName] = useState(currentUser?.displayName || '');

    // Preferences State
    const [notificationsEnabled, setNotificationsEnabled] = useState(true);

    // Initial Fetch for Preferences
    useEffect(() => {
        const fetchPreferences = async () => {
            if (currentUser) {
                try {
                    setNotificationsEnabled(
                        (localStorage.getItem('spidey_notifications')
                            ?? localStorage.getItem('kage_notifications')) !== 'false'
                    );
                } catch (error) {
                    console.error("Error fetching settings:", error);
                }
            }
        };
        fetchPreferences();
    }, [currentUser]);

    const showMessage = (type, text) => {
        setMessage({ type, text });
        setTimeout(() => setMessage({ type: '', text: '' }), 3000);
    };

    // --- Handlers ---

    const handleUpdateProfile = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        try {
            currentUser.displayName = displayName;

            showMessage('success', 'Profile updated successfully.');
            setIsProfileModalOpen(false);
        } catch (error) {
            console.error(error);
            showMessage('error', 'Failed to update profile.');
        } finally {
            setIsLoading(false);
        }
    };

    const handleResetPassword = async () => {
        setModalConfig({
            isOpen: true,
            title: 'Reset Password',
            message: 'Password reset is managed by the local S.P.I.D.E.Y assistant.',
            type: 'default',
            onConfirm: async () => {
                setIsLoading(true);
                try {
                    showMessage('success', 'S.P.I.D.E.Y uses local authentication for this installation.');
                } catch (error) {
                    console.error(error);
                    showMessage('error', 'Failed to send reset email. Try again later.');
                } finally {
                    setIsLoading(false);
                }
            }
        });
    };

    const handleToggleNotifications = async () => {
        const newState = !notificationsEnabled;

        if (newState) {
            const perm = await requestPermission();
            if (perm !== 'granted') {
                showMessage('error', 'Notification permission denied.');
                return;
            }
        }

        setNotificationsEnabled(newState); // Optimistic update

        try {
            localStorage.setItem('spidey_notifications', String(newState));
        } catch (error) {
            console.error(error);
            setNotificationsEnabled(!newState); // Revert
            showMessage('error', 'Failed to save preference.');
        }
    };

    const handleThemeChange = (event) => {
        setTheme(event.target.value);
        showMessage('success', 'Theme updated.');
    };

    const handleDeleteAccount = async () => {
        setModalConfig({
            isOpen: true,
            title: 'Delete Account',
            message: 'Are you sure you want to delete your account? This action cannot be undone. All data will be permanently erased.',
            type: 'danger',
            confirmString: 'DELETE',
            onConfirm: async () => {
                setIsLoading(true);
                try {
                    showMessage('success', 'Local profile data is retained by S.P.I.D.E.Y.');
                    logout();
                } catch (error) {
                    console.error(error);
                    showMessage('error', 'Failed to delete account. You may need to re-login first.');
                    setIsLoading(false);
                }
            }
        });
    };

    // --- Components ---

    const SettingSection = ({ title, children }) => (
        <div className="glass-card mb-6 border border-white/5 overflow-hidden">
            <h2 className="text-xs font-bold text-textMuted uppercase tracking-widest mb-4 px-4 pt-4">{title}</h2>
            <div className="space-y-1">
                {children}
            </div>
        </div>
    );

    const SettingItem = ({ icon: Icon, label, value, onClick, destructive = false, toggle = null }) => (
        <div
            onClick={onClick}
            className={`flex items-center justify-between p-4 transition-colors cursor-pointer group ${destructive ? 'hover:bg-red-500/10' : 'hover:bg-white/5'
                }`}
        >
            <div className="flex items-center space-x-4">
                <div className={`p-2 rounded-lg ${destructive ? 'bg-red-500/10 text-red-400' : 'bg-surfaceHighlight text-textMuted group-hover:text-white transition-colors'
                    }`}>
                    <Icon size={18} />
                </div>
                <span className={`text-sm font-medium ${destructive ? 'text-red-400' : 'text-text group-hover:text-white transition-colors'}`}>
                    {label}
                </span>
            </div>

            <div className="flex items-center space-x-3">
                {value && <span className="text-xs text-textMuted font-mono">{value}</span>}

                {toggle !== null ? (
                    <div className={`w-8 h-4 rounded-full relative transition-colors ${toggle ? 'bg-white' : 'bg-white/20'}`}>
                        <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-black rounded-full transition-transform ${toggle ? 'translate-x-4' : 'translate-x-0'}`} />
                    </div>
                ) : (
                    !destructive && <ChevronRight size={16} className="text-textMuted group-hover:text-white transition-colors" />
                )}
            </div>
        </div>
    );

    return (
        <div className="max-w-2xl mx-auto pb-20 relative">
            <h1 className="text-3xl font-bold text-white uppercase tracking-widest glow-white mb-2">Settings</h1>
            <p className="text-textMuted text-xs font-mono mb-8">Make this space feel like yours.</p>

            <SettingSection title="Appearance">
                <div className="flex items-center justify-between p-4">
                    <div>
                        <p className="text-sm font-medium text-text">Theme</p>
                        <p className="text-xs text-textMuted mt-1">Choose the dashboard visual mode.</p>
                    </div>
                    <select value={theme} onChange={handleThemeChange} className="bg-surfaceHighlight border border-white/10 rounded-lg p-2 text-white text-sm">
                        <option value="dark">Shadow</option>
                        <option value="midnight">Midnight</option>
                        <option value="light">Light</option>
                    </select>
                </div>
            </SettingSection>

            {/* Feedback Message */}
            <AnimatePresence>
                {message.text && (
                    <motion.div
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-lg shadow-lg border ${message.type === 'error' ? 'bg-red-900/90 border-red-500 text-white' : 'bg-green-900/90 border-green-500 text-white'
                            }`}
                    >
                        <p className="text-sm font-bold flex items-center gap-2">
                            {message.type === 'error' && <AlertTriangle size={16} />}
                            {message.text}
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Profile Card */}
            <div className="flex items-center space-x-4 mb-8 p-6 glass-card border border-white/10 bg-gradient-to-r from-white/5 to-transparent">
                <div className="w-16 h-16 rounded-full bg-white text-black flex items-center justify-center text-2xl font-bold shadow-[0_0_20px_rgba(255,255,255,0.2)]">
                    {currentUser?.photoURL ? (
                        <img src={currentUser.photoURL} alt="Profile" className="w-full h-full rounded-full object-cover" />
                    ) : (
                        currentUser?.displayName?.charAt(0) || 'K'
                    )}
                </div>
                <div>
                    <h2 className="text-xl font-bold text-white flex items-center gap-2">
                        {currentUser?.displayName || 'User'}
                    </h2>
                    <p className="text-textMuted text-sm font-mono">{currentUser?.email}</p>
                    <p className="text-textMuted text-xs mt-1">
                        Joined {currentUser?.metadata?.creationTime ? format(new Date(currentUser.metadata.creationTime), 'MMMM yyyy') : 'Recently'}
                    </p>
                </div>
            </div>

            <SettingSection title="Account">
                <SettingItem
                    icon={User}
                    label="Edit Profile"
                    onClick={() => setIsProfileModalOpen(true)}
                />
                <SettingItem
                    icon={Shield}
                    label="Reset Password"
                    onClick={handleResetPassword}
                />
            </SettingSection>

            <SettingSection title="Preferences">
                <SettingItem
                    icon={Bell}
                    label="Notifications"
                    toggle={notificationsEnabled}
                    onClick={handleToggleNotifications}
                />
            </SettingSection>

            <SettingSection title="Session">
                <SettingItem icon={LogOut} label="Log Out" onClick={logout} destructive />
            </SettingSection>

            <div className="mt-12">
                <SettingSection title="Danger Zone">
                    <SettingItem
                        icon={AlertTriangle}
                        label="Delete Account"
                        onClick={handleDeleteAccount}
                        destructive
                    />
                </SettingSection>
            </div>

            <div className="text-center mt-12 mb-8">
                <p className="text-[10px] text-textMuted uppercase tracking-widest opacity-50">S.P.I.D.E.Y System v1.1.0</p>
            </div>

            {/* Edit Profile Modal */}
            <AnimatePresence>
                {isProfileModalOpen && (
                    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.95 }}
                            className="glass-card w-full max-w-sm relative border border-white/10 shadow-[0_0_30px_rgba(0,0,0,0.5)] p-6"
                        >
                            <button
                                onClick={() => setIsProfileModalOpen(false)}
                                className="absolute top-4 right-4 text-textMuted hover:text-white transition-colors"
                            >
                                <X size={20} />
                            </button>

                            <h2 className="text-lg font-bold mb-6 text-white uppercase tracking-wide flex items-center gap-2">
                                <User size={20} />
                                Update Profile
                            </h2>

                            <form onSubmit={handleUpdateProfile} className="space-y-6">
                                <div>
                                    <label className="block text-xs font-bold text-textMuted uppercase mb-2">Display Name</label>
                                    <input
                                        type="text"
                                        value={displayName}
                                        onChange={(e) => setDisplayName(e.target.value)}
                                        className="w-full bg-surfaceHighlight border border-white/5 rounded-full py-3 px-6 text-white focus:outline-none focus:border-white/50 focus:ring-1 focus:ring-white/50 transition-all font-mono"
                                        placeholder="Enter your name"
                                        autoFocus
                                    />
                                </div>

                                <button
                                    type="submit"
                                    disabled={isLoading || !displayName.trim()}
                                    className="w-full bg-white hover:bg-gray-200 text-black font-bold py-3 rounded-full transition-colors shadow-[0_0_15px_rgba(255,255,255,0.1)] disabled:opacity-50 disabled:cursor-not-allowed uppercase tracking-wider text-sm flex items-center justify-center gap-2"
                                >
                                    {isLoading ? <Loader size={18} className="animate-spin" /> : <Save size={18} />}
                                    Save Changes
                                </button>
                            </form>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
            {/* Confirmation Modal */}
            <ConfirmationModal
                isOpen={modalConfig.isOpen}
                onClose={() => setModalConfig({ ...modalConfig, isOpen: false })}
                onConfirm={modalConfig.onConfirm}
                title={modalConfig.title}
                message={modalConfig.message}
                type={modalConfig.type}
                confirmString={modalConfig.confirmString}
            />
        </div>
    );
};

export default Settings;
