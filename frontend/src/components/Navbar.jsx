import { Link, useLocation } from 'react-router-dom';
import { Home, CheckSquare, ListTodo, Bell, BarChart2, Settings, Flame } from 'lucide-react';

const Navbar = () => {
    const location = useLocation();
    const navItems = [
        { path: '/dashboard', icon: Home, label: 'Dashboard' },
        { path: '/habits', icon: CheckSquare, label: 'Habits' },
        { path: '/todos', icon: ListTodo, label: 'Tasks' },
        { path: '/reminders', icon: Bell, label: 'Reminders & Memory' },
        { path: '/analytics', icon: BarChart2, label: 'Analytics' },
        { path: '/settings', icon: Settings, label: 'Settings' },
    ];

    return (
        <nav className="app-sidebar fixed bottom-0 left-0 right-0 md:top-0 md:left-0 md:bottom-0 md:w-64 md:h-screen bg-surface border-t md:border-t-0 md:border-r border-white/5 z-50 transition-all duration-300">
            <div className="flex md:flex-col items-center justify-around md:items-stretch md:gap-y-2 h-full p-2 md:p-6">

                {/* Logo (Desktop only) */}
                <div className="hidden md:flex items-center gap-3 mb-10 px-1">
                    <div className="w-12 h-12 rounded-2xl overflow-hidden border border-orange-400/30 shadow-[0_0_24px_rgba(255,106,26,0.18)]">
                        <img src="/logo.png" alt="S.P.I.D.E.Y logo" className="w-full h-full object-cover" />
                    </div>
                    <div><strong className="block text-sm tracking-[0.24em] text-white">S.P.I.D.E.Y</strong><span className="text-[10px] uppercase tracking-[0.18em] text-orange-300">personal assistant</span></div>
                </div>

                <p className="hidden md:block px-3 mb-3 text-[10px] font-bold uppercase tracking-[0.2em] text-textMuted">Workspace</p>

                {navItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = location.pathname === item.path;

                    return (
                        <Link
                            key={item.path}
                            to={item.path}
                            className={`relative group flex items-center justify-center md:justify-start gap-3 w-12 h-12 md:w-full md:px-4 md:rounded-xl rounded-full transition-all duration-300 ${isActive
                                ? 'bg-orange-500/15 text-orange-300 border border-orange-400/25 shadow-[0_0_24px_rgba(255,106,26,0.12)]'
                                : 'text-textMuted hover:text-white hover:bg-white/5'
                                }`}
                        >
                            <Icon size={20} strokeWidth={isActive ? 2.5 : 1.7} />
                            <span className="hidden md:block text-sm font-medium">{item.label}</span>

                            {/* Tooltip (Desktop) */}
                            <span className="absolute left-14 bg-surface px-3 py-1.5 rounded-full border border-white/10 text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity hidden whitespace-nowrap pointer-events-none tracking-wide">
                                {item.label}
                            </span>

                            {/* Active Indicator (Mobile) */}
                            {isActive && (
                                <span className="absolute -bottom-1 w-1 h-1 bg-orange-400 rounded-full md:hidden" />
                            )}
                        </Link>
                    );
                })}

                {/* Logout (Desktop) */}
                <div className="hidden md:flex mt-auto items-center gap-2 rounded-xl border border-orange-400/15 bg-orange-500/5 px-4 py-3 text-xs text-textMuted">
                    <Flame size={16} className="text-orange-400" /> Local workspace
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
