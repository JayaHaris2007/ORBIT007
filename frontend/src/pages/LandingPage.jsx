import { Link } from 'react-router-dom';
import { ArrowRight, CheckCircle, BarChart2, Shield, Zap } from 'lucide-react';

const LandingPage = () => {
    const fadeIn = {
        hidden: { opacity: 0, y: 20 },
        visible: { opacity: 1, y: 0, transition: { duration: 0.6 } }
    };

    const staggerContainer = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.2
            }
        }
    };

    return (
        <div className="min-h-screen bg-black text-white selection:bg-white selection:text-black overflow-x-hidden">
            {/* Navigation */}
            <nav className="fixed top-0 w-full z-50 bg-black/50 backdrop-blur-lg border-b border-white/10">
                <div className="max-w-7xl mx-auto px-4 md:px-6 h-20 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <img src="/logo.png" alt="Logo" className="w-8 h-8 rounded-full" />
                        <span className="text-xl font-bold tracking-wider sm:hidden">S.P.I.D.E.Y</span>
                        <span className="text-xl font-bold tracking-wider hidden sm:block">S.P.I.D.E.Y - A calmer way to grow</span>
                    </div>
                    <div className="flex items-center gap-2 md:gap-4">
                        <Link to="/login" className="text-sm font-medium text-textMuted hover:text-white transition-colors whitespace-nowrap">
                            Log In
                        </Link>
                        <Link
                            to="/signup"
                            className="bg-white text-black px-3 md:px-5 py-2 rounded-full text-xs md:text-sm font-bold hover:bg-gray-200 transition-colors whitespace-nowrap"
                        >
                            Get Started
                        </Link>
                    </div>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative pt-32 pb-20 px-6 min-h-screen flex items-center justify-center overflow-hidden">
                {/* Background Gradients */}
                <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
                    <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-purple-900/20 rounded-full blur-[100px]" />
                    <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-blue-900/20 rounded-full blur-[100px]" />
                </div>

                <div className="max-w-4xl mx-auto text-center relative z-10">
                    <motion.div
                        initial="hidden"
                        animate="visible"
                        variants={staggerContainer}
                    >
                        <motion.div variants={fadeIn} className="mb-6 inline-block px-4 py-1.5 rounded-full border border-white/10 bg-white/5 backdrop-blur-sm">
                            <span className="text-xs font-mono uppercase tracking-widest text-textMuted">v2.0 Now Available</span>
                        </motion.div>

                        <motion.h1
                            variants={fadeIn}
                            className="text-5xl md:text-7xl font-bold mb-8 leading-tight tracking-tight"
                        >
                            Master Your <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-500">Better days, one step at a time</span>
                        </motion.h1>

                        <motion.p
                            variants={fadeIn}
                            className="text-lg md:text-xl text-textMuted mb-10 max-w-2xl mx-auto leading-relaxed"
                        >
                            Build unbreakable habits with advanced analytics, streak tracking, and a distraction-free interface designed for high performers.
                        </motion.p>

                        <motion.div
                            variants={fadeIn}
                            className="flex flex-col sm:flex-row items-center justify-center gap-4"
                        >
                            <Link
                                to="/signup"
                                className="group relative px-8 py-4 bg-white text-black rounded-full font-bold text-lg overflow-hidden transition-all hover:scale-105"
                            >
                                <span className="relative z-10 flex items-center gap-2">
                                    Start Now <ArrowRight size={20} className="group-hover:translate-x-1 transition-transform" />
                                </span>
                                <div className="absolute inset-0 bg-gray-200 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-300" />
                            </Link>

                            <Link
                                to="/login"
                                className="px-8 py-4 rounded-full font-medium text-lg text-white border border-white/10 hover:bg-white/5 transition-all"
                            >
                                Live Demo
                            </Link>
                        </motion.div>
                    </motion.div>
                </div>
            </section>

            {/* Features Grid */}
            <section className="py-24 bg-zinc-950/50">
                <div className="max-w-7xl mx-auto px-6">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6 }}
                        className="grid grid-cols-1 md:grid-cols-3 gap-8"
                    >
                        <FeatureCard
                            icon={<Zap className="text-yellow-400" />}
                            title="Streak Tracking"
                            description="Visual momentum to keep you consistent. Don't break the chain."
                        />
                        <FeatureCard
                            icon={<BarChart2 className="text-purple-400" />}
                            title="Deep Analytics"
                            description="GitHub-style heatmaps and monthly reports to visualize your progress."
                        />
                        <FeatureCard
                            icon={<Shield className="text-green-400" />}
                            title="Data Privacy"
                            description="Your data is yours. Securely stored and never sold to third parties."
                        />
                    </motion.div>
                </div>
            </section>

            {/* Footer */}
            <footer className="py-10 border-t border-white/10 text-center text-textMuted text-sm">
                <p>&copy; {new Date().getFullYear()} S.P.I.D.E.Y. All rights reserved.</p>
            </footer>
        </div>
    );
};

const FeatureCard = ({ icon, title, description }) => (
    <div className="p-8 rounded-2xl bg-white/5 border border-white/5 hover:border-white/10 transition-colors group">
        <div className="mb-4 p-3 bg-black rounded-lg inline-block text-white group-hover:scale-110 transition-transform duration-300">
            {icon}
        </div>
        <h3 className="text-xl font-bold mb-3 text-white">{title}</h3>
        <p className="text-textMuted leading-relaxed">
            {description}
        </p>
    </div>
);

export default LandingPage;
