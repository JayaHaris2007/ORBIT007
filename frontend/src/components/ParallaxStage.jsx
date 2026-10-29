import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const PAGE_VARIANTS = {
    initial: {
        opacity: 0,
        y: 18,
        scale: 0.992,
    },
    animate: {
        opacity: 1,
        y: 0,
        scale: 1,
        transition: {
            duration: 0.42,
            ease: [0.16, 1, 0.3, 1],
        },
    },
    exit: {
        opacity: 0,
        y: -12,
        scale: 0.995,
        transition: {
            duration: 0.22,
            ease: [0.4, 0, 1, 1],
        },
    },
};

/**
 * Wraps the main workspace content with:
 * 1) Pointer + scroll driven 3D parallax ambient background layers
 * 2) Smooth route transition animations on navigation and initial app opening
 */
const ParallaxStage = ({ children }) => {
    const location = useLocation();
    const stageRef = useRef(null);
    const rafRef = useRef(0);

    useEffect(() => {
        const el = stageRef.current;
        if (!el) return undefined;

        const prefersReduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
        if (prefersReduced) return undefined;

        let targetX = 0;
        let targetY = 0;
        let scrollY = window.scrollY || 0;

        const updateVars = () => {
            el.style.setProperty('--app-px', `${targetX.toFixed(1)}px`);
            el.style.setProperty('--app-py', `${targetY.toFixed(1)}px`);
            el.style.setProperty('--app-scroll-y', `${(scrollY * -0.12).toFixed(1)}px`);
        };

        const onPointerMove = (e) => {
            const nx = (e.clientX / Math.max(window.innerWidth, 1) - 0.5) * 2;
            const ny = (e.clientY / Math.max(window.innerHeight, 1) - 0.5) * 2;
            targetX = nx * 22;
            targetY = ny * 22;
            cancelAnimationFrame(rafRef.current);
            rafRef.current = requestAnimationFrame(updateVars);
        };

        const onScroll = () => {
            scrollY = window.scrollY || 0;
            cancelAnimationFrame(rafRef.current);
            rafRef.current = requestAnimationFrame(updateVars);
        };

        window.addEventListener('pointermove', onPointerMove, { passive: true });
        window.addEventListener('scroll', onScroll, { passive: true });

        return () => {
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('scroll', onScroll);
            cancelAnimationFrame(rafRef.current);
        };
    }, []);

    return (
        <div ref={stageRef} className="spidey-parallax-stage relative min-h-screen flex-1 flex flex-col">
            {/* Ambient Multi-Depth Parallax Layers */}
            <div aria-hidden="true" className="pointer-events-none fixed inset-0 overflow-hidden z-0">
                <div className="spidey-stage-grid absolute inset-[-8%] opacity-20" />
                <div className="spidey-stage-glow-primary absolute -top-24 right-[18%] h-[420px] w-[420px] rounded-full bg-orange-500/10 blur-[120px]" />
                <div className="spidey-stage-glow-secondary absolute bottom-[-10%] left-[22%] h-[360px] w-[360px] rounded-full bg-amber-500/5 blur-[110px]" />
            </div>

            {/* Animated Route Container */}
            <main className="relative z-10 flex-1 p-4 md:p-8 max-w-7xl mx-auto w-full mb-16 md:mb-0">
                <AnimatePresence mode="wait" initial={true}>
                    <motion.div
                        key={location.pathname}
                        variants={PAGE_VARIANTS}
                        initial="initial"
                        animate="animate"
                        exit="exit"
                        className="mt-4 md:mt-0"
                    >
                        {children}
                    </motion.div>
                </AnimatePresence>
            </main>
        </div>
    );
};

export default ParallaxStage;
