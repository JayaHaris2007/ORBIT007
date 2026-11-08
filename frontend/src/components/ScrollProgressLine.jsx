import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Top scroll progress indicator line in S.P.I.D.E.Y orange theme.
 * Smoothly tracks viewport scroll progress with glowing orange gradient and leading photon head.
 */
const ScrollProgressLine = () => {
    const location = useLocation();
    const [progress, setProgress] = useState(0);
    const [isScrollable, setIsScrollable] = useState(false);

    useEffect(() => {
        let rafId = 0;

        const updateProgress = () => {
            const scrollTop = window.scrollY || document.documentElement.scrollTop || 0;
            const scrollHeight = document.documentElement.scrollHeight || 0;
            const clientHeight = window.innerHeight || document.documentElement.clientHeight || 1;
            const maxScroll = scrollHeight - clientHeight;

            if (maxScroll > 10) {
                setIsScrollable(true);
                const current = Math.min(1, Math.max(0, scrollTop / maxScroll));
                setProgress(current);
            } else {
                setIsScrollable(false);
                setProgress(0);
            }
        };

        const handleScrollOrResize = () => {
            cancelAnimationFrame(rafId);
            rafId = requestAnimationFrame(updateProgress);
        };

        handleScrollOrResize();
        window.addEventListener('scroll', handleScrollOrResize, { passive: true });
        window.addEventListener('resize', handleScrollOrResize, { passive: true });

        const timer = setTimeout(handleScrollOrResize, 200);

        return () => {
            window.removeEventListener('scroll', handleScrollOrResize);
            window.removeEventListener('resize', handleScrollOrResize);
            cancelAnimationFrame(rafId);
            clearTimeout(timer);
        };
    }, [location.pathname]);

    return (
        <div
            className={`fixed top-0 left-0 right-0 h-[3px] z-[99] pointer-events-none transition-opacity duration-300 ${
                isScrollable ? 'opacity-100' : 'opacity-0'
            }`}
            aria-hidden="true"
        >
            {/* Ambient track glow */}
            <div className="absolute inset-0 bg-orange-500/10 backdrop-blur-[1px]" />

            {/* Glowing active progress line */}
            <div
                className="h-full bg-gradient-to-r from-orange-600 via-orange-500 to-amber-300 shadow-[0_0_12px_rgba(255,106,26,0.9),0_0_24px_rgba(255,106,26,0.5)] origin-left will-change-transform"
                style={{
                    transform: `scaleX(${progress})`,
                    transition: 'transform 80ms ease-out',
                }}
            />

            {/* Glowing photon head at current scroll position */}
            {progress > 0.01 && progress < 0.99 && (
                <div
                    className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-amber-200 blur-[0.5px] shadow-[0_0_8px_#ff6a1a,0_0_16px_#ff8c28] pointer-events-none"
                    style={{
                        left: `calc(${progress * 100}% - 6px)`,
                        transition: 'left 80ms ease-out',
                    }}
                />
            )}
        </div>
    );
};

export default ScrollProgressLine;

