import { motion } from "framer-motion";

const CircularProgress = ({ value, max, size = 120, strokeWidth = 8, color = "#FF6A1A" }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const progress = Math.min(value / max, 1);
    const dashoffset = circumference - progress * circumference;

    return (
        <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
            {/* Background Circle */}
            <svg width={size} height={size} className="transform -rotate-90">
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke="currentColor"
                    strokeWidth={strokeWidth}
                    fill="transparent"
                    className="text-surfaceHighlight/50"
                />
                {/* Progress Circle */}
                <motion.circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={color}
                    strokeWidth={strokeWidth}
                    fill="transparent"
                    strokeLinecap="round"
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: dashoffset }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    style={{ strokeDasharray: circumference }}
                    className="drop-shadow-[0_0_15px_rgba(255,255,255,0.25)]"
                />
            </svg>
            {/* Percentage Text */}
            <div className="absolute flex flex-col items-center">
                <span className="text-3xl font-bold text-white tracking-tighter">{Math.round(progress * 100)}%</span>
            </div>
        </div>
    );
};

export default CircularProgress;
