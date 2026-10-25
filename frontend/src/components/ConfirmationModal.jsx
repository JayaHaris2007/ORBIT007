import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, X } from 'lucide-react';

const ConfirmationModal = ({ isOpen, onClose, onConfirm, title, message, type = 'danger', confirmString }) => {
    const [inputValue, setInputValue] = useState('');

    // Reset input when modal opens
    if (!isOpen && inputValue) setInputValue('');

    // If confirmation string is required, check match
    const isConfirmDisabled = confirmString && inputValue !== confirmString;

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="glass-card w-full max-w-sm border border-white/10 shadow-[0_0_30px_rgba(0,0,0,0.5)] p-6 relative overflow-hidden"
                >
                    {/* Background Glow */}
                    <div className={`absolute top-0 left-0 w-full h-1 ${type === 'danger' ? 'bg-red-500' : 'bg-primary'}`} />

                    <button
                        onClick={onClose}
                        className="absolute top-4 right-4 text-textMuted hover:text-white transition-colors"
                    >
                        <X size={20} />
                    </button>

                    <div className="flex flex-col items-center text-center space-y-4">
                        <div className={`p-4 rounded-full ${type === 'danger' ? 'bg-red-500/10 text-red-500' : 'bg-primary/10 text-primary'}`}>
                            <AlertTriangle size={32} />
                        </div>

                        <div>
                            <h3 className="text-xl font-bold text-white uppercase tracking-wide mb-2">{title}</h3>
                            <p className="text-textMuted text-sm font-mono leading-relaxed mb-4">{message}</p>

                            {confirmString && (
                                <div className="mt-4">
                                    <p className="text-[10px] uppercase font-bold text-textMuted mb-2">
                                        Type <span className="text-white select-all">{confirmString}</span> to confirm
                                    </p>
                                    <input
                                        type="text"
                                        value={inputValue}
                                        onChange={(e) => setInputValue(e.target.value)}
                                        className="w-full bg-black/30 border border-white/10 rounded px-3 py-2 text-center text-white text-sm font-mono focus:border-white/30 outline-none transition-colors"
                                        placeholder={confirmString}
                                    />
                                </div>
                            )}
                        </div>

                        <div className="flex w-full gap-3 mt-4">
                            <button
                                onClick={onClose}
                                className="flex-1 py-3 px-4 rounded-lg border border-white/10 text-textMuted hover:text-white hover:bg-white/5 transition-all uppercase text-xs font-bold tracking-wider"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => {
                                    onConfirm();
                                    onClose();
                                }}
                                disabled={isConfirmDisabled}
                                className={`flex-1 py-3 px-4 rounded-lg text-white shadow-[0_0_15px_rgba(0,0,0,0.2)] transition-all uppercase text-xs font-bold tracking-wider ${isConfirmDisabled ? 'opacity-50 cursor-not-allowed bg-gray-700' :
                                        type === 'danger'
                                            ? 'bg-red-600 hover:bg-red-500'
                                            : 'bg-primary hover:bg-violet-500'
                                    }`}
                            >
                                Confirm
                            </button>
                        </div>
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
};

export default ConfirmationModal;
