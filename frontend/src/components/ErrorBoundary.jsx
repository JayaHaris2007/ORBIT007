import { Component } from 'react';

/**
 * Error boundary for the Learning module. If anything inside the learning
 * page throws while rendering, the rest of the app keeps working and the
 * user sees a friendly card instead of a blank screen.
 */
class ErrorBoundary extends Component {
    state = { hasError: false, message: '' };

    static getDerivedStateFromError(error) {
        return { hasError: true, message: error?.message || 'Unknown error' };
    }

    componentDidCatch(error, info) {
        console.error('Learning module crashed:', error, info);
    }

    handleReset = () => {
        this.setState({ hasError: false, message: '' });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="glass-card p-8 flex flex-col items-center justify-center text-center space-y-4">
                    <div className="p-3 bg-orange-500/15 text-orange-300 rounded-full">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                            <line x1="12" y1="9" x2="12" y2="13" />
                            <line x1="12" y1="17" x2="12.01" y2="17" />
                        </svg>
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold text-white">The learning tracker hit a snag</h2>
                        <p className="text-sm text-textMuted mt-1">Your other pages are unaffected.</p>
                    </div>
                    <button
                        onClick={this.handleReset}
                        className="px-4 py-2 rounded-xl bg-orange-500/90 text-white text-sm font-semibold hover:bg-orange-500 transition-colors"
                    >
                        Try again
                    </button>
                </div>
            );
        }
        return this.props.children;
    }
}

export default ErrorBoundary;
