import { Component } from 'react';

/**
 * ErrorBoundary — reusable React error boundary.
 *
 * Catches render-time exceptions anywhere beneath it and displays a friendly
 * error card (with the failing message and a Reset button) instead of letting
 * the whole window unmount to a blank white screen. Resetting simply clears
 * the captured error so the subtree re-renders; a `key` on this component (or
 * a navigation event) remounts it and clears the error automatically.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, componentStack: '' };
    this.handleReset = this.handleReset.bind(this);
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, errorInfo) {
    // Keep the full detail in the devtools console for debugging.
    console.error('[AeroTwin] Render error caught by ErrorBoundary:', error, errorInfo);
    this.setState({ componentStack: errorInfo?.componentStack || '' });
  }

  handleReset() {
    this.setState({ error: null, componentStack: '' });
    if (typeof this.props.onReset === 'function') {
      this.props.onReset();
    }
  }

  render() {
    const { error, componentStack } = this.state;
    if (error) {
      return (
        <div
          role="alert"
          className="rounded-xl border border-red-200 bg-white p-6 shadow-sm"
        >
          <h2 className="text-sm font-semibold text-red-700">
            Something went wrong in this view
          </h2>
          <p className="mt-2 break-words text-xs text-slate-600">
            {String(error?.message || error)}
          </p>
          {componentStack ? (
            <details className="mt-3">
              <summary className="cursor-pointer text-[11px] text-slate-400">
                Technical details
              </summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-2 text-[10px] text-slate-500">
                {componentStack}
              </pre>
            </details>
          ) : null}
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-4 rounded-lg bg-sky-600 px-4 py-2 text-xs font-semibold text-white hover:bg-sky-700"
          >
            Reset view
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}