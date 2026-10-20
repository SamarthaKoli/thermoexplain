import type { BackendConnectionState } from "../types/health";

interface BackendStatusCardProps {
  state: BackendConnectionState;
  message: string;
  onRetry: () => void;
}

const stateLabels: Record<BackendConnectionState, string> = {
  checking: "Checking",
  connected: "Connected",
  unavailable: "Unavailable",
};

export function BackendStatusCard({
  state,
  message,
  onRetry,
}: BackendStatusCardProps) {
  return (
    <section className="status-card" aria-labelledby="backend-status-heading">
      <div className="status-card__top">
        <div>
          <p className="eyebrow">SYSTEM STATUS</p>
          <h2 id="backend-status-heading">Backend connection</h2>
        </div>
        <span className={`status-pill status-pill--${state}`}>
          <span className="status-pill__dot" aria-hidden="true" />
          {stateLabels[state]}
        </span>
      </div>
      <p className="status-card__message" role="status">
        {message}
      </p>
      {state === "unavailable" && (
        <button className="retry-button" onClick={onRetry} type="button">
          Try again
        </button>
      )}
    </section>
  );
}
