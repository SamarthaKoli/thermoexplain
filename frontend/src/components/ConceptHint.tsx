import type { ReactNode } from "react";

interface ConceptHintProps {
  term: string;
  children: ReactNode;
}

export function ConceptHint({ term, children }: ConceptHintProps) {
  return (
    <details className="concept-hint">
      <summary aria-label={`What does ${term} mean?`} title={`What does ${term} mean?`}>
        <span aria-hidden="true">i</span>
      </summary>
      <div className="concept-hint__content">
        <strong>{term}</strong>
        <p>{children}</p>
      </div>
    </details>
  );
}
