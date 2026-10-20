import { useState } from "react";

import { BackendStatusCard } from "../components/BackendStatusCard";
import { useBackendHealth } from "../hooks/useBackendHealth";
import { useDatasetStatus } from "../hooks/useDatasetStatus";
import { AppLayout, type PageId } from "../layouts/AppLayout";
import { CfdExplorer } from "./CfdExplorer";
import { EvaluationValidation } from "./EvaluationValidation";
import { ExplainableAiLab } from "./ExplainableAiLab";
import { GenerativeAiLab } from "./GenerativeAiLab";
import { ThermalKnowledgeAssistant } from "./ThermalKnowledgeAssistant";

export default function App() {
  const backend = useBackendHealth();
  const dataset = useDatasetStatus();
  const [activePage, setActivePage] = useState<PageId>("overview");

  return (
    <AppLayout activePage={activePage} onNavigate={setActivePage}>
      {activePage === "overview" ? (
        <>
          <section className="intro">
            <p className="eyebrow">THERMAL MANAGEMENT · DECISION SUPPORT</p>
            <h1>Understand the thermal story.</h1>
            <p className="intro__copy">
              A research workspace for inspecting computational fluid dynamics
              (CFD) evidence and testing explainable machine-learning methods for
              EV battery cooling. This application does not run CFD simulations.
            </p>
          </section>
          <BackendStatusCard
            state={backend.state}
            message={backend.message}
            onRetry={backend.retry}
          />
          {dataset.error ? (
            <p className="message message--error" role="alert">{dataset.error}</p>
          ) : dataset.scan ? (
            <section className="overview-data" aria-label="Dataset summary">
              <h2>Dataset inspection</h2>
              <div className="dataset-summary">
                <div><span className="eyebrow">SCAN STATUS</span><strong>{dataset.scan.status.replace(/_/g, " ")}</strong></div>
                <div><span className="eyebrow">CFD DATA FILES</span><strong>{dataset.scan.summary.candidate_count}</strong></div>
                <div><span className="eyebrow">CASES</span><strong>{dataset.cases.length}</strong></div>
                <div><span className="eyebrow">FIELDS</span><strong>{new Set(dataset.cases.flatMap((item) => item.fields.map((field) => field.name))).size}</strong></div>
              </div>
              <button
                className="text-button"
                onClick={() => setActivePage("cfd")}
                type="button"
              >
                Open CFD Explorer
              </button>
            </section>
          ) : (
            <p className="muted">{dataset.isLoading ? "Loading dataset status…" : "Dataset status is unavailable."}</p>
          )}
          <section className="overview-guide panel" aria-labelledby="overview-guide-title">
            <div>
              <p className="eyebrow">START HERE</p>
              <h2 id="overview-guide-title">What this workspace is for</h2>
              <p className="muted">
                Begin by checking the scanned CFD cases, then inspect a real field.
                Optional ML, generative, and language-model tools are separate
                experiments; they never replace the source data or a validated solver.
              </p>
            </div>
            <ol>
              <li>Confirm dataset and backend status above.</li>
              <li>Open CFD Explorer to check cases and field statistics.</li>
              <li>Use Evaluation &amp; Validation to review model evidence and limitations.</li>
            </ol>
          </section>
          <p className="foundation-note">
            Dataset availability and metadata are reported only after an actual scan;
            no model predictions are shown.
          </p>
        </>
      ) : activePage === "cfd" ? (
        <CfdExplorer dataset={dataset} />
      ) : activePage === "explainability" ? (
        <ExplainableAiLab />
      ) : activePage === "generative" ? (
        <GenerativeAiLab />
      ) : activePage === "knowledge" ? (
        <ThermalKnowledgeAssistant />
      ) : (
        <EvaluationValidation dataset={dataset} />
      )}
    </AppLayout>
  );
}
