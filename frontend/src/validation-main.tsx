import React from "react";
import ReactDOM from "react-dom/client";

import { useDatasetStatus } from "./hooks/useDatasetStatus";
import { EvaluationValidation } from "./pages/EvaluationValidation";
import "./styles/global.css";

function ValidationPage() {
  const dataset = useDatasetStatus();
  return <EvaluationValidation dataset={dataset} />;
}

const host = document.querySelector("main");
if (!host) {
  throw new Error("The Evaluation & Validation page content host is missing.");
}

host.className = "validation-main";
host.replaceChildren();

const topbar = host.parentElement?.querySelector("header");
if (topbar) {
  const label = document.createElement("span");
  label.className = "font-caption text-caption text-on-surface-variant";
  label.textContent = "RESEARCH WORKSPACE · EVALUATION & VALIDATION";
  topbar.replaceChildren(label);
}

ReactDOM.createRoot(host).render(
  <React.StrictMode>
    <ValidationPage />
  </React.StrictMode>,
);
