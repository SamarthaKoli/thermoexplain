import React from "react";
import ReactDOM from "react-dom/client";

import { ExplainableAiLab } from "./pages/ExplainableAiLab";
import "./styles/global.css";

const host = document.querySelector("main");
if (!host) {
  throw new Error("The Explainable AI page content host is missing.");
}

host.className = "explainable-main";
host.replaceChildren();
const topbar = host.parentElement?.querySelector("header");
if (topbar) {
  const label = document.createElement("span");
  label.className = "font-caption text-caption text-on-surface-variant";
  label.textContent = "RESEARCH WORKSPACE · EXPLAINABLE AI";
  topbar.replaceChildren(label);
}

ReactDOM.createRoot(host).render(
  <React.StrictMode>
    <ExplainableAiLab />
  </React.StrictMode>,
);
