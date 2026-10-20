import React from "react";
import ReactDOM from "react-dom/client";

import { ThermalKnowledgeAssistant } from "./pages/ThermalKnowledgeAssistant";
import "./styles/global.css";

const host = document.querySelector("main");
if (!host) {
  throw new Error("The Thermal Knowledge Assistant page content host is missing.");
}

host.className = "thermal-main";
host.replaceChildren();

const topbar = host.parentElement?.querySelector("header");
if (topbar) {
  const label = document.createElement("span");
  label.className = "font-caption text-caption text-on-surface-variant";
  label.textContent = "RESEARCH WORKSPACE · THERMAL KNOWLEDGE";
  topbar.replaceChildren(label);
}

ReactDOM.createRoot(host).render(
  <React.StrictMode>
    <ThermalKnowledgeAssistant />
  </React.StrictMode>,
);
