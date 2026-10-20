import React from "react";
import ReactDOM from "react-dom/client";

import { GenerativeAiLab } from "./pages/GenerativeAiLab";
import "./styles/global.css";

const host = document.querySelector("main");
if (!host) {
  throw new Error("The Generative AI page content host is missing.");
}

host.className = "generative-main";
host.replaceChildren();

const topbar = host.parentElement?.querySelector("header");
if (topbar) {
  const label = document.createElement("span");
  label.className = "font-caption text-caption text-on-surface-variant";
  label.textContent = "RESEARCH WORKSPACE · GENERATIVE AI";
  topbar.replaceChildren(label);
}

ReactDOM.createRoot(host).render(
  <React.StrictMode>
    <GenerativeAiLab />
  </React.StrictMode>,
);
