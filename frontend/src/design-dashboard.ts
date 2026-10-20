import type { CfdFieldOption, CfdFieldOptions, CfdSpatialVisualization, CfdStatistics, DatasetCase, DatasetScanStatus, GptOssAnalysis, MeshComparisonResult } from "./types/dataset";
import type { GptOssStatus } from "./types/provider";
import {
  analyzeWithGptOss,
  compareMeshes,
  getCfdFieldOptions,
  getCfdSpatialVisualization,
  getCfdStatistics,
  getDatasetCases,
  getDatasetScan,
} from "./services/datasetApi";
import { getGptOssStatus, providerTestFailure, testGptOssConnection } from "./services/providerApi";

const root = document.querySelector<HTMLElement>("main > div > div");
const caseSelect = document.getElementById("caseSelector") as HTMLSelectElement | null;
const meshButtons = [...document.querySelectorAll<HTMLButtonElement>(".mesh-toggle-btn")];
const showResultsButton = document.getElementById("btnShowResults") as HTMLButtonElement | null;
const resetButton = document.getElementById("btnResetSelection") as HTMLButtonElement | null;
const fieldRadioGroup = document.querySelector<HTMLInputElement>('input[name="fieldVariable"]')?.closest("label")?.parentElement ?? null;
const fieldStatsHeading = [...document.querySelectorAll<HTMLElement>("h3")].find((item) => item.textContent?.includes("Computed Slice Statistics"));
const fieldStatsSection = fieldStatsHeading?.parentElement?.parentElement ?? null;
const meshHeading = [...document.querySelectorAll<HTMLElement>("h3")].find((item) => item.textContent?.includes("Mesh Sensitivity"));
const meshSection = meshHeading?.closest(".bg-surface-container-lowest") as HTMLElement | null;
const gptOssCard = findGptOssCard(document.getElementById("btnTestGptOss") as HTMLButtonElement | null);
const pageContainer = root;
const apiShell = pageContainer?.children[1] as HTMLElement | undefined;
const setupColumn = apiShell?.querySelector<HTMLElement>("[class*='lg:col-span-4']");
const workspace = setupColumn?.parentElement ?? null;
const resultsColumn = workspace?.children[1] as HTMLElement | undefined;

let scan: DatasetScanStatus | null = null;
let cases: DatasetCase[] = [];
let selectedCase: DatasetCase | null = null;
let fieldOptions: CfdFieldOptions | null = null;
let selectedField: CfdFieldOption | null = null;
let selectedComponent = "value";
let statistics: CfdStatistics | null = null;
let statisticsError: string | null = null;
let gptOssStatus: GptOssStatus | null = null;
let gptOssAnalysis: GptOssAnalysis | null = null;
let currentRequest = 0;
let fieldOptionsRequest = 0;
let meshDetailsRequest = 0;
let visualizationRequest = 0;
let visualizationController: AbortController | null = null;
let activeVisualization: CfdSpatialVisualization | null = null;
let visualizationRotation = { x: -0.48, y: 0.58 };
let visualizationZoom = 1;
let visualizationPan = { x: 0, y: 0 };
let defaultCaseId = "";
let sourceReference = "";
let generatedDetailId = 0;

if (root && caseSelect && workspace && resultsColumn) {
  installStyles();
  prepareSharedHeader();
  prepareHeader();
  prepareWorkflow();
  prepareControls();
  prepareViewport();
  prepareStatistics();
  prepareMeshDetails();
  prepareEducationSections();
  prepareGptOssCard();
  bindControls();
  bindGptOssActions();
  void initialize();
}

function findGptOssCard(button: HTMLButtonElement | null): HTMLElement | null {
  let node = button?.parentElement ?? null;
  while (node && !node.classList.contains("border-l-4")) node = node.parentElement;
  return node;
}

function installStyles() {
  if (document.getElementById("design-dashboard-styles")) return;
  const style = document.createElement("style");
  style.id = "design-dashboard-styles";
  style.textContent = `
    .design-top-meta,.design-education-banner,.design-step-banner { display:none !important; }
    .design-page-root { gap:16px !important; padding-top:18px !important; }
    .design-page-header h1 { font-size:clamp(22px,2.2vw,30px) !important; line-height:1.2 !important; }
    .design-page-header p { margin-top:5px !important; font-size:14px !important; line-height:1.5 !important; }
    .design-workspace { grid-template-columns:minmax(340px,.92fr) minmax(0,1.55fr) !important; gap:20px !important; align-items:start; }
    .design-workspace > * { grid-column:auto !important; min-width:0; }
    .design-setup-card { box-sizing:border-box; padding:20px !important; gap:18px !important; }
    .design-setup-card > .flex.flex-col.gap-space-xs { min-width:0; gap:10px !important; }
    .design-setup-card > .flex.items-center.justify-between { flex-wrap:wrap; gap:8px 12px; padding-bottom:12px !important; }
    #design-selection-state { max-width:100%; font:600 12px/1.4 Inter,sans-serif; text-align:left; }
    #design-case-count { max-width:100%; text-align:right; white-space:normal; }
    .design-setup-card label { min-width:0; }
    .design-setup-card label span { overflow-wrap:anywhere; }
    .design-setup-card .mesh-toggle-btn { box-sizing:border-box; width:100%; min-width:0; min-height:68px; padding:12px 14px !important; gap:12px; }
    .design-setup-card .mesh-toggle-btn > div { min-width:0; flex:1 1 auto; gap:4px; }
    .design-setup-card .mesh-toggle-btn > div > span:first-child { font:600 14px/1.4 Inter,sans-serif; }
    .design-setup-card .mesh-toggle-btn > div > span:nth-child(2) { display:block; white-space:normal; line-height:1.4; }
    .design-setup-card .mesh-toggle-btn > span:last-child { flex:0 0 auto; max-width:40%; padding:4px 9px; font-size:11px; white-space:normal; text-align:center; }
    #meshButtonGroup { display:grid; grid-template-columns:minmax(0,1fr); gap:8px; }
    .design-setup-card input[name="reynolds"] + div { min-width:0; }
    .design-setup-card input[name="reynolds"] { flex:none; margin-top:3px; }
    .design-setup-card input[name="reynolds"] + div > span:first-child { font-size:14px; line-height:1.4; }
    .design-setup-card input[name="reynolds"] + div > span:nth-child(2) { font-size:12px; line-height:1.4; }
    .design-setup-card input[name="reynolds"] { accent-color:#004ac6; }
    #design-flow-options { grid-template-columns:minmax(0,1fr); gap:8px; }
    .design-field-picker { display:grid; gap:8px; }
    .design-field-picker select,.design-component-picker { width:100%; min-height:40px; padding:9px 12px; border:1px solid #c7d6e9; border-radius:8px; color:#0b1c30; background:#eff4ff; font:13px Inter,sans-serif; }
    .design-hint { margin:0; color:#536579; font:12px/1.5 Inter,sans-serif; }
    .design-control-heading { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .design-term-help { position:relative; flex:none; }
    .design-term-help > summary { width:25px; height:25px; display:grid; place-items:center; border-radius:50%; color:#52677e; cursor:pointer; list-style:none; }
    .design-term-help > summary::-webkit-details-marker { display:none; }
    .design-term-help > summary:focus-visible { outline:3px solid #2680ff; outline-offset:2px; }
    .design-term-help[open] { z-index:30; }
    .design-term-help .design-term-copy { position:absolute; right:0; top:calc(100% + 5px); width:min(260px,70vw); padding:10px 12px; border:1px solid #d8e3f2; border-radius:8px; background:#fff; box-shadow:0 8px 24px rgba(11,28,48,.14); color:#43566b; font:12px/1.5 Inter,sans-serif; }
    .design-inline-details { border:1px solid #d8e3f2; border-radius:8px; background:#fff; }
    .design-inline-details > summary { display:flex; justify-content:space-between; align-items:center; gap:10px; padding:11px 13px; cursor:pointer; list-style:none; font:600 13px/1.4 Inter,sans-serif; color:#0b1c30; }
    .design-inline-details > summary::-webkit-details-marker { display:none; }
    .design-inline-details > summary:focus-visible,.design-stat-toggle:focus-visible,.design-expand-button:focus-visible,.design-retry:focus-visible { outline:3px solid #2680ff; outline-offset:3px; }
    .design-arrow { display:inline-block; transition:transform .18s ease; }
    .design-inline-details[open] > summary .design-arrow { transform:rotate(180deg); }
    .design-details-body { padding:0 13px 13px; color:#536579; font:12px/1.55 Inter,sans-serif; animation:design-reveal .16s ease-out; }
    .design-main-view { min-height:460px; position:relative; display:grid; place-items:center; padding:0; color:#536579; background:#f6f9ff; border-top:1px solid #dbe6f5; overflow:hidden; }
    .design-canvas-wrap { position:absolute; inset:0; }
    #design-viz-canvas { display:block; width:100%; height:100%; touch-action:none; cursor:grab; }
    #design-viz-canvas:active { cursor:grabbing; }
    .design-viz-controls { position:absolute; right:12px; top:12px; z-index:2; display:flex; gap:6px; }
    .design-viz-controls[hidden] { display:none; }
    .design-viz-controls button { min-width:36px; min-height:36px; padding:6px 9px; border:1px solid #c7d6e9; border-radius:8px; background:rgba(255,255,255,.94); color:#19344f; font:600 12px Inter,sans-serif; cursor:pointer; }
    .design-viz-controls button:hover { background:#eaf1fc; }
    .design-viz-help { position:absolute; left:12px; bottom:10px; margin:0; color:#43566b; font:11px Inter,sans-serif; background:rgba(255,255,255,.88); padding:5px 8px; border-radius:6px; }
    .design-loading { display:flex; align-items:center; gap:10px; padding:18px; font:13px Inter,sans-serif; }
    .design-spinner { width:17px; height:17px; border:2px solid #d5e1f0; border-top-color:#1768d2; border-radius:50%; animation:design-spin .75s linear infinite; }
    .design-diagnostics { width:100%; border-top:1px solid #dbe6f5; }
    .design-diagnostics summary { padding:9px 14px; cursor:pointer; color:#425970; font:600 11px Inter,sans-serif; }
    .design-diagnostics p { margin:0; padding:0 14px 12px; color:#536579; font:11px/1.5 Inter,sans-serif; }
    .design-visualization-error { padding:18px; text-align:center; }
    .design-visualization-error h3 { margin:0 0 5px; color:#0b1c30; font:600 14px/1.4 Inter,sans-serif; }
    .design-visualization-error p { margin:0; max-width:52ch; font:12px/1.5 Inter,sans-serif; }
    .design-active-context { display:flex; flex-wrap:wrap; gap:5px 12px; padding:10px 14px; color:#31465d; background:#eff4ff; font:600 11px Inter,sans-serif; }
    .design-unavailable { max-width:520px; }
    .design-unavailable h3 { margin:0 0 7px; color:#0b1c30; font:600 16px/1.35 Inter,sans-serif; }
    .design-unavailable p { max-width:64ch; margin:0; font:12px/1.5 Inter,sans-serif; }
    .design-legend { display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px 16px; padding:10px 14px; color:#536579; background:#eff4ff; font:12px Inter,sans-serif; }
    .design-color-legend { display:grid; grid-template-columns:auto minmax(100px,220px) auto; align-items:center; gap:8px; width:100%; max-width:380px; font:11px 'JetBrains Mono',monospace; }
    .design-color-legend[hidden] { display:none; }
    .design-color-bar { height:9px; border-radius:999px; background:linear-gradient(90deg,#2457c5 0%,#28b7d7 33%,#f1db51 66%,#c63837 100%); }
    .design-stats-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 8px; }
    .design-stats-head h3 { margin:0; font-size:15px !important; }
    .design-stats-note { color:#536579; font:11px Inter,sans-serif; }
    .design-stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }
    .design-stat-card { min-width:0; padding:11px; border:1px solid #d8e3f2; border-radius:8px; background:#fff; }
    .design-stat-top { display:flex; align-items:center; justify-content:space-between; gap:5px; color:#536579; font:11px Inter,sans-serif; }
    .design-stat-toggle { width:26px; height:26px; display:grid; place-items:center; flex:none; border:1px solid #dbe4f0; border-radius:50%; background:#fff; color:#34516e; cursor:pointer; }
    .design-stat-toggle[aria-expanded="true"] .design-arrow { transform:rotate(180deg); }
    .design-stat-value { display:block; margin-top:7px; color:#0b1c30; font:600 20px/1.2 'JetBrains Mono',monospace; overflow-wrap:anywhere; }
    .design-stat-unit { display:block; margin-top:3px; color:#61758a; font:11px Inter,sans-serif; }
    .design-stat-detail { margin-top:8px; padding-top:8px; border-top:1px solid #e7edf5; color:#536579; font:11px/1.5 Inter,sans-serif; }
    .design-stat-detail[hidden] { display:none; }
    .design-stat-message,.design-status-message { margin:8px 0 0; color:#536579; font:12px/1.5 Inter,sans-serif; }
    .design-error { color:#a33434; }
    #design-dataset-status { display:inline-flex; width:max-content; max-width:100%; margin-top:10px; padding:5px 9px; border-radius:999px; background:#e5f7f2; color:#126c5d; font:600 11px/1.4 Inter,sans-serif; }
    #design-dataset-status.design-error { background:#fff0ef; }
    .design-statistics-section { padding:12px; border-radius:12px; background:#fff; box-shadow:0 1px 4px rgba(11,28,48,.05); }
    .design-mesh-section { gap:7px !important; padding:14px !important; }
    .design-mesh-section .design-status-message { margin-top:0; }
    .design-retry { border:0; padding:0; color:#004ac6; background:none; font:600 12px Inter,sans-serif; cursor:pointer; }
    .design-mesh-summary { display:flex; justify-content:space-between; align-items:center; gap:10px; }
    .design-mesh-summary h3 { margin:0; font:600 15px Inter,sans-serif; }
    .design-mesh-table-wrap { max-width:100%; overflow-x:auto; }
    .design-mesh-table { width:100%; min-width:760px; border-collapse:collapse; text-align:left; font:12px Inter,sans-serif; }
    .design-mesh-table th,.design-mesh-table td { padding:9px 11px; border-bottom:1px solid #e1eaf5; white-space:nowrap; }
    .design-mesh-table th { color:#31465d; background:#eff4ff; }
    .design-mesh-caption { margin:0 0 10px; color:#536579; font:12px/1.5 Inter,sans-serif; }
    .design-education-grid { display:grid; gap:8px; }
    .design-gpt-oss-card { border-left-width:3px !important; gap:8px !important; }
    .design-gpt-oss-head { display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap; }
    .design-gpt-oss-title { display:flex; align-items:center; gap:10px; }
    .design-gpt-oss-title h2 { margin:0; color:#0b1c30; font:600 15px Inter,sans-serif; }
    .design-gpt-oss-subtitle { margin:3px 0 0; color:#536579; font:11px Inter,sans-serif; }
    .design-gpt-oss-actions { display:flex; gap:8px; flex-wrap:wrap; }
    .design-gpt-oss-actions button { min-height:34px; }
    .design-gpt-oss-result { border:1px solid #d8e3f2; border-radius:8px; background:#f7f9fd; }
    .design-gpt-oss-result[hidden] { display:none; }
    .design-gpt-oss-result summary { padding:11px 13px; cursor:pointer; font:600 13px Inter,sans-serif; }
    .design-gpt-oss-result summary:focus-visible { outline:3px solid #2680ff; outline-offset:2px; }
    .design-gpt-oss-result[open] .design-arrow { transform:rotate(180deg); }
    .design-gpt-oss-result-content { padding:0 13px 13px; }
    .design-gpt-oss-result-content h3 { margin:10px 0 4px; color:#0b1c30; font:600 12px Inter,sans-serif; }
    .design-gpt-oss-result-content p { margin:0; color:#536579; font:12px/1.55 Inter,sans-serif; white-space:pre-wrap; }
    .design-gpt-oss-result summary { display:flex; justify-content:space-between; align-items:center; gap:8px; list-style:none; }
    .design-gpt-oss-result summary::-webkit-details-marker { display:none; }
    @keyframes design-reveal { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:translateY(0); } }
    @media (max-width:1080px) { .design-workspace { grid-template-columns:minmax(0,1fr) !important; } .design-setup-card { width:100%; } .design-stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    .design-stat-top > span:first-child { min-width:0; overflow-wrap:anywhere; }
    @media (max-width:640px) { body > aside.fixed.left-0 { width:56px !important; } body > aside.fixed.left-0 > div:first-child > div:first-child { justify-content:center; padding-left:0; padding-right:0; } body > aside.fixed.left-0 > div:first-child > div:first-child > div:last-child,body > aside.fixed.left-0 > div:first-child > div:nth-child(2),body > aside.fixed.left-0 > div:last-child { display:none !important; } body > aside.fixed.left-0 nav { padding-left:6px; padding-right:6px; } body > aside.fixed.left-0 nav a { justify-content:center; padding-left:0 !important; padding-right:0 !important; font-size:0 !important; } body > aside.fixed.left-0 nav a > span:last-child { display:none !important; } body > .pl-72 { padding-left:56px !important; } .pl-72 > header { left:56px !important; padding-left:12px !important; padding-right:12px !important; } }
    @keyframes design-spin { to { transform:rotate(360deg); } }
    @media (max-width:560px) { .design-stat-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .design-main-view { min-height:320px; padding:0; } .design-legend { display:grid; grid-template-columns:1fr; } .design-gpt-oss-actions { width:100%; } .design-gpt-oss-actions button { flex:1; } .design-gpt-oss-head { align-items:flex-start; } #design-flow-options { grid-template-columns:1fr; } }
    @media (prefers-reduced-motion:reduce) { .design-arrow,.design-details-body { transition:none; animation:none; } }
  `;
  document.head.append(style);
}

function prepareHeader() {
  if (!root) return;
  root.classList.add("design-page-root");
  const header = root.children[0] as HTMLElement | undefined;
  if (!header) return;
  header.classList.add("design-page-header");
  const reference = [...header.querySelectorAll<HTMLElement>("span")].find((element) =>
    element.textContent?.trim().startsWith("Dataset Reference:"),
  );
  sourceReference = reference?.textContent?.trim() ?? "";
  const metaRows = [...header.querySelectorAll<HTMLElement>("div")];
  for (const element of metaRows) {
    if (element.textContent?.includes("Simulation Diagnostic Module") || element.textContent?.includes("Dataset Reference:")) {
      element.classList.add("design-top-meta");
    }
  }
  const title = header.querySelector("h1");
  if (title) title.textContent = "CFD Explorer & Design Analysis";
  const description = title?.parentElement?.querySelector("p");
  if (description) description.textContent = "Explore battery-cooling simulations and understand how temperature and coolant flow vary across the pack.";
  for (const button of header.querySelectorAll<HTMLButtonElement>("button")) {
    if (/Export Slice VTK|Share Case Report/.test(button.textContent ?? "")) button.remove();
  }
  const education = [...header.querySelectorAll<HTMLElement>("div")].find((element) => element.textContent?.includes("What are we exploring?") && element.className.includes("bg-surface-container-lowest"));
  if (education) education.classList.add("design-education-banner");
  const progress = [...header.querySelectorAll<HTMLElement>("div")].find((element) => element.className.includes("min-w-[760px]"));
  progress?.parentElement?.classList.add("design-step-banner");

  const status = document.createElement("span");
  status.id = "design-dataset-status";
  status.className = "design-status-message";
  status.textContent = "Checking dataset status…";
  status.setAttribute("role", "status");
  header.append(status);
}

function prepareSharedHeader() {
  document.querySelectorAll<HTMLAnchorElement>("body > aside nav a").forEach((link) => {
    const label = link.querySelector("span:last-child")?.textContent?.trim() ?? link.textContent?.trim();
    if (label) link.setAttribute("aria-label", label);
  });
  const quickDocs = [...document.querySelectorAll<HTMLAnchorElement>("body > aside a")].find((link) => link.textContent?.includes("Quick Docs"));
  quickDocs?.addEventListener("click", (event) => {
    event.preventDefault();
    const firstEducation = document.querySelector<HTMLDetailsElement>(".design-education-grid details");
    if (!firstEducation) return;
    firstEducation.open = true;
    firstEducation.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  });
  const header = document.querySelector<HTMLElement>(".pl-72 > header");
  if (!header) return;
  const left = header.children[0] as HTMLElement | undefined;
  const study = left?.querySelector<HTMLElement>("span");
  if (study) study.textContent = "Battery cooling CHT dataset";
  left?.querySelector("div.hidden.md\\:flex")?.remove();
  const right = header.children[1] as HTMLElement | undefined;
  if (!right) return;
  const datasetBadge = [...right.querySelectorAll<HTMLElement>("div")].find((item) => item.textContent?.includes("Dataset: Ready"));
  datasetBadge?.remove();
  const hardware = [...right.querySelectorAll<HTMLElement>("div")].find((item) => item.querySelector("span")?.textContent?.trim() === "bolt");
  hardware?.remove();
  right.querySelector("button[title='Help & Thermal Glossary']")?.parentElement?.remove();
  right.querySelector("img[alt='Profile']")?.parentElement?.remove();
}

function prepareWorkflow() {
  if (!root || !workspace || !setupColumn) return;
  workspace.classList.add("design-workspace");
  setupColumn.classList.add("design-setup-card");
  const title = [...setupColumn.querySelectorAll<HTMLElement>("span")].find((element) => element.textContent?.trim() === "Simulation Selector");
  if (title) title.textContent = "Simulation Setup";
  const stateLabel = [...setupColumn.querySelectorAll<HTMLElement>("span")].find((element) => element.textContent?.includes("State: Synchronized"));
  if (stateLabel) {
    stateLabel.id = "design-selection-state";
    stateLabel.textContent = "Loading scanned cases…";
  }
  const caseCount = [...setupColumn.querySelectorAll<HTMLElement>("span")].find((element) => element.textContent?.includes("2 cases loaded"));
  if (caseCount) caseCount.id = "design-case-count";
  const setupIntro = setupColumn.querySelector("p.font-body-sm");
  if (setupIntro && setupIntro.textContent?.includes("10-Cell Conjugate")) setupIntro.textContent = "Choose a verified case, flow condition, mesh, and field.";
  for (const hint of [...setupColumn.querySelectorAll<HTMLElement>("p.font-caption")]) {
    if (hint.textContent?.includes("10-Cell Conjugate")) hint.textContent = "Select from the cases currently in the dataset scan.";
  }
}

function prepareControls() {
  if (!root || !fieldRadioGroup || !meshButtons.length) return;
  const fieldLabel = fieldRadioGroup.closest("div.flex.flex-col") as HTMLElement | null;
  if (fieldLabel) {
    fieldLabel.innerHTML = `<div class="flex flex-col gap-2"><div class="design-control-heading"><label class="font-headline-sm text-body-md text-on-surface font-semibold" for="design-field-select">4. Select a physical field</label><details class="design-term-help"><summary aria-label="About physical fields"><span class="material-symbols-outlined" aria-hidden="true">info</span></summary><div class="design-term-copy">A physical field is a value stored by the solver, such as temperature (T), velocity (U), or pressure (p).</div></details></div><div class="design-field-picker"><select id="design-field-select" aria-label="Physical field" disabled><option>Loading fields…</option></select><select id="design-component-select" class="design-component-picker" aria-label="Field component" hidden></select></div><p class="design-hint" id="design-field-hint">Select a stored field to summarize.</p></div>`;
  }
  const flowGroup = setupColumn?.querySelector<HTMLElement>(".grid.grid-cols-1.gap-2");
  if (flowGroup?.querySelector('input[name="reynolds"]')) flowGroup.id = "design-flow-options";
  for (const [label, explanation] of [
    ["Choose Flow Condition", "Reynolds number helps describe the flow regime. It is read from the selected case metadata, not calculated as a performance score."],
    ["Choose Mesh Resolution", "Mesh resolution describes how finely the simulation domain was divided for calculation. Only resolutions present in the dataset scan can be selected."],
  ]) {
    const labelElement = [...(setupColumn?.querySelectorAll<HTMLElement>("label") ?? [])].find((element) =>
      element.textContent?.includes(label),
    );
    const header = labelElement?.parentElement;
    const tooltip = header?.querySelector<HTMLElement>(".group");
    if (!header || !tooltip) continue;
    const info = document.createElement("details");
    info.className = "design-term-help";
    const summary = document.createElement("summary");
    summary.setAttribute("aria-label", `About ${label === "Choose Flow Condition" ? "Reynolds number" : "mesh resolution"}`);
    summary.innerHTML = '<span class="material-symbols-outlined" aria-hidden="true">info</span>';
    const text = document.createElement("div");
    text.className = "design-term-copy";
    text.textContent = explanation;
    info.append(summary, text);
    tooltip.replaceWith(info);
  }
  const setupDetails = [...(setupColumn?.querySelectorAll<HTMLDetailsElement>("details") ?? [])]
    .find((details) => details.querySelector("summary")?.textContent?.includes("Advanced Slice & Plane Options"));
  if (setupDetails) {
    setupDetails.innerHTML = `<summary>Simulation information <span class="material-symbols-outlined text-[18px] design-arrow" aria-hidden="true">expand_more</span></summary><div class="design-details-body" id="design-simulation-information"></div>`;
    setupDetails.classList.add("design-inline-details");
    setupDetails.id = "design-simulation-details";
  }
}

function prepareViewport() {
  if (!root || !resultsColumn) return;
  const visualizationCard = resultsColumn.children[0] as HTMLElement | undefined;
  if (!visualizationCard) return;
  visualizationCard.innerHTML = `<div class="px-space-md py-space-sm bg-surface-container-low flex items-center justify-between gap-space-xs border-b border-surface-container-high/40"><h2 class="font-headline-sm text-headline-sm text-on-surface">Simulation Visualization</h2><span id="design-visualization-kind" class="font-label-code text-caption text-on-surface-variant">Loading simulation data…</span></div><div id="design-active-context" class="design-active-context"><span id="design-active-case">Case: Loading</span><span id="design-active-flow">Flow: Loading</span><span id="design-active-mesh">Mesh: Loading</span><span id="design-active-field">Field: Loading</span></div><div class="design-main-view"><canvas id="design-viz-canvas" role="img" aria-label="Interactive scientific boundary-surface visualization" hidden></canvas><div id="design-visualization-state" class="design-visualization-error" role="status"><div class="design-loading"><span class="design-spinner" aria-hidden="true"></span><span>Loading simulation data…</span></div></div><div id="design-viz-controls" class="design-viz-controls" hidden><button id="design-viz-zoom-in" type="button" aria-label="Zoom in">+</button><button id="design-viz-zoom-out" type="button" aria-label="Zoom out">−</button><button id="design-viz-reset" type="button" aria-label="Reset view">Reset view</button></div><p id="design-viz-help" class="design-viz-help" hidden>Drag to rotate · Shift-drag to pan · Scroll to zoom</p></div><div class="design-legend"><span id="design-legend-field">Field: Unavailable</span><span id="design-legend-units">Units: Unavailable</span><span id="design-legend-state">Waiting for the selected case and field.</span><div id="design-color-legend" class="design-color-legend" hidden><span id="design-color-min"></span><span class="design-color-bar" aria-hidden="true"></span><span id="design-color-max"></span></div></div><details class="design-diagnostics"><summary>Visualization diagnostics</summary><p id="design-visualization-diagnostic">Select a scanned case and field to load its boundary surface.</p></details>`;
  bindVisualizationControls();
}

function bindVisualizationControls() {
  const canvas = document.getElementById("design-viz-canvas") as HTMLCanvasElement | null;
  if (!canvas) return;
  let drag: { x: number; y: number; pan: boolean } | null = null;
  canvas.addEventListener("pointerdown", (event) => {
    drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2 };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (drag.pan) {
      visualizationPan.x += deltaX;
      visualizationPan.y += deltaY;
    } else {
      visualizationRotation.y += deltaX * 0.009;
      visualizationRotation.x += deltaY * 0.009;
    }
    drawVisualization();
  });
  const endDrag = () => { drag = null; };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    visualizationZoom = Math.max(0.35, Math.min(8, visualizationZoom * (event.deltaY < 0 ? 1.12 : 0.89)));
    drawVisualization();
  }, { passive: false });
  document.getElementById("design-viz-zoom-in")?.addEventListener("click", () => {
    visualizationZoom = Math.min(8, visualizationZoom * 1.2);
    drawVisualization();
  });
  document.getElementById("design-viz-zoom-out")?.addEventListener("click", () => {
    visualizationZoom = Math.max(0.35, visualizationZoom / 1.2);
    drawVisualization();
  });
  document.getElementById("design-viz-reset")?.addEventListener("click", resetVisualizationView);
  window.addEventListener("resize", drawVisualization);
  if ("ResizeObserver" in window) {
    new ResizeObserver(drawVisualization).observe(canvas);
  }
}

function resetVisualizationView() {
  visualizationRotation = { x: -0.48, y: 0.58 };
  visualizationZoom = 1;
  visualizationPan = { x: 0, y: 0 };
  drawVisualization();
}

function drawVisualization() {
  const canvas = document.getElementById("design-viz-canvas") as HTMLCanvasElement | null;
  const data = activeVisualization;
  if (!canvas || !data || canvas.hidden) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  if (!context) {
    setVisualizationMessage("Visualization unavailable", "This browser could not create a 2D rendering surface.");
    return;
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  const bounds = data.bounds;
  const center = [(bounds[0] + bounds[3]) / 2, (bounds[1] + bounds[4]) / 2, (bounds[2] + bounds[5]) / 2];
  const span = Math.max(bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2], Number.EPSILON);
  const scale = Math.min(rect.width, rect.height) * 0.43 * visualizationZoom / span;
  const cosY = Math.cos(visualizationRotation.y);
  const sinY = Math.sin(visualizationRotation.y);
  const cosX = Math.cos(visualizationRotation.x);
  const sinX = Math.sin(visualizationRotation.x);
  const projected = data.points.map(([rawX, rawY, rawZ]) => {
    const x = rawX - center[0];
    const y = rawY - center[1];
    const z = rawZ - center[2];
    const rotatedX = x * cosY + z * sinY;
    const rotatedZ = -x * sinY + z * cosY;
    const rotatedY = y * cosX - rotatedZ * sinX;
    return {
      x: rect.width / 2 + visualizationPan.x + rotatedX * scale,
      y: rect.height / 2 + visualizationPan.y - rotatedY * scale,
      z: y * sinX + rotatedZ * cosX,
    };
  });
  const hasMappedField = data.values.length === data.points.length && data.scalar_range !== null;
  const scalarRange = data.scalar_range;
  const polygons = data.triangles.map((triangle, index) => ({
    triangle,
    depth: (projected[triangle[0]].z + projected[triangle[1]].z + projected[triangle[2]].z) / 3,
    value: hasMappedField
      ? (data.values[triangle[0]] + data.values[triangle[1]] + data.values[triangle[2]]) / 3
      : 0.5,
    index,
  })).sort((left, right) => left.depth - right.depth);
  for (const polygon of polygons) {
    const [a, b, c] = polygon.triangle.map((index) => projected[index]);
    const relative = !hasMappedField || !scalarRange || scalarRange[0] === scalarRange[1]
      ? 0.5
      : (polygon.value - scalarRange[0]) / (scalarRange[1] - scalarRange[0]);
    context.beginPath();
    context.moveTo(a.x, a.y);
    context.lineTo(b.x, b.y);
    context.lineTo(c.x, c.y);
    context.closePath();
    context.fillStyle = hasMappedField ? visualizationColor(relative) : "#6c91bd";
    context.fill();
  }
}

function visualizationColor(value: number) {
  const stops = [
    [36, 87, 197],
    [40, 183, 215],
    [241, 219, 81],
    [198, 56, 55],
  ];
  const position = Math.max(0, Math.min(0.999999, value)) * (stops.length - 1);
  const first = Math.floor(position);
  const fraction = position - first;
  const color = stops[first].map((channel, index) =>
    Math.round(channel + (stops[first + 1][index] - channel) * fraction),
  );
  return `rgb(${color[0]} ${color[1]} ${color[2]})`;
}

function prepareStatistics() {
  if (!fieldStatsSection || !fieldStatsHeading) return;
  fieldStatsSection.classList.add("design-statistics-section");
  fieldStatsHeading.textContent = "Selected field statistics";
  const oldDomainLabel = [...fieldStatsSection.querySelectorAll<HTMLElement>("span")].find((element) => element.textContent?.trim() === "Valid CHT Domain");
  if (oldDomainLabel) oldDomainLabel.textContent = "Stored solver values";
  const oldGrid = [...fieldStatsSection.querySelectorAll<HTMLElement>("div")].find((element) => element.className.includes("grid-cols-2") && element.textContent?.includes("Minimum Value"));
  if (oldGrid) {
    oldGrid.id = "design-stat-grid";
    oldGrid.className = "design-stat-grid";
    oldGrid.innerHTML = [
      ["min", "Minimum value", "ac_unit", "Minimum"],
      ["max", "Maximum value", "local_fire_department", "Maximum"],
      ["mean", "Average value", "equalizer", "Average"],
      ["count", "Domain values", "grid_4x4", "Values"],
    ].map(([id, label, icon, detail]) => `<article class="design-stat-card"><div class="design-stat-top"><span${id === "count" ? ' id="stat-count-label"' : ""}>${label}</span><span class="material-symbols-outlined" aria-hidden="true">${icon}</span><button class="design-stat-toggle" type="button" data-stat-toggle="${id}-detail" aria-controls="${id}-detail" aria-expanded="false" aria-label="Show ${label.toLowerCase()} details"><span class="material-symbols-outlined design-arrow" aria-hidden="true">expand_more</span></button></div><strong class="design-stat-value" id="stat-${id}">Unavailable</strong><span class="design-stat-unit" id="stat-${id}-unit">Choose a field and show results</span><div class="design-stat-detail" id="${id}-detail" hidden>${detail}</div></article>`).join("");
  }
  for (const element of [...fieldStatsSection.querySelectorAll<HTMLElement>("div,p")]) {
    if (element.textContent?.includes("What do these numbers tell us?") || element.textContent?.includes("total temperature spread")) element.remove();
  }
  const status = document.createElement("p");
  status.id = "design-statistics-status";
  status.className = "design-stat-message";
  status.setAttribute("role", "status");
  status.textContent = "Choose a case and field, then select Show Results.";
  fieldStatsSection.append(status);
}

function prepareMeshDetails() {
  if (!meshSection) return;
  meshSection.classList.add("design-mesh-section");
  meshSection.innerHTML = `<div class="design-mesh-summary"><h3>Mesh Sensitivity &amp; Verification</h3><button id="mesh-expand" class="design-stat-toggle" type="button" aria-controls="mesh-details" aria-expanded="false" aria-label="Show mesh sensitivity details"><span class="material-symbols-outlined design-arrow" aria-hidden="true">expand_more</span></button></div><p id="mesh-summary" class="design-status-message" role="status">Waiting for scanned mesh metadata.</p><div id="mesh-details" hidden><p class="design-mesh-caption">Mesh sensitivity checks whether computed values change across resolutions. Similar results are not proof of physical accuracy. Cross-mesh mapping is not validated by the current backend.</p><div id="mesh-comparison-status" class="design-status-message">Expand to load available measurements.</div><div class="design-mesh-table-wrap"><table class="design-mesh-table"><thead><tr><th>Mesh level</th><th>Cell count</th><th>Maximum T (K)</th><th>Pressure drop</th><th>Discretization difference</th><th>Status</th></tr></thead><tbody id="mesh-comparison-rows"><tr><td colspan="6">No comparison loaded.</td></tr></tbody></table></div></div>`;
}

function prepareEducationSections() {
  if (!resultsColumn) return;
  const education = document.createElement("div");
  education.className = "design-education-grid";
  education.innerHTML = [
    ["What is CFD?", "Computational Fluid Dynamics uses numerical methods to estimate how fluids move and transfer heat. This page inspects saved solver output; it does not run a new simulation."],
    ["Understanding the cooling system", "Battery cells generate heat. Coolant flows through channels in the pack and carries some of that heat away. The selected case and field determine which stored values are summarized."],
    ["Understanding Reynolds number", "Reynolds number helps describe the nature of fluid flow. The values here come from scanned case metadata; they are not performance scores."],
    ["Understanding temperature uniformity", "Temperature differences across cells may matter for pack operation. This page reports field statistics, not a safety threshold or engineering conclusion."],
    ["How to read the color map", "The color map shows the selected solver field on the extracted boundary surface. Cell-associated values are interpolated to surface points for display; the statistics above are calculated from the original internal solver values. This is not a volume rendering."],
  ].map(([title, description]) => `<details class="design-inline-details"><summary>${title}<span class="material-symbols-outlined design-arrow" aria-hidden="true">expand_more</span></summary><div class="design-details-body">${description}</div></details>`).join("");
  const position = meshSection?.nextSibling;
  resultsColumn.insertBefore(education, position ?? null);
  bindExpandableDetails(education);
}

function bindControls() {
  caseSelect?.addEventListener("change", () => {
    selectCase(caseSelect.value);
  });
  document.querySelectorAll<HTMLInputElement>('input[name="reynolds"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const candidate = cases.find((item) => item.metadata.reynolds_number === Number(radio.value) && item.metadata.mesh_resolution === selectedCase?.metadata.mesh_resolution)
        ?? cases.find((item) => item.metadata.reynolds_number === Number(radio.value));
      if (candidate) selectCase(candidate.case_id);
    });
  });
  for (const button of meshButtons) {
    button.addEventListener("click", () => {
      const candidate = cases.find((item) => item.metadata.reynolds_number === selectedCase?.metadata.reynolds_number && item.metadata.mesh_resolution === button.dataset.mesh);
      if (candidate) selectCase(candidate.case_id);
    });
  }
  document.getElementById("design-field-select")?.addEventListener("change", (event) => {
    const target = event.currentTarget as HTMLSelectElement;
    selectedField = fieldOptions?.fields.find((field) => fieldKey(field) === target.value) ?? null;
    selectedComponent = selectedField?.components[0] ?? "value";
    renderComponentSelect();
    updateViewportMetadata();
    clearGptOssResult();
    void loadSpatialVisualization();
    void loadStatistics();
  });
  document.getElementById("design-component-select")?.addEventListener("change", (event) => {
    selectedComponent = (event.currentTarget as HTMLSelectElement).value;
    updateViewportMetadata();
    clearGptOssResult();
    void loadSpatialVisualization();
    void loadStatistics();
  });
  showResultsButton?.addEventListener("click", () => {
    void loadSpatialVisualization();
    void loadStatistics();
  });
  resetButton?.addEventListener("click", () => {
    if (defaultCaseId) {
      resetVisualizationView();
      selectCase(defaultCaseId);
    }
  });
  for (const button of document.querySelectorAll<HTMLButtonElement>(".design-stat-toggle:not(#mesh-expand)")) {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.getAttribute("aria-controls") ?? "");
      if (!target) return;
      const expanded = button.getAttribute("aria-expanded") === "true";
      button.setAttribute("aria-expanded", String(!expanded));
      target.hidden = expanded;
    });
  }
  document.getElementById("mesh-expand")?.addEventListener("click", (event) => {
    const button = event.currentTarget as HTMLButtonElement;
    const panel = document.getElementById("mesh-details");
    if (!panel) return;
    const expanded = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!expanded));
    panel.hidden = expanded;
    if (!expanded) void loadMeshDetails();
  });
  for (const details of document.querySelectorAll<HTMLDetailsElement>("#design-simulation-details, #gpt-oss-result")) {
    bindExpandableDetails(details.parentElement ?? details);
  }
}

function bindExpandableDetails(container: ParentNode) {
  for (const details of container.querySelectorAll<HTMLDetailsElement>("details")) {
    const summary = details.querySelector<HTMLElement>(":scope > summary");
    if (!summary) continue;
    if (!details.id) details.id = `design-details-${++generatedDetailId}`;
    summary.setAttribute("aria-expanded", String(details.open));
    const content = [...details.children].find((child) => child !== summary) as HTMLElement | undefined;
    if (content && !content.id) content.id = `${details.id}-content`;
    summary.setAttribute("aria-controls", content?.id ?? `${details.id}-content`);
    details.addEventListener("toggle", () => {
      summary.setAttribute("aria-expanded", String(details.open));
    });
  }
}

async function initialize() {
  if (!caseSelect) return;
  setStatus("Loading verified dataset cases…");
  try {
    const [scanResult, casesResult, gptOssResult] = await Promise.allSettled([
      getDatasetScan(),
      getDatasetCases(),
      getGptOssStatus(),
    ]);
    if (casesResult.status === "rejected") throw casesResult.reason;
    cases = casesResult.value.cases;
    scan = scanResult.status === "fulfilled" ? scanResult.value : null;
    gptOssStatus = gptOssResult.status === "fulfilled" ? gptOssResult.value : null;
    renderGptOssStatus(gptOssResult.status === "rejected" ? errorMessage(gptOssResult.reason) : undefined);
    const caseIds = new Set(cases.map((item) => item.case_id));
    const initialCase = cases.find((item) => caseIds.has("Re560/coarse") && item.case_id === "Re560/coarse") ?? cases[0];
    if (!initialCase) {
      setStatus("No scanned CFD cases are available. Configure the dataset and refresh its scan.", true);
      setSelectionState("No scanned cases");
      return;
    }
    defaultCaseId = initialCase.case_id;
    renderCaseSelect();
    selectCase(initialCase.case_id);
    setText("#design-case-count", `${cases.length} scanned cases`);
    setStatus(scan?.status === "scanned" ? `${cases.length} scanned cases available.` : scan ? `Dataset status: ${scan.status}.` : "Dataset scan unavailable.", scanResult.status === "rejected");
  } catch (error) {
    setStatus(errorMessage(error), true);
    setSelectionState("Dataset unavailable");
    renderGptOssStatus(errorMessage(error));
  }
}

function renderCaseSelect() {
  if (!caseSelect) return;
  caseSelect.replaceChildren();
  for (const item of cases) {
    const option = document.createElement("option");
    option.value = item.case_id;
    option.textContent = `${item.case_id} · Re ${item.metadata.reynolds_number ?? "unknown"} · ${item.metadata.mesh_resolution ?? "mesh unknown"}`;
    caseSelect.append(option);
  }
}

function selectCase(caseId: string) {
  const nextCase = cases.find((item) => item.case_id === caseId);
  if (!nextCase) return;
  visualizationController?.abort();
  visualizationController = null;
  selectedCase = nextCase;
  visualizationRequest += 1;
  activeVisualization = null;
  setVisualizationMessage("Loading simulation data…", `Loading data for ${nextCase.case_id}.`);
  resetVisualizationView();
  meshDetailsRequest += 1;
  if (caseSelect) caseSelect.value = caseId;
  updateFlowRadios();
  updateMeshButtons();
  updateViewportMetadata();
  renderSimulationInformation();
  clearGptOssResult();
  void loadFieldOptions();
  const meshButton = document.getElementById("mesh-expand");
  if (meshButton?.getAttribute("aria-expanded") === "true") void loadMeshDetails();
}

async function loadFieldOptions() {
  const activeCase = selectedCase;
  if (!activeCase) return;
  const requestId = ++fieldOptionsRequest;
  currentRequest += 1;
  const select = document.getElementById("design-field-select") as HTMLSelectElement | null;
  if (select) {
    select.disabled = true;
    select.replaceChildren(new Option("Loading fields…", ""));
  }
  fieldOptions = null;
  selectedField = null;
  statistics = null;
  statisticsError = null;
  const component = document.getElementById("design-component-select") as HTMLSelectElement | null;
  if (component) component.hidden = true;
  renderStatistics();
  setVisualizationMessage("Loading simulation data…", "Reading fields and preparing the selected case surface.");
  try {
    const result = await getCfdFieldOptions(activeCase.case_id);
    if (requestId !== fieldOptionsRequest || selectedCase?.case_id !== activeCase.case_id) return;
    fieldOptions = result;
    const fields = result.fields.filter((field) => field.statistics_supported);
    if (select) {
      select.replaceChildren();
      for (const field of fields) {
        const option = new Option(`${field.region} / ${field.name}${field.units ? ` (${field.units})` : ""}`, fieldKey(field));
        select.add(option);
      }
      select.disabled = fields.length === 0;
    }
    if (fields.length) {
      selectedField = fields.find((field) => field.region === "coolant" && field.name === "T") ?? fields[0];
      if (select && selectedField) select.value = fieldKey(selectedField);
      selectedComponent = selectedField?.components[0] ?? "value";
      renderComponentSelect();
      updateViewportMetadata();
      renderSimulationInformation();
      if (document.getElementById("design-field-hint")) document.getElementById("design-field-hint")!.textContent = `${fields.length} available fields.`;
      void loadSpatialVisualization();
      await loadStatistics();
    } else {
      setVisualizationMessage("No supported field", "No statistics-supported physical field was found in this case.");
      renderStatistics("No supported field is available for this case.");
    }
  } catch (error) {
    if (requestId !== fieldOptionsRequest || selectedCase?.case_id !== activeCase.case_id) return;
    setVisualizationMessage("Field metadata unavailable", errorMessage(error));
    renderStatistics(errorMessage(error), true);
    setStatus(errorMessage(error), true);
  }
}

async function loadStatistics() {
  if (!selectedCase || !selectedField) return;
  const activeCase = selectedCase;
  const activeField = selectedField;
  const activeComponent = selectedComponent;
  const requestId = ++currentRequest;
  statisticsError = null;
  gptOssAnalysis = null;
  renderStatistics();
  updateGptOssButtons();
  const message = document.getElementById("design-statistics-status");
  if (message) {
    message.textContent = "Computing statistics from stored OpenFOAM values…";
    message.classList.remove("design-error");
  }
  if (showResultsButton) {
    showResultsButton.disabled = true;
    showResultsButton.setAttribute("aria-busy", "true");
  }
  try {
    const result = await getCfdStatistics(activeCase.case_id, activeField.region, activeField.name, activeComponent);
    if (requestId !== currentRequest || selectedCase?.case_id !== activeCase.case_id || fieldKey(selectedField ?? activeField) !== fieldKey(activeField) || selectedComponent !== activeComponent) return;
    statistics = result;
    statisticsError = null;
    const actualCount = result.sample_count;
    if (message) message.textContent = actualCount === null
      ? `${result.statistic_scope}; mesh-element count is not inferred for a uniform definition.`
      : `${result.statistic_scope}; ${actualCount.toLocaleString()} values included at solution time ${result.time}.`;
  } catch (error) {
    if (requestId !== currentRequest || selectedCase?.case_id !== activeCase.case_id || fieldKey(selectedField ?? activeField) !== fieldKey(activeField) || selectedComponent !== activeComponent) return;
    statistics = null;
    statisticsError = errorMessage(error);
    if (message) message.textContent = statisticsError;
    message?.classList.add("design-error");
  } finally {
    if (showResultsButton && requestId === currentRequest) {
      showResultsButton.disabled = false;
      showResultsButton.removeAttribute("aria-busy");
    }
    if (requestId === currentRequest) {
      renderStatistics(statisticsError ?? undefined, Boolean(statisticsError));
      updateGptOssButtons();
    }
  }
}

function renderStatistics(message?: string, error = false) {
  const values = statistics;
  const units = values?.units ?? selectedField?.units ?? "Units not stated";
  const association = values?.association ?? selectedField?.association;
  setText("#stat-min", values ? formatNumber(values.minimum) : "Unavailable");
  setText("#stat-max", values ? formatNumber(values.maximum) : "Unavailable");
  setText("#stat-mean", values ? formatNumber(values.mean) : "Unavailable");
  setText("#stat-count", values?.sample_count === null || values?.sample_count === undefined ? "Unavailable" : values.sample_count.toLocaleString());
  const countLabel = association === "face" ? "Domain faces" : association === "cell" ? "Domain cells" : "Domain values";
  setText("#stat-count-label", countLabel);
  const countButton = document.querySelector<HTMLButtonElement>('[data-stat-toggle="count-detail"]');
  countButton?.setAttribute("aria-label", `Show ${countLabel.toLowerCase()} details`);
  for (const id of ["min", "max", "mean"]) setText(`#stat-${id}-unit`, values ? units : "Waiting for computed values");
  setText("#stat-count-unit", values?.sample_count == null ? "Unavailable" : association === "face" ? "Faces included" : association === "cell" ? "Cells included" : "Values included");
  setText("#min-detail", values ? `Minimum across ${values.statistic_scope}. The API does not return the location of this value.` : "Minimum value is unavailable until a calculation succeeds.");
  setText("#max-detail", values ? `Maximum across ${values.statistic_scope}. The API does not return the location of this value.` : "Maximum value is unavailable until a calculation succeeds.");
  setText("#mean-detail", values ? `Arithmetic mean of the stored internal values. ${values.statistic_scope}.` : "Average is unavailable until a calculation succeeds.");
  setText("#count-detail", values?.sample_count == null ? "A uniform field definition does not provide a count of mesh elements." : `${values.sample_count.toLocaleString()} stored ${association === "face" ? "face" : "cell"} values were included in the selected field summary.`);
  const status = document.getElementById("design-statistics-status");
  if (status && message) {
    status.textContent = message;
    status.classList.toggle("design-error", error);
  }
}

function renderComponentSelect() {
  const component = document.getElementById("design-component-select") as HTMLSelectElement | null;
  if (!component || !selectedField) return;
  component.replaceChildren();
  for (const value of selectedField.components) component.add(new Option(componentLabel(value, selectedField), value));
  component.hidden = selectedField.components.length <= 1;
  component.value = selectedComponent;
}

function updateFlowRadios() {
  const flowValues = [...new Set(cases.map((item) => item.metadata.reynolds_number).filter((value): value is number => value !== null))].sort((a, b) => a - b);
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="reynolds"]')) {
    const value = Number(radio.value);
    const supported = flowValues.includes(value);
    radio.disabled = !supported;
    radio.checked = supported && value === selectedCase?.metadata.reynolds_number;
    const label = radio.closest("label");
    label?.classList.toggle("opacity-50", !supported);
    const text = label?.querySelector("span.font-body-sm");
    if (text && supported) text.textContent = `Re = ${value}`;
    const description = label?.querySelector("span.font-caption");
    if (description && supported) description.textContent = "Recorded condition";
  }
}

function updateMeshButtons() {
  const meshCases = cases.filter((item) => item.metadata.reynolds_number === selectedCase?.metadata.reynolds_number);
  const available = new Set(meshCases.map((item) => item.metadata.mesh_resolution).filter((value): value is string => Boolean(value)));
  const selectedMesh = selectedCase?.metadata.mesh_resolution;
  for (const button of meshButtons) {
    const level = button.dataset.mesh ?? "";
    const isAvailable = available.has(level);
    const isSelected = level === selectedMesh;
    button.disabled = !isAvailable;
    button.classList.toggle("opacity-50", !isAvailable);
    button.classList.toggle("cursor-not-allowed", !isAvailable);
    button.classList.toggle("bg-surface-container-high", isSelected);
    button.classList.toggle("bg-surface-container-low", !isSelected && isAvailable);
    const badge = button.lastElementChild;
    if (badge) badge.textContent = isSelected ? "Selected" : isAvailable ? "Available" : "Unavailable";
    const title = button.querySelector("span.font-body-sm");
    if (title) title.textContent = `${capitalize(level)} mesh`;
    const description = button.querySelector("span.font-caption");
    if (description) description.textContent = isAvailable ? "Present in scanned cases" : "Not present in this flow condition";
  }
  const label = document.querySelector<HTMLElement>("#design-selection-state");
  if (label) label.textContent = `${available.size} mesh levels available`;
  const summary = document.getElementById("mesh-summary");
  if (summary) {
    const levels = [...available].sort((a, b) => meshOrder(a) - meshOrder(b));
    const flow = selectedCase?.metadata.reynolds_number == null ? "" : ` for Re ${selectedCase.metadata.reynolds_number}`;
    summary.textContent = `Verification not established · ${levels.length} mesh levels${flow}: ${levels.map(capitalize).join(", ") || "none"} available. Primary metric: maximum coolant temperature; cross-mesh convergence delta is not calculated.`;
  }
}

function updateViewportMetadata() {
  if (!selectedCase) return;
  setText("#design-active-case", `Case: ${selectedCase.case_id}`);
  setText("#design-active-flow", selectedCase.metadata.reynolds_number === null ? "Flow: unavailable" : `Flow: Re ${selectedCase.metadata.reynolds_number}`);
  setText("#design-active-mesh", selectedCase.metadata.mesh_resolution ? `Mesh: ${capitalize(selectedCase.metadata.mesh_resolution)}` : "Mesh: unavailable");
  setText("#design-active-field", selectedField ? `Field: ${selectedField.region}/${selectedField.name}${selectedField.units ? ` (${selectedField.units})` : ""}` : "Field: loading");
  setText("#design-legend-field", selectedField ? `Field: ${selectedField.region}/${selectedField.name}` : "Field: Unavailable");
  setText("#design-legend-units", `Units: ${selectedField?.units ?? "not stated"}`);
}

async function loadSpatialVisualization() {
  const activeCase = selectedCase;
  const activeField = selectedField;
  const activeComponent = selectedComponent;
  if (!activeCase || !activeField) {
    setVisualizationMessage("No selected physical field", "Choose a supported field to load its case surface.");
    return;
  }
  visualizationController?.abort();
  const controller = new AbortController();
  visualizationController = controller;
  const requestId = ++visualizationRequest;
  setVisualizationMessage("Loading simulation data…", `Reading ${activeField.region}/${activeField.name} for ${activeCase.case_id}.`);
  try {
    const result = await getCfdSpatialVisualization(
      activeCase.case_id,
      activeField.region,
      activeField.name,
      activeComponent,
      controller.signal,
    );
    if (
      requestId !== visualizationRequest
      || selectedCase?.case_id !== activeCase.case_id
      || fieldKey(selectedField ?? activeField) !== fieldKey(activeField)
      || selectedComponent !== activeComponent
    ) return;
    if (!isValidVisualization(result)) {
      throw new Error("The visualization API returned incomplete or invalid geometry.");
    }
    renderSpatialVisualization(result);
  } catch (error) {
    if (controller.signal.aborted) return;
    if (
      requestId !== visualizationRequest
      || selectedCase?.case_id !== activeCase.case_id
      || fieldKey(selectedField ?? activeField) !== fieldKey(activeField)
      || selectedComponent !== activeComponent
    ) return;
    setVisualizationMessage(
      "Visualization unavailable",
      errorMessage(error),
    );
    setText("#design-visualization-kind", "Unable to render selected data");
  } finally {
    if (visualizationController === controller) visualizationController = null;
  }
}

function isValidVisualization(value: CfdSpatialVisualization) {
  if (
    !Array.isArray(value.points)
    || value.points.length === 0
    || !Array.isArray(value.triangles)
    || value.triangles.length === 0
    || !Array.isArray(value.values)
  ) return false;

  const fieldDataIsValid = value.geometry_type === "surface_geometry"
    ? value.region === null
      && value.field === null
      && value.values.length === 0
      && value.scalar_range === null
    : value.region !== null
      && value.field !== null
      && value.values.length === value.points.length
      && Array.isArray(value.scalar_range)
      && value.scalar_range.length === 2
      && value.scalar_range.every(Number.isFinite);
  return fieldDataIsValid
    && value.triangles.every((triangle) =>
      triangle.length === 3
      && triangle.every((index) =>
        Number.isInteger(index) && index >= 0 && index < value.points.length,
      ),
    )
    && value.points.every((point) =>
      point.length === 3 && point.every(Number.isFinite),
    )
    && value.values.every(Number.isFinite)
    && value.bounds.length === 6
    && value.bounds.every(Number.isFinite);
}

function renderSpatialVisualization(data: CfdSpatialVisualization) {
  const canvas = document.getElementById("design-viz-canvas") as HTMLCanvasElement | null;
  const state = document.getElementById("design-visualization-state");
  if (!canvas || !state) {
    setVisualizationMessage("Visualization unavailable", "The rendering surface could not be initialized.");
    return;
  }
  activeVisualization = data;
  state.hidden = true;
  state.classList.remove("design-visualization-error");
  canvas.hidden = false;
  const controls = document.getElementById("design-viz-controls");
  if (controls) controls.hidden = false;
  const help = document.getElementById("design-viz-help");
  if (help) help.hidden = false;
  const hasMappedField = data.geometry_type === "boundary_surface" && data.scalar_range !== null;
  setText(
    "#design-visualization-kind",
    hasMappedField ? "Boundary surface · selected solver field" : "Surface Geometry · no physical field mapped",
  );
  setText(
    "#design-active-field",
    `Field: ${selectedField ? `${selectedField.region}/${selectedField.name}${selectedField.units ? ` (${selectedField.units})` : ""}` : "unavailable"}`,
  );
  setText("#design-legend-field", hasMappedField ? `Field: ${data.region}/${data.field}${data.component && data.component !== "value" ? ` · ${data.component}` : ""}` : "Field: Not mapped to surface");
  setText("#design-legend-units", hasMappedField ? `Units: ${data.units ?? "not stated"}` : "Units: not applicable");
  setText("#design-legend-state", hasMappedField
    ? `${data.mapping_method}; boundary surface only, not a volume rendering.`
    : "Geometry only; the selected physical field is not present on this surface. No color scale is shown.");
  if (data.scalar_range) {
    setText("#design-color-min", `${formatNumber(data.scalar_range[0])}${data.units ? ` ${data.units}` : ""}`);
    setText("#design-color-max", `${formatNumber(data.scalar_range[1])}${data.units ? ` ${data.units}` : ""}`);
  }
  const arrayDetails = data.available_arrays?.length
    ? ` VTK arrays found: ${data.available_arrays.join(", ")}.`
    : "";
  setText(
    "#design-visualization-diagnostic",
    `${data.diagnostic} Rendered ${data.triangle_count.toLocaleString()} surface triangles${data.simplified ? ` from ${data.source_triangle_count.toLocaleString()} boundary triangles` : ""}.${arrayDetails}${hasMappedField ? ` Values are mapped from the selected ${data.source_association}-associated ${data.field} field using ${data.mapping_method}.` : ""}`,
  );
  const legend = document.getElementById("design-color-legend");
  if (legend) legend.hidden = !hasMappedField;
  drawVisualization();
}

function renderSimulationInformation() {
  if (!selectedCase) return;
  const information = document.getElementById("design-simulation-information");
  if (!information) return;
  information.replaceChildren();
  const selected = document.createElement("p");
  selected.textContent = `Selected case: ${selectedCase.case_id} · Reynolds number ${selectedCase.metadata.reynolds_number ?? "not provided"} · ${selectedCase.metadata.mesh_resolution ?? "mesh resolution not provided"} · solution time ${fieldOptions?.time ?? selectedCase.metadata.solution_time ?? "not provided"}.`;
  const partitions = document.createElement("p");
  partitions.textContent = `Processor partitions: ${fieldOptions?.processor_count ?? selectedCase.metadata.processor_count ?? "not provided"}. This page inspects saved results; it does not run a solver.`;
  information.append(selected, partitions);
  if (sourceReference) {
    const reference = document.createElement("p");
    reference.textContent = `${sourceReference} (label from the Stitch page; not independently verified by the dataset scan).`;
    information.append(reference);
  }
}

function setVisualizationMessage(title: string, reason: string) {
  activeVisualization = null;
  const canvas = document.getElementById("design-viz-canvas") as HTMLCanvasElement | null;
  if (canvas) canvas.hidden = true;
  const controls = document.getElementById("design-viz-controls");
  if (controls) controls.hidden = true;
  const help = document.getElementById("design-viz-help");
  if (help) help.hidden = true;
  const legend = document.getElementById("design-color-legend");
  if (legend) legend.hidden = true;
  const kind = document.getElementById("design-visualization-kind");
  if (kind) kind.textContent = title === "Loading simulation data…" ? "Loading…" : "No renderable field";
  setText("#design-legend-state", reason);
  setText("#design-visualization-diagnostic", reason);
  const message = document.getElementById("design-visualization-state");
  if (!message) return;
  message.replaceChildren();
  if (title === "Loading simulation data…") {
    const loading = document.createElement("div");
    loading.className = "design-loading";
    loading.setAttribute("aria-label", title);
    const spinner = document.createElement("span");
    spinner.className = "design-spinner";
    spinner.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.textContent = title;
    loading.append(spinner, text);
    message.append(loading);
    return;
  }
  message.classList.add("design-visualization-error");
  const heading = document.createElement("h3");
  heading.textContent = title;
  const text = document.createElement("p");
  text.textContent = reason;
  message.append(heading, text);
}

async function loadMeshDetails() {
  const activeCase = selectedCase;
  if (!activeCase) return;
  const requestId = ++meshDetailsRequest;
  const currentRe = activeCase.metadata.reynolds_number;
  const meshCases = cases.filter((item) => item.metadata.reynolds_number === currentRe).sort((a, b) => meshOrder(a.metadata.mesh_resolution) - meshOrder(b.metadata.mesh_resolution));
  const status = document.getElementById("mesh-comparison-status");
  const rows = document.getElementById("mesh-comparison-rows");
  if (status) status.textContent = "Loading stored temperature statistics for the scanned meshes…";
  if (!rows) return;
  rows.replaceChildren();
  const reference = meshCases.find((item) => item.case_id !== activeCase.case_id);
  let comparison: MeshComparisonResult | null = null;
  let comparisonError: string | null = null;
  if (reference) {
    try {
      comparison = await compareMeshes(activeCase.case_id, reference.case_id);
    } catch (error) {
      comparisonError = errorMessage(error);
    }
  }
  const temperatureRows = await Promise.all(meshCases.map(async (item) => {
    try {
      const options = await getCfdFieldOptions(item.case_id);
      const temperature = options.fields.find((field) => field.region === "coolant" && field.name === "T" && field.statistics_supported);
      if (!temperature) return { item, stats: null, error: "Coolant temperature statistics are unavailable." };
      return { item, stats: await getCfdStatistics(item.case_id, temperature.region, temperature.name, "value"), error: null };
    } catch (error) {
      return { item, stats: null, error: errorMessage(error) };
    }
  }));
  if (requestId !== meshDetailsRequest || selectedCase?.case_id !== activeCase.case_id) return;
  for (const { item, stats, error } of temperatureRows) {
    const row = document.createElement("tr");
    const cellValues = [
      capitalize(item.metadata.mesh_resolution ?? "Unknown"),
      stats?.association === "cell" && stats.sample_count !== null ? stats.sample_count.toLocaleString() : "Unavailable",
      stats ? `${formatNumber(stats.maximum)} ${stats.units ?? ""}`.trim() : "Unavailable",
      "Unavailable",
      "Not calculated",
      stats ? "Statistic computed; mesh verification not established" : error ?? "Unavailable",
    ];
    for (const value of cellValues) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    rows.append(row);
  }
  const failedRows = temperatureRows.filter((row) => row.error).length;
  if (status) {
    const comparisonStatus = comparison?.reason ?? comparisonError ?? (reference ? "Cross-mesh mapping is not validated." : "No second mesh at the same Reynolds number is available.");
    status.textContent = failedRows ? `${comparisonStatus} Temperature statistics could not be loaded for ${failedRows} mesh level${failedRows === 1 ? "" : "s"}.` : comparisonStatus;
  }
}

function prepareGptOssCard() {
  if (!gptOssCard) return;
  gptOssCard.classList.add("design-gpt-oss-card");
  gptOssCard.innerHTML = `<div class="design-gpt-oss-head"><div class="design-gpt-oss-title"><div class="w-8 h-8 rounded-lg bg-primary-container text-on-primary flex items-center justify-center"><span class="material-symbols-outlined text-[20px]">auto_awesome</span></div><div><h2>GPT-OSS Data Interpretation</h2><p class="design-gpt-oss-subtitle"><span id="gpt-oss-provider-name">GPT-OSS</span> · <span id="gpt-oss-model-name">Checking model…</span> · <span id="gpt-oss-provider-state">Checking status…</span></p></div></div><div class="design-gpt-oss-actions"><button class="px-space-sm py-1.5 rounded-xl bg-surface-container text-on-surface hover:bg-surface-container-high transition-colors font-body-sm text-caption" id="btnTestGptOss" type="button">Test Inference</button><button class="px-space-md py-1.5 rounded-xl bg-primary text-on-primary hover:bg-primary-container transition-colors font-headline-sm text-caption" id="btnRunGptOss" type="button" disabled>Explain Selected Case</button></div></div><p id="gpt-oss-action-status" class="design-status-message" role="status">GPT-OSS interpretation uses selected-case metadata and backend-computed field statistics.</p><details id="gpt-oss-result" class="design-gpt-oss-result" hidden><summary>AI-generated explanation <span class="material-symbols-outlined design-arrow" aria-hidden="true">expand_more</span></summary><div class="design-gpt-oss-result-content"><h3>Summary</h3><p id="gpt-oss-summary-text"></p><h3>Physical interpretation</h3><p id="gpt-oss-explanation-text"></p><h3>Limitations</h3><p id="gpt-oss-limitations">The language model interprets the values supplied by the backend. It does not create CFD values, establish causality, verify mesh convergence, or validate an engineering design.</p><h3>Next steps</h3><p><a href="validation.html">Review evaluation evidence</a> or select another scanned case and field.</p></div></details>`;
  const status = document.createElement("span");
  status.id = "gpt-oss-connection-state";
  status.hidden = true;
  gptOssCard.append(status);
}

function bindGptOssActions() {
  document.getElementById("btnTestGptOss")?.addEventListener("click", () => void runGptOssConnectionTest());
  document.getElementById("btnRunGptOss")?.addEventListener("click", () => void runGptOssAnalysis());
}

async function runGptOssConnectionTest() {
  const button = document.getElementById("btnTestGptOss") as HTMLButtonElement | null;
  const state = document.getElementById("gpt-oss-provider-state");
  const output = document.getElementById("gpt-oss-action-status");
  if (!button) return;
  button.disabled = true;
  button.textContent = "Testing…";
  if (state) state.textContent = "Testing live connection…";
  if (output) output.textContent = "Sending a controlled inference probe to GPT-OSS…";
  try {
    const result = await testGptOssConnection();
    if (state) state.textContent = "Connected · live test succeeded";
    if (output) output.textContent = `${result.provider} inference succeeded using ${result.model} at ${result.tested_at}; measured latency ${result.latency_ms ?? "not reported"} ms.`;
  } catch (error) {
    const result = providerTestFailure(gptOssStatus?.provider ?? "GPT-OSS", gptOssStatus?.model ?? null, error);
    const notConfigured = result.diagnostic.toLowerCase().includes("not configured");
    if (state) state.textContent = notConfigured ? "Not configured" : result.status === "failed" ? "Failed" : "Not tested";
    if (output) output.textContent = result.diagnostic;
  } finally {
    button.disabled = false;
    button.textContent = "Test Connection";
  }
}

async function runGptOssAnalysis() {
  const button = document.getElementById("btnRunGptOss") as HTMLButtonElement | null;
  const output = document.getElementById("gpt-oss-action-status");
  const resultDetails = document.getElementById("gpt-oss-result") as HTMLDetailsElement | null;
  const resultText = document.getElementById("gpt-oss-explanation-text");
  if (!button || !selectedCase || !selectedField || !statistics) return;
  button.disabled = true;
  button.textContent = "Explaining…";
  if (output) output.textContent = "Computing selected-case statistics and requesting a cautious interpretation…";
  setText("#gpt-oss-summary-text", selectedCaseSummary());
  if (resultDetails) {
    resultDetails.hidden = false;
    resultDetails.open = true;
  }
  if (resultText) resultText.textContent = "Loading GptOss interpretation…";
  try {
    gptOssAnalysis = await analyzeWithGptOss({
      case_id: selectedCase.case_id,
      region: selectedField.region,
      field: selectedField.name,
      component: selectedComponent,
    });
    if (resultText) resultText.textContent = gptOssAnalysis.interpretation.interpretation;
    if (output) output.textContent = `AI-generated interpretation · ${gptOssAnalysis.interpretation.provider ?? "GptOss"} · ${gptOssAnalysis.interpretation.model} · ${gptOssAnalysis.interpretation.generated_at}`;
  } catch (error) {
    if (resultText) resultText.textContent = "No interpretation was returned.";
    if (output) {
      output.textContent = `${errorMessage(error)} `;
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "design-retry";
      retry.textContent = "Retry";
      retry.addEventListener("click", () => void runGptOssAnalysis());
      output.append(retry);
    }
  } finally {
    button.disabled = false;
    button.textContent = "Explain Selected Case";
    updateGptOssButtons();
  }
}

function renderGptOssStatus(error?: string) {
  setText("#gpt-oss-model-name", gptOssStatus?.model ?? "No model configured");
  setText("#gpt-oss-provider-state", error ? "Unavailable" : gptOssStatus?.status === "reachable" ? "Reachable" : gptOssStatus?.status === "unavailable" ? "Unavailable" : "Not Verified");
  updateGptOssButtons();
}

function updateGptOssButtons() {
  const button = document.getElementById("btnRunGptOss") as HTMLButtonElement | null;
  if (button) button.disabled = !statistics || !selectedCase || !selectedField || gptOssStatus?.status !== "reachable";
}

function clearGptOssResult() {
  gptOssAnalysis = null;
  const panel = document.getElementById("gpt-oss-result") as HTMLDetailsElement | null;
  if (panel) {
    panel.hidden = true;
    panel.open = false;
  }
  const output = document.getElementById("gpt-oss-action-status");
  if (output) output.textContent = "GPT-OSS interpretation uses selected-case metadata and backend-computed field statistics.";
  setText("#gpt-oss-summary-text", "");
  updateGptOssButtons();
}

function selectedCaseSummary() {
  if (!selectedCase || !selectedField || !statistics) return "Selected-case statistics are unavailable.";
  const unit = statistics.units ? ` ${statistics.units}` : " (units not stated)";
  return `${selectedCase.case_id}, Re ${selectedCase.metadata.reynolds_number ?? "unknown"}, ${selectedCase.metadata.mesh_resolution ?? "mesh unknown"}: ${selectedField.region}/${selectedField.name} ${selectedComponent} has a minimum of ${formatNumber(statistics.minimum)}${unit}, a maximum of ${formatNumber(statistics.maximum)}${unit}, and an average of ${formatNumber(statistics.mean)}${unit}. These values are computed from stored solver values.`;
}

function setStatus(message: string, error = false) {
  const element = document.getElementById("design-dataset-status");
  if (!element) return;
  element.textContent = message;
  element.classList.toggle("design-error", error);
}

function setSelectionState(message: string) {
  const element = document.getElementById("design-selection-state");
  if (element) element.textContent = message;
}

function setText(selector: string, value: string) {
  const element = document.querySelector<HTMLElement>(selector);
  if (element) element.textContent = value;
}

function fieldKey(field: CfdFieldOption) {
  return `${field.region}/${field.name}`;
}

function componentLabel(component: string, field: CfdFieldOption) {
  if (component === "magnitude" && field.value_type.includes("Tensor")) return "Tensor magnitude";
  return component === "value" ? "Value" : capitalize(component);
}

function meshOrder(value: string | null) {
  return ({ coarse: 0, medium: 1, fine: 2 } as Record<string, number>)[value ?? ""] ?? 99;
}

function capitalize(value: string) {
  return value ? value[0].toUpperCase() + value.slice(1) : value;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 7 }).format(value);
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "The request could not be completed.";
}