interface ApiResult<T> {
  value: T | null;
  error: string | null;
}

interface DatasetScan {
  status: string;
  source_kind: string;
  generated_at: string | null;
  summary: { file_count: number; candidate_count: number; total_bytes: number };
  warnings: string[];
}

interface DatasetCase {
  case_id: string;
  fields: Array<{ name: string; units: string | null }>;
  metadata: { reynolds_number: number | null; mesh_resolution: string | null };
}

interface SurrogateModel {
  model_id: string;
  model_name: string;
  available: boolean;
  unavailable_reason?: string;
  cases: Array<{ case_id: string; reynolds_number: number; mesh_resolution: string }>;
  target: { region: string; field: string; units: string | null };
  features: Array<{ name: string }>;
  metrics: {
    extra_trees_spatial_validation: { mae: number; rmse: number; r2: number | null };
    metric_scope: string;
  };
  explainability: {
    shap_global: { method: string; sample_count: number; additivity_checked: boolean };
    lime: { method: string; approximation_quality_metric: string };
  };
  limitations: string[];
}

interface ModelsResponse {
  models: SurrogateModel[];
  empty_reason: string | null;
}

interface ProviderStatus {
  provider?: string;
  configured: boolean;
  model: string | null;
  status?: "reachable" | "unavailable" | "not_verified";
  checked_at?: string;
  diagnostic?: string;
}

interface KnowledgeStatus {
  corpus: { indexing_status: string; document_count: number; passage_count: number; warnings: string[] };
  bert: { status: string; reason: string | null; label_set: string[] };
  gpt_oss: { configured: boolean; model: string | null; provider: string };
}

interface GenerativeStatus {
  torch_available: boolean;
  models: Array<{ model_type: string; available: boolean; model_id: string }>;
  error?: string;
}

interface ProviderTestResult {
  status?: string;
  model?: string;
  tested_at?: string;
  latency_ms?: number;
  diagnostic?: string;
  error?: string;
}

const root = document.getElementById("overview-dashboard-root");
const apiBase = (import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api").replace(/\/$/, "");

if (root) {
  addDashboardStyles();
  document.querySelectorAll<HTMLAnchorElement>("body > aside nav a").forEach((link) => {
    const label = link.querySelector("span:last-child")?.textContent?.trim();
    if (label) link.setAttribute("aria-label", label);
  });
  root.innerHTML = `
    <div id="overview-dashboard" class="overview-dashboard">
      <header class="overview-topline">
        <div>
          <p class="overview-kicker">THERMOEXPLAIN AI</p>
          <h1>Battery cooling CFD dataset</h1>
        </div>
        <div class="overview-top-status">
          <div><span>Dataset</span><strong id="header-dataset-state">Loading</strong></div>
          <div><span>System</span><strong id="header-system-state">Checking</strong></div>
          <button class="overview-text-button" type="button" data-overview-toggle="system-info" aria-controls="system-info" aria-expanded="false">System information <span class="overview-arrow" aria-hidden="true">⌄</span></button>
        </div>
      </header>

      <section class="overview-welcome" aria-labelledby="welcome-title">
        <div>
          <p class="overview-kicker">RESEARCH WORKSPACE</p>
          <h2 id="welcome-title">Understand Battery Cooling with AI</h2>
          <p>Explore battery-cooling data and use AI to analyze and explain results.</p>
        </div>
        <div class="overview-actions">
          <a class="overview-primary" href="design.html">Explore CFD <span aria-hidden="true">→</span></a>
          <button class="overview-secondary" type="button" data-open-guide>Quick Guide</button>
        </div>
      </section>

      <section id="project-guide" class="overview-guide" hidden aria-labelledby="guide-title">
        <div class="overview-guide-heading">
          <div><p class="overview-kicker">QUICK GUIDE</p><h2 id="guide-title">How does this project work?</h2></div>
          <button class="overview-icon-button" type="button" data-overview-toggle="project-guide" aria-controls="project-guide" aria-expanded="true" aria-label="Hide quick guide"><span class="overview-arrow" aria-hidden="true">⌃</span></button>
        </div>
        <div class="overview-steps">
          <article><span>1</span><div><h3>Explore the data</h3><p>Inspect the available battery-cooling simulation cases.</p></div></article>
          <article><span>2</span><div><h3>Train and predict</h3><p>Use a trained AI model to estimate selected thermal outputs.</p></div></article>
          <article><span>3</span><div><h3>Understand and evaluate</h3><p>Use SHAP/LIME and language models to explain results and inspect limitations.</p></div></article>
        </div>
        <p class="overview-guide-note">This helps make simulation evidence easier to inspect while keeping model limits visible.</p>
      </section>

      <section class="overview-card-grid" aria-label="Project status summaries">
        <article class="overview-card" aria-labelledby="dataset-card-title">
          <div class="overview-card-heading"><div><p class="overview-kicker">DATA</p><h2 id="dataset-card-title">Dataset</h2></div><button class="overview-icon-button overview-toggle" type="button" data-overview-toggle="dataset-details" aria-controls="dataset-details" aria-expanded="false" aria-label="Show dataset details"><span class="overview-arrow" aria-hidden="true">⌄</span></button></div>
          <div class="overview-card-summary"><strong class="overview-state" id="dataset-card-state" data-state="loading">Loading</strong><span id="dataset-case-count">Checking available cases…</span></div>
          <div class="overview-details" id="dataset-details" hidden>
            <dl class="overview-detail-list"><div><dt>Dataset size</dt><dd id="dataset-size">Loading</dd></div><div><dt>Files scanned</dt><dd id="dataset-files">Loading</dd></div><div><dt>Candidate data files</dt><dd id="dataset-candidates">Loading</dd></div><div><dt>Available fields</dt><dd id="dataset-field-count">Loading</dd></div><div><dt>Expected cases</dt><dd>Not provided by the API</dd></div><div><dt>Data processing</dt><dd id="dataset-processing">Loading</dd></div></dl>
            <p class="overview-detail-copy" id="dataset-limitations">Loading dataset notes…</p><ul class="overview-list" id="dataset-field-list"></ul><ul class="overview-list overview-warning-list" id="dataset-warnings"></ul>
          </div>
        </article>

        <article class="overview-card" aria-labelledby="model-card-title">
          <div class="overview-card-heading"><div><p class="overview-kicker">PREDICTION</p><h2 id="model-card-title">Prediction Model</h2></div><button class="overview-icon-button overview-toggle" type="button" data-overview-toggle="model-details" aria-controls="model-details" aria-expanded="false" aria-label="Show prediction model details"><span class="overview-arrow" aria-hidden="true">⌄</span></button></div>
          <div class="overview-card-summary"><strong class="overview-state" id="model-card-state" data-state="loading">Loading</strong><span id="model-card-name">Checking model inventory…</span></div>
          <div class="overview-details" id="model-details" hidden>
            <dl class="overview-detail-list"><div><dt>Model type</dt><dd id="model-type">Loading</dd></div><div><dt>Model name</dt><dd id="model-name-detail">Loading</dd></div><div><dt>Prediction target</dt><dd id="model-target">Loading</dd></div><div><dt>Training data</dt><dd id="model-training">Loading</dd></div></dl>
            <div class="overview-metrics" id="model-metrics"></div><p class="overview-detail-copy" id="model-metric-scope"></p><ul class="overview-list" id="model-limitations"></ul>
          </div>
        </article>

        <article class="overview-card" aria-labelledby="xai-card-title">
          <div class="overview-card-heading"><div><p class="overview-kicker">EXPLANATIONS</p><h2 id="xai-card-title">Explainable AI</h2></div><button class="overview-icon-button overview-toggle" type="button" data-overview-toggle="xai-details" aria-controls="xai-details" aria-expanded="false" aria-label="Show explainability details"><span class="overview-arrow" aria-hidden="true">⌄</span></button></div>
          <div class="overview-card-summary"><strong class="overview-state" id="xai-card-state" data-state="loading">Loading</strong><span>Understand which inputs influence a prediction.</span></div>
          <div class="overview-details" id="xai-details" hidden>
            <dl class="overview-detail-list"><div><dt>SHAP</dt><dd id="shap-state">Loading</dd></div><div><dt>LIME</dt><dd id="lime-state">Loading</dd></div><div><dt>Required model</dt><dd id="xai-model-required">Loading</dd></div><div><dt>Available features</dt><dd id="xai-feature-list">Loading</dd></div></dl>
            <p class="overview-detail-copy">SHAP and LIME explain model behavior; they do not establish physical causality. The current explanation endpoints require a compatible, current surrogate.</p>
          </div>
        </article>

        <article class="overview-card" aria-labelledby="assistants-card-title">
          <div class="overview-card-heading"><div><p class="overview-kicker">LANGUAGE TOOLS</p><h2 id="assistants-card-title">AI Assistants</h2></div><button class="overview-icon-button overview-toggle" type="button" data-overview-toggle="assistant-details" aria-controls="assistant-details" aria-expanded="false" aria-label="Show AI assistant details"><span class="overview-arrow" aria-hidden="true">⌄</span></button></div>
          <div class="overview-provider-summary"><div><span>GPT-OSS</span><strong class="overview-state" id="gpt-oss-card-state" data-state="loading">Checking</strong></div></div>
          <div class="overview-details" id="assistant-details" hidden>
            <div class="overview-provider-detail"><div><h3>GPT-OSS</h3><p id="gpt-oss-provider-info">Loading</p><p id="gpt-oss-test-result">Inference verification has not been run.</p></div><button class="overview-test-button" id="test-gpt-oss" type="button">Test inference</button></div>
            <p class="overview-detail-copy">GPT-OSS is the sole language-model provider for CFD, Explainable AI, and evidence-grounded Thermal Knowledge Assistant explanations. Numerical and scientific models remain independent.</p>
          </div>
        </article>
      </section>

      <section class="overview-next-step" aria-labelledby="next-step-title">
        <div><p class="overview-kicker">RECOMMENDED NEXT STEP</p><h2 id="next-step-title">Checking project readiness…</h2><p id="next-step-copy">Reading current dataset and model status.</p></div><a class="overview-primary" id="next-step-link" href="design.html">Explore CFD <span aria-hidden="true">→</span></a>
      </section>

      <section class="overview-more" aria-label="Additional project information">
        <section class="overview-accordion" id="system-info" hidden aria-label="System information"><div class="overview-accordion-content"><p id="system-info-copy">Backend and dataset state are reported from the API. Hardware details are not provided.</p><div class="overview-readiness-grid" id="readiness-details"></div></div></section>
        <details class="overview-accordion"><summary>Guided research journey <span class="overview-arrow" aria-hidden="true">⌄</span></summary><div class="overview-accordion-content"><ol class="overview-journey"><li><strong>Explore the data</strong><span>Review scanned cases and measured fields.</span><a href="design.html">Open CFD Explorer</a></li><li><strong>Make a prediction</strong><span>Use only a current saved model and its supported inputs.</span><a href="explainable.html">Open Explainable AI</a></li><li><strong>Explain the result</strong><span>Inspect SHAP/LIME outputs and their limitations.</span><a href="explainable.html">Review explanations</a></li><li><strong>Evaluate evidence</strong><span>Review recorded validation results before interpreting them.</span><a href="validation.html">Open Evaluation</a></li></ol></div></details>
        <details class="overview-accordion"><summary>Learn the basics <span class="overview-arrow" aria-hidden="true">⌄</span></summary><div class="overview-accordion-content overview-basics"><article><h3>What is CFD?</h3><p>Computational Fluid Dynamics uses numerical methods to estimate how fluids move and exchange heat. This project inspects saved simulations; it does not run a CFD solver.</p></article><article><h3>What is a coolant channel?</h3><p>It is a passage that carries coolant through the battery pack to remove heat. The actual geometry and fields depend on the selected case.</p></article><article><h3>What is Reynolds number?</h3><p>A dimensionless measure comparing inertial and viscous effects in a flow. The scanned cases expose their actual Reynolds values in the CFD Explorer.</p></article><article><h3>What is a surrogate model?</h3><p>A numerical model trained to approximate selected simulation outputs. Its predictions are not new CFD results and must be considered with the recorded validation metrics.</p></article></div></details>
        <details class="overview-accordion"><summary>Simulation history <span class="overview-arrow" aria-hidden="true">⌄</span></summary><div class="overview-accordion-content"><p>The current backend does not expose a simulation-run history endpoint. Existing CFD cases can be inspected in the CFD Explorer.</p><a class="overview-inline-link" href="design.html">Explore available cases</a></div></details>
      </section>
    </div>`;

  bindInteractions(root);
  void loadDashboard(root);
}

function addDashboardStyles() {
  if (document.getElementById("overview-dashboard-styles")) return;
  const style = document.createElement("style");
  style.id = "overview-dashboard-styles";
  style.textContent = `
    .pl-72 > header { display:none !important; }
    .pl-72 > main.pt-16 { padding-top:0 !important; }
    #overview-dashboard { --ov-ink:#0b1c30; --ov-muted:#536579; --ov-line:#d8e3f2; --ov-blue:#004ac6; --ov-blue-soft:#eff4ff; --ov-surface:#fff; display:grid; gap:18px; color:var(--ov-ink); }
    #overview-dashboard * { box-sizing:border-box; }
    .overview-topline,.overview-welcome,.overview-card,.overview-next-step,.overview-guide,.overview-accordion { border:1px solid var(--ov-line); border-radius:10px; background:var(--ov-surface); }
    .overview-topline { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:14px 18px; }
    .overview-topline h1 { margin:2px 0 0; font:600 18px/1.3 Inter,sans-serif; color:var(--ov-ink); }
    .overview-kicker { margin:0 0 4px; color:#52677f; font:600 10px/1.3 Inter,sans-serif; }
    .overview-top-status { display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
    .overview-top-status div { display:grid; gap:2px; }
    .overview-top-status span,.overview-provider-summary span { color:var(--ov-muted); font-size:12px; }
    .overview-top-status strong { font-size:13px; font-weight:600; }
    .overview-text-button,.overview-inline-link { border:0; padding:0; background:none; color:var(--ov-blue); font:600 12px/1.4 Inter,sans-serif; cursor:pointer; text-decoration:none; }
    .overview-welcome { display:flex; align-items:center; justify-content:space-between; gap:24px; padding:20px 22px; background:linear-gradient(110deg,#fff 0%,#f5f8ff 100%); }
    .overview-welcome h2,.overview-next-step h2,.overview-guide h2 { margin:0; font:600 22px/1.25 Inter,sans-serif; color:var(--ov-ink); }
    .overview-welcome p:not(.overview-kicker),.overview-next-step p:not(.overview-kicker) { margin:6px 0 0; color:var(--ov-muted); font-size:14px; line-height:1.5; }
    .overview-actions { display:flex; align-items:center; gap:9px; flex-shrink:0; }
    .overview-primary,.overview-secondary,.overview-test-button { display:inline-flex; min-height:38px; align-items:center; justify-content:center; gap:9px; padding:8px 14px; border-radius:7px; font:600 13px/1.2 Inter,sans-serif; cursor:pointer; text-decoration:none; }
    .overview-primary { border:1px solid var(--ov-blue); color:#fff; background:var(--ov-blue); }
    .overview-secondary,.overview-test-button { border:1px solid #c7d5e7; color:var(--ov-ink); background:#fff; }
    .overview-primary:focus-visible,.overview-secondary:focus-visible,.overview-test-button:focus-visible,.overview-icon-button:focus-visible,.overview-text-button:focus-visible,.overview-accordion summary:focus-visible,.overview-inline-link:focus-visible { outline:3px solid #2780ff; outline-offset:3px; }
    .overview-guide { padding:16px 18px; background:#f5f8ff; }
    .overview-guide[hidden],.overview-details[hidden],.overview-accordion[hidden] { display:none; }
    .overview-guide-heading,.overview-card-heading { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }
    .overview-guide-heading { align-items:center; margin-bottom:14px; }
    .overview-guide h2 { font-size:17px; }
    .overview-steps { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
    .overview-steps article { display:flex; align-items:flex-start; gap:10px; padding:12px; border-radius:7px; background:#fff; }
    .overview-steps article>span { display:grid; width:23px; height:23px; place-items:center; border-radius:50%; color:var(--ov-blue); background:#e3edff; font-weight:700; font-size:12px; flex:none; }
    .overview-steps h3,.overview-provider-detail h3 { margin:0 0 3px; font:600 13px/1.35 Inter,sans-serif; }
    .overview-steps p,.overview-guide-note { margin:0; color:var(--ov-muted); font-size:12px; line-height:1.5; }
    .overview-guide-note { margin-top:12px; }
    .overview-card-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; align-items:start; }
    .overview-card { min-width:0; padding:15px; box-shadow:0 2px 8px rgba(15,42,80,.035); }
    .overview-card-heading { align-items:center; }
    .overview-card h2 { margin:0; font:600 16px/1.3 Inter,sans-serif; }
    .overview-card .overview-kicker { margin-bottom:3px; }
    .overview-icon-button { display:grid; width:30px; height:30px; flex:none; place-items:center; border:1px solid #dbe4f0; border-radius:50%; color:#38536f; background:#fff; cursor:pointer; }
    .overview-arrow { display:inline-block; font-size:18px; line-height:1; transition:transform .18s ease; }
    .overview-icon-button[aria-expanded="true"] .overview-arrow { transform:rotate(180deg); }
    .overview-icon-button[data-overview-toggle="project-guide"][aria-expanded="true"] .overview-arrow { transform:rotate(180deg); }
    .overview-card-summary { display:grid; gap:5px; margin-top:15px; min-height:48px; align-content:start; }
    .overview-card-summary>span { color:var(--ov-muted); font-size:12px; line-height:1.45; }
    .overview-state { width:max-content; max-width:100%; color:#31465d; font-size:12px; font-weight:650; }
    .overview-state[data-state="ready"],.overview-state[data-state="connected"] { color:#087766; }
    .overview-state[data-state="warning"],.overview-state[data-state="configured"] { color:#8a5800; }
    .overview-state[data-state="unavailable"],.overview-state[data-state="failed"] { color:#a33434; }
    .overview-provider-summary { display:grid; gap:8px; margin-top:13px; }
    .overview-provider-summary div { display:flex; justify-content:space-between; gap:8px; align-items:center; }
    .overview-provider-summary strong { font-size:11px; text-align:right; }
    .overview-details { margin-top:14px; padding-top:12px; border-top:1px solid #e6edf6; animation:overview-open .18s ease-out both; }
    .overview-detail-list { display:grid; gap:9px; margin:0; }
    .overview-detail-list div { display:grid; grid-template-columns:minmax(90px,.8fr) minmax(0,1.2fr); gap:8px; align-items:start; }
    .overview-detail-list dt { color:var(--ov-muted); font-size:11px; line-height:1.4; }
    .overview-detail-list dd { margin:0; font-size:11px; line-height:1.45; overflow-wrap:anywhere; }
    .overview-detail-copy { margin:10px 0 0; color:var(--ov-muted); font-size:11px; line-height:1.55; }
    .overview-list { display:grid; gap:4px; margin:9px 0 0; padding-left:17px; color:var(--ov-muted); font-size:11px; line-height:1.45; }
    .overview-warning-list { color:#8a5800; }
    .overview-metrics { display:grid; gap:6px; margin-top:10px; }
    .overview-metric { display:flex; justify-content:space-between; gap:8px; color:var(--ov-muted); font-size:11px; }
    .overview-metric strong { color:var(--ov-ink); font-weight:600; text-align:right; }
    .overview-provider-detail { display:flex; justify-content:space-between; align-items:flex-start; gap:10px; padding:9px 0; border-bottom:1px solid #e6edf6; }
    .overview-provider-detail p { margin:3px 0 0; color:var(--ov-muted); font-size:10px; line-height:1.45; overflow-wrap:anywhere; }
    .overview-test-button { min-height:30px; padding:6px 9px; font-size:11px; white-space:nowrap; }
    .overview-test-button:disabled { opacity:.6; cursor:wait; }
    .overview-next-step { display:flex; align-items:center; justify-content:space-between; gap:18px; padding:15px 18px; }
    .overview-next-step h2 { font-size:16px; }
    .overview-next-step p:not(.overview-kicker) { font-size:12px; }
    .overview-more { display:grid; gap:8px; }
    .overview-accordion { overflow:hidden; }
    .overview-accordion summary { display:flex; align-items:center; justify-content:space-between; gap:12px; min-height:46px; padding:11px 15px; color:var(--ov-ink); font:600 13px/1.4 Inter,sans-serif; cursor:pointer; list-style:none; }
    .overview-accordion summary::-webkit-details-marker { display:none; }
    .overview-accordion[open] summary .overview-arrow { transform:rotate(180deg); }
    .overview-accordion-content { padding:0 15px 15px; color:var(--ov-muted); font-size:12px; line-height:1.55; animation:overview-open .18s ease-out both; }
    .overview-accordion-content p { margin:0 0 8px; }
    .overview-readiness-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .overview-readiness-item { display:flex; justify-content:space-between; gap:10px; padding:8px 10px; border-radius:6px; background:#f5f8fc; font-size:11px; }
    .overview-readiness-item strong { color:var(--ov-ink); text-align:right; }
    .overview-journey { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px 20px; margin:0; padding-left:20px; }
    .overview-journey li { padding-left:3px; }
    .overview-journey strong,.overview-journey span,.overview-journey a { display:block; }
    .overview-journey strong { color:var(--ov-ink); }
    .overview-journey a { margin-top:4px; color:var(--ov-blue); font-weight:600; text-decoration:none; }
    .overview-basics { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
    .overview-basics h3 { margin:0 0 4px; color:var(--ov-ink); font-size:12px; }
    .overview-basics p { margin:0; }
    @keyframes overview-open { from { opacity:0; transform:translateY(-3px); } to { opacity:1; transform:translateY(0); } }
    @media (max-width:1150px) { .overview-card-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:760px) { .overview-topline,.overview-welcome,.overview-next-step { align-items:flex-start; flex-direction:column; } .overview-top-status { gap:12px; } .overview-welcome { padding:17px; } .overview-steps { grid-template-columns:1fr; } }
    @media (max-width:640px) { body > aside.fixed.left-0 { width:56px !important; } body > aside.fixed.left-0 > div:first-child > div:first-child { justify-content:center; padding-left:0; padding-right:0; } body > aside.fixed.left-0 > div:first-child > div:first-child > div:last-child, body > aside.fixed.left-0 > div:first-child > div:nth-child(2), body > aside.fixed.left-0 > div:last-child { display:none !important; } body > aside.fixed.left-0 nav { padding-left:6px; padding-right:6px; } body > aside.fixed.left-0 nav a { justify-content:center; padding-left:0 !important; padding-right:0 !important; font-size:0 !important; } body > aside.fixed.left-0 nav a > span:last-child { display:none !important; } body > .pl-72 { padding-left:56px !important; } }
    @media (max-width:560px) { .overview-card-grid,.overview-basics,.overview-journey,.overview-readiness-grid { grid-template-columns:1fr; } .overview-actions { width:100%; } .overview-actions>* { flex:1; } .overview-next-step>.overview-primary { width:100%; } .overview-topline { gap:12px; } .overview-top-status { width:100%; justify-content:space-between; } }
    @media (prefers-reduced-motion:reduce) { .overview-arrow,.overview-details,.overview-accordion-content { transition:none; animation:none; } }
  `;
  document.head.append(style);
}

function bindInteractions(container: HTMLElement) {
  container.querySelectorAll<HTMLButtonElement>("[data-overview-toggle]").forEach((button) => {
    button.addEventListener("click", () => {
      const target = document.getElementById(button.dataset.overviewToggle ?? "");
      if (!target) return;
      const expanded = button.getAttribute("aria-expanded") === "true";
      button.setAttribute("aria-expanded", String(!expanded));
      target.hidden = expanded;
      const label = button.dataset.overviewToggle === "project-guide" ? "quick guide" : "details";
      button.setAttribute("aria-label", `${expanded ? "Show" : "Hide"} ${label}`);
    });
  });

  container.querySelector<HTMLButtonElement>("[data-open-guide]")?.addEventListener("click", () => {
    const guide = container.querySelector<HTMLElement>("#project-guide");
    const toggle = container.querySelector<HTMLButtonElement>('[data-overview-toggle="project-guide"]');
    if (!guide || !toggle) return;
    guide.hidden = false;
    toggle.setAttribute("aria-expanded", "true");
    toggle.setAttribute("aria-label", "Hide quick guide");
    guide.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  });

  const quickDocs = document.querySelector<HTMLAnchorElement>("body > aside a[href=\"#\"]");
  quickDocs?.addEventListener("click", (event) => {
    event.preventDefault();
    container.querySelector<HTMLButtonElement>("[data-open-guide]")?.click();
  });

  container.querySelector<HTMLButtonElement>("#test-gpt-oss")?.addEventListener("click", (event) => {
    void runProviderTest(event.currentTarget as HTMLButtonElement, "#gpt-oss-card-state", "#gpt-oss-test-result");
  });
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, init);
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const message = typeof payload === "object" && payload !== null && "error" in payload && typeof payload.error === "string"
      ? payload.error
      : `Request failed with HTTP ${response.status}.`;
    throw new Error(message);
  }
  return payload as T;
}

async function settled<T>(path: string): Promise<ApiResult<T>> {
  try {
    return { value: await fetchJson<T>(path), error: null };
  } catch (error) {
    return { value: null, error: error instanceof Error ? error.message : "Status unavailable." };
  }
}

async function loadDashboard(container: HTMLElement) {
  const [health, scan, cases, models, generative, knowledge, gptOss, preprocessing] = await Promise.all([
    settled<{ status: string }>("/health"),
    settled<DatasetScan>("/dataset/scan"),
    settled<{ cases: DatasetCase[] }>("/dataset/cases"),
    settled<ModelsResponse>("/ml/models"),
    settled<GenerativeStatus>("/generative/status"),
    settled<KnowledgeStatus>("/knowledge/status"),
    settled<ProviderStatus>("/llm/status"),
    settled<{ status: string; message?: string }>("/dataset/preprocessing"),
  ]);

  const currentCases = cases.value?.cases ?? [];
  const currentModel = models.value?.models.find((model) => model.available) ?? null;
  const staleModels = models.value?.models.filter((model) => !model.available) ?? [];
  const uniqueFields = [...new Set(currentCases.flatMap((item) => item.fields.map((field) => field.name)))].sort();
  const datasetReady = scan.value?.status === "scanned" && !scan.error;
  const fieldsByName = currentCases.flatMap((item) => item.fields);
  const fieldUnits = new Map(fieldsByName.map((field) => [field.name, field.units]));

  setText(container, "#header-dataset-state", scan.error ? "Unavailable" : datasetReady ? "Ready" : scan.value?.status ?? "Unknown");
  setText(container, "#header-system-state", health.error ? "Unavailable" : health.value?.status === "ok" && datasetReady ? "Available · research limits" : "Partially available");
  setState(container, "#dataset-card-state", scan.error ? "Status unavailable" : datasetReady ? "Ready" : scan.value?.status ?? "Unknown", scan.error ? "unavailable" : datasetReady ? "ready" : "warning");
  setText(container, "#dataset-case-count", cases.error ? "Case count unavailable" : `${currentCases.length} available cases`);
  setText(container, "#dataset-size", scan.value ? formatBytes(scan.value.summary.total_bytes) : "Status unavailable");
  setText(container, "#dataset-files", scan.value ? scan.value.summary.file_count.toLocaleString() : "Status unavailable");
  setText(container, "#dataset-candidates", scan.value ? scan.value.summary.candidate_count.toLocaleString() : "Status unavailable");
  setText(container, "#dataset-field-count", cases.error ? "Status unavailable" : String(uniqueFields.length));
  setText(container, "#dataset-processing", preprocessing.value ? `${preprocessing.value.status}${preprocessing.value.message ? ` · ${preprocessing.value.message}` : ""}` : "Status unavailable");
  setText(container, "#dataset-limitations", "Expected-case totals are not reported by the API. Spatial visualization and validated cross-mesh comparisons are not available.");
  renderList(container, "#dataset-field-list", uniqueFields.map((name) => `${name}${fieldUnits.get(name) ? ` (${fieldUnits.get(name)})` : ""}`), "No scanned field names are available.");
  renderList(container, "#dataset-warnings", scan.value?.warnings ?? [], "No scan warnings were reported.");

  const modelStatus = models.error ? "Status unavailable" : currentModel ? "Trained · exploratory" : staleModels.length ? "Saved but stale" : "Not trained";
  setState(container, "#model-card-state", modelStatus, models.error ? "unavailable" : currentModel ? "ready" : "warning");
  setText(container, "#model-card-name", models.error ? "Model status unavailable" : currentModel?.model_name ?? models.value?.empty_reason ?? "No current prediction model");
  const modelType = currentModel?.model_name.toLowerCase().includes("extra trees") ? "Extra Trees regressor" : currentModel ? "Not identified by model inventory" : models.error ? "Status unavailable" : "No current model";
  setText(container, "#model-type", modelType);
  setText(container, "#model-name-detail", currentModel?.model_name ?? (models.error ? "Status unavailable" : "No current model"));
  setText(container, "#model-target", currentModel ? `${currentModel.target.region}/${currentModel.target.field}${currentModel.target.units ? ` (${currentModel.target.units})` : ""}` : "Not available");
  setText(container, "#model-training", currentModel ? `${currentModel.cases.length} scanned cases; ${currentModel.metrics.metric_scope}` : "No current training dataset metadata.");
  setText(container, "#model-metric-scope", currentModel?.metrics.metric_scope ?? "Validation metrics are unavailable without a current model.");
  if (currentModel) {
    const metrics = currentModel.metrics.extra_trees_spatial_validation;
    renderMetrics(container, [
      ["Held-out MAE", `${formatMetric(metrics.mae)} ${currentModel.target.units ?? ""}`.trim()],
      ["Held-out RMSE", `${formatMetric(metrics.rmse)} ${currentModel.target.units ?? ""}`.trim()],
      ["Held-out R²", metrics.r2 === null ? "Not available" : formatMetric(metrics.r2)],
    ]);
    renderList(container, "#model-limitations", currentModel.limitations, "No model limitations were provided.");
  } else {
    renderMetrics(container, []);
    const limitations = staleModels.map((model) => model.unavailable_reason ?? `${model.model_name} does not match the current dataset.`);
    renderList(container, "#model-limitations", limitations, "Model metrics and limitations are not available.");
  }

  const shapReady = Boolean(currentModel?.explainability?.shap_global);
  const limeReady = Boolean(currentModel?.explainability?.lime);
  setState(container, "#xai-card-state", models.error ? "Status unavailable" : currentModel && shapReady && limeReady ? "Available · exploratory" : "Unavailable", models.error ? "unavailable" : currentModel ? "ready" : "warning");
  setText(container, "#shap-state", models.error ? "Status unavailable" : shapReady ? `Available · ${currentModel?.explainability.shap_global.sample_count} global samples` : "Unavailable");
  setText(container, "#lime-state", models.error ? "Status unavailable" : limeReady ? "Available for local explanation" : "Unavailable");
  setText(container, "#xai-model-required", currentModel ? currentModel.model_name : staleModels.length ? "A current model is required; saved models are stale." : "A trained, current surrogate is required.");
  setText(container, "#xai-feature-list", currentModel?.features.map((feature) => feature.name).join(", ") ?? "Unavailable");

  const gptOssState = providerLabel(gptOss);
  setState(container, "#gpt-oss-card-state", gptOssState.label, gptOssState.state);
  setText(container, "#gpt-oss-provider-info", providerDetails(gptOss.value, gptOss.error));

  const overallReady = health.value?.status === "ok" && datasetReady;
  renderReadiness(container, [
    ["Backend", health.error ? "Status unavailable" : health.value?.status === "ok" ? "Available" : "Unavailable"],
    ["Dataset scan", scan.error ? "Status unavailable" : scan.value?.status ?? "Unknown"],
    ["Prediction model", modelStatus],
    ["Generative models", generative.error ? "Status unavailable" : `${generative.value?.models.filter((model) => model.available).length ?? 0} current checkpoints`],
    ["Knowledge documents", knowledge.error ? "Status unavailable" : `${knowledge.value?.corpus.document_count ?? 0} documents · ${knowledge.value?.corpus.passage_count ?? 0} passages`],
    ["BERT classifier", knowledge.error ? "Status unavailable" : knowledge.value?.bert.status === "configured" ? "Configured" : "Not configured"],
    ["GPT-OSS", gptOssState.label],
    ["PyTorch", generative.error ? "Status unavailable" : generative.value?.torch_available ? "Installed" : "Not installed"],
  ]);

  const recommendationTitle = !overallReady
    ? "Check dataset readiness"
    : currentModel
      ? "Review the model evidence"
      : "Explore the scanned data";
  const recommendationCopy = !overallReady
    ? "The backend or dataset scan is unavailable. Check the status details before continuing."
    : currentModel
      ? "A current exploratory model is available. Review its held-out metrics and limits before using explanations."
      : "The dataset is scanned, but there is no current trained prediction model. Start by inspecting real cases and fields.";
  const recommendationHref = currentModel ? "validation.html" : "design.html";
  const recommendationLabel = currentModel ? "Review evaluation" : "Explore CFD";
  setText(container, "#next-step-title", recommendationTitle);
  setText(container, "#next-step-copy", recommendationCopy);
  const nextLink = container.querySelector<HTMLAnchorElement>("#next-step-link");
  if (nextLink) {
    nextLink.href = recommendationHref;
    nextLink.innerHTML = `${recommendationLabel} <span aria-hidden="true">→</span>`;
  }
}

function providerLabel(result: ApiResult<ProviderStatus>): { label: string; state: string } {
  if (result.error) return { label: "Unavailable", state: "unavailable" };
  if (!result.value) return { label: "Unknown", state: "warning" };
  if (result.value.status === "reachable") return { label: "Reachable", state: "connected" };
  if (result.value.status === "unavailable") return { label: "Unavailable", state: "unavailable" };
  return { label: "Not Verified", state: "warning" };
}

function providerDetails(status: ProviderStatus | null, error: string | null) {
  if (error) return `Status unavailable: ${error}`;
  if (!status) return "Status unknown.";
  return `${status.provider ?? "Provider"} · ${status.model ?? "No model configured"} · ${status.checked_at ? `Verified ${status.checked_at}. ` : ""}${status.diagnostic ?? ""}`;
}

async function runProviderTest(button: HTMLButtonElement, stateSelector: string, resultSelector: string) {
  const container = button.closest<HTMLElement>("#overview-dashboard");
  if (!container) return;
  const state = container.querySelector<HTMLElement>(stateSelector);
  const result = container.querySelector<HTMLElement>(resultSelector);
  button.disabled = true;
  button.textContent = "Testing…";
  if (state) setState(container, stateSelector, "Testing", "loading");
  if (result) result.textContent = "Live connection test in progress…";
  try {
    const response = await fetchJson<ProviderTestResult>("/llm/connection-test", { method: "POST" });
    const connected = response.status === "connected";
    if (state) setState(container, stateSelector, connected ? "Connected" : "Unknown", connected ? "connected" : "warning");
    if (result) result.textContent = `Inference succeeded at ${response.tested_at ?? "time unavailable"}${response.latency_ms === undefined ? "" : ` · ${response.latency_ms} ms`}. ${response.diagnostic ?? ""}`;
  } catch (error) {
    const message = error instanceof Error ? error.message : "Connection test failed.";
    const stateLabel = message.toLowerCase().includes("not configured") ? "Not configured" : "Failed";
    if (state) setState(container, stateSelector, stateLabel, stateLabel === "Failed" ? "failed" : "warning");
    if (result) result.textContent = `Last live test failed: ${message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Test inference";
  }
}

function setText(container: ParentNode, selector: string, value: string) {
  const element = container.querySelector<HTMLElement>(selector);
  if (element) element.textContent = value;
}

function setState(container: ParentNode, selector: string, label: string, state: string) {
  const element = container.querySelector<HTMLElement>(selector);
  if (!element) return;
  element.textContent = label;
  element.dataset.state = state;
}

function renderList(container: ParentNode, selector: string, items: string[], emptyMessage: string) {
  const list = container.querySelector<HTMLUListElement>(selector);
  if (!list) return;
  list.replaceChildren();
  const values = items.length ? items : [emptyMessage];
  for (const value of values) {
    const item = document.createElement("li");
    item.textContent = value;
    list.append(item);
  }
}

function renderMetrics(container: ParentNode, metrics: Array<[string, string]>) {
  const holder = container.querySelector<HTMLElement>("#model-metrics");
  if (!holder) return;
  holder.replaceChildren();
  for (const [label, value] of metrics) {
    const row = document.createElement("div");
    row.className = "overview-metric";
    const name = document.createElement("span");
    name.textContent = label;
    const result = document.createElement("strong");
    result.textContent = value;
    row.append(name, result);
    holder.append(row);
  }
}

function renderReadiness(container: ParentNode, items: Array<[string, string]>) {
  const holder = container.querySelector<HTMLElement>("#readiness-details");
  if (!holder) return;
  holder.replaceChildren();
  for (const [label, status] of items) {
    const row = document.createElement("div");
    row.className = "overview-readiness-item";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("strong");
    value.textContent = status;
    row.append(name, value);
    holder.append(row);
  }
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes < 0) return "Not available";
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function formatMetric(value: number) {
  return Number.isFinite(value) ? value.toFixed(4) : "Not available";
}