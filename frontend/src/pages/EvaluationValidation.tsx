import { useEffect, useState, type ReactNode } from "react";

import type { useDatasetStatus } from "../hooks/useDatasetStatus";
import { getGptOssStatus, testGptOssConnection } from "../services/providerApi";
import { getGenerativeStatus } from "../services/generativeApi";
import { getKnowledgeStatus } from "../services/knowledgeApi";
import { getSurrogateModels } from "../services/mlApi";
import type { GenerativeModel, GenerativeStatus } from "../types/generative";
import type { KnowledgeStatus } from "../types/knowledge";
import type {
  GlobalShapResponse,
  SurrogateModel,
  SurrogateModelsResponse,
} from "../types/ml";
import type { GptOssStatus, ProviderTestResult } from "../types/provider";

interface LoadResult<T> {
  value: T | null;
  error: string | null;
}

interface EvaluationValidationProps {
  dataset: ReturnType<typeof useDatasetStatus>;
}

const emptyResult = <T,>(): LoadResult<T> => ({ value: null, error: null });

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "The evaluation status could not be loaded.";
}

function metric(value: number | null | undefined, units = "") {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Not available";
  return `${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}${units ? ` ${units}` : ""}`;
}

function DetailDisclosure({ children, id, onExpand, title }: { children: ReactNode; id: string; onExpand?: () => void; title: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section className="validation-disclosure">
      <h3>
        <button
          aria-controls={id}
          aria-expanded={expanded}
          className="validation-disclosure__toggle"
          onClick={() => {
            if (!expanded) onExpand?.();
            setExpanded(!expanded);
          }}
          type="button"
        >
          <span>{title}</span>
          <span aria-hidden="true" className="validation-disclosure__chevron">⌄</span>
        </button>
      </h3>
      <div className="validation-disclosure__content" hidden={!expanded} id={id}>
        {children}
      </div>
    </section>
  );
}

function ValidationCard({
  children,
  details,
  id,
  status,
  summary,
  title,
}: {
  children: ReactNode;
  details: ReactNode;
  id: string;
  status: "available" | "caution" | "unavailable" | "neutral";
  summary: ReactNode;
  title: string;
}) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section className="panel validation-card" aria-labelledby={`${id}-title`}>
      <div className="validation-card__heading">
        <h2 id={`${id}-title`}>
          <button aria-controls={`${id}-details`} aria-expanded={expanded} className="validation-card__toggle" type="button" onClick={() => setExpanded((value) => !value)}>
            <span>{title}</span>
            <span aria-hidden="true" className="validation-card__chevron">⌄</span>
          </button>
        </h2>
        <span className={`validation-status validation-status--${status}`}>
          {status === "available" ? "Evidence available" : status === "caution" ? "Caveats apply" : status === "unavailable" ? "Unavailable" : "Not evaluated"}
        </span>
      </div>
      <div className="validation-card__summary">{summary}</div>
      {children}
      <div className="validation-card__details" hidden={!expanded} id={`${id}-details`}>{details}</div>
    </section>
  );
}

function MetricValue({ label, value }: { label: string; value: string | number }) {
  return <div className="validation-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function ModelDetails({ model }: { model: SurrogateModel }) {
  const validation = model.metrics.extra_trees_spatial_validation;
  return (
    <>
      <p className="muted">
        {model.model_name} · {model.target.region}/{model.target.field}
        {model.target.units ? ` (${model.target.units})` : ""} · recorded {model.split.validation_kind}.
        Scores are not independent-case test results.
      </p>
      <div className="validation-metrics">
        <MetricValue label="MAE" value={metric(validation.mae, model.target.units ?? "")} />
        <MetricValue label="R²" value={metric(validation.r2)} />
        <MetricValue label="RMSE" value={metric(validation.rmse, model.target.units ?? "")} />
        <MetricValue label="Training samples" value={model.split.training_sample_count} />
        <MetricValue label="Held-out samples" value={model.split.validation_sample_count} />
        <MetricValue label="Excluded buffer" value={model.split.excluded_buffer_sample_count} />
      </div>
      <p className="muted">{model.metrics.metric_scope}</p>
      <h3>Baseline comparison</h3>
      <div className="validation-metrics">
        <MetricValue label="Mean baseline MAE" value={metric(model.metrics.baseline_mean_predictor.mae, model.target.units ?? "")} />
        <MetricValue label="Mean baseline RMSE" value={metric(model.metrics.baseline_mean_predictor.rmse, model.target.units ?? "")} />
        <MetricValue label="Mean baseline R²" value={metric(model.metrics.baseline_mean_predictor.r2)} />
      </div>
      <h3>Split and model setup</h3>
      <p className="muted">{model.split.method} · axis {model.split.axis} · {model.dataset.case_count} source cases.</p>
      <p className="muted">Features: {model.features.map((feature) => `${feature.name}${feature.units ? ` (${feature.units})` : ""}`).join(", ")}.</p>
      {model.limitations.length > 0 && <ul className="validation-detail-list">{model.limitations.map((item) => <li key={item}>{item}</li>)}</ul>}
    </>
  );
}

function InterpretabilityDetails({ model }: { model: SurrogateModel }) {
  const shap = model.explainability.shap_global;
  return (
    <>
      <p className="muted">Method: {shap.method} · sample count: {shap.sample_count} · additivity checked: {shap.additivity_checked ? `yes (max error ${metric(shap.max_absolute_additivity_error)})` : "no"}.</p>
      <FeatureList features={shap.features} />
      <h3>LIME metadata</h3>
      <p className="muted">Method: {model.explainability.lime.method} · quality metric: {model.explainability.lime.approximation_quality_metric}. No LIME score or SHAP/LIME agreement result is reported by model metadata.</p>
      <p className="muted">Feature attribution describes model behavior; it does not demonstrate physical causation.</p>
      {model.limitations.length > 0 && <ul className="validation-detail-list">{model.limitations.map((item) => <li key={item}>{item}</li>)}</ul>}
    </>
  );
}

function FeatureList({ features }: { features: GlobalShapResponse["features"] }) {
  return (
    <div className="validation-table-wrap">
      <table className="validation-table">
        <thead><tr><th>Feature</th><th>Mean |SHAP|</th><th>Mean SHAP</th></tr></thead>
        <tbody>{features.map((feature) => (
          <tr key={feature.feature}>
            <td>{feature.feature}</td><td>{metric(feature.mean_absolute_shap)}</td><td>{metric(feature.mean_shap)}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function GenerativeDetails({ model }: { model: GenerativeModel }) {
  const metrics = Object.entries(model.metrics);
  return (
    <>
      <p className="muted">{model.model_type.toUpperCase()} · {model.source.case_id} · {model.configuration.epochs} epochs · seed {model.configuration.seed} · {model.configuration.device}</p>
      <p className="muted">{model.split.patch_counts.train} training / {model.split.patch_counts.validation} held-out / {model.split.patch_counts.buffer} buffer patches.</p>
      <div className="validation-metrics">
        {metrics.map(([key, value]) => (
          <MetricValue key={key} label={key.replace(/_/g, " ")} value={typeof value === "number" ? metric(value) : typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)} />
        ))}
      </div>
      <p className="muted">Representation: {model.representation.method}; grid {model.representation.grid_size.join(" × ")}; patch {model.configuration.patch_shape.join(" × ")}.</p>
      <ul className="validation-detail-list">
        {model.limitations.map((item) => <li key={item}>{item}</li>)}
        {model.model_type === "gan" && (
          <li>Physics enforcement is not established by this metadata. Conservation, PDE compliance, and physical safety must not be inferred from distribution metrics.</li>
        )}
      </ul>
    </>
  );
}

function ProviderCard({
  connection,
  details,
  error,
  id,
  label,
  onTest,
  pending,
  status,
}: {
  connection: LoadResult<ProviderTestResult>;
  details: string | null;
  error: string | null;
  id: string;
  label: string;
  onTest: () => void;
  pending: boolean;
  status: LoadResult<GptOssStatus>;
}) {
  const [expanded, setExpanded] = useState(false);
  const connectionState = status.error || status.value?.status === "unavailable"
    ? "Unavailable"
    : status.value?.status === "reachable"
      ? "Reachable"
      : "Not Verified";
  return (
    <section className="validation-provider">
      <h3 className="validation-provider__heading">
        <button aria-label={`Toggle ${label} details`} aria-controls={`${id}-details`} aria-expanded={expanded} onClick={() => setExpanded((value) => !value)} type="button">
          <span>{label}</span>
          <span className={`validation-status validation-status--${connection.value ? "available" : connection.error ? "unavailable" : "neutral"}`}>{connectionState}</span>
          <span aria-hidden="true" className="validation-provider__chevron">⌄</span>
        </button>
      </h3>
      <p className="muted">{status.value?.diagnostic ?? "Service reachability has not been verified."}</p>
      <div className="validation-provider__details" hidden={!expanded} id={`${id}-details`}>
        <button className="secondary-button" disabled={pending} onClick={onTest} type="button">{pending ? "Testing…" : "Test connection"}</button>
        {error && <p className="message message--error" role="alert">{error}</p>}
        {connection.error && <p className="message message--error" role="alert">{connection.error}</p>}
        {status.error && <p className="message message--error" role="alert">{status.error}</p>}
        <dl className="validation-detail-grid">
          <div><dt>Service status</dt><dd>{connectionState}</dd></div>
          <div><dt>Model</dt><dd>{status.value?.model ?? connection.value?.model ?? "Not reported"}</dd></div>
          <div><dt>Provider</dt><dd>{status.value && "provider" in status.value ? status.value.provider ?? "Not reported" : connection.value?.provider ?? "Not reported"}</dd></div>
          {status.value?.checked_at && <div><dt>Reachability checked</dt><dd>{new Date(status.value.checked_at).toLocaleString()}</dd></div>}
          {connection.value && <div><dt>Inference probe</dt><dd>{connection.value.diagnostic}</dd></div>}
          {connection.value && <div><dt>Inference latency</dt><dd>{connection.value.latency_ms === undefined ? "Not reported" : `${connection.value.latency_ms} ms`}</dd></div>}
          {connection.value && <div><dt>Inference tested</dt><dd>{new Date(connection.value.tested_at).toLocaleString()}</dd></div>}
          {connection.error && <div><dt>Inference probe</dt><dd>{connection.error}</dd></div>}
          {details && <div><dt>Diagnostic</dt><dd>{details}</dd></div>}
        </dl>
        <p className="muted">Service reachability and inference generation are tested separately. Neither validates generated answers or physical correctness. Credentials are never displayed.</p>
      </div>
    </section>
  );
}

export function EvaluationValidation({ dataset }: EvaluationValidationProps) {
  const [models, setModels] = useState<LoadResult<SurrogateModelsResponse>>(emptyResult);
  const [generative, setGenerative] = useState<LoadResult<GenerativeStatus>>(emptyResult);
  const [knowledge, setKnowledge] = useState<LoadResult<KnowledgeStatus>>(emptyResult);
  const [gptOss, setGptOss] = useState<LoadResult<GptOssStatus>>(emptyResult);
  const [gptOssConnection, setGptOssConnection] = useState<LoadResult<ProviderTestResult>>(emptyResult);
  const [gptOssTesting, setGptOssTesting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);

  useEffect(() => {
    let active = true;
    setIsRefreshing(true);
    Promise.all([
      getSurrogateModels().then((value) => ({ value, error: null })).catch((error: unknown) => ({ value: null, error: errorMessage(error) })),
      getGenerativeStatus().then((value) => ({ value, error: null })).catch((error: unknown) => ({ value: null, error: errorMessage(error) })),
      getKnowledgeStatus().then((value) => ({ value, error: null })).catch((error: unknown) => ({ value: null, error: errorMessage(error) })),
      getGptOssStatus().then((value) => ({ value, error: null })).catch((error: unknown) => ({ value: null, error: errorMessage(error) })),
    ]).then(([modelResult, generativeResult, knowledgeResult, gptOssResult]) => {
      if (!active) return;
      setModels(modelResult);
      setGenerative(generativeResult);
      setKnowledge(knowledgeResult);
      setGptOss(gptOssResult);
    }).finally(() => {
      if (active) setIsRefreshing(false);
    });
    return () => { active = false; };
  }, [refreshToken]);

  const availableModels = generative.value?.models.filter((model) => model.available) ?? [];
  const selectedSurrogate = models.value?.models.find((model) => model.available);
  const shapFeatures = selectedSurrogate?.explainability.shap_global.features ?? [];
  const topShapFeature = [...shapFeatures].sort((left, right) => right.mean_absolute_shap - left.mean_absolute_shap)[0];
  const vaeModels = availableModels.filter((model) => model.model_type === "vae");
  const ganModels = availableModels.filter((model) => model.model_type === "gan");
  const feedsLoaded = [models.value, generative.value, knowledge.value, gptOss.value].filter(Boolean).length;
  const dataStatus = dataset.error
    ? "Scan status unavailable"
    : dataset.scan?.status === "scanned"
      ? `${dataset.cases.length} scanned cases`
      : dataset.scan?.status ?? (dataset.isLoading ? "Checking…" : "Not scanned");
  const topModelMetric = selectedSurrogate?.metrics.extra_trees_spatial_validation;
  async function testGptOss() {
    setGptOssTesting(true);
    setGptOssConnection(emptyResult);
    try {
      setGptOssConnection({ value: await testGptOssConnection(), error: null });
    } catch (error) {
      setGptOssConnection({ value: null, error: errorMessage(error) });
    } finally {
      setGptOssTesting(false);
    }
  }

  return (
    <section className="evaluation-page evaluation-page--compact">
      <header className="page-heading validation-page-heading">
        <div>
          <h1>Evaluation &amp; Validation</h1>
          <p className="intro__copy">Evaluate model accuracy, physical consistency, interpretability, and evidence grounding.</p>
          <p className="validation-baseline">Baseline case: Not selected · Dataset: {dataStatus} · Protocol: {selectedSurrogate?.split.validation_kind ?? "No full validation protocol reported"}</p>
        </div>
        <div className="validation-header-actions">
          <button className="secondary-button" disabled={isRefreshing} onClick={() => {
            void dataset.refresh();
            setRefreshToken((current) => current + 1);
          }} type="button">{isRefreshing ? "Refreshing…" : "Refresh validation data"}</button>
          <button className="primary-button" onClick={() => window.print()} type="button">Export Audit Report (PDF)</button>
        </div>
      </header>

      <section className="panel validation-matrix" aria-labelledby="validation-matrix-title">
        <div className="validation-card__heading">
          <h2 id="validation-matrix-title">System Reliability Matrix</h2>
          <span className="validation-status validation-status--unavailable">Full suite: not reported</span>
        </div>
        <div className="validation-matrix__metrics">
          <MetricValue label="Verification coverage" value="Not available" />
          <MetricValue label="Modules assessed" value="Not reported" />
          <MetricValue label="Verified" value="Not reported" />
          <MetricValue label="Qualified with caveats" value="Not reported" />
          <MetricValue label="Restricted" value="Not reported" />
          <MetricValue label="Status feeds loaded" value={`${feedsLoaded} / 5`} />
        </div>
        <p className="muted">The backend has no aggregate validation-suite endpoint. Counts and coverage are not inferred from service configuration or model availability.</p>
        <div className="validation-matrix__highlights">
          <p><strong>Numerical mesh independence:</strong> Not evaluated by the current validation API.</p>
          <p><strong>Surrogate prediction fidelity:</strong> {selectedSurrogate ? `Recorded spatial-holdout MAE ${metric(topModelMetric?.mae, selectedSurrogate.target.units ?? "")}; exploratory, not independently validated.` : "No current evaluation result."}</p>
          <p><strong>Generative physical constraints:</strong> {ganModels.length ? "Not established; PDE and conservation checks are not reported." : "No GAN model available to assess."}</p>
        </div>
      </section>

      <DetailDisclosure
        id="validation-audit-details"
        onExpand={() => void dataset.loadAudit()}
        title="Audit Details"
      >
        <div className="validation-detail-grid">
          <div><dt>Audit status</dt><dd>{dataset.audit?.status ?? "Unavailable"}</dd></div>
          <div><dt>Audit identifier / hash</dt><dd>Not returned by the current API.</dd></div>
          <div><dt>Scan generated</dt><dd>{dataset.scan?.generated_at ?? "Not reported"}</dd></div>
          <div><dt>Source kind</dt><dd>{dataset.scan?.source_kind ?? "Not reported"}</dd></div>
          <div><dt>Current cases</dt><dd>{dataset.cases.map((item) => item.case_id).join(", ") || "None reported"}</dd></div>
          <div><dt>Recorded split protocol</dt><dd>{selectedSurrogate?.split.method ?? "Not reported"}</dd></div>
          <div><dt>Build / solver version</dt><dd>Not returned by current validation endpoints.</dd></div>
        </div>
        {dataset.error && <p className="message message--error" role="alert">{dataset.error}</p>}
        {dataset.isLoadingAudit && <p className="muted" role="status">Loading audit report…</p>}
        {dataset.auditError && <p className="message message--error" role="alert">{dataset.auditError}</p>}
        {dataset.audit?.report && <pre className="validation-audit-report">{dataset.audit.report}</pre>}
      </DetailDisclosure>

      <ValidationCard
        id="validation-cfd"
        title="1. CFD Numerical Consistency"
        status="unavailable"
        summary={<><strong>Mesh convergence:</strong> Not evaluated · GCI: Not available</>}
        details={<>
          <p className="muted">The current backend exposes mesh-comparison input, but its response contract reports the comparison as unavailable. No GCI, Richardson extrapolation, or convergence status is returned.</p>
          <p className="muted">Scanned cases: {dataset.cases.length ? dataset.cases.map((item) => `${item.case_id} (${item.mesh_info?.cell_count ?? "cell count not reported"})`).join("; ") : "No cases reported."}</p>
          <p className="muted">Mesh-level peak temperatures, pressure drops, tested Reynolds comparisons, scalar field convergence, and solver versions are not provided by the current dashboard API.</p>
          {dataset.scan?.warnings.map((warning) => <p className="message message--notice" key={warning}>{warning}</p>)}
        </>}
      >
        <div className="validation-metrics">
          <MetricValue label="Scanned cases" value={dataset.cases.length} />
          <MetricValue label="GCI" value="Not reported" />
        </div>
      </ValidationCard>

      <ValidationCard
        id="validation-surrogate"
        title="2. Surrogate Model Evaluation"
        status={models.error ? "unavailable" : selectedSurrogate ? "caution" : "unavailable"}
        summary={selectedSurrogate ? <><strong>MAE:</strong> {metric(topModelMetric?.mae, selectedSurrogate.target.units ?? "")} <span>·</span> <strong>R²:</strong> {metric(topModelMetric?.r2)}</> : <strong>No current surrogate evaluation result</strong>}
        details={models.error ? <p className="message message--error" role="alert">{models.error}</p> : selectedSurrogate ? <ModelDetails model={selectedSurrogate} /> : <p className="muted">{models.value?.empty_reason ?? "Model inventory is loading or unavailable."} No evaluation values are substituted.</p>}
      >
        <p className="muted">{selectedSurrogate ? "Recorded spatial-holdout metrics are exploratory and do not establish performance on independent geometries or operating conditions." : models.error ? "Model status is unavailable." : models.value ? models.value.empty_reason ?? "No eligible model is available." : "Checking model inventory…"}</p>
        {selectedSurrogate && topModelMetric?.r2 !== null && topModelMetric?.r2 !== undefined && topModelMetric.r2 <= 0 && (
          <p className="message message--notice" role="status">Non-positive held-out R²: this model has not demonstrated useful predictive skill against its mean reference.</p>
        )}
      </ValidationCard>

      <ValidationCard
        id="validation-interpretability"
        title="3. SHAP &amp; LIME Interpretability"
        status={topShapFeature ? "caution" : "unavailable"}
        summary={topShapFeature ? <><strong>Leading global SHAP feature:</strong> {topShapFeature.feature} · SHAP/LIME agreement: Not evaluated</> : <strong>No current explanation results available</strong>}
        details={selectedSurrogate ? <InterpretabilityDetails model={selectedSurrogate} /> : <p className="muted">Global SHAP and LIME explanation data are not available without a current surrogate model.</p>}
      >
        <p className="muted">{topShapFeature ? "Feature attribution describes model behavior; it does not demonstrate physical causation. No LIME local score or agreement test is present in the current status response." : "No feature ranking, explanation quality, or SHAP/LIME agreement result is fabricated."}</p>
      </ValidationCard>

      <ValidationCard
        id="validation-generative"
        title="4. Generative AI Fidelity &amp; Physics"
        status={availableModels.length ? "caution" : generative.error ? "unavailable" : "neutral"}
        summary={<div className="validation-generative-summaries">
          <div><strong>VAE reconstruction</strong><span>{vaeModels.length ? `MAE ${metric(typeof vaeModels[0].metrics.mae_K === "number" ? vaeModels[0].metrics.mae_K : null, "K")}` : "No current VAE result"}</span></div>
          <div><strong>GAN field generation</strong><span>{ganModels.length ? "Physical validity not established" : "No current GAN result"}</span></div>
        </div>}
        details={generative.error ? <p className="message message--error" role="alert">{generative.error}</p> : availableModels.length ? <div className="validation-model-list">{availableModels.map((model) => <article key={model.model_id}><h3>{model.model_type.toUpperCase()} · {model.source.case_id}</h3><GenerativeDetails model={model} /></article>)}</div> : <p className="muted">{generative.value?.empty_reason ?? "Checking saved generative model status."}</p>}
      >
        {ganModels.length > 0 && <p className="message message--notice" role="status">GAN outputs are synthetic samples, not CFD solutions. Statistical similarity does not establish conservation, PDE compliance, or engineering validity.</p>}
        {vaeModels.length > 0 && <p className="muted">A reconstruction error metric is not a physical validity test.</p>}
      </ValidationCard>

      <ValidationCard
        id="validation-grounding"
        title="5. NLP Intent &amp; Citation Grounding"
        status={knowledge.error ? "unavailable" : knowledge.value?.corpus.passage_count ? "caution" : "neutral"}
        summary={<div className="validation-generative-summaries">
          <div><strong>Intent classification</strong><span>{knowledge.value?.bert?.status === "configured" ? "Configured · benchmark F1 not returned" : "Not configured / evaluated"}</span></div>
          <div><strong>Retrieval &amp; grounding</strong><span>{knowledge.value ? `${knowledge.value.corpus.document_count} documents · ${knowledge.value.corpus.passage_count} passages · traceability not evaluated` : "Status unavailable"}</span></div>
        </div>}
        details={knowledge.error ? <p className="message message--error" role="alert">{knowledge.error}</p> : knowledge.value ? <>
          <div className="validation-detail-grid">
            <div><dt>BERT status</dt><dd>{knowledge.value.bert.status}</dd></div>
            <div><dt>Benchmark F1 / precision / recall</dt><dd>Not returned</dd></div>
            <div><dt>Held-out test size</dt><dd>{knowledge.value.bert.held_out_test_size}</dd></div>
            <div><dt>Supported labels</dt><dd>{knowledge.value.bert.label_set.join(", ") || "None reported"}</dd></div>
            <div><dt>Retrieval method</dt><dd>{knowledge.value.retrieval.method}</dd></div>
            <div><dt>Similarity threshold</dt><dd>{knowledge.value.retrieval.minimum_cosine_similarity}</dd></div>
            <div><dt>Chunk size / overlap</dt><dd>{knowledge.value.retrieval.chunk_characters} / {knowledge.value.retrieval.chunk_overlap_characters} characters</dd></div>
            <div><dt>Citation traceability</dt><dd>Not evaluated by the current status endpoint</dd></div>
          </div>
          {knowledge.value.bert.reason && <p className="muted">{knowledge.value.bert.reason}</p>}
          <p className="muted">Retrieval metadata and citation traceability do not prove that answer content is factually correct.</p>
        </> : <p className="muted">Checking retrieval and classifier status…</p>}
      >
        <p className="muted">{knowledge.value ? "The knowledge service reports configuration and index inventory, but this dashboard has no benchmark or citation-grounding evaluation endpoint." : "No benchmark or traceability values are invented."}</p>
      </ValidationCard>

      <ValidationCard
        id="validation-providers"
        title="6. GPT-OSS Provider Status"
        status="neutral"
        summary={<div className="validation-generative-summaries">
          <div><strong>GPT-OSS</strong><span>{gptOss.error || gptOss.value?.status === "unavailable" ? "Unavailable" : gptOss.value?.status === "reachable" ? "Reachable" : "Not Verified"}</span></div>
        </div>}
        details={<div className="validation-provider-grid">
          <ProviderCard id="validation-gpt-oss" label="GPT-OSS" status={gptOss} connection={gptOssConnection} pending={gptOssTesting} onTest={() => void testGptOss()} error={gptOssConnection.error} details={gptOss.value?.diagnostic ?? null} />
        </div>}
      >
        <p className="muted">GPT-OSS is the sole configured language-model provider. A successful inference probe does not establish scientific validity.</p>
      </ValidationCard>

      <DetailDisclosure id="validation-scope-details" title="Validation scope and limitations">
        <ul className="validation-detail-list">
          <li>This dashboard reports data from existing status and model-inventory endpoints; it does not run a complete validation suite.</li>
          <li>No CFD mesh convergence result or GCI is returned by current validation status.</li>
          <li>Surrogate scores are recorded spatial-holdout results and may not generalize to new geometries or operating conditions.</li>
          <li>GAN statistical comparisons do not establish conservation, PDE compliance, or physical validity.</li>
          <li>Provider status configuration does not indicate live availability until a connection test succeeds.</li>
          <li>Use source CFD evidence and engineering review for safety-critical decisions.</li>
        </ul>
      </DetailDisclosure>

      <footer className="validation-footer">
        <span>ThermoExplain AI · Evidence status from current backend responses</span>
        <span>{dataset.scan?.generated_at ? `Dataset scan: ${dataset.scan.generated_at}` : "Dataset scan timestamp unavailable"}</span>
      </footer>
    </section>
  );
}
