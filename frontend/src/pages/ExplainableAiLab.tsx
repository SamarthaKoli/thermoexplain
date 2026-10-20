import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { ContributionBars } from "../components/ContributionBars";
import { ConceptHint } from "../components/ConceptHint";
import {
  getGlobalShap,
  getLocalExplanation,
  getSurrogateModels,
  interpretWithGptOss,
  predictField,
} from "../services/mlApi";
import { getGptOssStatus, providerTestFailure, testGptOssConnection } from "../services/providerApi";
import type {
  GlobalShapResponse,
  GptOssInterpretation,
  LocalExplanationResponse,
  PredictionInput,
  SurrogateModel,
  SurrogatePrediction,
  SurrogateModelsResponse,
} from "../types/ml";
import type { GptOssStatus, ProviderTestResult } from "../types/provider";

type ExplanationTab = "shap-global" | "shap-local" | "lime-local";

interface CompactExplainableAiViewProps {
  activeTab: ExplanationTab;
  availableModels: SurrogateModel[];
  changeCase: (caseId: string) => void;
  changeCoordinate: (axis: "x" | "y" | "z", value: string) => void;
  changeModel: (modelId: string) => void;
  changeTarget: (target: string) => void;
  coordinates: { x: string; y: string; z: string };
  explanationError: string | null;
  globalShap: GlobalShapResponse | null;
  globalShapError: string | null;
  gptOssConnection: ProviderTestResult | null;
  gptOssError: string | null;
  gptOssStatus: GptOssStatus | null;
  gptOssStatusError: string | null;
  interpretWithGptOss: () => Promise<void>;
  interpretation: GptOssInterpretation | null;
  isInterpreting: boolean;
  isLoadingExplanation: boolean;
  isPredicting: boolean;
  isTestingGptOss: boolean;
  localExplanation: LocalExplanationResponse | null;
  modelError: string | null;
  modelResponse: SurrogateModelsResponse | null;
  modelsForTarget: SurrogateModel[];
  prediction: SurrogatePrediction | null;
  predictionError: string | null;
  runPrediction: () => Promise<void>;
  selectedCase: SurrogateModel["cases"][number] | null;
  selectedModel: SurrogateModel | null;
  selectedTarget: string;
  selectExplanationTab: (tab: ExplanationTab) => Promise<void>;
  staleModels: SurrogateModel[];
  targetOptions: string[];
  testGptOssConnection: () => Promise<void>;
}

function CompactExplainableAiView({
  activeTab,
  availableModels,
  changeCase,
  changeCoordinate,
  changeModel,
  changeTarget,
  coordinates,
  explanationError,
  globalShap,
  globalShapError,
  gptOssConnection,
  gptOssError,
  gptOssStatus,
  gptOssStatusError,
  interpretWithGptOss,
  interpretation,
  isInterpreting,
  isLoadingExplanation,
  isPredicting,
  isTestingGptOss,
  localExplanation,
  modelError,
  modelResponse,
  modelsForTarget,
  prediction,
  predictionError,
  runPrediction,
  selectedCase,
  selectedModel,
  selectedTarget,
  selectExplanationTab,
  staleModels,
  targetOptions,
  testGptOssConnection,
}: CompactExplainableAiViewProps) {
  const [showAllGlobalShap, setShowAllGlobalShap] = useState(false);
  const [showAllLocal, setShowAllLocal] = useState(false);
  const activeExplanation = localExplanation?.explanation;

  return (
    <section className="explorer explainable-lab explainable-lab--compact">
      <header className="page-heading xai-page-heading">
        <div>
          <h1>Explainable AI Lab</h1>
          <p className="intro__copy">
            Understand what the model predicts and which inputs influence its result.
          </p>
        </div>
        <span className={`status-pill${availableModels.length ? "" : " status-pill--unavailable"}`}>
          {modelError ? "Model status unavailable" : !modelResponse ? "Checking model…" : availableModels.length ? "Model available" : "No current model"}
        </span>
      </header>

      <Disclosure id="xai-meaning" title="What is Explainable AI?">
        <p className="muted">
          Explainable AI helps describe how a machine-learning model reached an
          output. These tools explain this model’s behavior; they do not prove
          physical cause and effect or replace a CFD simulation.
        </p>
      </Disclosure>

      {modelError && <p className="message message--error" role="alert">{modelError}</p>}
      {!modelResponse && !modelError && <p className="muted" role="status">Loading available models…</p>}
      {modelResponse && availableModels.length === 0 && (
        <section className="panel empty-state">
          <h2>No current trained model is available</h2>
          <p className="muted">
            {staleModels.length ? "Saved model metadata no longer matches the current dataset." : modelResponse.empty_reason}
          </p>
          <p className="muted">
            Train a model only after verifying the dataset: <code>python -m backend.scripts.train_surrogate</code>.
          </p>
        </section>
      )}

      {selectedModel && selectedCase && (
        <>
          <section className="panel xai-setup" aria-labelledby="xai-setup-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">PREDICTION SETUP</p>
                <h2 id="xai-setup-title">Choose a case and location</h2>
              </div>
              <span className="status-pill status-pill--unavailable">Exploratory</span>
            </div>
            <div className="guided-selectors xai-target-selectors">
              <label className="selector-field">
                <span>Prediction target</span>
                <select aria-label="Target field" onChange={(event) => changeTarget(event.target.value)} value={selectedTarget}>
                  {targetOptions.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
              <label className="selector-field">
                <span>Trained model</span>
                <select aria-label="Trained model" onChange={(event) => changeModel(event.target.value)} value={selectedModel.model_id}>
                  {modelsForTarget.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_name}</option>)}
                </select>
              </label>
              <label className="selector-field">
                <span>CFD case</span>
                <select aria-label="Verified case" onChange={(event) => changeCase(event.target.value)} value={selectedCase.case_id}>
                  {selectedModel.cases.map((item) => (
                    <option key={item.case_id} value={item.case_id}>
                      {item.case_id} · Re {item.reynolds_number} · {item.mesh_resolution}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <div className="guided-selectors guided-selectors--coordinates xai-coordinates">
              {(["x", "y", "z"] as const).map((axis, index) => (
                <label className="selector-field" key={axis}>
                  <span>{axis.toUpperCase()} coordinate · {selectedModel.coordinate_units ?? "unit not declared"}</span>
                  <input
                    aria-label={`${axis.toUpperCase()} coordinate`}
                    onChange={(event) => changeCoordinate(axis, event.target.value)}
                    step="any"
                    type="number"
                    value={coordinates[axis]}
                  />
                  <small>Range: {formatNumber(selectedCase.bounds[index][0])} to {formatNumber(selectedCase.bounds[index][1])}</small>
                </label>
              ))}
            </div>
            <p className="muted xai-reynolds">
              Reynolds number: <strong>{selectedCase.reynolds_number}</strong> (from the selected case)
            </p>
            <div className="xai-run-row">
              <button
                className="primary-button"
                disabled={isPredicting}
                onClick={() => void runPrediction()}
                type="button"
              >
                {isPredicting ? "Running prediction…" : "Run Prediction"}
              </button>
              {predictionError && <p className="message message--error" role="alert">{predictionError}</p>}
            </div>
            <Disclosure id="xai-advanced" title="Advanced settings and feature details">
              <dl className="metadata-list">
                <div><dt>Trained model</dt><dd>{selectedModel.model_name}</dd></div>
                <div><dt>Model target</dt><dd>{selectedModel.target.region}/{selectedModel.target.field} · {selectedModel.target.units ?? "unit not available"}</dd></div>
                <div><dt>Solution time</dt><dd>{selectedCase.solution_time}</dd></div>
                <div><dt>Cell count</dt><dd>{selectedCase.cell_count.toLocaleString()}</dd></div>
                <div><dt>Coordinate source</dt><dd>{selectedModel.coordinate_method}</dd></div>
                <div><dt>Model inputs</dt><dd>{selectedModel.features.map((feature) => `${feature.name}${feature.units ? ` (${feature.units})` : ""}`).join(", ")}</dd></div>
              </dl>
            </Disclosure>
          </section>

          <section className="panel xai-result" aria-labelledby="xai-result-title" aria-live="polite">
            <div className="section-heading">
              <div>
                <p className="eyebrow">PREDICTION RESULT</p>
                <h2 id="xai-result-title">{prediction ? `Predicted ${prediction.target.region}/${prediction.target.field}` : "Your result will appear here"}</h2>
              </div>
              {prediction && <span className="status-pill status-pill--unavailable">Exploratory estimate</span>}
            </div>
            {prediction ? (
              <>
                <p className="prediction-value">{formatNumber(prediction.prediction)} <span>{prediction.units ?? "unit not available"}</span></p>
                <p className="muted">
                  This estimates the selected model target for {prediction.case_id} at the entered location. It is not a new CFD result.
                </p>
              </>
            ) : (
              <p className="muted">Enter the location above and select <strong>Run Prediction</strong> to see the model’s result.</p>
            )}
            <Disclosure id="xai-validation-details" title="View validation details">
              <p className="muted">{selectedModel.metrics.metric_scope}</p>
              <div className="model-metrics">
                <div>
                  <h3>Model validation</h3>
                  <p>MAE: {formatNumber(selectedModel.metrics.extra_trees_spatial_validation.mae)} {selectedModel.target.units ?? ""}</p>
                  <p>RMSE: {formatNumber(selectedModel.metrics.extra_trees_spatial_validation.rmse)} {selectedModel.target.units ?? ""}</p>
                  <p>R²: {selectedModel.metrics.extra_trees_spatial_validation.r2 === null ? "Not available" : formatNumber(selectedModel.metrics.extra_trees_spatial_validation.r2)}</p>
                </div>
                <div>
                  <h3>Validation coverage</h3>
                  <p>{selectedModel.split.validation_sample_count.toLocaleString()} held-out samples</p>
                  <p>{selectedModel.split.validation_kind}</p>
                </div>
                <div>
                  <h3>Reference comparison</h3>
                  <p>{prediction ? "Not available for this selected coordinate." : "Available only when a matched reference result exists."}</p>
                  <p>Prediction error: Not available</p>
                </div>
              </div>
              {selectedModel.limitations.length > 0 && (
                <>
                  <h3>Known limitations</h3>
                  <ul className="limitation-list">{selectedModel.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
                </>
              )}
              <p className="message message--notice">
                Metrics describe the recorded validation split only. Results outside the scanned cases, feature ranges, or geometry are not established.
              </p>
            </Disclosure>
          </section>

          <section className="panel xai-explanation" aria-labelledby="xai-shap-title">
            <p className="eyebrow">SHAP · FEATURE ATTRIBUTION</p>
            <h2 id="xai-shap-title">What influenced this prediction?</h2>
            <p className="muted">SHAP estimates how each input moves the model’s output. Positive effects raise it; negative effects lower it. This describes the model, not physical causation.</p>
            {!prediction ? (
              <p className="xai-empty-note">Run a prediction to calculate feature effects for that result.</p>
            ) : (
              <>
                <button className="secondary-button" disabled={isLoadingExplanation} onClick={() => void selectExplanationTab("shap-local")} type="button">
                  {isLoadingExplanation && activeTab === "shap-local" ? "Calculating SHAP…" : "Calculate SHAP for this prediction"}
                </button>
                {activeTab === "shap-local" && activeExplanation && (
                  <div className="xai-chart-block">
                    <h3>Effects for this prediction</h3>
                    <p className="muted">Contribution values are in {prediction.units ?? "the target’s unit"}.</p>
                    <ContributionBars items={[...activeExplanation.contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, showAllLocal ? undefined : 5)} />
                    {activeExplanation.contributions.length > 5 && <button className="text-button" onClick={() => setShowAllLocal((show) => !show)} type="button">{showAllLocal ? "Show top five" : "View all SHAP features"}</button>}
                  </div>
                )}
                {activeTab === "shap-local" && isLoadingExplanation && <p className="muted" role="status">Calculating feature effects for this prediction…</p>}
                {activeTab === "shap-local" && activeExplanation && (
                  <Disclosure id="xai-shap-technical" title="How SHAP works">
                    <p className="muted">{activeExplanation.baseline_definition ?? "The baseline definition is not available."}</p>
                    <p className="muted">
                      Reconstruction: {activeExplanation.reconstructed_prediction === undefined ? "Not available" : `${formatNumber(activeExplanation.reconstructed_prediction)} ${prediction.units ?? ""}`} · Additivity check: {activeExplanation.additivity_passed === undefined ? "Not available" : activeExplanation.additivity_passed ? "passed" : "not passed"}.
                    </p>
                  </Disclosure>
                )}
              </>
            )}
            <Disclosure id="xai-global-shap" title="Global feature importance">
              {globalShap ? (
                <>
                  <p className="muted">Mean absolute SHAP values from {globalShap.sample_count} held-out samples. These summarize model behavior across samples, not this individual prediction.</p>
                  <ContributionBars items={[...globalShap.features].sort((a, b) => b.mean_absolute_shap - a.mean_absolute_shap).slice(0, showAllGlobalShap ? undefined : 5).map((item) => ({ feature: item.feature, contribution: item.mean_absolute_shap, meanAbsoluteShap: item.mean_absolute_shap, meanShap: item.mean_shap }))} global />
                  {globalShap.features.length > 5 && <button className="text-button" onClick={() => setShowAllGlobalShap((show) => !show)} type="button">{showAllGlobalShap ? "Show top five" : "View all SHAP features"}</button>}
                </>
              ) : globalShapError ? <p className="message message--error" role="alert">{globalShapError}</p> : <p className="muted" role="status">Loading global feature importance…</p>}
            </Disclosure>
            {explanationError && activeTab === "shap-local" && <p className="message message--error" role="alert">{explanationError}</p>}
          </section>

          <section className="panel xai-lime" aria-labelledby="xai-lime-title">
            <p className="eyebrow">LIME · LOCAL APPROXIMATION</p>
            <h2 id="xai-lime-title">Why did the model predict this?</h2>
            <p className="muted">LIME fits a simple approximation near this one input. It is different from SHAP and does not show physical cause and effect.</p>
            {!prediction ? <p className="xai-empty-note">Run a prediction before requesting a LIME explanation.</p> : (
              <>
                <button className="secondary-button" disabled={isLoadingExplanation} onClick={() => void selectExplanationTab("lime-local")} type="button">
                  {isLoadingExplanation && activeTab === "lime-local" ? "Calculating LIME…" : "Calculate LIME explanation"}
                </button>
                {activeTab === "lime-local" && activeExplanation && (
                  <div className="xai-chart-block">
                    <p className="muted">A positive local effect raises the approximation; a negative effect lowers it.</p>
                    <ContributionBars items={[...activeExplanation.contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, showAllLocal ? undefined : 3)} />
                    {activeExplanation.contributions.length > 3 && <button className="text-button" onClick={() => setShowAllLocal((show) => !show)} type="button">{showAllLocal ? "Show top three" : "View all LIME features"}</button>}
                    <Disclosure id="xai-lime-details" title="View LIME details">
                      <p className="muted">{activeExplanation.caveat ?? "No additional LIME caveat is available."}</p>
                      <p className="muted">Local R²: {activeExplanation.local_r2 === undefined ? "Not available" : formatNumber(activeExplanation.local_r2)} · Approximation: {activeExplanation.local_prediction === undefined ? "Not available" : `${formatNumber(activeExplanation.local_prediction)} ${prediction.units ?? ""}`}</p>
                    </Disclosure>
                  </div>
                )}
                {activeTab === "lime-local" && isLoadingExplanation && <p className="muted" role="status">Calculating the local approximation…</p>}
              </>
            )}
            {explanationError && activeTab === "lime-local" && <p className="message message--error" role="alert">{explanationError}</p>}
          </section>

          <section className="panel xai-gptOss" aria-labelledby="xai-gptOss-title">
            <div className="section-heading">
              <div>
                <p className="eyebrow">{gptOssStatus?.provider ?? "GPT-OSS"} · PLAIN-LANGUAGE SUMMARY</p>
                <h2 id="xai-gptOss-title">Explain my prediction in plain English</h2>
              </div>
              <span className={`status-pill${gptOssStatus?.status === "reachable" ? " status-pill--connected" : gptOssStatus?.status === "unavailable" || gptOssStatusError ? " status-pill--unavailable" : ""}`}>
                {gptOssStatusError ? "Unavailable" : gptOssStatus?.status === "reachable" ? "Reachable" : gptOssStatus?.status === "unavailable" ? "Unavailable" : gptOssStatus ? "Not Verified" : "Checking…"}
              </span>
            </div>
            <p className="muted">The configured provider explains the actual prediction and computed SHAP or LIME evidence. It does not set or change the numeric result.</p>
            <div className="gpt-oss-actions">
              <button className="secondary-button" disabled={isTestingGptOss} onClick={() => void testGptOssConnection()} type="button">
                {isTestingGptOss ? "Testing connection…" : "Test GPT-OSS Connection"}
              </button>
              <button
                className="primary-button"
                disabled={!prediction || gptOssStatus?.status !== "reachable" || !activeExplanation || (activeTab !== "shap-local" && activeTab !== "lime-local") || isInterpreting}
                onClick={() => void interpretWithGptOss()}
                type="button"
              >
                {isInterpreting ? "Requesting explanation…" : "Explain Prediction"}
              </button>
            </div>
            {gptOssConnection && (
              <div className={`provider-test provider-test--${gptOssConnection.status}`} role={gptOssConnection.status === "connected" ? "status" : "alert"}>
                <strong>{gptOssConnection.provider}: {gptOssConnection.status.replace("_", " ")}</strong>
                <span>Model {gptOssConnection.model ?? "not configured"}</span>
                <time dateTime={gptOssConnection.tested_at}>{new Date(gptOssConnection.tested_at).toLocaleString()}</time>
                <span>{gptOssConnection.diagnostic}{gptOssConnection.latency_ms === undefined ? "" : ` · ${gptOssConnection.latency_ms} ms`}</span>
              </div>
            )}
            {gptOssStatusError && <p className="message message--error" role="alert">{gptOssStatusError}</p>}
            {gptOssError && <p className="message message--error" role="alert">{gptOssError}</p>}
            {gptOssStatus && !gptOssStatus.configured && <p className="muted">{gptOssStatus.diagnostic ?? "The provider is not configured on the backend."}</p>}
            {interpretation && (
              <div className="ai-interpretation" aria-live="polite">
                <p>{interpretation.interpretation.interpretation}</p>
                <small>{interpretation.interpretation.provider ?? "GPT-OSS"} · {interpretation.interpretation.model} · {new Date(interpretation.interpretation.generated_at).toLocaleString()}</small>
              </div>
            )}
            <Disclosure id="xai-gptOss-technical" title="Technical details">
              <p className="muted">The backend receives the selected case, coordinates, model prediction, and requested explanation method. Credentials remain on the backend.</p>
              <p className="muted">Configured model: {gptOssStatus?.model ?? "Not available"}</p>
              {gptOssStatus?.diagnostic && <p className="muted">{gptOssStatus.diagnostic}</p>}
            </Disclosure>
          </section>
        </>
      )}
    </section>
  );
}

interface DisclosureProps {
  children: ReactNode;
  id: string;
  title: string;
}

function Disclosure({ children, id, title }: DisclosureProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <section className="xai-disclosure">
      <h3>
        <button
          aria-controls={id}
          aria-expanded={isOpen}
          className="xai-disclosure__toggle"
          onClick={() => setIsOpen((open) => !open)}
          type="button"
        >
          <span>{title}</span>
          <span aria-hidden="true" className="xai-disclosure__arrow">⌄</span>
        </button>
      </h3>
      <div hidden={!isOpen} id={id}>
        {children}
      </div>
    </section>
  );
}

function targetKey(model: SurrogateModel) {
  return `${model.target.region}/${model.target.field}`;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 7 }).format(
    value,
  );
}

function coordinatesAtCenter(modelCase: SurrogateModel["cases"][number]) {
  return {
    x: String((modelCase.bounds[0][0] + modelCase.bounds[0][1]) / 2),
    y: String((modelCase.bounds[1][0] + modelCase.bounds[1][1]) / 2),
    z: String((modelCase.bounds[2][0] + modelCase.bounds[2][1]) / 2),
  };
}

export function ExplainableAiLab() {
  const resultContext = useRef(0);
  const [modelResponse, setModelResponse] = useState<SurrogateModelsResponse | null>(
    null,
  );
  const [modelError, setModelError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [selectedTarget, setSelectedTarget] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [coordinates, setCoordinates] = useState({ x: "", y: "", z: "" });
  const [prediction, setPrediction] = useState<SurrogatePrediction | null>(null);
  const [predictionInput, setPredictionInput] = useState<PredictionInput | null>(
    null,
  );
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [activeTab, setActiveTab] = useState<ExplanationTab>("shap-global");
  const [globalShap, setGlobalShap] = useState<GlobalShapResponse | null>(null);
  const [globalShapError, setGlobalShapError] = useState<string | null>(null);
  const [localExplanation, setLocalExplanation] =
    useState<LocalExplanationResponse | null>(null);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  const [isLoadingExplanation, setIsLoadingExplanation] = useState(false);
  const [gptOssStatus, setGptOssStatus] = useState<GptOssStatus | null>(null);
  const [gptOssStatusError, setGptOssStatusError] = useState<string | null>(null);
  const [gptOssConnection, setGptOssConnection] =
    useState<ProviderTestResult | null>(null);
  const [gptOssError, setGptOssError] = useState<string | null>(null);
  const [isTestingGptOss, setIsTestingGptOss] = useState(false);
  const [interpretation, setInterpretation] =
    useState<GptOssInterpretation | null>(null);
  const [isInterpreting, setIsInterpreting] = useState(false);

  const availableModels = useMemo(
    () => modelResponse?.models.filter((model) => model.available) ?? [],
    [modelResponse],
  );
  const targetOptions = useMemo(
    () => [...new Set(availableModels.map(targetKey))],
    [availableModels],
  );
  const modelsForTarget = useMemo(
    () => availableModels.filter((model) => targetKey(model) === selectedTarget),
    [availableModels, selectedTarget],
  );
  const selectedModel =
    modelsForTarget.find((model) => model.model_id === selectedModelId) ??
    modelsForTarget[0] ??
    null;
  const selectedCase =
    selectedModel?.cases.find((item) => item.case_id === selectedCaseId) ??
    selectedModel?.cases[0] ??
    null;

  useEffect(() => {
    getSurrogateModels()
      .then((result) => {
        setModelResponse(result);
        setModelError(null);
        const firstAvailable = result.models.find((model) => model.available);
        if (firstAvailable) {
          setSelectedModelId(firstAvailable.model_id);
          setSelectedTarget(targetKey(firstAvailable));
        }
      })
      .catch((error: Error) => setModelError(error.message));
    getGptOssStatus()
      .then((status) => {
        setGptOssStatus(status);
        setGptOssStatusError(null);
      })
      .catch((error: Error) => {
        setGptOssStatus(null);
        setGptOssStatusError(error.message);
      });
  }, []);

  useEffect(() => {
    setGlobalShap(null);
    setGlobalShapError(null);
    if (!selectedModel) return;
    let active = true;
    getGlobalShap(selectedModel.model_id)
      .then((result) => {
        if (active) setGlobalShap(result);
      })
      .catch((error: Error) => {
        if (active) setGlobalShapError(error.message);
      });
    return () => {
      active = false;
    };
  }, [selectedModel?.model_id]);

  useEffect(() => {
    if (!selectedCase) return;
    setSelectedCaseId(selectedCase.case_id);
    setCoordinates(coordinatesAtCenter(selectedCase));
    setPrediction(null);
    setPredictionInput(null);
    setLocalExplanation(null);
    setInterpretation(null);
  }, [selectedCase?.case_id]);

  function clearComputedResults() {
    resultContext.current += 1;
    setPrediction(null);
    setPredictionInput(null);
    setLocalExplanation(null);
    setInterpretation(null);
    setPredictionError(null);
    setExplanationError(null);
    setGptOssError(null);
    setIsPredicting(false);
    setIsLoadingExplanation(false);
    setIsInterpreting(false);
  }

  function changeModel(modelId: string) {
    const model = availableModels.find((item) => item.model_id === modelId);
    if (!model) return;
    setSelectedModelId(modelId);
    setSelectedTarget(targetKey(model));
    clearComputedResults();
  }

  function changeTarget(value: string) {
    setSelectedTarget(value);
    const model = availableModels.find((item) => targetKey(item) === value);
    if (model) setSelectedModelId(model.model_id);
    clearComputedResults();
  }

  function changeCase(caseId: string) {
    const modelCase = selectedModel?.cases.find((item) => item.case_id === caseId);
    if (!modelCase) return;
    setSelectedCaseId(caseId);
    setCoordinates(coordinatesAtCenter(modelCase));
    clearComputedResults();
  }

  function changeCoordinate(name: "x" | "y" | "z", value: string) {
    setCoordinates((current) => ({ ...current, [name]: value }));
    clearComputedResults();
  }

  function buildInput(): PredictionInput | null {
    if (!selectedModel || !selectedCase) return null;
    const values = [coordinates.x, coordinates.y, coordinates.z].map((value) =>
      value.trim() === "" ? Number.NaN : Number(value),
    );
    if (!values.every(Number.isFinite)) {
      setPredictionError("Enter a finite number for each coordinate.");
      return null;
    }
    return {
      model_id: selectedModel.model_id,
      case_id: selectedCase.case_id,
      coordinates: { x: values[0], y: values[1], z: values[2] },
    };
  }

  async function runPrediction() {
    const input = buildInput();
    if (!input) return;
    const context = resultContext.current;
    setPredictionError(null);
    setIsPredicting(true);
    try {
      const result = await predictField(input);
      if (context !== resultContext.current) return;
      setPrediction(result);
      setPredictionInput(input);
      setLocalExplanation(null);
      setInterpretation(null);
    } catch (error) {
      if (context !== resultContext.current) return;
      setPredictionError(
        error instanceof Error ? error.message : "Prediction request failed.",
      );
    } finally {
      if (context === resultContext.current) setIsPredicting(false);
    }
  }

  async function selectExplanationTab(tab: ExplanationTab) {
    setActiveTab(tab);
    setExplanationError(null);
    if (tab === "shap-global") {
      setLocalExplanation(null);
      return;
    }
    if (!predictionInput) {
      setExplanationError("Run a prediction before requesting a local explanation.");
      return;
    }
    const context = resultContext.current;
    setIsLoadingExplanation(true);
    setLocalExplanation(null);
    try {
      const result = await getLocalExplanation(
        tab === "lime-local" ? "lime" : "shap",
        predictionInput,
      );
      if (context !== resultContext.current) return;
      setLocalExplanation(result);
    } catch (error) {
      if (context !== resultContext.current) return;
      setExplanationError(
        error instanceof Error ? error.message : "Explanation request failed.",
      );
    } finally {
      if (context === resultContext.current) setIsLoadingExplanation(false);
    }
  }

  async function runGptOssConnectionTest() {
    setGptOssError(null);
    setGptOssConnection(null);
    setIsTestingGptOss(true);
    try {
      const result = await testGptOssConnection();
      setGptOssConnection(result);
    } catch (error) {
      setGptOssConnection(
        providerTestFailure(
          gptOssStatus?.provider ?? "GPT-OSS",
          gptOssStatus?.model ?? null,
          error,
        ),
      );
      setGptOssError(
        error instanceof Error ? error.message : "GPT-OSS connection test failed.",
      );
    } finally {
      setIsTestingGptOss(false);
    }
  }

  async function runGptOssInterpretation() {
    if (!predictionInput || !prediction) return;
    const context = resultContext.current;
    setGptOssError(null);
    setInterpretation(null);
    setIsInterpreting(true);
    const method = activeTab === "lime-local" ? "lime" : "shap";
    try {
      const result = await interpretWithGptOss(method, predictionInput);
      if (context !== resultContext.current) return;
      setInterpretation(result);
    } catch (error) {
      if (context !== resultContext.current) return;
      setGptOssError(
        error instanceof Error ? error.message : "GPT-OSS interpretation failed.",
      );
    } finally {
      if (context === resultContext.current) setIsInterpreting(false);
    }
  }

  const staleModels = modelResponse?.models.filter((model) => !model.available) ?? [];
  const activeExplanation = localExplanation?.explanation;

  return window.location.pathname === "/explainable.html" ? (
    <CompactExplainableAiView
      activeTab={activeTab}
      availableModels={availableModels}
      changeCase={changeCase}
      changeCoordinate={changeCoordinate}
      changeModel={changeModel}
      changeTarget={changeTarget}
      coordinates={coordinates}
      explanationError={explanationError}
      globalShap={globalShap}
      globalShapError={globalShapError}
      gptOssConnection={gptOssConnection}
      gptOssError={gptOssError}
      gptOssStatus={gptOssStatus}
      gptOssStatusError={gptOssStatusError}
      interpretWithGptOss={runGptOssInterpretation}
      interpretation={interpretation}
      isInterpreting={isInterpreting}
      isLoadingExplanation={isLoadingExplanation}
      isPredicting={isPredicting}
      isTestingGptOss={isTestingGptOss}
      localExplanation={localExplanation}
      modelError={modelError}
      modelResponse={modelResponse}
      modelsForTarget={modelsForTarget}
      prediction={prediction}
      predictionError={predictionError}
      runPrediction={runPrediction}
      selectedCase={selectedCase}
      selectedModel={selectedModel}
      selectedTarget={selectedTarget}
      selectExplanationTab={selectExplanationTab}
      staleModels={staleModels}
      targetOptions={targetOptions}
      testGptOssConnection={runGptOssConnectionTest}
    />
  ) : (
    <section className="explorer explainable-lab">
      <header className="page-heading">
        <div>
          <p className="eyebrow">EXPLAINABLE MACHINE LEARNING</p>
          <h1>Explainable AI Lab</h1>
          <p className="intro__copy">
            Explore an experimental surrogate trained only on scanned CFD cells.
            It approximates stored results; it does not run a new CFD simulation.
          </p>
        </div>
      </header>

      <section className="panel experiment-note" aria-labelledby="experiment-heading">
        <h2 id="experiment-heading">What this experiment can—and cannot—learn</h2>
        <p className="muted">
          A surrogate model is a learned approximation of existing simulation
          outputs; it is quicker to query but does not solve the CFD equations.
          The target here is coolant temperature <code>coolant/T</code> in
          kelvin. Inputs are the verified OpenFOAM cell-center coordinates and the
          Reynolds number taken from the selected scanned case. Cell centers come
          from the decomposed OpenFOAM mesh, and target values are checked against
          that region’s cell count.
        </p>
        <p className="muted">
          The available data contains six mesh/case folders for one geometry, only
          two Reynolds numbers, and three mesh resolutions. Validation holds out
          spatial blocks across all related meshes and leaves a coordinate buffer;
          it is not independent-case validation. Coordinate length units are not
          declared in the scanned metadata. Treat every output as exploratory and
          specific to these cases.
        </p>
      </section>

      {modelError && <p className="message message--error" role="alert">{modelError}</p>}
      {!modelResponse && !modelError && (
        <p className="muted" role="status">Loading trained model inventory…</p>
      )}
      {modelResponse && availableModels.length === 0 && (
        <section className="panel empty-state">
          <h2>No current trained model is available</h2>
          <p className="muted">
            {staleModels.length > 0
              ? "Saved model metadata no longer matches the current dataset scan. Retrain to prevent stale predictions."
              : modelResponse.empty_reason}
          </p>
          <p className="muted">
            From the project root, run{" "}
            <code>python -m backend.scripts.train_surrogate</code>. Training uses
            a seeded sample and writes a local model artifact; no model or metrics
            are shown until that command completes successfully.
          </p>
          {staleModels.length > 0 && (
            <ul className="case-list">
              {staleModels.map((model) => (
                <li key={model.model_id}>
                  <span className="case-button">
                    <strong>{model.model_name}</strong>
                    <span>{model.unavailable_reason}</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {selectedModel && selectedCase && (
        <>
          <section className="panel selection-panel" aria-labelledby="model-selection-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">STEP 1 · SELECT VERIFIED DATA</p>
                <h2 id="model-selection-heading">Model and target</h2>
              </div>
              <span className="status-pill status-pill--unavailable">Exploratory</span>
            </div>
            <div className="guided-selectors">
              <label className="selector-field">
                <span>Target field</span>
                <select
                  aria-label="Target field"
                  onChange={(event) => changeTarget(event.target.value)}
                  value={selectedTarget}
                >
                  {targetOptions.map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
              </label>
              <label className="selector-field">
                <span>Trained model</span>
                <select
                  aria-label="Trained model"
                  onChange={(event) => changeModel(event.target.value)}
                  value={selectedModel.model_id}
                >
                  {modelsForTarget.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="selector-field">
                <span>Verified case</span>
                <select
                  aria-label="Verified case"
                  onChange={(event) => changeCase(event.target.value)}
                  value={selectedCase.case_id}
                >
                  {selectedModel.cases.map((item) => (
                    <option key={item.case_id} value={item.case_id}>
                      {item.case_id} · Re {item.reynolds_number} · {item.mesh_resolution}
                    </option>
                  ))}
                </select>
              </label>
              <div className="selector-field">
                <span>Stored solution time</span>
                <output className="selector-output">{selectedCase.solution_time}</output>
              </div>
            </div>
            <p className="muted">
              Target units: {selectedModel.target.units ?? "not established"} ·
              Coordinate units: {selectedModel.coordinate_units ?? "not declared"}
            </p>
          </section>

          <section className="panel prediction-input-panel" aria-labelledby="input-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">STEP 2 · SET A CELL LOCATION</p>
                <h2 id="input-heading">Prediction inputs</h2>
              </div>
              <span className="muted">
                Reynolds number is supplied by the verified case.
              </span>
            </div>
            <div className="guided-selectors guided-selectors--coordinates">
              {(["x", "y", "z"] as const).map((axis) => (
                <label className="selector-field" key={axis}>
                  <span>{axis.toUpperCase()} coordinate · {selectedModel.coordinate_units ?? "unit unknown"}</span>
                  <input
                    aria-label={`${axis.toUpperCase()} coordinate`}
                    onChange={(event) => changeCoordinate(axis, event.target.value)}
                    step="any"
                    type="number"
                    value={coordinates[axis]}
                  />
                  <small>
                    Verified range: {formatNumber(selectedCase.bounds[axis === "x" ? 0 : axis === "y" ? 1 : 2][0])} to{" "}
                    {formatNumber(selectedCase.bounds[axis === "x" ? 0 : axis === "y" ? 1 : 2][1])}
                  </small>
                </label>
              ))}
              <div className="selector-field">
                <span>Reynolds number</span>
                <output className="selector-output">{selectedCase.reynolds_number}</output>
                <small>Not editable; read from this case’s scanned metadata.</small>
              </div>
            </div>
            <button
              className="primary-button"
              disabled={isPredicting}
              onClick={() => void runPrediction()}
              type="button"
            >
              {isPredicting ? "Computing prediction…" : "Predict coolant temperature"}
            </button>
            {predictionError && <p className="message message--error" role="alert">{predictionError}</p>}
          </section>

          <section className="panel model-evaluation" aria-labelledby="metrics-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">HELD-OUT SPATIAL VALIDATION</p>
                <h2 id="metrics-heading">How this model performed</h2>
              </div>
              <span className="status-pill">Not independent-case test metrics</span>
            </div>
            <p className="muted">{selectedModel.metrics.metric_scope}</p>
            <div className="model-metrics">
              <div>
                <h3>Mean baseline</h3>
                <p>MAE {formatNumber(selectedModel.metrics.baseline_mean_predictor.mae)}</p>
                <p>RMSE {formatNumber(selectedModel.metrics.baseline_mean_predictor.rmse)}</p>
                <p>R² {selectedModel.metrics.baseline_mean_predictor.r2 === null ? "not defined" : formatNumber(selectedModel.metrics.baseline_mean_predictor.r2)}</p>
              </div>
              <div>
                <h3>Extra Trees</h3>
                <p>MAE {formatNumber(selectedModel.metrics.extra_trees_spatial_validation.mae)} {selectedModel.target.units ?? ""}</p>
                <p>RMSE {formatNumber(selectedModel.metrics.extra_trees_spatial_validation.rmse)} {selectedModel.target.units ?? ""}</p>
                <p>R² {selectedModel.metrics.extra_trees_spatial_validation.r2 === null ? "not defined" : formatNumber(selectedModel.metrics.extra_trees_spatial_validation.r2)}</p>
              </div>
              <div>
                <h3>Split coverage</h3>
                <p>{selectedModel.split.training_sample_count.toLocaleString()} training samples</p>
                <p>{selectedModel.split.validation_sample_count.toLocaleString()} validation samples</p>
                <p>{selectedModel.split.excluded_buffer_sample_count.toLocaleString()} excluded buffer samples</p>
              </div>
            </div>
            <p className="muted">
              The mean predictor is a simple baseline. Extra Trees metrics are
              measured only on the buffered spatial holdout, never on the model’s
              training rows. No claim is made about other pack designs or operating
              conditions.
            </p>
            {selectedModel.metrics.extra_trees_spatial_validation.r2 !== null &&
              selectedModel.metrics.extra_trees_spatial_validation.r2 <= 0 && (
                <p className="message message--notice" role="status">
                  The spatial-validation R² is non-positive. On this held-out
                  region, this surrogate has not demonstrated reliable predictive
                  skill over a validation-mean reference; do not use it for
                  engineering decisions.
                </p>
              )}
            <h3>Recorded limitations</h3>
            <ul className="limitation-list">
              {selectedModel.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
          </section>

          {prediction && (
            <section className="panel prediction-result" aria-labelledby="prediction-heading">
              <p className="eyebrow">EXPLORATORY SURROGATE OUTPUT</p>
              <h2 id="prediction-heading">
                Predicted {prediction.target.region}/{prediction.target.field}
              </h2>
              <p className="prediction-value">
                {formatNumber(prediction.prediction)}{" "}
                <span>{prediction.units ?? "unit unknown"}</span>
              </p>
              <p className="muted">
                Case {prediction.case_id} · Re {prediction.reynolds_number} ·
                coordinates ({formatNumber(prediction.coordinates.x)},{" "}
                {formatNumber(prediction.coordinates.y)},{" "}
                {formatNumber(prediction.coordinates.z)}) · exploratory model
                approximation, not a new CFD result.
              </p>
            </section>
          )}

          <section className="panel explain-panel" aria-labelledby="explain-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">STEP 3 · INSPECT THE MODEL</p>
                <h2 id="explain-heading">Feature explanations</h2>
                <p className="muted">
                  SHAP attributes prediction changes to features; LIME fits a
                  simple local approximation around one input. Neither demonstrates
                  physical cause-and-effect.
                </p>
              </div>
            </div>
            <div className="explanation-tabs" role="tablist" aria-label="Explanation methods">
              {([
                ["shap-global", "SHAP Global"],
                ["shap-local", "SHAP Local"],
                ["lime-local", "LIME Local"],
              ] as const).map(([tab, label]) => (
                <button
                  aria-selected={activeTab === tab}
                  className={activeTab === tab ? "explanation-tabs__active" : ""}
                  key={tab}
                  onClick={() => void selectExplanationTab(tab)}
                  role="tab"
                  type="button"
                >
                  {label}
                </button>
              ))}
            </div>
            <p className="muted">
              SHAP values describe how model features move its prediction from a
              model-specific baseline. Positive contributions raise the prediction;
              negative contributions lower it. These describe the model, not
              physical causality.
              {" "}
              <ConceptHint term="SHAP and LIME">
                SHAP shares a prediction difference across features relative to a
                chosen baseline. LIME approximates the model near one selected
                input; its local quality is reported when available.
              </ConceptHint>
            </p>
            {activeTab === "shap-global" && (
              globalShap ? (
                <>
                  <p className="muted">
                    Mean absolute SHAP values from {globalShap.sample_count} held-out
                    spatial samples. Larger magnitudes indicate more influence on
                    model output in this validation sample; signs shown are average
                    signed contributions.
                  </p>
                  <p className="message message--notice">{globalShap.baseline_definition}</p>
                  <ContributionBars items={globalShap.features.map((item) => ({
                    feature: item.feature,
                    contribution: item.mean_absolute_shap,
                    meanAbsoluteShap: item.mean_absolute_shap,
                    meanShap: item.mean_shap,
                  }))} global />
                </>
              ) : (
                <p className="muted" role="status">Loading computed SHAP global values…</p>
              )
            )}
            {activeTab !== "shap-global" && !prediction && (
              <p className="muted">Run a prediction to request a local explanation.</p>
            )}
            {activeTab !== "shap-global" && isLoadingExplanation && (
              <p className="muted" role="status">Computing local explanation…</p>
            )}
            {activeTab !== "shap-global" && activeExplanation && (
              <>
                <p className="muted">
                  Explained sample: {selectedCase.case_id} at x{" "}
                  {formatNumber(activeExplanation.sample?.x ?? prediction?.coordinates.x ?? 0)},
                  y {formatNumber(activeExplanation.sample?.y ?? prediction?.coordinates.y ?? 0)},
                  z {formatNumber(activeExplanation.sample?.z ?? prediction?.coordinates.z ?? 0)}.
                </p>
                {activeTab === "shap-local" && (
                  <>
                    <p className="message message--notice">
                      Baseline {formatNumber(activeExplanation.baseline ?? 0)}{" "}
                      {prediction?.units ?? ""}: {activeExplanation.baseline_definition}
                    </p>
                    <p className="muted">
                      SHAP reconstruction {formatNumber(activeExplanation.reconstructed_prediction ?? 0)}{" "}
                      {prediction?.units ?? ""}; absolute additivity error{" "}
                      {formatNumber(activeExplanation.additivity_error ?? 0)}.
                      Additivity check: {activeExplanation.additivity_passed ? "passed" : "failed"}.
                    </p>
                  </>
                )}
                {activeTab === "lime-local" && (
                  <p className="message message--notice">
                    LIME local R² {formatNumber(activeExplanation.local_r2 ?? 0)} ·
                    local approximation {formatNumber(activeExplanation.local_prediction ?? 0)}{" "}
                    {prediction?.units ?? ""} vs model prediction{" "}
                    {formatNumber(activeExplanation.prediction)} {prediction?.units ?? ""}.
                    {activeExplanation.caveat}
                  </p>
                )}
                <ContributionBars items={activeExplanation.contributions} />
              </>
            )}
            {explanationError && <p className="message message--error" role="alert">{explanationError}</p>}
          </section>

          <section className="panel gpt-oss-panel" aria-labelledby="gpt-oss-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">SEPARATE GENERATIVE ASSISTANT</p>
                <h2 id="gpt-oss-heading">GPT-OSS interpretation</h2>
              </div>
              <span className={`status-pill${gptOssStatus?.status === "reachable" ? "" : gptOssStatusError || gptOssStatus?.status === "unavailable" ? " status-pill--unavailable" : ""}`}>
                {gptOssStatusError
                  ? "Unavailable"
                  : gptOssStatus?.status === "reachable"
                    ? "Reachable"
                    : gptOssStatus?.status === "unavailable"
                      ? "Unavailable"
                      : gptOssStatus
                        ? "Not Verified"
                      : "Checking…"}
              </span>
            </div>
            <p className="muted">
              The backend sends the actual model prediction and computed SHAP or
              LIME values to the configured GPT-OSS model for plain-English context.
              It cannot produce the numeric prediction or explanation values.
            </p>
            <div className="gpt-oss-actions">
              <button
                className="secondary-button"
                disabled={isTestingGptOss}
                onClick={() => void runGptOssConnectionTest()}
                type="button"
              >
                {isTestingGptOss ? "Testing live connection…" : "Test GPT-OSS Connection"}
              </button>
              <button
                className="primary-button"
                disabled={!prediction || gptOssStatus?.status !== "reachable" || isInterpreting}
                onClick={() => void runGptOssInterpretation()}
                type="button"
              >
                {isInterpreting ? "Requesting interpretation…" : "Interpret computed explanation"}
              </button>
            </div>
            {gptOssConnection && (
              <div
                className={`provider-test provider-test--${gptOssConnection.status}`}
                role={gptOssConnection.status === "connected" ? "status" : "alert"}
              >
                <strong>{gptOssConnection.provider}: {gptOssConnection.status.replace("_", " ")}</strong>
                <span>Model {gptOssConnection.model ?? "not configured"}</span>
                <time dateTime={gptOssConnection.tested_at}>
                  {new Date(gptOssConnection.tested_at).toLocaleString()}
                </time>
                <span>{gptOssConnection.diagnostic}{gptOssConnection.latency_ms === undefined ? "" : ` · ${gptOssConnection.latency_ms} ms`}</span>
              </div>
            )}
            {gptOssStatusError && <p className="message message--error" role="alert">{gptOssStatusError}</p>}
            {gptOssError && <p className="message message--error" role="alert">{gptOssError}</p>}
            {gptOssStatus && !gptOssStatus.configured && (
              <p className="muted">
                {gptOssStatus.diagnostic}
                Credentials are never sent to the browser.
              </p>
            )}
            {interpretation && (
              <div className="ai-interpretation" aria-live="polite">
                <span className="ai-interpretation__label">{interpretation.interpretation.label}</span>
                <p>{interpretation.interpretation.interpretation}</p>
                <small>
                  {interpretation.interpretation.provider ?? "GPT-OSS"} · Model {interpretation.interpretation.model} ·{" "}
                  {new Date(interpretation.interpretation.generated_at).toLocaleString()}
                </small>
              </div>
            )}
          </section>
        </>
      )}
    </section>
  );
}
