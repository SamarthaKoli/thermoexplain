import { useEffect, useMemo, useState } from "react";

import { ConceptHint } from "../components/ConceptHint";
import type { useDatasetStatus } from "../hooks/useDatasetStatus";
import {
  analyzeWithGptOss,
  compareMeshes,
  getCfdFieldOptions,
  getCfdStatistics,
  preprocessCsv,
} from "../services/datasetApi";
import { getGptOssStatus, providerTestFailure, testGptOssConnection } from "../services/providerApi";
import type {
  CfdFieldOptions,
  CfdStatistics,
  DatasetCase,
  GptOssAnalysis,
  MeshComparisonResult,
} from "../types/dataset";
import type { GptOssStatus, ProviderTestResult } from "../types/provider";

interface CfdExplorerProps {
  dataset: ReturnType<typeof useDatasetStatus>;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat(undefined, { maximumSignificantDigits: 7 }).format(value);
}

function getCaseId(item: DatasetCase) {
  return item.case_id;
}

export function CfdExplorer({ dataset }: CfdExplorerProps) {
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [selectedFieldKey, setSelectedFieldKey] = useState("");
  const [selectedComponent, setSelectedComponent] = useState("");
  const [fieldOptions, setFieldOptions] = useState<CfdFieldOptions | null>(null);
  const [fieldsCaseId, setFieldsCaseId] = useState("");
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [isLoadingFields, setIsLoadingFields] = useState(false);
  const [statistics, setStatistics] = useState<CfdStatistics | null>(null);
  const [statisticsError, setStatisticsError] = useState<string | null>(null);
  const [isLoadingStatistics, setIsLoadingStatistics] = useState(false);
  const [referenceCaseId, setReferenceCaseId] = useState("");
  const [comparison, setComparison] = useState<MeshComparisonResult | null>(null);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [gptOssStatus, setGptOssStatus] = useState<GptOssStatus | null>(null);
  const [gptOssStatusError, setGptOssStatusError] = useState<string | null>(null);
  const [gptOssConnection, setGptOssConnection] =
    useState<ProviderTestResult | null>(null);
  const [gptOssError, setGptOssError] = useState<string | null>(null);
  const [isTestingGptOss, setIsTestingGptOss] = useState(false);
  const [gptOssAnalysis, setGptOssAnalysis] = useState<GptOssAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [preprocessFile, setPreprocessFile] = useState("");
  const [sampleSize, setSampleSize] = useState("100000");
  const [preprocessMessage, setPreprocessMessage] = useState<string | null>(null);
  const [isPreprocessing, setIsPreprocessing] = useState(false);

  const selectedCase = dataset.cases.find((item) => item.case_id === selectedCaseId) ?? null;
  const reynoldsNumbers = useMemo(
    () =>
      [...new Set(dataset.cases.map((item) => item.metadata.reynolds_number).filter(
        (value): value is number => value !== null,
      ))].sort((a, b) => a - b),
    [dataset.cases],
  );
  const availableMeshes = useMemo(
    () =>
      selectedCase
        ? dataset.cases.filter(
            (item) =>
              item.metadata.reynolds_number === selectedCase.metadata.reynolds_number,
          )
        : [],
    [dataset.cases, selectedCase],
  );
  const uniqueResolutions = useMemo(
    () => [...new Set(availableMeshes.map((item) => item.metadata.mesh_resolution).filter(
      (value): value is string => value !== null,
    ))].sort(),
    [availableMeshes],
  );
  const selectedOption =
    fieldsCaseId === selectedCaseId
      ? fieldOptions?.fields.find(
          (option) => `${option.region}/${option.name}` === selectedFieldKey,
        ) ?? null
      : null;
  const availableFields =
    fieldsCaseId === selectedCaseId
      ? fieldOptions?.fields.filter((option) => option.statistics_supported) ?? []
      : [];
  const availableReferences = availableMeshes.filter(
    (item) => item.case_id !== selectedCaseId &&
      item.metadata.mesh_resolution !== selectedCase?.metadata.mesh_resolution,
  );

  useEffect(() => {
    if (!selectedCaseId && dataset.cases.length > 0) {
      setSelectedCaseId(getCaseId(dataset.cases[0]));
    }
    if (selectedCaseId && !dataset.cases.some((item) => item.case_id === selectedCaseId)) {
      setSelectedCaseId(dataset.cases[0]?.case_id ?? "");
    }
  }, [dataset.cases, selectedCaseId]);

  useEffect(() => {
    setFieldOptions(null);
    setFieldsCaseId("");
    setSelectedFieldKey("");
    setSelectedComponent("");
    setStatistics(null);
    setStatisticsError(null);
    setFieldError(null);
    setComparison(null);
    setReferenceCaseId("");
    if (!selectedCaseId) return;
    let active = true;
    setIsLoadingFields(true);
    getCfdFieldOptions(selectedCaseId)
      .then((result) => {
        if (!active) return;
        setFieldOptions(result);
        setFieldsCaseId(selectedCaseId);
        const preferred = result.fields.find(
          (field) => field.name === "T" && field.region === "coolant",
        ) ?? result.fields.find((field) => field.statistics_supported);
        if (preferred) {
          setSelectedFieldKey(`${preferred.region}/${preferred.name}`);
          setSelectedComponent(preferred.components[0] ?? "value");
        }
      })
      .catch((error: Error) => {
        if (active) setFieldError(error.message);
      })
      .finally(() => {
        if (active) setIsLoadingFields(false);
      });
    return () => {
      active = false;
    };
  }, [selectedCaseId]);

  useEffect(() => {
    setStatistics(null);
    setStatisticsError(null);
    setGptOssAnalysis(null);
    if (!selectedCaseId || !selectedOption || !selectedComponent) return;
    let active = true;
    setIsLoadingStatistics(true);
    getCfdStatistics(
      selectedCaseId,
      selectedOption.region,
      selectedOption.name,
      selectedComponent,
    )
      .then((result) => {
        if (active) setStatistics(result);
      })
      .catch((error: Error) => {
        if (active) setStatisticsError(error.message);
      })
      .finally(() => {
        if (active) setIsLoadingStatistics(false);
      });
    return () => {
      active = false;
    };
  }, [selectedCaseId, selectedOption, selectedComponent]);

  useEffect(() => {
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

  function selectReynolds(value: string) {
    const candidate = dataset.cases.find(
      (item) => String(item.metadata.reynolds_number ?? "") === value,
    );
    if (candidate) setSelectedCaseId(candidate.case_id);
  }

  function selectMesh(value: string) {
    const candidate = availableMeshes.find(
      (item) => item.metadata.mesh_resolution === value,
    );
    if (candidate) setSelectedCaseId(candidate.case_id);
  }

  function selectField(value: string) {
    const option = fieldOptions?.fields.find(
      (field) => `${field.region}/${field.name}` === value,
    );
    setSelectedFieldKey(value);
    setSelectedComponent(option?.components[0] ?? "");
  }

  async function runMeshComparison() {
    setComparison(null);
    setComparisonError(null);
    setIsComparing(true);
    try {
      const result = await compareMeshes(selectedCaseId, referenceCaseId);
      setComparison(result);
    } catch (error) {
      setComparisonError(error instanceof Error ? error.message : "Mesh comparison failed.");
    } finally {
      setIsComparing(false);
    }
  }

  async function runGptOssConnectionTest() {
    setGptOssConnection(null);
    setGptOssError(null);
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
    } finally {
      setIsTestingGptOss(false);
    }
  }

  async function runGptOssAnalysis() {
    if (!selectedCase || !selectedOption) return;
    setGptOssAnalysis(null);
    setGptOssError(null);
    setIsAnalyzing(true);
    try {
      const result = await analyzeWithGptOss({
        case_id: selectedCase.case_id,
        region: selectedOption.region,
        field: selectedOption.name,
        component: selectedComponent,
      });
      setGptOssAnalysis(result);
    } catch (error) {
      setGptOssError(error instanceof Error ? error.message : "GPT-OSS analysis failed.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function runPreprocessing() {
    setPreprocessMessage(null);
    setIsPreprocessing(true);
    try {
      const parsedSize = sampleSize.trim() ? Number(sampleSize) : undefined;
      if (parsedSize !== undefined && (!Number.isInteger(parsedSize) || parsedSize < 1)) {
        throw new Error("Sample size must be a positive whole number.");
      }
      const result = await preprocessCsv(preprocessFile, parsedSize);
      setPreprocessMessage(
        `Processed ${result.output_rows} of ${result.source_rows} rows. No train/test split was created.`,
      );
      await dataset.refresh();
    } catch (error) {
      setPreprocessMessage(error instanceof Error ? error.message : "Preprocessing failed.");
    } finally {
      setIsPreprocessing(false);
    }
  }

  return (
    <section className="explorer">
      <header className="page-heading">
        <div>
          <p className="eyebrow">CFD EXPLORER &amp; DESIGN ANALYSIS</p>
          <h1>Explore the simulated flow.</h1>
          <p className="intro__copy">
            Computational fluid dynamics (CFD) uses numerical methods to estimate how
            fluids move and transfer heat through a designed space. Explore the
            dataset’s measured solver fields; this page does not run a new simulation.
          </p>
        </div>
        <button
          className="primary-button"
          disabled={dataset.isScanning}
          onClick={() => void dataset.runScan()}
          type="button"
        >
          {dataset.isScanning ? "Scanning…" : "Refresh dataset"}
        </button>
      </header>

      {dataset.error && <p className="message message--error" role="alert">{dataset.error}</p>}
      {dataset.isLoading && <p className="muted">Loading the scanned dataset…</p>}
      {dataset.scan?.warnings.map((warning) => (
        <p className="message message--notice" key={warning}>{warning}</p>
      ))}

      {dataset.isLoading && dataset.cases.length === 0 ? (
        <section className="panel empty-state" aria-label="Loading CFD cases">
          <p className="loading-state" role="status">Loading discovered CFD cases…</p>
        </section>
      ) : dataset.cases.length === 0 ? (
        <section className="panel empty-state" aria-label="No CFD cases available">
          <span className="empty-state__icon" aria-hidden="true">i</span>
          <h2>No scanned CFD cases are available</h2>
          <p>
            Add or configure the real dataset, then refresh the scan. No sample case
            or simulated result is substituted here.
          </p>
        </section>
      ) : (
        <>
          <section className="panel selection-panel" aria-labelledby="selection-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">GUIDED CASE SELECTION</p>
                <h2 id="selection-heading">Choose the run to inspect</h2>
              </div>
              {fieldOptions && (
                <span className="status-pill">
                  Solution time {fieldOptions.time} · {fieldOptions.processor_count} partitions
                </span>
              )}
            </div>
            <div className="guided-selectors">
              <label className="selector-field">
                <span>
                  <b>1</b> Reynolds number
                  <ConceptHint term="Reynolds number">
                    A dimensionless value that describes the balance between
                    momentum and viscosity in a flow. Choose only values recorded
                    in the scanned CFD cases.
                  </ConceptHint>
                </span>
                <select
                  aria-label="Reynolds number"
                  onChange={(event) => selectReynolds(event.target.value)}
                  value={String(selectedCase?.metadata.reynolds_number ?? "")}
                >
                  <option value="" disabled>Select a scanned value</option>
                  {reynoldsNumbers.map((value) => (
                    <option key={value} value={value}>Re {value}</option>
                  ))}
                </select>
                <small>Choose a flow condition present in the dataset.</small>
              </label>
              <label className="selector-field">
                <span>
                  <b>2</b> Mesh resolution
                  <ConceptHint term="Mesh resolution">
                    The mesh divides the simulated region into computational
                    cells. A finer mesh has more cells, but does not automatically
                    mean a more accurate result.
                  </ConceptHint>
                </span>
                <select
                  aria-label="Mesh resolution"
                  disabled={!selectedCase}
                  onChange={(event) => selectMesh(event.target.value)}
                  value={selectedCase?.metadata.mesh_resolution ?? ""}
                >
                  <option value="" disabled>Select an available mesh</option>
                  {uniqueResolutions.map((value) => (
                    <option key={value} value={value}>{value}</option>
                  ))}
                </select>
                <small>Coarser and finer meshes change spatial detail.</small>
              </label>
              <label className="selector-field">
                <span><b>3</b> Case</span>
                <select
                  aria-label="CFD case"
                  onChange={(event) => setSelectedCaseId(event.target.value)}
                  value={selectedCaseId}
                >
                  {dataset.cases.map((item) => (
                    <option key={item.case_id} value={item.case_id}>{item.case_id}</option>
                  ))}
                </select>
                <small>Only cases present in the latest manifest are shown.</small>
              </label>
              <label className="selector-field">
                <span><b>4</b> Region and field</span>
                <select
                  aria-label="CFD field"
                  disabled={isLoadingFields || availableFields.length === 0}
                  onChange={(event) => selectField(event.target.value)}
                  value={selectedFieldKey}
                >
                  <option value="" disabled>
                    {isLoadingFields ? "Reading field headers…" : "Select an available field"}
                  </option>
                  {availableFields.map((field) => (
                    <option
                      key={`${field.region}/${field.name}`}
                      value={`${field.region}/${field.name}`}
                    >
                      {field.region} / {field.name} · {field.field_class}
                    </option>
                  ))}
                </select>
                <small>Fields are read from the case’s latest common solver time.</small>
              </label>
            </div>
            {selectedOption && selectedOption.components.length > 1 && (
              <label className="selector-field selector-field--component">
                <span>Vector / tensor component</span>
                <select
                  aria-label="Field component"
                  onChange={(event) => setSelectedComponent(event.target.value)}
                  value={selectedComponent}
                >
                  {selectedOption.components.map((component) => (
                    <option key={component} value={component}>
                      {component === "magnitude" &&
                      (selectedOption.value_type.includes("Tensor") ||
                        selectedOption.value_type === "tensor")
                        ? "Frobenius magnitude"
                        : component}
                    </option>
                  ))}
                </select>
                <small>Vector magnitude is computed from the three stored components.</small>
              </label>
            )}
            {fieldError && <p className="message message--error" role="alert">{fieldError}</p>}
          </section>

          <div className="analysis-grid">
            <section className="panel visualization-panel" aria-labelledby="visual-heading">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">SCIENTIFIC VIEW</p>
                  <h2 id="visual-heading">Spatial field visualization</h2>
                </div>
                <span className="status-pill status-pill--unavailable">Unavailable</span>
              </div>
              {fieldOptions?.visualization.reason ? (
                <div className="visualization-unavailable" role="status">
                  <div className="visualization-unavailable__symbol" aria-hidden="true">∿</div>
                  <h3>Visualization unavailable</h3>
                  <p>{fieldOptions.visualization.reason}</p>
                  <p className="muted">
                    The available <code>layerInfo.vtk</code> files describe a surface
                    layer export, not a volume mesh carrying the selected physical field.
                    We do not draw an unrelated or interpolated temperature map.
                  </p>
                </div>
              ) : (
                <div className="visualization-unavailable" role="status">
                  <h3>{isLoadingFields ? "Checking supported data…" : "No renderable field data"}</h3>
                  <p>{isLoadingFields ? "Reading real case metadata." : "Select a case with verified field data."}</p>
                </div>
              )}
              <div className="legend-unavailable">
                <span>Field: {selectedOption?.name ?? "—"}</span>
                <span>Units: {selectedOption?.units ?? "not stated"}</span>
                <span>Colour scale unavailable until a spatial field is supported</span>
              </div>
            </section>

            <section className="panel statistics-panel" aria-labelledby="statistics-heading">
              <p className="eyebrow">COMPUTED FROM SOLVER VALUES</p>
              <h2 id="statistics-heading">Field statistics</h2>
              {!selectedCase ? (
                <p className="muted">Select a scanned case to calculate its field statistics.</p>
              ) : isLoadingStatistics ? (
                <p className="loading-state" role="status">Reading partition values and calculating statistics…</p>
              ) : statisticsError ? (
                <p className="message message--error" role="alert">{statisticsError}</p>
              ) : statistics ? (
                <>
                  <div className="statistic-grid">
                    <div><span>Minimum</span><strong>{formatNumber(statistics.minimum)}</strong></div>
                    <div><span>Maximum</span><strong>{formatNumber(statistics.maximum)}</strong></div>
                    <div><span>Mean</span><strong>{formatNumber(statistics.mean)}</strong></div>
                  </div>
                  <p className="statistic-units">
                    {statistics.units ?? "Units not stated"}
                    {statistics.component !== "value" ? ` · ${statistics.component}` : ""}
                  </p>
                  <dl className="metadata-list">
                    <div><dt>Case</dt><dd>{statistics.case_id}</dd></div>
                    <div><dt>Region / field</dt><dd>{statistics.region} / {statistics.field}</dd></div>
                    <div><dt>Field type</dt><dd>{statistics.field_class}</dd></div>
                    <div><dt>Solution time</dt><dd>{statistics.time}</dd></div>
                    <div><dt>Values included</dt><dd>{statistics.sample_count?.toLocaleString() ?? "Uniform definition"}</dd></div>
                  </dl>
                  <p className="muted statistic-scope">{statistics.statistic_scope}.</p>
                </>
              ) : (
                <p className="muted">
                  {isLoadingFields ? "Available solver fields are loading." : "Choose an available field to calculate statistics."}
                </p>
              )}
            </section>
          </div>

          <section className="panel comparison-panel" aria-labelledby="comparison-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">DESIGN ANALYSIS</p>
                <h2 id="comparison-heading">Compare mesh resolutions</h2>
                <p className="muted">
                  Mesh refinement can change local values and numerical detail. A fair
                  comparison requires spatial interpolation between the two geometries,
                  not matching cell arrays by index.
                </p>
              </div>
            </div>
            <div className="comparison-controls">
              <label className="selector-field">
                <span>Reference mesh · same Reynolds number</span>
                <select
                  aria-label="Reference mesh"
                  disabled={availableReferences.length === 0}
                  onChange={(event) => {
                    setReferenceCaseId(event.target.value);
                    setComparison(null);
                    setComparisonError(null);
                  }}
                  value={referenceCaseId}
                >
                  <option value="">Choose another available resolution</option>
                  {availableReferences.map((item) => (
                    <option key={item.case_id} value={item.case_id}>
                      {item.metadata.mesh_resolution} · {item.case_id}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="secondary-button"
                disabled={!referenceCaseId || isComparing}
                onClick={() => void runMeshComparison()}
                type="button"
              >
                {isComparing ? "Checking comparison…" : "Check comparison readiness"}
              </button>
            </div>
            {comparisonError && <p className="message message--error" role="alert">{comparisonError}</p>}
            {comparison && (
              <p className="message message--notice" role="status">{comparison.reason}</p>
            )}
          </section>

          <section className="panel reading-card reading-card--wide" aria-labelledby="reading-heading">
            <p className="eyebrow">QUICK GUIDE</p>
            <h2 id="reading-heading">How to read this view</h2>
            <ol>
              <li>Choose a Reynolds number and one of its available mesh resolutions.</li>
              <li>Select a region and field; scalar fields show values, vectors offer components or magnitude.</li>
              <li>Statistics summarize stored internal values at the reported solution time, with units only when the source dimensions establish them.</li>
              <li>A colour map needs field values spatially attached to mesh geometry. It remains unavailable until that association can be verified.</li>
            </ol>
          </section>
        </>
      )}

      <section className="panel gpt-oss-panel gpt-oss-panel--wide" aria-labelledby="gpt-oss-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">SEPARATE GENERATIVE ASSISTANT</p>
            <h2 id="gpt-oss-heading">GPT-OSS CFD interpretation</h2>
          </div>
          <span className={`status-pill${gptOssStatus?.configured ? "" : gptOssStatus || gptOssStatusError ? " status-pill--unavailable" : ""}`}>
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
          GptOss receives the selected case metadata and backend-computed field
          statistics. Its response is an AI interpretation, not a simulation or
          validated engineering conclusion.
        </p>
        <div className="gpt-oss-actions">
          <button
            className="secondary-button"
            disabled={isTestingGptOss}
            onClick={() => void runGptOssConnectionTest()}
            type="button"
          >
            {isTestingGptOss ? "Testing inference…" : "Test GPT-OSS Inference"}
          </button>
          <button
            className="primary-button"
            disabled={!statistics || isAnalyzing || gptOssStatus?.status !== "reachable"}
            onClick={() => void runGptOssAnalysis()}
            type="button"
          >
            {isAnalyzing ? "Requesting analysis…" : "Interpret selected statistics"}
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
            <span>{gptOssConnection.diagnostic}</span>
          </div>
        )}
        {gptOssError && <p className="message message--error" role="alert">{gptOssError}</p>}
        {gptOssStatusError && <p className="message message--error" role="alert">{gptOssStatusError}</p>}
        {gptOssStatus && (
          <p className="muted">
            {gptOssStatus.diagnostic}
            {" "}Credentials are never sent to the browser.
          </p>
        )}
        {gptOssAnalysis && (
          <div className="ai-interpretation" aria-live="polite">
            <span className="ai-interpretation__label">{gptOssAnalysis.interpretation.label}</span>
            <p>{gptOssAnalysis.interpretation.interpretation}</p>
            <small>
              {gptOssAnalysis.interpretation.provider ?? "GPT-OSS"} · Model {gptOssAnalysis.interpretation.model} ·{" "}
              {new Date(gptOssAnalysis.interpretation.generated_at).toLocaleString()}
            </small>
          </div>
        )}
      </section>

      <section className="panel preprocessing-panel" aria-labelledby="preprocessing-heading">
        <div className="section-heading">
          <div>
            <h2 id="preprocessing-heading">CSV point-table preprocessing</h2>
            <p className="muted">
              This separate utility is only for scanned CSV files with explicit x, y,
              and z columns. OpenFOAM mesh fields are not flattened into a table.
            </p>
          </div>
          <span className="status-pill">{dataset.preprocessing?.status ?? "not run"}</span>
        </div>
        <details className="advanced-controls">
          <summary>Advanced: preprocess a supported CSV point table</summary>
          <p className="muted">
            Optional utility. It streams the selected CSV and writes a separate
            derived file; it does not convert OpenFOAM meshes or modify source data.
          </p>
          <div className="preprocess-form">
            <label className="selector-field">
              <span>Scanned CSV relative path</span>
              <input
                aria-label="Scanned CSV relative path"
                onChange={(event) => setPreprocessFile(event.target.value)}
                placeholder="case/points.csv"
                value={preprocessFile}
              />
            </label>
            <label className="selector-field selector-field--sample">
              <span>Maximum sample rows</span>
              <input
                aria-label="Maximum sample rows"
                min="1"
                onChange={(event) => setSampleSize(event.target.value)}
                type="number"
                value={sampleSize}
              />
            </label>
            <button
              className="primary-button"
              disabled={!preprocessFile || isPreprocessing}
              onClick={() => void runPreprocessing()}
              type="button"
            >
              {isPreprocessing ? "Preprocessing…" : "Preprocess"}
            </button>
          </div>
          {preprocessMessage && <p className="message message--notice" role="status">{preprocessMessage}</p>}
        </details>
      </section>

      <details
        className="panel audit-panel"
      >
        <summary id="audit-heading" onClick={() => void dataset.loadAudit()}>
          Data audit report
        </summary>
        {dataset.isLoadingAudit && <p className="muted" role="status">Loading audit report…</p>}
        {dataset.auditError && <p className="message message--error" role="alert">{dataset.auditError}</p>}
        {dataset.audit?.report ? (
          <pre>{dataset.audit.report}</pre>
        ) : !dataset.isLoadingAudit && !dataset.auditError ? (
          <p className="muted">Audit report is unavailable until a scan has been run.</p>
        ) : null}
      </details>
    </section>
  );
}
