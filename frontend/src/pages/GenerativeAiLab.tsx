import { useEffect, useMemo, useState, type ReactNode } from "react";

import { ConceptHint } from "../components/ConceptHint";
import {
  getGptOssStatus,
  interpretFeatureEvidence,
} from "../services/providerApi";
import {
  generatePatch,
  getGenerativeSamples,
  getGenerativeStatus,
  reconstructPatch,
  trainGenerativeModel,
} from "../services/generativeApi";
import type {
  FeatureInterpretation,
  GptOssStatus,
} from "../types/provider";
import type {
  GenerativeModel,
  GenerativeModelType,
  GenerativeSample,
  GenerativeStatus,
  PatchResult,
} from "../types/generative";

type AsyncState = "idle" | "working";

function formatMetric(value: unknown, digits = 4) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Unavailable";
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function plausibilityLabel(value: unknown) {
  return typeof value === "boolean" ? (value ? "Passed" : "Failed") : "Unavailable";
}

function patchSlice(
  values: Array<Array<Array<number | null>>>,
  zIndex: number,
) {
  return values.map((xValues) => xValues.map((yValues) => yValues[zIndex]));
}

function Heatmap({
  label,
  values,
  difference = false,
  range,
}: {
  label: string;
  values: Array<Array<number | null>>;
  difference?: boolean;
  range?: [number, number];
}) {
  const finiteValues = values.flat().filter((value): value is number => value !== null);
  if (finiteValues.length === 0) {
    return (
      <div className="generative-heatmap">
        <h4>{label}</h4>
        <p className="muted">No valid coolant voxels in this slice.</p>
      </div>
    );
  }
  const extent = difference
    ? Math.max(1e-9, ...finiteValues.map(Math.abs))
    : Math.max(1e-9, ...finiteValues) - Math.min(...finiteValues);
  const minimum = range?.[0] ?? (difference ? -extent : Math.min(...finiteValues));
  const maximum = range?.[1] ?? (difference ? extent : Math.max(...finiteValues));

  return (
    <div className="generative-heatmap">
      <h4>{label}</h4>
      <div
        className="generative-heatmap__grid"
        role="img"
        aria-label={`${label}, middle z slice`}
        style={{ gridTemplateColumns: `repeat(${values[0]?.length ?? 1}, 1fr)` }}
      >
        {values.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => {
            const fraction =
              value === null ? 0 : Math.max(0, Math.min(1, (value - minimum) / (maximum - minimum || 1)));
            const background = value === null
              ? "#e5e9e8"
              : difference
                ? fraction < 0.5
                  ? `rgb(${Math.round(40 + fraction * 100)}, ${Math.round(118 + fraction * 100)}, 154)`
                  : `rgb(${Math.round(140 + (fraction - 0.5) * 180)}, ${Math.round(218 - (fraction - 0.5) * 250)}, ${Math.round(154 - (fraction - 0.5) * 220)})`
                : `hsl(${205 - fraction * 190} 72% ${70 - fraction * 23}%)`;
            return (
              <span
                key={`${rowIndex}-${columnIndex}`}
                title={value === null ? "Outside the coolant region" : `${formatMetric(value, 5)} ${difference ? "K difference" : "K"}`}
                style={{ backgroundColor: background }}
              />
            );
          }),
        )}
      </div>
      <p className="generative-heatmap__range">
        {difference ? `${formatMetric(minimum)} to ${formatMetric(maximum)} K` : `${formatMetric(Math.min(...finiteValues))}–${formatMetric(Math.max(...finiteValues))} K`}
      </p>
    </div>
  );
}

function ResultsView({
  modelType,
  result,
}: {
  modelType: GenerativeModelType;
  result: PatchResult;
}) {
  if (modelType === "vae" && !Array.isArray(result.values_K)) {
    const volumes = result.values_K;
    if (!("original" in volumes) || !volumes.original || !volumes.reconstructed || !volumes.difference) {
      return <p className="message message--error">The VAE response did not include all reconstruction fields.</p>;
    }
    const originalSlice = patchSlice(volumes.original, 4);
    const reconstructedSlice = patchSlice(volumes.reconstructed, 4);
    const comparableValues = [...originalSlice.flat(), ...reconstructedSlice.flat()]
      .filter((value): value is number => value !== null);
    const sharedRange: [number, number] | undefined =
      comparableValues.length > 0
        ? [Math.min(...comparableValues), Math.max(...comparableValues)]
        : undefined;
    return (
      <>
        <h3>{result.label}</h3>
        <p className="muted">
          {result.sample.case_id} · Re {result.sample.reynolds_number} · {result.sample.mesh_resolution} ·
          solution time {result.sample.solution_time} · patch center{" "}
          {result.sample.patch_center.map((coordinate) => formatMetric(coordinate, 5)).join(", ")}{" "}
          (coordinate units unknown)
        </p>
        <p className="muted">Middle z slice of the selected held-out patch; masked gray cells are outside the coolant mesh.</p>
        <div className="generative-heatmaps">
          <Heatmap label="Original CFD field" values={originalSlice} range={sharedRange} />
          <Heatmap label="VAE reconstruction" values={reconstructedSlice} range={sharedRange} />
          <Heatmap label="Reconstruction difference" values={patchSlice(volumes.difference, 4)} difference />
        </div>
        <div className="generative-metrics">
          <Metric label="Patch MAE" value={result.metrics.mae_K} units="K" />
          <Metric label="Patch RMSE" value={result.metrics.rmse_K} units="K" />
          <Metric label="Positive-K check" value={plausibilityLabel(result.metrics.positivity_passed)} />
        </div>
      </>
    );
  }
  if (modelType === "gan" && Array.isArray(result.values_K)) {
    return (
      <>
        <h3>{result.label}</h3>
        <p className="muted">
          {result.sample.case_id} · Re {result.sample.reynolds_number} · {result.sample.mesh_resolution} ·
          solution time {result.sample.solution_time} · patch {result.sample.sample_id} · seed {result.generation_seed}
        </p>
        <p className="muted">
          Synthetic output conditioned on the selected real patch's coolant mask. It is not a CFD solution.
        </p>
        <div className="generative-heatmaps generative-heatmaps--single">
          <Heatmap label="Synthetic temperature patch" values={patchSlice(result.values_K, 4)} />
        </div>
        <div className="generative-metrics">
          <Metric label="Wasserstein distance to held-out T" value={result.metrics.temperature_wasserstein_distance_K} units="K" />
          <Metric label="Held-out mean T" value={result.metrics.heldout_mean_temperature_K} units="K" />
          <Metric label="Synthetic mean T" value={result.metrics.generated_mean_temperature_K} units="K" />
          <Metric label="Held-out standard deviation" value={result.metrics.heldout_temperature_std_K} units="K" />
          <Metric label="Synthetic standard deviation" value={result.metrics.generated_temperature_std_K} units="K" />
          <Metric label="Held-out neighbor jump" value={result.metrics.heldout_mean_neighbor_jump_K_per_voxel} units="K/voxel" />
          <Metric label="Synthetic neighbor jump" value={result.metrics.generated_mean_neighbor_jump_K_per_voxel} units="K/voxel" />
          <Metric label="Neighbor-jump difference" value={result.metrics.mean_neighbor_jump_difference_K_per_voxel} units="K/voxel" />
          <Metric label="Positive-K check" value={plausibilityLabel(result.metrics.positivity_passed)} />
        </div>
        <p className="foundation-note">
          Similarity statistics compare generated values with held-out real voxels. They do not establish physical
          validity, convergence, a new simulation, or a safe cooling design.
        </p>
      </>
    );
  }
  return <p className="message message--error">The backend returned a result incompatible with this experiment.</p>;
}

function Metric({ label, value, units }: { label: string; value: unknown; units?: string }) {
  const rendered = typeof value === "string" ? value : formatMetric(value);
  return (
    <div className="generative-metric">
      <span>{label}</span>
      <strong>{rendered}{units && typeof value === "number" ? ` ${units}` : ""}</strong>
    </div>
  );
}

interface DisclosureProps {
  children: ReactNode;
  id: string;
  title: string;
}

function Disclosure({ children, id, title }: DisclosureProps) {
  const [open, setOpen] = useState(false);

  return (
    <section className="gen-disclosure">
      <h3>
        <button
          aria-controls={id}
          aria-expanded={open}
          className="gen-disclosure__toggle"
          onClick={() => setOpen((value) => !value)}
          type="button"
        >
          <span>{title}</span>
          <span aria-hidden="true" className="gen-disclosure__chevron">⌄</span>
        </button>
      </h3>
      <div className="gen-disclosure__body" hidden={!open} id={id}>
        {children}
      </div>
    </section>
  );
}

function metricValue(metrics: Record<string, unknown>, name: string) {
  return metrics[name];
}

function maxAbsolute(values: Array<Array<Array<number | null>>>) {
  const numbers = values.flat(2).filter((value): value is number => value !== null && Number.isFinite(value));
  return numbers.length ? Math.max(...numbers.map(Math.abs)) : null;
}

function peakDifference(
  original: Array<Array<Array<number | null>>>,
  reconstructed: Array<Array<Array<number | null>>>,
) {
  const originalValues = original.flat(2).filter((value): value is number => value !== null && Number.isFinite(value));
  const reconstructedValues = reconstructed.flat(2).filter((value): value is number => value !== null && Number.isFinite(value));
  if (!originalValues.length || !reconstructedValues.length) return null;
  return Math.abs(Math.max(...originalValues) - Math.max(...reconstructedValues));
}

function CompactResults({
  modelType,
  result,
  model,
}: {
  modelType: GenerativeModelType;
  result: PatchResult;
  model: GenerativeModel | null;
}) {
  const sampleSummary = `${result.sample.case_id} · Re ${result.sample.reynolds_number} · ${result.sample.mesh_resolution} · time ${result.sample.solution_time}`;
  const values = result.values_K;

  if (modelType === "vae" && !Array.isArray(values) && values.original && values.reconstructed && values.difference) {
    const originalSlice = patchSlice(values.original, 4);
    const reconstructionSlice = patchSlice(values.reconstructed, 4);
    const comparison = [...originalSlice.flat(), ...reconstructionSlice.flat()]
      .filter((value): value is number => value !== null);
    const range: [number, number] | undefined = comparison.length
      ? [Math.min(...comparison), Math.max(...comparison)]
      : undefined;
    const maxError = maxAbsolute(values.difference);
    const peakDelta = peakDifference(values.original, values.reconstructed);
    const absoluteDifference = values.difference.map((row) =>
      row.map((column) => column.map((value) => value === null ? null : Math.abs(value))),
    );

    return (
      <section className="panel gen-results" aria-labelledby="gen-result-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">RECONSTRUCTION RESULT</p>
            <h2 id="gen-result-title">Original, reconstructed and residual fields</h2>
          </div>
          <span className="status-pill">Exploratory</span>
        </div>
        <p className="muted gen-result-context">{sampleSummary} · Middle z slice · gray cells are outside the coolant mesh.</p>
        <div className="generative-heatmaps gen-results__vae">
          <Heatmap label="Original CFD field" values={originalSlice} range={range} />
          <Heatmap label="VAE reconstruction" values={reconstructionSlice} range={range} />
          <Heatmap label="Absolute residual / error" values={patchSlice(absoluteDifference, 4)} difference range={[0, maxError ?? 0]} />
        </div>
        <div className="generative-metrics gen-results__summary">
          <Metric label="Reconstruction MAE" value={metricValue(result.metrics, "mae_K")} units="K" />
          <Metric label="Maximum absolute error" value={maxError} units="K" />
          <Metric label="Peak temperature difference" value={peakDelta} units="K" />
        </div>
        <Disclosure id="gen-vae-result-details" title="Reconstruction metrics">
          <div className="generative-metrics">
            <Metric label="RMSE" value={metricValue(result.metrics, "rmse_K")} units="K" />
            <Metric label="Positive-K check" value={plausibilityLabel(metricValue(result.metrics, "positivity_passed"))} />
            <Metric label="Positive Kelvin fraction" value={metricValue(result.metrics, "positive_kelvin_fraction")} />
            <Metric label="Point count" value={result.sample.valid_voxel_count} />
            <Metric label="Patch dimensions" value={model?.configuration.patch_shape?.join(" × ") ?? "Unavailable"} />
            <Metric label="Grid resolution" value={model?.representation.grid_size?.join(" × ") ?? "Unavailable"} />
          </div>
          <p className="muted">Peak temperature difference is the absolute difference between the maximum temperatures in the original and reconstructed valid voxels. Residual values are calculated from the actual returned fields.</p>
          <p className="muted">{sampleSummary} · Patch center: {result.sample.patch_center.map((coordinate) => formatMetric(coordinate, 5)).join(", ")} {result.sample.patch_center_units ?? "(source units not declared)"}</p>
        </Disclosure>
      </section>
    );
  }

  if (modelType === "gan" && Array.isArray(values)) {
    const checks = result.metrics.physical_plausibility;
    const plausibility = typeof checks === "object" && checks !== null
      ? checks as Record<string, unknown>
      : null;
    return (
      <section className="panel gen-results" aria-labelledby="gen-result-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">SYNTHETIC SAMPLE</p>
            <h2 id="gen-result-title">Generated thermal field</h2>
          </div>
          <span className="status-pill status-pill--unavailable">Not a CFD solution</span>
        </div>
        <p className="muted gen-result-context">{sampleSummary} · seed {result.generation_seed ?? "not returned"} · conditioned on a real held-out coolant mask.</p>
        <div className="gen-results__gan">
          <div className="gen-reference-unavailable">
            <h3>Real CFD reference field</h3>
            <p>The generation response includes the real sample’s mask and aggregate comparison metrics, but not its temperature field. No reference visualization is available from this endpoint.</p>
          </div>
          <div className="generative-heatmaps generative-heatmaps--single">
            <Heatmap label="GAN-generated field" values={patchSlice(values, 4)} />
          </div>
        </div>
        <div className="generative-metrics gen-results__summary">
          <Metric label="Generated mean temperature" value={metricValue(result.metrics, "generated_mean_temperature_K")} units="K" />
          <Metric label="Generated temperature spread" value={metricValue(result.metrics, "generated_temperature_std_K")} units="K" />
          <Metric label="Positive-K check" value={plausibilityLabel(metricValue(result.metrics, "positivity_passed"))} />
        </div>
        <Disclosure id="gen-gan-result-details" title="Statistical and physics checks">
          <p className="muted">These are real aggregate comparisons between generated and held-out voxels; they do not establish physical validity.</p>
          <div className="generative-metrics">
            <Metric label="Temperature Wasserstein distance" value={metricValue(result.metrics, "temperature_wasserstein_distance_K")} units="K" />
            <Metric label="Held-out mean temperature" value={metricValue(result.metrics, "heldout_mean_temperature_K")} units="K" />
            <Metric label="Held-out temperature spread" value={metricValue(result.metrics, "heldout_temperature_std_K")} units="K" />
            <Metric label="Generated neighbor jump" value={metricValue(result.metrics, "generated_mean_neighbor_jump_K_per_voxel")} units="K/voxel" />
            <Metric label="Held-out neighbor jump" value={metricValue(result.metrics, "heldout_mean_neighbor_jump_K_per_voxel")} units="K/voxel" />
            <Metric label="Neighbor-jump difference" value={metricValue(result.metrics, "mean_neighbor_jump_difference_K_per_voxel")} units="K/voxel" />
            <Metric label="Valid generated voxels" value={metricValue(result.metrics, "generated_valid_voxel_count")} />
            <Metric label="Valid held-out voxels" value={metricValue(result.metrics, "heldout_real_voxel_count")} />
            <Metric label="Positive Kelvin fraction" value={metricValue(result.metrics, "positive_kelvin_fraction")} />
          </div>
          <h3>Checks</h3>
          <ul className="gen-check-list">
            <li>Finite generated values: <strong>{plausibility ? plausibilityLabel(plausibility.all_valid_values_finite) : "Not reported"}</strong></li>
            <li>Absolute temperature above 0 K: <strong>{plausibilityLabel(metricValue(result.metrics, "positivity_passed"))}</strong></li>
            <li>Spatial neighbor-jump comparison: <strong>{metricValue(result.metrics, "mean_neighbor_jump_difference_K_per_voxel") == null ? "Not available" : "Measured; not a physics pass/fail test"}</strong></li>
            <li>Mass conservation: <strong>Not evaluated</strong></li>
            <li>Energy conservation: <strong>Not evaluated</strong></li>
            <li>Spatial gradient validity: <strong>Not evaluated</strong></li>
            <li>Temperature histogram: <strong>Not returned by the API</strong></li>
          </ul>
          <p className="muted">{typeof result.metrics.plausibility_caveat === "string" ? result.metrics.plausibility_caveat : "Statistical resemblance and positivity do not establish physical validity or CFD convergence."}</p>
        </Disclosure>
        {model && (
          <Disclosure id="gen-gan-sampling-details" title="Advanced sampling details">
            <ModelDetails model={model} />
            <p className="muted">The existing generation API accepts a reproducible random seed; it does not provide a truncation parameter.</p>
          </Disclosure>
        )}
      </section>
    );
  }

  return <p className="message message--error" role="alert">The backend returned a result incompatible with this experiment.</p>;
}

function ModelDetails({ model }: { model: GenerativeModel }) {
  return (
    <>
      <dl className="metadata-list">
        <div><dt>Model ID</dt><dd>{model.model_id}</dd></div>
        <div><dt>Architecture dimensions</dt><dd>Hidden {model.configuration.hidden_dimension} · latent {model.configuration.latent_dimension}</dd></div>
        <div><dt>Patch / grid</dt><dd>{model.configuration.patch_shape.join(" × ")} / {model.representation.grid_size.join(" × ")}</dd></div>
        <div><dt>Training</dt><dd>{model.configuration.epochs} epochs · seed {model.configuration.seed} · {model.configuration.device}</dd></div>
        <div><dt>Normalization</dt><dd>{model.configuration.normalization.method} · mean {formatMetric(model.configuration.normalization.mean_K)} K · standard deviation {formatMetric(model.configuration.normalization.standard_deviation_K)} K</dd></div>
        <div><dt>Spatial split</dt><dd>{model.split.method} · {model.split.patch_counts.train} train / {model.split.patch_counts.validation} validation / {model.split.patch_counts.buffer} buffer patches</dd></div>
        <div><dt>Representation</dt><dd>{model.representation.method}; interpolation: {model.representation.interpolation}; mask: {model.representation.mask}</dd></div>
        {model.configuration.vae_beta !== null && <div><dt>VAE beta</dt><dd>{model.configuration.vae_beta}</dd></div>}
      </dl>
      <p className="muted">Final recorded training values:</p>
      <pre>{JSON.stringify(model.training_history_summary.final, null, 2)}</pre>
      {model.limitations.length > 0 && <ul className="gen-check-list">{model.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>}
    </>
  );
}

interface CompactGenerativeViewProps {
  actionState: AsyncState;
  caseId: string;
  error: string | null;
  models: GenerativeModel[];
  modelType: GenerativeModelType;
  result: PatchResult | null;
  sampleId: string;
  samples: GenerativeSample[];
  samplesLoading: boolean;
  seed: string;
  epochs: string;
  selectedCase: GenerativeStatus["cases"][number] | null;
  selectedModel: GenerativeModel | null;
  status: GenerativeStatus | null;
  statusError: string | null;
  statusLoading: boolean;
  trainingState: AsyncState;
  busy: boolean;
  setCaseId: (value: string) => void;
  setModelId: (value: string) => void;
  setModelType: (value: GenerativeModelType) => void;
  setSampleId: (value: string) => void;
  setSeed: (value: string) => void;
  setEpochs: (value: string) => void;
  runAction: () => Promise<void>;
  interpretResult: () => Promise<void>;
  gptOssStatus: GptOssStatus | null;
  gptOssStatusError: string | null;
  gptOssInterpretation: FeatureInterpretation | null;
  gptOssInterpretationError: string | null;
  isInterpreting: boolean;
  train: () => Promise<void>;
}

function CompactGenerativeView({
  actionState,
  caseId,
  error,
  models,
  modelType,
  result,
  sampleId,
  samples,
  samplesLoading,
  seed,
  epochs,
  selectedCase,
  selectedModel,
  status,
  statusError,
  statusLoading,
  trainingState,
  busy,
  setCaseId,
  setModelId,
  setModelType,
  setSampleId,
  setSeed,
  setEpochs,
  runAction,
  interpretResult,
  gptOssStatus,
  gptOssStatusError,
  gptOssInterpretation,
  gptOssInterpretationError,
  isInterpreting,
  train,
}: CompactGenerativeViewProps) {
  const availableCount = status?.models.filter((item) => item.available).length ?? 0;
  const unavailableModel = status?.models.find(
    (item) => item.model_type === modelType && !item.available && item.unavailable_reason,
  );
  const statusText = statusLoading
    ? "Checking model status…"
    : statusError
      ? "Status unavailable"
      : !status?.torch_available
        ? "PyTorch runtime unavailable"
        : selectedModel
          ? "Trained model available"
          : availableCount
            ? `${availableCount} saved model${availableCount === 1 ? "" : "s"}`
            : "No trained model";

  return (
    <section className="generative-lab generative-lab--compact">
      <header className="page-heading gen-page-heading">
        <div>
          <h1>Generative AI Lab</h1>
          <p className="intro__copy">Reconstruct thermal fields and explore synthetic samples.</p>
        </div>
        <span className={`status-pill${statusError || !status?.torch_available ? " status-pill--unavailable" : selectedModel ? " status-pill--connected" : ""}`} role="status">
          {statusText}
        </span>
      </header>

      <Disclosure id="gen-about" title="About this lab">
        <p className="muted">Experiments use real, held-out coolant temperature patches from the scanned CFD data. VAE outputs are reconstructions; GAN outputs are synthetic and are not CFD simulations. A model’s statistical similarity or basic temperature checks do not establish physical validity or engineering reliability.</p>
        <p className="muted">The backend supports a GAN noise seed, but no truncation control. The generation API returns a synthetic field and aggregate comparisons, not the reference field values or a histogram.</p>
        {status && <p className="muted">Dataset: {status.dataset_status} · PyTorch: {status.torch_available ? `${status.torch_runtime.version} (${status.torch_runtime.device})` : "unavailable"} · Saved models: {availableCount} · Training device: {status.training_device ?? "not reported"}</p>}
        {statusError && <p className="message message--error" role="alert">{statusError}</p>}
      </Disclosure>

      <div className="generative-tabs" role="tablist" aria-label="Generative experiment">
        <button role="tab" aria-selected={modelType === "vae"} onClick={() => setModelType("vae")} type="button">VAE reconstruction</button>
        <button role="tab" aria-selected={modelType === "gan"} onClick={() => setModelType("gan")} type="button">GAN field generation</button>
      </div>

      <section className="panel gen-control-card" aria-labelledby="gen-controls-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">{modelType === "vae" ? "VAE" : "GAN"}</p>
            <h2 id="gen-controls-title">{modelType === "vae" ? "VAE Thermal Field Reconstruction" : "GAN Thermal Field Generation"}</h2>
          </div>
          <span className={`status-pill${selectedModel ? " status-pill--connected" : " status-pill--unavailable"}`}>
            {selectedModel ? "Model available" : "No matching trained model"}
          </span>
        </div>
        <div className="gen-controls-grid">
          <label className="selector-field">
            <span>Source CFD case</span>
            <select aria-label="Source CFD case" value={caseId} onChange={(event) => { setCaseId(event.target.value); setModelId(""); setSampleId(""); }}>
              <option value="">Select a scanned case</option>
              {status?.cases.map((item) => <option key={item.case_id} value={item.case_id}>{item.case_id} · Re {item.reynolds_number} · {item.mesh_resolution}</option>)}
            </select>
          </label>
          <label className="selector-field">
            <span>Trained {modelType.toUpperCase()} model</span>
            <select aria-label="Trained model" disabled={!models.length} value={selectedModel?.model_id ?? ""} onChange={(event) => setModelId(event.target.value)}>
              {!models.length && <option value="">No trained model for this case</option>}
              {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}
            </select>
          </label>
          {selectedModel && (
            <label className="selector-field">
              <span>Held-out CFD patch</span>
              <select aria-label="Held-out CFD patch" disabled={!samples.length} value={sampleId} onChange={(event) => setSampleId(event.target.value)}>
                {!samples.length && <option value="">No held-out patches available</option>}
                {samples.map((sample) => <option key={sample.sample_id} value={sample.sample_id}>{sample.sample_id} · {sample.valid_voxel_count}/512 valid voxels</option>)}
              </select>
            </label>
          )}
          {modelType === "gan" && (
            <label className="selector-field">
              <span>Noise seed</span>
              <input aria-label="Noise seed" min="0" step="1" type="number" value={seed} onChange={(event) => setSeed(event.target.value)} />
            </label>
          )}
        </div>
        {samplesLoading && <p className="muted" role="status">Loading held-out patch selections…</p>}
        {selectedCase && <p className="muted gen-case-context">Target: {selectedCase.target.region}/{selectedCase.target.field} ({selectedCase.target.units}) · solution time {selectedCase.solution_time}</p>}
        <div className="gen-action-row">
          <button className="primary-button" disabled={busy || !selectedModel || !sampleId || (modelType === "gan" && (!Number.isInteger(Number(seed)) || Number(seed) < 0))} onClick={() => void runAction()} type="button">
            {actionState === "working" ? modelType === "vae" ? "Reconstructing…" : "Generating sample…" : modelType === "vae" ? "Run Reconstruction" : "Generate Sample"}
          </button>
        </div>
        {error && <p className="message message--error" role="alert">{error}</p>}
        {status?.torch_available === false && <p className="message message--warning" role="status">{status.torch_runtime.diagnostic} No output is substituted.</p>}
        {!models.length && status?.torch_available && (
          <p className="muted">
            {unavailableModel?.unavailable_reason
              ?? status.empty_reason
              ?? `No saved ${modelType.toUpperCase()} model is available for this case yet. Training runs only when requested.`}
          </p>
        )}

        <Disclosure id="gen-advanced-controls" title={modelType === "vae" ? "Advanced reconstruction details" : "Advanced sampling details"}>
          <div className="gen-training-fields">
            <label className="selector-field">
              <span>Training random seed</span>
              <input min="0" step="1" type="number" value={seed} onChange={(event) => setSeed(event.target.value)} />
            </label>
            <label className="selector-field">
              <span>Training epochs (1–100)</span>
              <input min="1" max="100" step="1" type="number" value={epochs} onChange={(event) => setEpochs(event.target.value)} />
            </label>
          </div>
          <p className="muted">Training creates a new exploratory model for the selected case.</p>
          <button className="secondary-button" disabled={busy || !status?.torch_available || !selectedCase} onClick={() => void train()} type="button">
            {trainingState === "working" ? `Training ${modelType.toUpperCase()}…` : `Train ${modelType.toUpperCase()}`}
          </button>
          {selectedModel && <ModelDetails model={selectedModel} />}
          <p className="muted">Architecture descriptions, exact objective/loss configuration, and inference metadata are shown only when returned by the saved-model metadata.</p>
          {selectedModel && (
            <div className="generative-metrics">
              {Object.entries(selectedModel.metrics).filter(([, value]) => typeof value === "number" || typeof value === "string" || typeof value === "boolean").map(([key, value]) => <Metric key={key} label={key.replace(/_/g, " ")} value={value} />)}
            </div>
          )}
        </Disclosure>
      </section>

      {actionState === "working" && <p className="muted" role="status">Running the selected operation on the saved model and held-out data…</p>}
      {result && <CompactResults modelType={modelType} result={result} model={selectedModel} />}
      <section className="panel gen-gpt-oss" aria-labelledby="gen-gpt-oss-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">{gptOssStatus?.provider ?? "GPT-OSS"} · RESULT EXPLANATION</p>
            <h2 id="gen-gpt-oss-title">Explain this experiment</h2>
          </div>
          <span className={`status-pill${gptOssStatus?.status === "reachable" ? " status-pill--connected" : " status-pill--unavailable"}`}>
            {gptOssStatusError ? "Unavailable" : gptOssStatus?.status === "reachable" ? "Reachable" : gptOssStatus?.status === "unavailable" ? "Unavailable" : gptOssStatus ? "Not Verified" : "Checking…"}
          </span>
        </div>
        <p className="muted">GPT-OSS explains only the returned model result and metrics. Its text does not create predictions or establish physical validity.</p>
        <button
          className="secondary-button"
          disabled={!result || gptOssStatus?.status !== "reachable" || isInterpreting}
          onClick={() => void interpretResult()}
          type="button"
        >
          {isInterpreting ? "Requesting explanation…" : "Explain current result"}
        </button>
        {!result && <p className="muted">Run an experiment to enable evidence-based interpretation.</p>}
        {gptOssStatusError && <p className="message message--error" role="alert">{gptOssStatusError}</p>}
        {gptOssInterpretationError && <p className="message message--error" role="alert">{gptOssInterpretationError}</p>}
        {gptOssStatus?.diagnostic && gptOssStatus.status !== "reachable" && <p className="muted">{gptOssStatus.diagnostic}</p>}
        {gptOssInterpretation && (
          <div className="ai-interpretation" aria-live="polite">
            <p>{gptOssInterpretation.interpretation}</p>
            <small>{gptOssInterpretation.provider} · {gptOssInterpretation.model} · {new Date(gptOssInterpretation.generated_at).toLocaleString()}</small>
          </div>
        )}
      </section>
      {!result && selectedModel && (
        <section className="panel gen-empty-results">
          <h2>{modelType === "vae" ? "Reconstruction results" : "Generated sample"}</h2>
          <p className="muted">Run the selected experiment to view its actual field visualization and metrics.</p>
        </section>
      )}

      <section className="gen-compare" aria-label="VAE and GAN comparison">
        <div className="gen-compare__row">
          <div><strong>VAE</strong><span>Reconstructs an existing thermal field.</span></div>
          <div><strong>GAN</strong><span>Generates synthetic thermal field samples.</span></div>
        </div>
        <Disclosure id="gen-compare-details" title="Compare VAE and GAN">
          <div className="gen-compare__details">
            <div><h3>VAE · reconstruction</h3><p>Encodes a real patch into a compact latent representation and decodes a reconstruction. The existing implementation records masked reconstruction loss and KL loss; the saved metadata identifies the beta coefficient.</p><p>Use it to measure how closely a model reconstructs held-out patches.</p></div>
            <div><h3>GAN · synthetic generation</h3><p>A generator and discriminator are trained in competition. Inference samples random noise conditioned on a real held-out coolant mask. The saved training record includes generator and discriminator losses.</p><p>Use it to explore generated samples and aggregate distribution comparisons, not as a new CFD run.</p></div>
          </div>
        </Disclosure>
      </section>

      <section className="gen-validity">
        <p>Generated and reconstructed fields are approximations. Validate physical constraints before using results for engineering decisions.</p>
        <Disclosure id="gen-validity-details" title="Physical validity and engineering limitations">
          <ul className="gen-check-list">
            <li>The field is represented on a coarse regular grid using piecewise-constant cell values; original unstructured-cell topology is not preserved.</li>
            <li>Voxels outside the coolant region are masked and excluded from loss and metrics; gray visualization cells are not field values.</li>
            <li>Training and held-out regions are separated with excluded buffer patches; evaluation is limited to the selected case and does not establish cross-case generalization.</li>
            <li>A positive absolute temperature is a minimal numerical plausibility check, not proof of physical validity.</li>
            <li>Spatial gradient, mass-conservation, and energy-conservation checks are not implemented by this experiment.</li>
            <li>GAN distribution comparisons and neighbor-jump statistics do not establish CFD convergence, a new simulation, or a safe cooling design.</li>
          </ul>
          {selectedModel?.limitations.map((limitation) => <p className="muted" key={limitation}>{limitation}</p>)}
        </Disclosure>
      </section>
    </section>
  );
}

export function GenerativeAiLab() {
  const [status, setStatus] = useState<GenerativeStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [modelType, setModelType] = useState<GenerativeModelType>("vae");
  const [caseId, setCaseId] = useState("");
  const [modelId, setModelId] = useState("");
  const [sampleId, setSampleId] = useState("");
  const [samples, setSamples] = useState<GenerativeSample[]>([]);
  const [samplesLoading, setSamplesLoading] = useState(false);
  const [seed, setSeed] = useState("42");
  const [epochs, setEpochs] = useState("30");
  const [trainingState, setTrainingState] = useState<AsyncState>("idle");
  const [actionState, setActionState] = useState<AsyncState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PatchResult | null>(null);
  const [gptOssStatus, setGptOssStatus] = useState<GptOssStatus | null>(null);
  const [gptOssStatusError, setGptOssStatusError] = useState<string | null>(null);
  const [gptOssInterpretation, setGptOssInterpretation] = useState<FeatureInterpretation | null>(null);
  const [gptOssInterpretationError, setGptOssInterpretationError] = useState<string | null>(null);
  const [isInterpreting, setIsInterpreting] = useState(false);

  const models = useMemo(
    () =>
      status?.models.filter(
        (model) => model.available && model.model_type === modelType && model.source.case_id === caseId,
      ) ?? [],
    [caseId, modelType, status?.models],
  );
  const selectedModel = models.find((model) => model.model_id === modelId) ?? models[0] ?? null;
  const selectedCase = status?.cases.find((item) => item.case_id === caseId) ?? null;

  useEffect(() => {
    getGenerativeStatus()
      .then((response) => {
        setStatus(response);
        setStatusError(response.error ?? null);
        const modelCaseId = response.models.find(
          (model) => model.available && response.cases.some((item) => item.case_id === model.source.case_id),
        )?.source.case_id;
        setCaseId(modelCaseId ?? response.cases[0]?.case_id ?? "");
        setModelId("");
      })
      .catch((requestError: Error) => setStatusError(requestError.message))
      .finally(() => setStatusLoading(false));
  }, []);

  useEffect(() => {
    getGptOssStatus()
      .then(setGptOssStatus)
      .catch((requestError: Error) => setGptOssStatusError(requestError.message));
  }, []);

  useEffect(() => {
    setSamples([]);
    setSampleId("");
    setResult(null);
    if (!selectedModel) {
      setSamplesLoading(false);
      return;
    }
    let active = true;
    setSamplesLoading(true);
    getGenerativeSamples(selectedModel.model_id)
      .then((response) => {
        if (!active) return;
        setSamples(response.samples);
        setSampleId(response.samples[0]?.sample_id ?? "");
      })
      .catch((requestError: Error) => {
        if (active) setError(requestError.message);
      })
      .finally(() => {
        if (active) setSamplesLoading(false);
      });
    return () => {
      active = false;
    };
  }, [selectedModel?.model_id]);

  function updateModels(newModel: GenerativeModel) {
    setStatus((current) =>
      current
        ? {
            ...current,
            models: [
              newModel,
              ...current.models.filter((item) => item.model_id !== newModel.model_id),
            ],
          }
        : current,
    );
    setModelId(newModel.model_id);
  }

  async function train() {
    const numericSeed = Number(seed);
    const numericEpochs = Number(epochs);
    if (!Number.isInteger(numericSeed) || numericSeed < 0 || !Number.isInteger(numericEpochs) || numericEpochs < 1) {
      setError("Enter a non-negative whole-number seed and a positive whole-number epoch count.");
      return;
    }
    if (!caseId) {
      setError("Select one scanned case before training.");
      return;
    }
    setTrainingState("working");
    setError(null);
    setResult(null);
    try {
      const response = await trainGenerativeModel({
        model_type: modelType,
        case_id: caseId,
        seed: numericSeed,
        epochs: numericEpochs,
      });
      updateModels(response.model);
      setStatusError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Training failed.");
    } finally {
      setTrainingState("idle");
    }
  }

  async function runAction() {
    if (!selectedModel || !sampleId) {
      setError("Choose a trained model and a real held-out patch first.");
      return;
    }
    const numericSeed = Number(seed);
    if (modelType === "gan" && (!Number.isInteger(numericSeed) || numericSeed < 0)) {
      setError("The generation seed must be a non-negative whole number.");
      return;
    }
    setActionState("working");
    setError(null);
    setResult(null);
    setGptOssInterpretation(null);
    setGptOssInterpretationError(null);
    try {
      const response =
        modelType === "vae"
          ? await reconstructPatch({ model_id: selectedModel.model_id, sample_id: sampleId })
          : await generatePatch({
              model_id: selectedModel.model_id,
              sample_id: sampleId,
              seed: numericSeed,
            });
      setResult(response);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The experiment action failed.");
    } finally {
      setActionState("idle");
    }
  }

  async function interpretResult() {
    if (!result || !selectedModel) return;
    setIsInterpreting(true);
    setGptOssInterpretation(null);
    setGptOssInterpretationError(null);
    try {
      const interpretation = await interpretFeatureEvidence("generative", {
        model_type: modelType,
        model_id: selectedModel.model_id,
        result_label: result.label,
        status: result.status,
        source: result.sample,
        target: result.target,
        metrics: result.metrics,
        generation_seed: result.generation_seed ?? null,
        limitations: result.limitations ?? selectedModel.limitations,
      });
      setGptOssInterpretation(interpretation);
    } catch (requestError) {
      setGptOssInterpretationError(requestError instanceof Error ? requestError.message : "GPT-OSS interpretation failed.");
    } finally {
      setIsInterpreting(false);
    }
  }

  const busy = trainingState === "working" || actionState === "working";

  return window.location.pathname === "/generative.html" ? (
    <CompactGenerativeView
      actionState={actionState}
      caseId={caseId}
      error={error}
      models={models}
      modelType={modelType}
      result={result}
      sampleId={sampleId}
      samples={samples}
      samplesLoading={samplesLoading}
      seed={seed}
      epochs={epochs}
      selectedCase={selectedCase}
      selectedModel={selectedModel}
      status={status}
      statusError={statusError}
      statusLoading={statusLoading}
      trainingState={trainingState}
      busy={busy}
      setCaseId={(value) => { setCaseId(value); setGptOssInterpretation(null); setGptOssInterpretationError(null); }}
      setModelId={(value) => { setModelId(value); setResult(null); setGptOssInterpretation(null); setGptOssInterpretationError(null); }}
      setModelType={(value) => { setModelType(value); setResult(null); setGptOssInterpretation(null); setGptOssInterpretationError(null); setError(null); }}
      setSampleId={(value) => { setSampleId(value); setResult(null); setGptOssInterpretation(null); setGptOssInterpretationError(null); }}
      setSeed={setSeed}
      setEpochs={setEpochs}
      runAction={runAction}
      interpretResult={interpretResult}
      gptOssStatus={gptOssStatus}
      gptOssStatusError={gptOssStatusError}
      gptOssInterpretation={gptOssInterpretation}
      gptOssInterpretationError={gptOssInterpretationError}
      isInterpreting={isInterpreting}
      train={train}
    />
  ) : (
    <main className="generative-lab">
      <section className="page-heading">
        <div>
          <p className="eyebrow">RESEARCH EXPERIMENT · NOT A VALIDATED SIMULATION</p>
          <h1>Generative AI Lab</h1>
          <p className="intro__copy">
            Test whether compact generative models can reconstruct or statistically imitate small regions of the
            verified coolant temperature field.
          </p>
        </div>
      </section>

      {statusLoading && <p className="loading-state" role="status">Checking the latest scanned data and saved generative models…</p>}
      {statusError && <p className="message message--error" role="alert">{statusError}</p>}
      {status && (
        <section className="generative-status" aria-label="Generative experiment status">
          <div>
            <span className="eyebrow">VERIFIED DATA</span>
            <strong>{status.dataset_status === "available" ? `${status.cases.length} cases with coolant/T` : "Unavailable"}</strong>
          </div>
          <div>
            <span className="eyebrow">PYTORCH RUNTIME</span>
            <strong>{status.torch_available ? `${status.torch_runtime.version} · ${status.torch_runtime.device}` : "Unavailable"}</strong>
          </div>
          <div>
            <span className="eyebrow">SAVED MODELS</span>
            <strong>{status.models.filter((model) => model.available).length}</strong>
          </div>
        </section>
      )}

      <div className="generative-tabs" role="tablist" aria-label="Generative model">
        <button role="tab" aria-selected={modelType === "vae"} onClick={() => { setModelType("vae"); setResult(null); setError(null); }} type="button">
          VAE reconstruction
        </button>
        <button role="tab" aria-selected={modelType === "gan"} onClick={() => { setModelType("gan"); setResult(null); setError(null); }} type="button">
          GAN field generation
        </button>
      </div>

      <section className="generative-explainer panel">
        <div>
          <p className="eyebrow">{modelType === "vae" ? "VARIATIONAL AUTOENCODER" : "GENERATIVE ADVERSARIAL NETWORK"}</p>
          <h2>{modelType === "vae" ? "Can a compact model rebuild a held-out patch?" : "Can a generator imitate held-out field statistics?"}</h2>
          <ConceptHint term={modelType === "vae" ? "Variational autoencoder (VAE)" : "Generative adversarial network (GAN)"}>
            {modelType === "vae"
              ? "A VAE learns a compressed representation and attempts to reconstruct a held-out real data patch. Reconstruction similarity does not establish physical accuracy."
              : "A GAN trains a generator and discriminator in competition. Its output is explicitly synthetic and is not a new validated CFD solution."}
          </ConceptHint>
          <p>
            {modelType === "vae"
              ? "A VAE compresses a real temperature patch into a small latent representation and decodes it. We compare its reconstruction against a spatially held-out CFD patch."
              : "A GAN trains a generator against a discriminator. It receives a real held-out coolant mask and creates a new temperature-valued patch; the result is synthetic, not a CFD run."}
          </p>
        </div>
        <p className="generative-warning">
          Statistical resemblance is not physical validation. These models do not solve the governing equations,
          establish convergence, or identify a safe cooling design.
        </p>
      </section>

      <section className="panel generative-controls" aria-label="Experiment controls">
        <div className="section-heading">
          <div>
            <p className="eyebrow">REPRODUCIBLE PILOT</p>
            <h2>Choose data and run the experiment</h2>
          </div>
          <span className={`status-pill ${selectedModel ? "status-pill--connected" : ""}`}>
            {selectedModel ? "Trained model available" : "No matching trained model"}
          </span>
        </div>
        <div className="explorer-controls">
          <label>
            Verified case
            <select value={caseId} onChange={(event) => { setCaseId(event.target.value); setModelId(""); setSamples([]); setResult(null); }}>
              <option value="">Select a scanned case</option>
              {status?.cases.map((item) => (
                <option value={item.case_id} key={item.case_id}>
                  {item.case_id} · Re {item.reynolds_number} · {item.mesh_resolution}
                </option>
              ))}
            </select>
          </label>
          <label>
            Target field
            <input aria-label="Target field" readOnly value={selectedCase ? "coolant/T (K)" : "Unavailable"} />
          </label>
          {models.length > 0 && (
            <label>
              Saved {modelType.toUpperCase()} model
              <select value={selectedModel?.model_id ?? ""} onChange={(event) => { setModelId(event.target.value); setResult(null); }}>
                {models.map((model) => <option key={model.model_id} value={model.model_id}>{model.model_id}</option>)}
              </select>
            </label>
          )}
          {samplesLoading && <p className="muted" role="status">Loading real held-out patch selections…</p>}
          {selectedModel && samples.length > 0 && (
            <label>
              Held-out patch
              <select value={sampleId} onChange={(event) => { setSampleId(event.target.value); setResult(null); }}>
                {samples.map((sample) => (
                  <option key={sample.sample_id} value={sample.sample_id}>
                    {sample.sample_id} · {sample.valid_voxel_count}/512 coolant voxels
                  </option>
                ))}
              </select>
            </label>
          )}
        </div>
        <details className="advanced-controls">
          <summary>Advanced training settings</summary>
          <p className="muted">
            Defaults are reproducible. Change these only when comparing
            experiments; every run records its seed and epoch count.
          </p>
          <div className="advanced-controls__fields">
            <label>
              Random seed
              <input type="number" min="0" step="1" value={seed} onChange={(event) => setSeed(event.target.value)} />
            </label>
            <label>
              Training epochs (1–100)
              <input type="number" min="1" max="100" step="1" value={epochs} onChange={(event) => setEpochs(event.target.value)} />
            </label>
          </div>
        </details>
        {!status?.torch_available && (
          <p className="message message--warning" role="status">
            {status?.torch_runtime.diagnostic ?? "The backend PyTorch runtime is unavailable."} In the same environment used to start Flask, run <code>python -m pip install -r requirements-generative.txt</code>. No model or training metrics are being substituted.
          </p>
        )}
        {selectedModel && (
          <div className="generative-trained-summary">
            <p>
              <strong>Saved experiment:</strong> {selectedModel.model_id} · {selectedModel.configuration.epochs} epochs ·
              seed {selectedModel.configuration.seed} · CPU
            </p>
            <p>
              Actual samples: {selectedModel.split.patch_counts.train} training,{" "}
              {selectedModel.split.patch_counts.validation} held-out validation,{" "}
              {selectedModel.split.patch_counts.buffer} excluded buffer patches.
            </p>
          </div>
        )}
        <div className="generative-actions">
          <button
            className="primary-button"
            disabled={busy || !status?.torch_available || !selectedCase}
            onClick={() => void train()}
            type="button"
          >
            {trainingState === "working" ? "Training from selected real case…" : `Train ${modelType.toUpperCase()}`}
          </button>
          <button
            className="secondary-button"
            disabled={busy || !selectedModel || !sampleId}
            onClick={() => void runAction()}
            type="button"
          >
            {actionState === "working"
              ? modelType === "vae" ? "Reconstructing…" : "Generating synthetic patch…"
              : modelType === "vae" ? "Reconstruct held-out patch" : "Generate synthetic patch"}
          </button>
        </div>
        {error && <p className="message message--error" role="alert">{error}</p>}
        {!models.length && status?.torch_available && (
          <p className="muted">
            {status.models.find(
              (item) => item.model_type === modelType && !item.available && item.unavailable_reason,
            )?.unavailable_reason
              ?? status.empty_reason
              ?? `No saved ${modelType.toUpperCase()} model is available for this case yet. Training runs only when you press the button.`}
          </p>
        )}
      </section>

      {selectedModel && (
        <section className="panel generative-evaluation">
          <p className="eyebrow">MEASURED {modelType.toUpperCase()} EXPERIMENT</p>
          <h2>Held-out evaluation</h2>
          {modelType === "vae" ? (
            <div className="generative-metrics">
              <Metric label="Held-out reconstruction MAE" value={selectedModel.metrics.mae_K} units="K" />
              <Metric label="Held-out reconstruction RMSE" value={selectedModel.metrics.rmse_K} units="K" />
              <Metric label="Positive-K check" value={plausibilityLabel(selectedModel.metrics.positivity_passed)} />
            </div>
          ) : (
            <div className="generative-metrics">
              <Metric label="Temperature Wasserstein distance" value={selectedModel.metrics.temperature_wasserstein_distance_K} units="K" />
              <Metric label="Held-out mean T" value={selectedModel.metrics.heldout_mean_temperature_K} units="K" />
              <Metric label="Synthetic mean T" value={selectedModel.metrics.generated_mean_temperature_K} units="K" />
              <Metric label="Held-out neighbor jump" value={selectedModel.metrics.heldout_mean_neighbor_jump_K_per_voxel} units="K/voxel" />
              <Metric label="Synthetic neighbor jump" value={selectedModel.metrics.generated_mean_neighbor_jump_K_per_voxel} units="K/voxel" />
              <Metric label="Neighbor-jump difference" value={selectedModel.metrics.mean_neighbor_jump_difference_K_per_voxel} units="K/voxel" />
              <Metric label="Positive-K check" value={selectedModel.metrics.positive_kelvin_fraction === 1 ? "Passed" : "Not passed"} />
            </div>
          )}
          <p className="muted">
            These are actual saved held-out results for this one case and representation. They are pilot statistics,
            not evidence of cross-case generalization or physical validity.
          </p>
          <details>
            <summary>Training history (recorded losses)</summary>
            <p>Final recorded epoch: {selectedModel.training_history_summary.record_count}</p>
            <pre>{JSON.stringify(selectedModel.training_history_summary.final, null, 2)}</pre>
          </details>
        </section>
      )}

      {actionState === "working" && <p className="muted" role="status">Running the selected operation against the saved model and real held-out data…</p>}
      {result && <section className="panel generative-results"><ResultsView modelType={modelType} result={result} /></section>}

      <section className="panel generative-method">
        <h2>How to read this experiment</h2>
        <ul>
          <li>Each patch is an 8 × 8 × 8 region cut from a coarse 32 × 32 × 32 grid over the selected mesh bounds.</li>
          <li>Each voxel center uses the value from the containing coolant cell; there is no interpolation or fabricated fill value.</li>
          <li>Gray cells are outside the coolant region and are excluded from model loss and evaluation.</li>
          <li>Spatially neighboring patches are kept apart: training and held-out regions are separated by excluded buffer patches.</li>
          <li>The field remains in kelvin. Coordinates remain in source units, which the dataset does not declare.</li>
        </ul>
      </section>
    </main>
  );
}
