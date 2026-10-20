import type {
  CfdFieldOptions,
  CfdStatistics,
  CfdSpatialVisualization,
  DatasetAudit,
  DatasetCase,
  DatasetScanStatus,
  GptOssAnalysis,
  MeshComparisonResult,
  PreprocessingStatus,
} from "../types/dataset";
import { throwApiError } from "./providerApi";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");
const datasetUrl = `${apiBaseUrl}/dataset`;

export function getDatasetScan(): Promise<DatasetScanStatus> {
  return getJson<DatasetScanStatus>(`${datasetUrl}/scan`);
}

export function getDatasetCases(): Promise<{ cases: DatasetCase[] }> {
  return getJson<{ cases: DatasetCase[] }>(`${datasetUrl}/cases`);
}

export function getCfdFieldOptions(caseId: string): Promise<CfdFieldOptions> {
  return getJson<CfdFieldOptions>(
    `${datasetUrl}/cfd-fields?case_id=${encodeURIComponent(caseId)}`,
  );
}

export function getCfdStatistics(
  caseId: string,
  region: string,
  field: string,
  component: string,
): Promise<CfdStatistics> {
  const query = new URLSearchParams({ case_id: caseId, region, field, component });
  return getJson<CfdStatistics>(`${datasetUrl}/field-statistics?${query}`);
}

export function getCfdSpatialVisualization(
  caseId: string,
  region: string,
  field: string,
  component: string,
  signal?: AbortSignal,
): Promise<CfdSpatialVisualization> {
  const query = new URLSearchParams({ case_id: caseId, region, field, component });
  return getJson<CfdSpatialVisualization>(
    `${datasetUrl}/spatial-visualization?${query}`,
    { signal },
  );
}

export function compareMeshes(
  caseId: string,
  referenceCaseId: string,
): Promise<MeshComparisonResult> {
  return getJson<MeshComparisonResult>(`${datasetUrl}/mesh-comparison`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId, reference_case_id: referenceCaseId }),
  });
}

export function analyzeWithGptOss(input: {
  case_id: string;
  region: string;
  field: string;
  component: string;
}): Promise<GptOssAnalysis> {
  return getJson<GptOssAnalysis>(`${apiBaseUrl}/llm/analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export function getPreprocessingStatus(): Promise<PreprocessingStatus> {
  return getJson<PreprocessingStatus>(`${datasetUrl}/preprocessing`);
}

export function getDatasetAudit(): Promise<DatasetAudit> {
  return getJson<DatasetAudit>(`${datasetUrl}/audit`);
}

export function scanDataset(): Promise<
  DatasetScanStatus & { case_count: number }
> {
  return getJson<DatasetScanStatus & { case_count: number }>(`${datasetUrl}/scan`, {
    method: "POST",
  });
}

export async function getCaseDetails(caseId: string) {
  const query = `?case_id=${encodeURIComponent(caseId)}`;
  const [metadata, fields, mesh] = await Promise.all([
    getJson<DatasetCase>(`${datasetUrl}/case-metadata${query}`),
    getJson<{ fields: DatasetCase["fields"] }>(`${datasetUrl}/fields${query}`),
    getJson<{
      case_id: string;
      mesh_info: DatasetCase["mesh_info"];
      mesh_info_by_file: DatasetCase["mesh_info_by_file"];
    }>(
      `${datasetUrl}/mesh-info${query}`,
    ),
  ]);
  return { metadata, fields: fields.fields, meshInfo: mesh.mesh_info };
}

export function preprocessCsv(file: string, sampleSize?: number, seed = 42) {
  return getJson<PreprocessingStatus>(`${datasetUrl}/preprocess`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file, sample_size: sampleSize, seed }),
  });
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error("Could not reach the dataset API. Check that the backend is running.");
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The dataset API returned an invalid response.");
  }
  if (!response.ok) {
    throwApiError(payload, response.status);
  }
  return payload as T;
}
