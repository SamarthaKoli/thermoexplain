import type {
  GlobalShapResponse,
  GptOssInterpretation,
  LocalExplanationResponse,
  PredictionInput,
  SurrogateModelsResponse,
  SurrogatePrediction,
} from "../types/ml";
import { throwApiError } from "./providerApi";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");
const mlUrl = `${apiBaseUrl}/ml`;

export function getSurrogateModels(): Promise<SurrogateModelsResponse> {
  return getJson<SurrogateModelsResponse>(`${mlUrl}/models`);
}

export function predictField(input: PredictionInput): Promise<SurrogatePrediction> {
  return postJson<SurrogatePrediction>(`${mlUrl}/predict`, input);
}

export function getGlobalShap(modelId: string): Promise<GlobalShapResponse> {
  return getJson<GlobalShapResponse>(
    `${mlUrl}/shap/global?model_id=${encodeURIComponent(modelId)}`,
  );
}

export function getLocalExplanation(
  method: "shap" | "lime",
  input: PredictionInput,
): Promise<LocalExplanationResponse> {
  return postJson<LocalExplanationResponse>(
    `${mlUrl}/${method}/local`,
    input,
  );
}

export function interpretWithGptOss(
  method: "shap" | "lime",
  input: PredictionInput,
): Promise<GptOssInterpretation> {
  return postJson<GptOssInterpretation>(`${mlUrl}/gpt-oss/interpretation`, {
    ...input,
    method,
  });
}

async function getJson<T>(url: string): Promise<T> {
  return requestJson<T>(url);
}

async function postJson<T>(url: string, payload: object): Promise<T> {
  return requestJson<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error("Could not reach the model API. Check that the backend is running.");
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The model API returned an invalid response.");
  }
  if (!response.ok) {
    throwApiError(payload, response.status);
  }
  return payload as T;
}
