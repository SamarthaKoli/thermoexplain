import type {
  GenerativeModel,
  GenerativeModelType,
  GenerativeStatus,
  PatchResult,
  PreparedSamples,
} from "../types/generative";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");
const generativeUrl = `${apiBaseUrl}/generative`;

export function getGenerativeStatus(): Promise<GenerativeStatus> {
  return requestJson<GenerativeStatus>(`${generativeUrl}/status`);
}

export function trainGenerativeModel(input: {
  model_type: GenerativeModelType;
  case_id: string;
  seed: number;
  epochs: number;
}): Promise<{ model: GenerativeModel }> {
  return postJson(`${generativeUrl}/train`, input);
}

export function getGenerativeSamples(modelId: string): Promise<PreparedSamples> {
  return requestJson(
    `${generativeUrl}/models/${encodeURIComponent(modelId)}/samples`,
  );
}

export function reconstructPatch(input: {
  model_id: string;
  sample_id: string;
}): Promise<PatchResult> {
  return postJson(`${generativeUrl}/reconstruct`, input);
}

export function generatePatch(input: {
  model_id: string;
  sample_id: string;
  seed: number;
}): Promise<PatchResult> {
  return postJson(`${generativeUrl}/generate`, input);
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
    throw new Error(
      "Could not reach the generative experiment API. Check that the backend is running.",
    );
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The generative API returned an invalid response.");
  }
  if (!response.ok) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "error" in payload &&
      typeof payload.error === "string"
        ? payload.error
        : `Request failed with HTTP ${response.status}.`;
    throw new Error(message);
  }
  return payload as T;
}
