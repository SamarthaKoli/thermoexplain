import type {
  FeatureInterpretation,
  GptOssStatus,
  ProviderTestResult,
} from "../types/provider";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");

export function getGptOssStatus(): Promise<GptOssStatus> {
  return providerJson<GptOssStatus>(`${apiBaseUrl}/llm/status`);
}

export function testGptOssConnection(): Promise<ProviderTestResult> {
  return providerJson<ProviderTestResult>(`${apiBaseUrl}/llm/connection-test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
}

export function interpretFeatureEvidence(
  feature: "generative" | "validation",
  context: Record<string, unknown>,
): Promise<FeatureInterpretation> {
  return providerJson<FeatureInterpretation>(`${apiBaseUrl}/llm/interpretation`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ feature, context }),
  });
}

async function providerJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error("Could not reach the GPT-OSS API. Check that the backend is running.");
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The GPT-OSS API returned an invalid response.");
  }
  if (!response.ok) throwApiError(payload, response.status);
  return payload as T;
}

export class ProviderTestApiError extends Error {
  readonly result: ProviderTestResult;

  constructor(
    message: string,
    result: ProviderTestResult,
    readonly statusCode: number,
  ) {
    super(message);
    this.name = "ProviderTestApiError";
    this.result = result;
  }
}

export function throwApiError(payload: unknown, statusCode: number): never {
  if (
    typeof payload === "object" &&
    payload !== null &&
    "provider" in payload &&
    typeof payload.provider === "string" &&
    "status" in payload &&
    payload.status === "failed" &&
    "tested_at" in payload &&
    typeof payload.tested_at === "string" &&
    "diagnostic" in payload &&
    typeof payload.diagnostic === "string"
  ) {
    const model =
      "model" in payload && typeof payload.model === "string"
        ? payload.model
        : null;
    throw new ProviderTestApiError(payload.diagnostic, {
      provider: payload.provider,
      model,
      status: "failed",
      tested_at: payload.tested_at,
      diagnostic: payload.diagnostic,
    }, statusCode);
  }
  const message =
    typeof payload === "object" &&
    payload !== null &&
    "error" in payload &&
    typeof payload.error === "string"
      ? payload.error
      : `Request failed with HTTP ${statusCode}.`;
  throw new Error(message);
}

export function providerTestFailure(
  provider: string,
  model: string | null,
  error: unknown,
): ProviderTestResult {
  if (error instanceof ProviderTestApiError) return error.result;
  return {
    provider,
    model,
    status: "not_tested",
    tested_at: new Date().toISOString(),
    diagnostic:
      error instanceof Error
        ? `Provider test was not completed: ${error.message}`
        : "Provider test was not completed because the backend could not be reached.",
  };
}
