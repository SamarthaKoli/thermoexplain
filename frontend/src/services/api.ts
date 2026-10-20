import type { HealthResponse } from "../types/health";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");

export async function getBackendHealth(): Promise<HealthResponse> {
  let response: Response;

  try {
    response = await fetch(`${apiBaseUrl}/health`);
  } catch {
    throw new Error("Could not reach the backend. Check that it is running.");
  }

  if (!response.ok) {
    throw new Error(`The backend returned HTTP ${response.status}.`);
  }

  const payload: unknown = await response.json();
  if (
    typeof payload !== "object" ||
    payload === null ||
    !("status" in payload) ||
    payload.status !== "ok"
  ) {
    throw new Error("The backend returned an unexpected health response.");
  }

  return payload as HealthResponse;
}
