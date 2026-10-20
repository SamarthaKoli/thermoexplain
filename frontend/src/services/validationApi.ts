import type { ValidationStatusResponse } from "../types/validation";
import { throwApiError } from "./providerApi";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");

export async function getValidationStatus(): Promise<ValidationStatusResponse> {
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}/validation/status`);
  } catch {
    throw new Error("Could not reach the validation status API. Check the backend.");
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The validation status API returned an invalid response.");
  }
  if (!response.ok) {
    throwApiError(payload, response.status);
  }
  return payload as ValidationStatusResponse;
}
