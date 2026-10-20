export interface HealthResponse {
  status: "ok";
}

export type BackendConnectionState = "checking" | "connected" | "unavailable";
