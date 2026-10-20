export interface ProviderTestResult {
  provider: string;
  model: string | null;
  status: "connected" | "failed" | "not_tested";
  tested_at: string;
  diagnostic: string;
  latency_ms?: number;
}

export interface GptOssStatus {
  provider: string;
  configured: boolean;
  model: string | null;
  status: "reachable" | "unavailable" | "not_verified";
  checked_at?: string;
  diagnostic: string;
}

export interface FeatureInterpretation {
  label: string;
  provider: string;
  model: string;
  interpretation: string;
  generated_at: string;
}
