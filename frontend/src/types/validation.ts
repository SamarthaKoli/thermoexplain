export type ValidationEvidenceStatus =
  | "passed"
  | "failed"
  | "partial"
  | "not_evaluated"
  | "unavailable"
  | "not_reported";

export interface ValidationModuleEvidence {
  id: string;
  title: string;
  status: ValidationEvidenceStatus;
  recorded_at: string | null;
  metrics: Record<string, unknown>;
  limitations: string[];
  diagnostic: string | null;
}

export interface ValidationStatusResponse {
  checked_at: string;
  summary: {
    full_suite_status: "not_evaluated";
    module_count: number;
    modules_with_recorded_evidence: number;
    status_counts: Record<ValidationEvidenceStatus, number>;
    limitations: string[];
  };
  dataset: {
    status: string;
    generated_at: string | null;
    case_count: number | null;
    warnings: string[];
    diagnostic: string | null;
  };
  modules: ValidationModuleEvidence[];
  provider: {
    provider: string;
    model: string | null;
    configured: boolean;
    status: "configured" | "not_configured";
    counts_as_validation: false;
  };
}
