export interface ValidationMetrics {
  mae: number;
  rmse: number;
  r2: number | null;
}

export interface SurrogateCase {
  case_id: string;
  reynolds_number: number;
  mesh_resolution: string;
  solution_time: string;
  cell_count: number;
  sample_count: number;
  bounds: [number, number][];
}

export interface SurrogateModel {
  model_id: string;
  model_name: string;
  available: boolean;
  unavailable_reason?: string;
  target: {
    region: string;
    field: string;
    units: string | null;
    field_class: string;
  };
  features: Array<{ name: string; units: string | null }>;
  coordinate_units: string | null;
  coordinate_method: string;
  cases: SurrogateCase[];
  dataset: {
    case_count: number;
    reynolds_numbers: number[];
    mesh_resolutions: string[];
    single_geometry: boolean;
    exploratory: boolean;
  };
  split: {
    method: string;
    axis: string;
    training_sample_count: number;
    validation_sample_count: number;
    excluded_buffer_sample_count: number;
    training_cases: string[];
    validation_cases: string[];
    validation_kind: string;
  };
  metrics: {
    baseline_mean_predictor: ValidationMetrics;
    extra_trees_spatial_validation: ValidationMetrics;
    metric_scope: string;
    validation_sample_count: number;
  };
  explainability: {
    shap_global: {
      method: string;
      baseline_definition: string;
      sample_count: number;
      additivity_checked: boolean;
      max_absolute_additivity_error: number;
      features: Array<{
        feature: string;
        mean_absolute_shap: number;
        mean_shap: number;
      }>;
    };
    lime: {
      method: string;
      approximation_quality_metric: string;
    };
  };
  limitations: string[];
}

export interface SurrogateModelsResponse {
  models: SurrogateModel[];
  status: string;
  empty_reason: string | null;
}

export interface PredictionInput {
  model_id: string;
  case_id: string;
  coordinates: { x: number; y: number; z: number };
}

export interface SurrogatePrediction {
  model_id: string;
  case_id: string;
  target: SurrogateModel["target"];
  coordinates: { x: number; y: number; z: number; units: string | null };
  reynolds_number: number;
  prediction: number;
  units: string | null;
  status: "exploratory";
  limitations: string[];
}

export interface LocalExplanation {
  method: string;
  prediction: number;
  contributions: Array<{
    feature: string;
    value: number;
    contribution: number;
  }>;
  baseline?: number;
  baseline_definition?: string;
  reconstructed_prediction?: number;
  additivity_error?: number;
  additivity_checked?: boolean;
  additivity_passed?: boolean;
  sample?: Record<string, number>;
  local_prediction?: number;
  intercept?: number;
  local_r2?: number;
  local_approximation_absolute_error?: number;
  num_samples?: number;
  caveat?: string;
}

export interface LocalExplanationResponse {
  model_id: string;
  case_id: string;
  target: SurrogateModel["target"];
  explanation: LocalExplanation;
}

export interface GlobalShapResponse {
  model_id: string;
  target: SurrogateModel["target"];
  baseline_definition: string;
  sample_count: number;
  additivity_checked: boolean;
  features: SurrogateModel["explainability"]["shap_global"]["features"];
}

export interface GptOssInterpretation {
  explanation_method: "shap" | "lime";
  interpretation: {
    label: string;
    provider?: string;
    model: string;
    interpretation: string;
    generated_at: string;
  };
}
