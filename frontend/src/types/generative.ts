export type GenerativeModelType = "vae" | "gan";

export interface GenerativeCase {
  case_id: string;
  reynolds_number: number;
  mesh_resolution: string;
  solution_time: string;
  target: {
    region: string;
    field: string;
    units: string;
    field_class: string;
  };
}

export interface GenerativeSample {
  sample_id: string;
  case_id: string;
  reynolds_number: number;
  mesh_resolution: string;
  solution_time: string;
  patch_index: [number, number, number];
  patch_center: [number, number, number];
  patch_center_units: string | null;
  valid_voxel_count: number;
  split: "train" | "validation" | "buffer";
}

export interface GenerativeModel {
  schema_version: number;
  model_id: string;
  model_type: GenerativeModelType;
  status: string;
  available: boolean;
  unavailable_reason?: string;
  created_at: string;
  source: {
    case_id: string;
    reynolds_number: number;
    mesh_resolution: string;
    solution_time: string;
    target: GenerativeCase["target"];
    cell_count: number;
    manifest_sha256: string;
  };
  representation: {
    method: string;
    interpolation: string;
    mask: string;
    grid_size: number[];
    patch_size: number[];
    stride: number;
    coordinate_units: string | null;
    limitations: string[];
  };
  split: {
    method: string;
    patch_counts: { train: number; validation: number; buffer: number };
  };
  configuration: {
    seed: number;
    epochs: number;
    device: string;
    latent_dimension: number;
    hidden_dimension: number;
    patch_shape: number[];
    normalization: {
      method: string;
      mean_K: number;
      standard_deviation_K: number;
    };
    vae_beta: number | null;
  };
  metrics: Record<string, unknown>;
  training_history: Array<Record<string, number>>;
  training_history_summary: { record_count: number; final: Record<string, number> };
  limitations: string[];
}

export interface GenerativeStatus {
  dataset_status: "available" | "unavailable";
  cases: GenerativeCase[];
  models: GenerativeModel[];
  torch_available: boolean;
  torch_runtime: {
    available: boolean;
    version: string | null;
    device: string | null;
    diagnostic: string;
  };
  training_device?: string;
  manifest_sha256?: string;
  status: string;
  empty_reason: string | null;
  error?: string;
}

export interface PreparedSamples {
  model_id: string;
  model_type: GenerativeModelType;
  samples: GenerativeSample[];
}

export interface PatchResult {
  label: string;
  synthetic_sample?: {
    label: string;
    conditioning_sample_id: string;
    source_case_id: string;
    source_mesh_resolution: string;
    source_solution_time: string;
    source_split: string;
    patch_index: number[];
    patch_center: number[];
    patch_center_units: string | null;
    generation_seed: number;
    field: { region: string; name: string; units: string };
    status: string;
  };
  sample: GenerativeSample;
  target: GenerativeCase["target"];
  values_K: {
    original?: Array<Array<Array<number | null>>>;
    reconstructed?: Array<Array<Array<number | null>>>;
    difference?: Array<Array<Array<number | null>>>;
  } | Array<Array<Array<number | null>>>;
  mask: boolean[][][];
  metrics: Record<string, number | string | boolean | null>;
  status: string;
  generation_seed?: number;
  limitations?: string[];
}
