export interface DatasetScanStatus {
  status: string;
  generated_at: string | null;
  source_kind: string;
  summary: {
    file_count: number;
    candidate_count: number;
    total_bytes: number;
  };
  warnings: string[];
}

export interface DatasetField {
  name: string;
  association: string;
  units: string | null;
}

export interface CfdFieldOption extends DatasetField {
  region: string;
  field_class: string;
  value_type: "scalar" | "vector" | "symmTensor" | "tensor";
  dimensions: string | null;
  components: string[];
  processor_count: number;
  statistics_supported: boolean;
}

export interface CfdFieldOptions {
  time: string;
  processor_count: number;
  fields: CfdFieldOption[];
  visualization: {
    status: "boundary_surface_on_request";
    reason: string;
  };
}

export interface CfdStatistics {
  case_id: string;
  reynolds_number: number | null;
  mesh_resolution: string | null;
  region: string | null;
  field: string | null;
  field_class: string;
  component: string;
  association: string;
  dimensions: string | null;
  units: string | null;
  time: string;
  processor_count: number;
  sample_count: number | null;
  statistic_scope: string;
  minimum: number;
  maximum: number;
  mean: number;
  visualization: {
    status: "boundary_surface_on_request";
    reason: string;
  };
}

export interface CfdSpatialVisualization {
  case_id: string;
  region: string | null;
  field: string | null;
  units: string | null;
  component: string | null;
  geometry_type: "boundary_surface" | "surface_geometry";
  source_format: string;
  source_association: "cell" | "point" | "none";
  render_association: "point" | "none";
  mapping_method: string;
  triangle_count: number;
  source_triangle_count: number;
  simplified: boolean;
  points: [number, number, number][];
  triangles: [number, number, number][];
  values: number[];
  scalar_range: [number, number] | null;
  bounds: [number, number, number, number, number, number];
  available_arrays?: string[];
  diagnostic: string;
}

export interface MeshComparisonResult {
  status: "unavailable";
  reason: string;
}

export interface GptOssAnalysis {
  statistics: CfdStatistics;
  interpretation: {
    label: string;
    provider?: string;
    model: string;
    interpretation: string;
    generated_at: string;
  };
}

export interface DatasetMeshInfo {
  point_count: number | null;
  cell_count: number | null;
  piece_count?: number | null;
  geometry: string;
}

export interface DatasetCase {
  case_id: string;
  files: string[];
  fields: DatasetField[];
  mesh_info: DatasetMeshInfo | null;
  mesh_info_by_file: Array<{
    file: string;
    mesh_info: DatasetMeshInfo | null;
  }>;
  metadata: {
    reynolds_number: number | null;
    mesh_resolution: string | null;
    units: string | null;
    solution_time?: string;
    processor_count?: number;
  };
}

export interface PreprocessingStatus {
  status: string;
  message?: string;
  output_rows?: number;
  source_rows?: number;
  sampling?: string;
  split_status?: string;
}

export interface DatasetAudit {
  status: string;
  report: string | null;
}
