import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExplainableAiLab } from "./ExplainableAiLab";
import type { SurrogateModel } from "../types/ml";

const api = vi.hoisted(() => ({
  getGlobalShap: vi.fn(),
  getGptOssStatus: vi.fn(),
  getLocalExplanation: vi.fn(),
  getSurrogateModels: vi.fn(),
  interpretWithGptOss: vi.fn(),
  predictField: vi.fn(),
  testGptOssConnection: vi.fn(),
}));

vi.mock("../services/mlApi", () => api);
vi.mock("../services/providerApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../services/providerApi")>()),
  getGptOssStatus: api.getGptOssStatus,
  testGptOssConnection: api.testGptOssConnection,
}));

const model: SurrogateModel = {
  model_id: "verified_coolant_t",
  model_name: "Extra Trees spatial surrogate",
  available: true,
  target: {
    region: "coolant",
    field: "T",
    units: "K",
    field_class: "volScalarField",
  },
  features: [
    { name: "x", units: null },
    { name: "y", units: null },
    { name: "z", units: null },
    { name: "reynolds_number", units: "dimensionless" },
  ],
  coordinate_units: null,
  coordinate_method: "VTK cell centers from the OpenFOAM internal mesh",
  cases: [
    {
      case_id: "Re560/coarse",
      reynolds_number: 560,
      mesh_resolution: "coarse",
      solution_time: "200",
      cell_count: 1000,
      sample_count: 500,
      bounds: [[0, 1], [0, 1], [0, 1]],
    },
  ],
  dataset: {
    case_count: 6,
    reynolds_numbers: [560, 11240],
    mesh_resolutions: ["coarse", "fine", "medium"],
    single_geometry: true,
    exploratory: true,
  },
  split: {
    method: "spatial_block_holdout_with_two_bin_buffer",
    axis: "x",
    training_sample_count: 100,
    validation_sample_count: 50,
    excluded_buffer_sample_count: 30,
    training_cases: ["Re560/coarse"],
    validation_cases: ["Re560/coarse"],
    validation_kind: "spatial holdout; not independent-case or new-design validation",
  },
  metrics: {
    baseline_mean_predictor: { mae: 1.2, rmse: 1.4, r2: -0.1 },
    extra_trees_spatial_validation: { mae: 0.6, rmse: 0.8, r2: 0.7 },
    metric_scope: "held-out spatial validation samples only",
    validation_sample_count: 50,
  },
  explainability: {
    shap_global: {
      method: "TreeExplainer",
      baseline_definition: "Mean model output over training background.",
      sample_count: 20,
      additivity_checked: true,
      max_absolute_additivity_error: 0,
      features: [
        { feature: "x", mean_absolute_shap: 0.5, mean_shap: 0.1 },
        { feature: "reynolds_number", mean_absolute_shap: 0.2, mean_shap: -0.1 },
      ],
    },
    lime: {
      method: "LimeTabularExplainer",
      approximation_quality_metric: "local R-squared",
    },
  },
  limitations: ["Exploratory test fixture."],
};

describe("ExplainableAiLab", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    window.history.replaceState({}, "", "/explainable.html");
    api.getSurrogateModels.mockResolvedValue({
      models: [model],
      status: "available",
      empty_reason: null,
    });
    api.getGptOssStatus.mockResolvedValue({
      provider: "GPT-OSS",
      configured: false,
      model: null,
      status: "not_verified",
      diagnostic: "Not verified.",
    });
    api.getGlobalShap.mockResolvedValue({
      model_id: model.model_id,
      target: model.target,
      baseline_definition: "Mean output from stored training background.",
      sample_count: 20,
      additivity_checked: true,
      features: model.explainability.shap_global.features,
    });
    api.predictField.mockResolvedValue({
      model_id: model.model_id,
      case_id: "Re560/coarse",
      target: model.target,
      coordinates: { x: 0.5, y: 0.5, z: 0.5, units: null },
      reynolds_number: 560,
      prediction: 291.25,
      units: "K",
      status: "exploratory",
      limitations: model.limitations,
    });
    api.getLocalExplanation.mockResolvedValue({
      model_id: model.model_id,
      case_id: "Re560/coarse",
      target: model.target,
      explanation: {
        method: "SHAP TreeExplainer",
        prediction: 291.25,
        baseline: 290.5,
        baseline_definition: "Mean output from stored training background.",
        reconstructed_prediction: 291.25,
        additivity_error: 0,
        additivity_checked: true,
        additivity_passed: true,
        contributions: [
          { feature: "x", value: 0.5, contribution: 0.75 },
          { feature: "reynolds_number", value: 560, contribution: 0 },
        ],
        local_r2: 0.82,
        local_prediction: 291.2,
        caveat: "Local approximation only.",
      },
    });
    api.testGptOssConnection.mockResolvedValue({
      provider: "GPT-OSS",
      status: "connected",
      model: "openai/gpt-oss-20b",
      tested_at: "2026-10-18T00:00:00+00:00",
      diagnostic: "Live test completed.",
    });
    api.interpretWithGptOss.mockResolvedValue({
      explanation_method: "shap",
      interpretation: {
        label: "AI-generated interpretation",
        model: "openai/gpt-oss-20b",
        interpretation: "This is an exploratory model output.",
        generated_at: "2026-10-18T00:00:00+00:00",
      },
    });
  });

  it("shows an honest empty state before a model is trained", async () => {
    api.getSurrogateModels.mockResolvedValue({
      models: [],
      status: "unavailable",
      empty_reason: "No trained surrogate is saved.",
    });
    render(<ExplainableAiLab />);

    expect(
      await screen.findByRole("heading", {
        name: "No current trained model is available",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/python -m backend\.scripts\.train_surrogate/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run Prediction" })).not.toBeInTheDocument();
  });

  it("uses backend predictions and local SHAP/LIME explanations", async () => {
    render(<ExplainableAiLab />);

    await screen.findByRole("combobox", { name: "Verified case" });
    fireEvent.click(screen.getByRole("button", { name: "Run Prediction" }));
    expect(await screen.findByText("291.25")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Calculate SHAP for this prediction" }));
    await waitFor(() => expect(api.getLocalExplanation).toHaveBeenCalledWith(
      "shap",
      expect.objectContaining({
        model_id: model.model_id,
        case_id: "Re560/coarse",
        coordinates: { x: 0.5, y: 0.5, z: 0.5 },
      }),
    ));
    expect(await screen.findByText("Effects for this prediction")).toBeInTheDocument();
    expect(screen.getByText("+0.75")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Calculate LIME explanation" }));
    await waitFor(() => expect(api.getLocalExplanation).toHaveBeenCalledWith(
      "lime",
      expect.objectContaining({ model_id: model.model_id }),
    ));
    fireEvent.click(screen.getByRole("button", { name: "View LIME details" }));
    expect(await screen.findByText(/Local R²: 0.82/)).toBeInTheDocument();
  });

  it("shows backend errors and never fabricates a prediction or interpretation", async () => {
    api.getGptOssStatus.mockResolvedValue({
      provider: "GPT-OSS",
      configured: true,
      model: "openai/gpt-oss-20b",
      status: "reachable",
      diagnostic: "Service reachable.",
    });
    api.predictField.mockRejectedValue(new Error("Coordinate x is outside the verified bounds."));
    api.testGptOssConnection.mockRejectedValue(new Error("Live GPT-OSS check failed."));
    render(<ExplainableAiLab />);

    await screen.findByRole("button", { name: "Run Prediction" });
    fireEvent.click(screen.getByRole("button", { name: "Run Prediction" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Coordinate x is outside the verified bounds.",
    );
    expect(screen.queryByRole("heading", { name: /Predicted coolant\/T/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Test GPT-OSS Connection" }));
    expect(await screen.findByText("Live GPT-OSS check failed.")).toBeInTheDocument();
    expect(screen.getByText(/GPT-OSS: not tested/)).toBeInTheDocument();
  });

  it("opens accessible disclosures without requesting an explanation", async () => {
    render(<ExplainableAiLab />);

    await screen.findByRole("button", { name: "Run Prediction" });
    const disclosure = screen.getByRole("button", { name: "What is Explainable AI?" });
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(disclosure).toHaveAttribute("aria-controls", "xai-meaning");
    fireEvent.click(disclosure);
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(api.getLocalExplanation).not.toHaveBeenCalled();
  });

  it("only requests a plain-English interpretation after a real prediction and attribution", async () => {
    api.getGptOssStatus.mockResolvedValue({
      configured: true,
      model: "openai/gpt-oss-20b",
      provider: "GPT-OSS",
      status: "reachable",
      diagnostic: "Service reachable.",
    });
    render(<ExplainableAiLab />);

    fireEvent.click(await screen.findByRole("button", { name: "Run Prediction" }));
    await screen.findByText("291.25");
    expect(screen.getByRole("button", { name: "Explain Prediction" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Calculate SHAP for this prediction" }));
    await screen.findByText("Effects for this prediction");
    const explain = screen.getByRole("button", { name: "Explain Prediction" });
    expect(explain).toBeEnabled();
    fireEvent.click(explain);

    await waitFor(() => expect(api.interpretWithGptOss).toHaveBeenCalledWith(
      "shap",
      expect.objectContaining({ model_id: model.model_id }),
    ));
    expect(await screen.findByText("This is an exploratory model output.")).toBeInTheDocument();
  });
});
