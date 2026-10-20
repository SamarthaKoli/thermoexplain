import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GenerativeAiLab } from "./GenerativeAiLab";
import type {
  GenerativeModel,
  GenerativeSample,
  GenerativeStatus,
  PatchResult,
} from "../types/generative";

const api = vi.hoisted(() => ({
  generatePatch: vi.fn(),
  getGenerativeSamples: vi.fn(),
  getGenerativeStatus: vi.fn(),
  reconstructPatch: vi.fn(),
  trainGenerativeModel: vi.fn(),
}));

vi.mock("../services/generativeApi", () => api);

const caseRecord = {
  case_id: "Re560/coarse",
  reynolds_number: 560,
  mesh_resolution: "coarse",
  solution_time: "200",
  target: {
    region: "coolant",
    field: "T",
    units: "K",
    field_class: "volScalarField",
  },
};

const sample: GenerativeSample = {
  sample_id: "x24_y0_z0",
  case_id: "Re560/coarse",
  reynolds_number: 560,
  mesh_resolution: "coarse",
  solution_time: "200",
  patch_index: [24, 0, 0],
  patch_center: [0.8, 0.1, 0.1],
  patch_center_units: null,
  valid_voxel_count: 400,
  split: "validation",
};

function makeModel(modelType: "vae" | "gan"): GenerativeModel {
  return {
    schema_version: 1,
    model_id: `${modelType}_re560_coarse_0123456789`,
    model_type: modelType,
    status: "trained_exploratory",
    available: true,
    created_at: "2026-10-18T00:00:00Z",
    source: {
      case_id: caseRecord.case_id,
      reynolds_number: caseRecord.reynolds_number,
      mesh_resolution: caseRecord.mesh_resolution,
      solution_time: caseRecord.solution_time,
      target: caseRecord.target,
      cell_count: 1000,
      manifest_sha256: "verified-test-digest",
    },
    representation: {
      method: "verified test patch representation",
      interpolation: "none",
      mask: "outside cells excluded",
      grid_size: [32, 32, 32],
      patch_size: [8, 8, 8],
      stride: 4,
      coordinate_units: null,
      limitations: ["test fixture"],
    },
    split: {
      method: "spatial slab with buffer",
      patch_counts: { train: 63, validation: 49, buffer: 196 },
    },
    configuration: {
      seed: 42,
      epochs: 1,
      device: "cpu",
      latent_dimension: 16,
      hidden_dimension: 256,
      patch_shape: [8, 8, 8],
      normalization: {
        method: "training valid voxels",
        mean_K: 290,
        standard_deviation_K: 1,
      },
      vae_beta: modelType === "vae" ? 0.001 : null,
    },
    metrics:
      modelType === "vae"
        ? { mae_K: 0.4, rmse_K: 0.7 }
        : {
            temperature_wasserstein_distance_K: 0.8,
            mean_neighbor_jump_difference_K_per_voxel: 0.12,
            positive_kelvin_fraction: 1,
            positivity_passed: true,
          },
    training_history: [{ epoch: 1, training_loss: 0.1 }],
    training_history_summary: {
      record_count: 1,
      final: { epoch: 1, training_loss: 0.1 },
    },
    limitations: ["single-case test model"],
  };
}

function makeStatus(
  models: GenerativeModel[] = [],
  torchAvailable = false,
): GenerativeStatus {
  return {
    dataset_status: "available",
    cases: [caseRecord],
    models,
    torch_available: torchAvailable,
    training_device: "CPU (bounded pilot)",
    manifest_sha256: "verified-test-digest",
    status: "available",
    empty_reason: null,
  };
}

function makeVolume(value: number) {
  return Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () =>
      Array.from({ length: 8 }, () => value),
    ),
  );
}

describe("GenerativeAiLab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.replaceState({}, "", "/");
    api.getGenerativeStatus.mockResolvedValue(makeStatus());
    api.getGenerativeSamples.mockResolvedValue({
      model_id: makeModel("vae").model_id,
      model_type: "vae",
      samples: [sample],
    });
    api.trainGenerativeModel.mockResolvedValue({ model: makeModel("vae") });
  });

  afterEach(() => {
    window.history.replaceState({}, "", "/");
  });

  it("shows the real dataset and an honest unavailable state without PyTorch", async () => {
    render(<GenerativeAiLab />);

    expect(await screen.findByRole("option", { name: /Re560\/coarse/ })).toBeInTheDocument();
    expect(screen.getByText(/PyTorch is not installed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Train VAE" })).toBeDisabled();
    expect(screen.getByText(/No model or training metrics are being substituted/)).toBeInTheDocument();
  });

  it("uses actual saved model metadata and renders real VAE reconstruction output", async () => {
    const model = makeModel("vae");
    api.getGenerativeStatus.mockResolvedValue(makeStatus([model], true));
    const result: PatchResult = {
      label: "VAE reconstruction of a held-out real CFD patch",
      sample,
      target: caseRecord.target,
      values_K: {
        original: makeVolume(290),
        reconstructed: makeVolume(289.5),
        difference: makeVolume(-0.5),
      },
      mask: Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => Array(8).fill(true))),
      metrics: { mae_K: 0.5, rmse_K: 0.5 },
      status: "exploratory",
    };
    api.reconstructPatch.mockResolvedValue(result);
    render(<GenerativeAiLab />);

    const reconstructButton = await screen.findByRole("button", { name: "Reconstruct held-out patch" });
    await waitFor(() => expect(reconstructButton).toBeEnabled());
    fireEvent.click(reconstructButton);
    expect(await screen.findByText("VAE reconstruction of a held-out real CFD patch")).toBeInTheDocument();
    expect(await screen.findByText(/Patch MAE/)).toBeInTheDocument();
    expect(api.reconstructPatch).toHaveBeenCalledWith({
      model_id: model.model_id,
      sample_id: sample.sample_id,
    });
  });

  it("requires a trained GAN, labels generated output synthetic, and shows measured comparison", async () => {
    const model = makeModel("gan");
    api.getGenerativeStatus.mockResolvedValue(makeStatus([model], true));
    api.getGenerativeSamples.mockResolvedValue({
      model_id: model.model_id,
      model_type: "gan",
      samples: [sample],
    });
    api.generatePatch.mockResolvedValue({
      label: "Synthetic GAN-generated field patch (not a CFD simulation)",
      sample,
      target: caseRecord.target,
      values_K: makeVolume(291),
      mask: Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => Array(8).fill(true))),
      metrics: {
        temperature_wasserstein_distance_K: 0.8,
        mean_neighbor_jump_difference_K_per_voxel: 0.12,
        positivity_passed: true,
      },
      generation_seed: 42,
      status: "synthetic_exploratory",
      limitations: model.limitations,
    });
    render(<GenerativeAiLab />);

    fireEvent.click(await screen.findByRole("tab", { name: "GAN field generation" }));
    const generateButton = await screen.findByRole("button", { name: "Generate synthetic patch" });
    await waitFor(() => expect(generateButton).toBeEnabled());
    fireEvent.click(generateButton);

    expect(await screen.findByText(/not a CFD simulation/)).toBeInTheDocument();
    expect(screen.getByText(/Similarity statistics compare generated values/)).toBeInTheDocument();
    expect(api.generatePatch).toHaveBeenCalledWith({
      model_id: model.model_id,
      sample_id: sample.sample_id,
      seed: 42,
    });
  });

  it("shows training and inference API failures without producing a result", async () => {
    api.getGenerativeStatus.mockResolvedValue(makeStatus([], true));
    api.trainGenerativeModel.mockRejectedValue(new Error("PyTorch could not allocate memory."));
    render(<GenerativeAiLab />);

    fireEvent.click(await screen.findByRole("button", { name: "Train VAE" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("PyTorch could not allocate memory.");
    expect(screen.queryByRole("heading", { name: "Held-out evaluation" })).not.toBeInTheDocument();
    await waitFor(() => expect(api.trainGenerativeModel).toHaveBeenCalled());
  });

  it("renders the compact Stitch VAE view with real reconstruction results and collapsed details", async () => {
    window.history.replaceState({}, "", "/generative.html");
    const model = makeModel("vae");
    api.getGenerativeStatus.mockResolvedValue(makeStatus([model], true));
    const difference = makeVolume(-0.5);
    difference[0][0][0] = 0.5;
    api.reconstructPatch.mockResolvedValue({
      label: "VAE reconstruction of a held-out real CFD patch",
      sample,
      target: caseRecord.target,
      values_K: {
        original: makeVolume(290),
        reconstructed: makeVolume(289.5),
        difference,
      },
      mask: Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => Array(8).fill(true))),
      metrics: { mae_K: 0.5, rmse_K: 0.5 },
      status: "exploratory",
    });
    render(<GenerativeAiLab />);

    expect(await screen.findByRole("heading", { name: "VAE Thermal Field Reconstruction" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run Reconstruction" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Train VAE" })).not.toBeInTheDocument();
    const advanced = screen.getByRole("button", { name: "Advanced reconstruction details" });
    expect(advanced).toHaveAttribute("aria-expanded", "false");
    expect(document.getElementById("gen-advanced-controls")).toHaveAttribute("hidden");

    fireEvent.click(screen.getByRole("button", { name: "Run Reconstruction" }));
    expect(await screen.findByRole("img", { name: "Original CFD field, middle z slice" })).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "VAE reconstruction, middle z slice" })).toBeInTheDocument();
    const residual = screen.getByRole("img", { name: "Absolute residual / error, middle z slice" });
    expect(residual).toBeInTheDocument();
    expect(residual.children[0].getAttribute("style")).toBe(residual.children[1].getAttribute("style"));
    expect(screen.getAllByText("0.5 K").length).toBeGreaterThanOrEqual(1);
    expect(api.reconstructPatch).toHaveBeenCalledWith({
      model_id: model.model_id,
      sample_id: sample.sample_id,
    });

    fireEvent.click(screen.getByRole("button", { name: "Reconstruction metrics" }));
    expect(screen.getByRole("button", { name: "Reconstruction metrics" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("RMSE")).toBeInTheDocument();
  });

  it("uses real GAN sample output and reports unsupported comparison checks honestly", async () => {
    window.history.replaceState({}, "", "/generative.html");
    const model = makeModel("gan");
    api.getGenerativeStatus.mockResolvedValue(makeStatus([model], true));
    api.getGenerativeSamples.mockResolvedValue({
      model_id: model.model_id,
      model_type: "gan",
      samples: [sample],
    });
    api.generatePatch.mockResolvedValue({
      label: "Synthetic GAN-generated field patch (not a CFD simulation)",
      sample,
      target: caseRecord.target,
      values_K: makeVolume(291),
      mask: Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => Array(8).fill(true))),
      metrics: {
        generated_mean_temperature_K: 291,
        generated_temperature_std_K: 0.2,
        temperature_wasserstein_distance_K: 0.8,
        positivity_passed: true,
      },
      generation_seed: 42,
      status: "synthetic_exploratory",
      limitations: model.limitations,
    });
    render(<GenerativeAiLab />);

    fireEvent.click(await screen.findByRole("tab", { name: "GAN field generation" }));
    expect(await screen.findByRole("heading", { name: "GAN Thermal Field Generation" })).toBeInTheDocument();
    expect(screen.getByRole("spinbutton", { name: "Noise seed" })).toHaveValue(42);
    expect(screen.queryByLabelText(/truncation/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Generate Sample" }));

    expect(await screen.findByRole("img", { name: "GAN-generated field, middle z slice" })).toBeInTheDocument();
    expect(screen.getByText("The generation response includes the real sample’s mask and aggregate comparison metrics, but not its temperature field. No reference visualization is available from this endpoint.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Statistical and physics checks" })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(screen.getByRole("button", { name: "Statistical and physics checks" }));
    expect(screen.getAllByText("Not evaluated").length).toBeGreaterThanOrEqual(1);
    expect(api.generatePatch).toHaveBeenCalledWith({
      model_id: model.model_id,
      sample_id: sample.sample_id,
      seed: 42,
    });
  });
});
