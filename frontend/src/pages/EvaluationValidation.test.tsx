import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EvaluationValidation } from "./EvaluationValidation";

const api = vi.hoisted(() => ({
  getGptOssStatus: vi.fn(),
  testGptOssConnection: vi.fn(),
  getGenerativeStatus: vi.fn(),
  getKnowledgeStatus: vi.fn(),
  getSurrogateModels: vi.fn(),
}));

vi.mock("../services/providerApi", () => ({
  getGptOssStatus: api.getGptOssStatus,
  testGptOssConnection: api.testGptOssConnection,
}));
vi.mock("../services/generativeApi", () => ({ getGenerativeStatus: api.getGenerativeStatus }));
vi.mock("../services/knowledgeApi", () => ({ getKnowledgeStatus: api.getKnowledgeStatus }));
vi.mock("../services/mlApi", () => ({
  getSurrogateModels: api.getSurrogateModels,
}));

const dataset = {
  scan: {
    status: "scanned",
    generated_at: "2026-10-18T00:00:00Z",
    source_kind: "directory",
    summary: { file_count: 100, candidate_count: 6, total_bytes: 1000 },
    warnings: [],
  },
  cases: Array.from({ length: 6 }, (_, index) => ({
    case_id: `case-${index + 1}`,
    files: [],
    fields: [],
    mesh_info: null,
    mesh_info_by_file: [],
    metadata: { reynolds_number: null, mesh_resolution: null, units: null },
  })),
  preprocessing: null,
  audit: { status: "available", report: null },
  error: null,
  isLoading: false,
  isScanning: false,
  auditError: null,
  isLoadingAudit: false,
  refresh: vi.fn(),
  runScan: vi.fn(),
  loadAudit: vi.fn(),
};

describe("EvaluationValidation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getSurrogateModels.mockResolvedValue({
      models: [],
      status: "unavailable",
      empty_reason: "No current trained surrogate is saved.",
    });
    api.getGenerativeStatus.mockResolvedValue({
      dataset_status: "available",
      cases: [],
      models: [],
      torch_available: false,
      status: "available",
      empty_reason: "No trained VAE or GAN model is saved.",
    });
    api.getKnowledgeStatus.mockResolvedValue({
      corpus: {
        indexing_status: "empty",
        corpus_version: null,
        indexed_at: null,
        document_count: 0,
        passage_count: 0,
        documents: [],
        warnings: [],
      },
      retrieval: {
        method: "TF-IDF test retrieval",
        minimum_cosine_similarity: 0.05,
        chunk_characters: 1200,
        chunk_overlap_characters: 200,
      },
      bert: {
        status: "not_configured",
        label_set: [],
        held_out_test_size: 0,
        reason: "No classifier configured.",
      },
      gpt_oss: {
        provider: "GPT-OSS",
        model: null,
        configured: false,
        status: "not_configured",
        diagnostic: "Not configured.",
      },
    });
    api.getGptOssStatus.mockResolvedValue({
      provider: "GPT-OSS",
      configured: true,
      model: "openai/gpt-oss-120b",
      status: "reachable",
      checked_at: "2026-10-19T00:00:00Z",
      diagnostic: "Service reachable.",
    });
    api.testGptOssConnection.mockResolvedValue({
      provider: "GPT-OSS",
      status: "connected",
      model: "openai/gpt-oss-120b",
      tested_at: "2026-10-19T00:00:00Z",
      diagnostic: "Connection test succeeded.",
      latency_ms: 120,
    });
  });

  it("shows actual unavailable states instead of creating evaluation metrics", async () => {
    render(<EvaluationValidation dataset={dataset} />);

    expect(await screen.findByText(/Dataset: 6 scanned cases/)).toBeInTheDocument();
    expect(screen.getByText("Full suite: not reported")).toBeInTheDocument();
    expect(screen.getByText(/No current surrogate evaluation result/)).toBeInTheDocument();
    expect(screen.getByText(/No current VAE result/)).toBeInTheDocument();
    expect(screen.queryByText(/MAE · lower is better/)).not.toBeInTheDocument();
  });

  it("surfaces status-request errors", async () => {
    api.getSurrogateModels.mockRejectedValue(new Error("Model inventory unavailable."));
    render(<EvaluationValidation dataset={dataset} />);

    const surrogateSection = screen.getByRole("heading", { name: "2. Surrogate Model Evaluation" }).closest("section");
    expect(surrogateSection).toHaveTextContent("Unavailable");
    fireEvent.click(surrogateSection?.querySelector(".validation-card__toggle")!);
    expect(await screen.findByText("Model inventory unavailable.")).toHaveAttribute("role", "alert");
  });

  it("keeps each validation area collapsed and expands only the selected section", async () => {
    render(<EvaluationValidation dataset={dataset} />);
    await screen.findByRole("heading", { name: "Evaluation & Validation" });
    const sectionTitles = [
      "1. CFD Numerical Consistency",
      "2. Surrogate Model Evaluation",
      "3. SHAP & LIME Interpretability",
      "4. Generative AI Fidelity & Physics",
      "5. NLP Intent & Citation Grounding",
      "6. GPT-OSS Provider Status",
    ];

    for (const title of sectionTitles) {
      const heading = screen.getByRole("heading", { name: title });
      const section = heading.closest("section");
      const toggle = section?.querySelector(".validation-card__toggle");
      expect(toggle).toHaveAttribute("aria-expanded", "false");
    }

    const cfdSection = screen.getByRole("heading", { name: sectionTitles[0] }).closest("section");
    const cfdToggle = cfdSection?.querySelector(".validation-card__toggle");
    fireEvent.click(cfdToggle!);
    expect(cfdToggle).toHaveAttribute("aria-expanded", "true");
    const surrogateSection = screen.getByRole("heading", { name: sectionTitles[1] }).closest("section");
    expect(surrogateSection?.querySelector(".validation-card__toggle")).toHaveAttribute("aria-expanded", "false");
  });

  it("loads the audit report only when audit details are opened", async () => {
    render(<EvaluationValidation dataset={dataset} />);

    expect(dataset.loadAudit).not.toHaveBeenCalled();
    fireEvent.click(await screen.findByRole("button", { name: "Audit Details" }));
    expect(dataset.loadAudit).toHaveBeenCalledOnce();
  });

  it("reports model reachability separately from measured inference tests", async () => {
    render(<EvaluationValidation dataset={dataset} />);
    const providerSection = screen.getByRole("heading", { name: "6. GPT-OSS Provider Status" }).closest("section");
    fireEvent.click(providerSection?.querySelector(".validation-card__toggle")!);

    expect((await screen.findAllByText("Reachable")).length).toBeGreaterThanOrEqual(1);
    fireEvent.click(screen.getByRole("button", { name: "Toggle GPT-OSS details" }));
    fireEvent.click(screen.getByRole("button", { name: "Test connection" }));

    expect(await screen.findByText("120 ms")).toBeInTheDocument();
    expect(api.testGptOssConnection).toHaveBeenCalledOnce();
  });

  it("exports the current report through the browser PDF print workflow", () => {
    const print = vi.spyOn(window, "print").mockImplementation(() => undefined);
    render(<EvaluationValidation dataset={dataset} />);

    fireEvent.click(screen.getByRole("button", { name: "Export Audit Report (PDF)" }));
    expect(print).toHaveBeenCalledOnce();
  });
});
