import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CfdExplorer } from "./CfdExplorer";

const api = vi.hoisted(() => ({
  analyzeWithGptOss: vi.fn(),
  compareMeshes: vi.fn(),
  getCfdFieldOptions: vi.fn(),
  getCfdStatistics: vi.fn(),
  getGptOssStatus: vi.fn(),
  preprocessCsv: vi.fn(),
  testGptOssConnection: vi.fn(),
}));

vi.mock("../services/datasetApi", () => api);
vi.mock("../services/providerApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../services/providerApi")>()),
  getGptOssStatus: api.getGptOssStatus,
  testGptOssConnection: api.testGptOssConnection,
}));

const cases = [
  {
    case_id: "Re560/coarse",
    files: [],
    fields: [],
    mesh_info: null,
    mesh_info_by_file: [],
    metadata: { reynolds_number: 560, mesh_resolution: "coarse", units: null },
  },
  {
    case_id: "Re560/fine",
    files: [],
    fields: [],
    mesh_info: null,
    mesh_info_by_file: [],
    metadata: { reynolds_number: 560, mesh_resolution: "fine", units: null },
  },
  {
    case_id: "Re11240/coarse",
    files: [],
    fields: [],
    mesh_info: null,
    mesh_info_by_file: [],
    metadata: { reynolds_number: 11240, mesh_resolution: "coarse", units: null },
  },
];

const options = {
  time: "200",
  processor_count: 12,
  fields: [
    {
      name: "T",
      region: "coolant",
      field_class: "volScalarField",
      value_type: "scalar" as const,
      association: "cell",
      dimensions: "0 0 0 1 0 0 0",
      units: "K",
      components: ["value"],
      processor_count: 12,
      statistics_supported: true,
    },
  ],
  visualization: {
    status: "unavailable" as const,
    reason: "A validated physical field mapping is unavailable.",
  },
};

const statistics = {
  case_id: "Re560/coarse",
  reynolds_number: 560,
  mesh_resolution: "coarse",
  region: "coolant",
  field: "T",
  field_class: "volScalarField",
  component: "value",
  association: "cell",
  dimensions: "0 0 0 1 0 0 0",
  units: "K",
  time: "200",
  processor_count: 12,
  sample_count: 3,
  statistic_scope: "all internal cell values across processor partitions",
  minimum: 300,
  maximum: 302,
  mean: 301,
  visualization: {
    status: "unavailable" as const,
    reason: "A validated physical field mapping is unavailable.",
  },
};

function makeDataset(caseList = cases) {
  return {
    scan: null,
    cases: caseList,
    preprocessing: null,
    audit: { status: "unavailable", report: null },
    auditError: null,
    error: null,
    isLoading: false,
    isScanning: false,
    isLoadingAudit: false,
    refresh: vi.fn(),
    runScan: vi.fn(),
    loadAudit: vi.fn(),
  };
}

describe("CfdExplorer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getCfdFieldOptions.mockResolvedValue(options);
    api.getCfdStatistics.mockResolvedValue(statistics);
    api.getGptOssStatus.mockResolvedValue({ configured: false, model: null });
    api.testGptOssConnection.mockResolvedValue({
      provider: "GPT-OSS",
      status: "connected",
      model: "model",
      tested_at: "2026-10-18T00:00:00+00:00",
      diagnostic: "Live test completed.",
    });
    api.analyzeWithGptOss.mockResolvedValue({
      statistics,
      interpretation: {
        label: "AI-generated interpretation",
        model: "model",
        interpretation: "The verified temperature values are close together.",
        generated_at: "2026-10-18T00:00:00+00:00",
      },
    });
    api.compareMeshes.mockResolvedValue({
      status: "unavailable",
      reason: "Spatial mapping is unavailable.",
    });
  });

  it("shows an honest empty state when there are no discovered cases", () => {
    render(<CfdExplorer dataset={makeDataset([])} />);

    expect(screen.getByRole("heading", { name: "No scanned CFD cases are available" })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "CFD case" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "GPT-OSS CFD interpretation" })).toBeInTheDocument();
  });

  it("populates selectors from scanned cases and changes case with Reynolds selection", async () => {
    render(<CfdExplorer dataset={makeDataset()} />);

    const reSelector = screen.getByRole("combobox", { name: "Reynolds number" });
    await waitFor(() => expect(reSelector).toHaveValue("560"));
    expect(screen.getByRole("combobox", { name: "Mesh resolution" })).toHaveValue("coarse");
    await waitFor(() =>
      expect(screen.getByRole("combobox", { name: "CFD field" })).toHaveDisplayValue(
        "coolant / T · volScalarField",
      ),
    );
    expect(await screen.findByText("301")).toBeInTheDocument();

    fireEvent.change(reSelector, { target: { value: "11240" } });
    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: "CFD case" })).toHaveValue(
        "Re11240/coarse",
      );
    });
    expect(api.getCfdFieldOptions).toHaveBeenCalledWith("Re11240/coarse");
  });

  it("shows field-loading API errors without replacing them with fabricated fields", async () => {
    api.getCfdFieldOptions.mockRejectedValueOnce(new Error("Could not read scanned fields."));
    render(<CfdExplorer dataset={makeDataset()} />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Could not read scanned fields.",
    );
    expect(screen.getByRole("combobox", { name: "CFD field" })).toBeDisabled();
  });

  it("loads the large audit report only when its disclosure is opened", async () => {
    const dataset = makeDataset();
    render(<CfdExplorer dataset={dataset} />);

    expect(dataset.loadAudit).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText("Data audit report"));
    expect(dataset.loadAudit).toHaveBeenCalledOnce();
  });

  it("surfaces GPT-OSS connection and analysis errors", async () => {
    api.getGptOssStatus.mockResolvedValue({
      provider: "GPT-OSS",
      configured: true,
      model: "test-model",
      status: "reachable",
      diagnostic: "Service reachable.",
    });
    api.testGptOssConnection.mockRejectedValue(new Error("Live GPT-OSS test failed."));
    api.analyzeWithGptOss.mockRejectedValue(new Error("GPT-OSS analysis failed."));
    render(<CfdExplorer dataset={makeDataset()} />);

    fireEvent.click(await screen.findByRole("button", { name: "Test GPT-OSS Inference" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Live GPT-OSS test failed.");
    expect(screen.getByText(/GPT-OSS: not tested/)).toBeInTheDocument();
    expect(screen.getByText(/provider test was not completed/i)).toBeInTheDocument();

    const analyzeButton = screen.getByRole("button", {
      name: "Interpret selected statistics",
    });
    await waitFor(() => expect(analyzeButton).toBeEnabled());
    fireEvent.click(analyzeButton);
    await waitFor(() =>
      expect(screen.getByText("GPT-OSS analysis failed.")).toBeInTheDocument(),
    );
    expect(api.analyzeWithGptOss).toHaveBeenCalledWith({
      case_id: "Re560/coarse",
      region: "coolant",
      field: "T",
      component: "value",
    });
  });
});
