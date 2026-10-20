import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ThermalKnowledgeAssistant } from "./ThermalKnowledgeAssistant";
import type {
  KnowledgeAnswer,
  KnowledgeStatus,
  RetrievalResult,
} from "../types/knowledge";

const api = vi.hoisted(() => ({
  buildKnowledgeIndex: vi.fn(),
  classifyQuestion: vi.fn(),
  evaluateBert: vi.fn(),
  getKnowledgeAnswer: vi.fn(),
  getKnowledgeStatus: vi.fn(),
  retrievePassages: vi.fn(),
  testKnowledgeConnection: vi.fn(),
  uploadKnowledgeDocument: vi.fn(),
}));

vi.mock("../services/knowledgeApi", () => api);

const status: KnowledgeStatus = {
  corpus: {
    indexing_status: "indexed",
    corpus_version: "verified-fixture-corpus-version",
    indexed_at: "2026-10-18T00:00:00Z",
    document_count: 1,
    passage_count: 1,
    documents: [],
    warnings: [],
  },
  retrieval: {
    method: "TF-IDF with unigram/bigram cosine similarity",
    minimum_cosine_similarity: 0.05,
    chunk_characters: 1200,
    chunk_overlap_characters: 200,
  },
  bert: {
    status: "not_configured",
    label_set: [],
    held_out_test_size: 0,
    reason: "No verified BERT classifier or held-out labeled set is configured.",
  },
  gpt_oss: {
    provider: "GPT-OSS",
    model: null,
    configured: false,
    status: "not_configured",
    diagnostic: "GPT-OSS is not configured.",
  },
};

const retrieved: RetrievalResult = {
  query: "How does coolant remove heat?",
  corpus_version: status.corpus.corpus_version!,
  passages: [
    {
      passage_id: "p-test-source-1",
      document_id: "test-source",
      title: "Cooling test fixture",
      file_name: "cooling-test.md",
      source_url: "https://example.org/test-source",
      page: null,
      section: "Cooling",
      start_character: 0,
      end_character: 44,
      text: "Coolant flow transfers heat from battery cells.",
      relevance_score: 0.42,
    },
  ],
  empty_reason: null,
};

const answer: KnowledgeAnswer = {
  label: "AI-generated answer grounded in retrieved passages",
  model: "openai/gpt-oss-120b",
  answer: "The test source says coolant flow transfers heat from cells.",
  citations: ["p-test-source-1"],
  insufficient_evidence: false,
};

describe("ThermalKnowledgeAssistant (mocked API unit tests)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getKnowledgeStatus.mockResolvedValue(status);
    api.retrievePassages.mockResolvedValue(retrieved);
    api.getKnowledgeAnswer.mockResolvedValue(answer);
    api.testKnowledgeConnection.mockResolvedValue({
      provider: "GPT-OSS",
      model: "openai/gpt-oss-120b",
      status: "connected",
      tested_at: "2026-10-18T00:00:00Z",
      diagnostic: "Live test completed.",
    });
    api.classifyQuestion.mockResolvedValue({
      classifier: "BERT",
      predicted_label: "test-category",
      confidence: 0.9,
      confidence_method: "softmax score; not calibrated",
      label_set: ["test-category"],
      status: "inference",
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows an honest empty classifier configuration without a prediction", async () => {
    render(<ThermalKnowledgeAssistant />);

    fireEvent.click(await screen.findByRole("button", { name: "System health and model details" }));
    expect((await screen.findAllByText("Not configured")).length).toBeGreaterThan(0);
    expect(
      screen.getByText(/No verified BERT classifier or held-out labeled set/),
    ).toBeInTheDocument();
    expect(api.classifyQuestion).not.toHaveBeenCalled();
  });

  it("displays retrieved passages and maps an answer citation to its source card", async () => {
    render(<ThermalKnowledgeAssistant />);
    await screen.findByRole("button", { name: "Submit Question" });
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "How does coolant remove heat?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit Question" }));

    expect(
      await screen.findByText(
        "The test source says coolant flow transfers heat from cells.",
      ),
    ).toBeInTheDocument();
    expect((await screen.findAllByText("Coolant flow transfers heat from battery cells.")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Source details" }));
    expect(screen.getByRole("link", { name: "[1]" })).toHaveAttribute(
      "href",
      "#source-p-test-source-1",
    );
    expect(screen.getByText("p-test-source-1").closest("article")).toHaveTextContent(
      /Similarity 0\.420/,
    );
    expect(api.getKnowledgeAnswer).toHaveBeenCalledWith({
      question: "How does coolant remove heat?",
      passage_ids: ["p-test-source-1"],
    });
  });

  it("does not call the LLM when retrieval returns no useful evidence", async () => {
    api.retrievePassages.mockResolvedValue({
      ...retrieved,
      passages: [],
      empty_reason: "No passage met the similarity threshold.",
    });
    render(<ThermalKnowledgeAssistant />);
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "A question without sources" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit Question" }));

    expect(await screen.findByText(/Insufficient evidence/)).toBeInTheDocument();
    expect(api.getKnowledgeAnswer).not.toHaveBeenCalled();
    expect(screen.queryByText("AI-generated answer grounded in retrieved passages")).not.toBeInTheDocument();
  });

  it("keeps real retrieved sources visible when answer generation fails", async () => {
    api.getKnowledgeAnswer.mockRejectedValue(
      new Error("GPT-OSS request failed. No interpretation was generated."),
    );
    render(<ThermalKnowledgeAssistant />);
    await screen.findByRole("button", { name: "Submit Question" });
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "How does coolant remove heat?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit Question" }));

    expect(
      await screen.findByText(/GPT-OSS request failed\. No interpretation was generated\./),
    ).toBeInTheDocument();
    expect((await screen.findAllByText("Coolant flow transfers heat from battery cells.")).length).toBeGreaterThan(0);
  });

  it("reports retrieval failure and supports an empty corpus state", async () => {
    api.getKnowledgeStatus.mockResolvedValue({
      ...status,
      corpus: {
        ...status.corpus,
        indexing_status: "empty",
        document_count: 0,
        passage_count: 0,
        documents: [],
        corpus_version: "empty-corpus-fixture",
      },
    });
    api.retrievePassages.mockRejectedValue(new Error("The knowledge index has not been built yet."));
    render(<ThermalKnowledgeAssistant />);
    fireEvent.click(await screen.findByRole("button", { name: "System health and model details" }));
    await screen.findByText("No supported, readable knowledge documents were found.");
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "Question" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Submit Question" }));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        "The knowledge index has not been built yet.",
      ),
    );
  });

  it("starts advanced sections collapsed and lets users select examples without submitting", async () => {
    render(<ThermalKnowledgeAssistant />);
    await screen.findByRole("heading", { name: "Ask a thermal engineering question" });

    for (const title of [
      "System health and model details",
      "How the assistant works",
      "More example questions",
      "Tips for better answers",
    ]) {
      const toggle = screen.getByRole("button", { name: title });
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      expect(document.getElementById(toggle.getAttribute("aria-controls")!)).toHaveAttribute("hidden");
    }

    fireEvent.click(screen.getByRole("button", { name: "How does coolant flow remove heat from battery cells?" }));
    expect(screen.getByLabelText("Your question")).toHaveValue("How does coolant flow remove heat from battery cells?");
    expect(api.retrievePassages).not.toHaveBeenCalled();
    expect(screen.getByText(/\d+\/1,000 characters/)).toBeInTheDocument();
  });

  it("submits with Ctrl+Enter and supports answer regeneration", async () => {
    render(<ThermalKnowledgeAssistant />);
    await screen.findByRole("button", { name: "Submit Question" });
    const textarea = screen.getByLabelText("Your question");
    fireEvent.change(textarea, { target: { value: "How does coolant remove heat?" } });
    fireEvent.keyDown(textarea, { key: "Enter", ctrlKey: true });

    expect(await screen.findByText(answer.answer)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Regenerate Answer" }));
    await waitFor(() => expect(api.retrievePassages).toHaveBeenCalledTimes(2));
    expect(api.getKnowledgeAnswer).toHaveBeenCalledTimes(2);
  });

  it("copies the answer and clears old evidence when the question changes", async () => {
    const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) };
    const oldClipboard = Object.getOwnPropertyDescriptor(navigator, "clipboard");
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: clipboard });

    try {
      render(<ThermalKnowledgeAssistant />);
      await screen.findByRole("button", { name: "Submit Question" });
      fireEvent.change(screen.getByLabelText("Your question"), {
        target: { value: "How does coolant remove heat?" },
      });
      fireEvent.click(screen.getByRole("button", { name: "Submit Question" }));
      expect(await screen.findByText(answer.answer)).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: "Copy Answer" }));
      await waitFor(() => expect(clipboard.writeText).toHaveBeenCalledWith(answer.answer));
      expect(await screen.findByText("Answer copied.")).toBeInTheDocument();

      fireEvent.change(screen.getByLabelText("Your question"), {
        target: { value: "A different question" },
      });
      expect(screen.queryByText(answer.answer)).not.toBeInTheDocument();
      expect(screen.getByText("Sources will appear here after you submit a question.")).toBeInTheDocument();
    } finally {
      if (oldClipboard) Object.defineProperty(navigator, "clipboard", oldClipboard);
      else Reflect.deleteProperty(navigator, "clipboard");
    }
  });
});
