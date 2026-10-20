import { afterEach, describe, expect, it, vi } from "vitest";

import {
  analyzeWithGptOss,
} from "./datasetApi";
import {
  interpretWithGptOss,
} from "./mlApi";
import { getGptOssStatus, testGptOssConnection } from "./providerApi";
import {
  getKnowledgeAnswer,
  retrievePassages,
  testKnowledgeConnection,
} from "./knowledgeApi";

const response = { ok: true, json: async () => ({}) };

describe("provider and RAG API routing", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("uses the centralized GPT-OSS status, inference, and CFD routes", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response);
    vi.stubGlobal("fetch", fetchMock);

    await getGptOssStatus();
    await testGptOssConnection();
    await analyzeWithGptOss({
      case_id: "Re560/coarse",
      region: "coolant",
      field: "T",
      component: "value",
    });

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method ?? "GET"])).toEqual([
      ["http://127.0.0.1:5000/api/llm/status", "GET"],
      ["http://127.0.0.1:5000/api/llm/connection-test", "POST"],
      ["http://127.0.0.1:5000/api/llm/analysis", "POST"],
    ]);
    expect(JSON.parse(fetchMock.mock.calls[2][1]?.body as string)).toEqual({
      case_id: "Re560/coarse",
      region: "coolant",
      field: "T",
      component: "value",
    });
  });

  it("reports provider API network failures without mislabeling them as status-only", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    await expect(getGptOssStatus()).rejects.toThrow(
      "Could not reach the GPT-OSS API. Check that the backend is running.",
    );
  });

  it("uses GPT-OSS for explainability and knowledge calls", async () => {
    const fetchMock = vi.fn().mockResolvedValue(response);
    vi.stubGlobal("fetch", fetchMock);

    await interpretWithGptOss("lime", {
      model_id: "verified-model",
      case_id: "Re560/coarse",
      coordinates: { x: 0.1, y: 0.2, z: 0.3 },
    });
    await retrievePassages({ question: "How does cooling work?" });
    await getKnowledgeAnswer({
      question: "How does cooling work?",
      passage_ids: ["p-source-1"],
    });
    await testKnowledgeConnection();

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method ?? "GET"])).toEqual([
      ["http://127.0.0.1:5000/api/ml/gpt-oss/interpretation", "POST"],
      ["http://127.0.0.1:5000/api/knowledge/retrieve", "POST"],
      ["http://127.0.0.1:5000/api/knowledge/answer", "POST"],
      ["http://127.0.0.1:5000/api/llm/connection-test", "POST"],
    ]);
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/gemini/"))).toBe(false);
    expect(JSON.parse(fetchMock.mock.calls[0][1]?.body as string)).toMatchObject({
      method: "lime",
      model_id: "verified-model",
    });
    expect(JSON.parse(fetchMock.mock.calls[2][1]?.body as string)).toEqual({
      question: "How does cooling work?",
      passage_ids: ["p-source-1"],
    });
  });
});
