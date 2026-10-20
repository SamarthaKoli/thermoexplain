import type {
  BertClassification,
  BertEvaluation,
  KnowledgeAnswer,
  KnowledgeStatus,
  RetrievalResult,
} from "../types/knowledge";
import type { ProviderTestResult } from "../types/provider";
import { testGptOssConnection, throwApiError } from "./providerApi";

const apiBaseUrl = (
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:5000/api"
).replace(/\/$/, "");
const knowledgeUrl = `${apiBaseUrl}/knowledge`;

export function getKnowledgeStatus(): Promise<KnowledgeStatus> {
  return requestJson(`${knowledgeUrl}/status`);
}

export function buildKnowledgeIndex(): Promise<KnowledgeStatus["corpus"]> {
  return requestJson(`${knowledgeUrl}/index`, { method: "POST" });
}

export function retrievePassages(input: {
  question: string;
  top_k?: number;
}): Promise<RetrievalResult> {
  return postJson(`${knowledgeUrl}/retrieve`, input);
}

export function getKnowledgeAnswer(input: {
  question: string;
  passage_ids: string[];
}): Promise<KnowledgeAnswer> {
  return postJson(`${knowledgeUrl}/answer`, input);
}

export function uploadKnowledgeDocument(file: File): Promise<{
  file_name: string;
  size_bytes: number;
  indexing_required: boolean;
}> {
  const body = new FormData();
  body.append("file", file);
  return requestJson(`${knowledgeUrl}/upload`, { method: "POST", body });
}

export function classifyQuestion(text: string): Promise<BertClassification> {
  return postJson(`${knowledgeUrl}/bert/classify`, { text });
}

export function evaluateBert(): Promise<BertEvaluation> {
  return requestJson(`${knowledgeUrl}/bert/evaluate`, { method: "POST" });
}

export function testKnowledgeConnection(): Promise<ProviderTestResult> {
  return testGptOssConnection();
}

async function postJson<T>(url: string, payload: object): Promise<T> {
  return requestJson<T>(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error("Could not reach the knowledge assistant API. Check the backend.");
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The knowledge assistant returned an invalid response.");
  }
  if (!response.ok) {
    throwApiError(payload, response.status);
  }
  return payload as T;
}
