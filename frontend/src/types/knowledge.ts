export interface KnowledgeDocument {
  document_id: string;
  title: string;
  file_name: string;
  source_url: string | null;
  page_count: number | null;
  sha256: string;
  chunk_count: number;
}

export interface RetrievedPassage {
  passage_id: string;
  document_id: string;
  title: string;
  file_name: string;
  source_url: string | null;
  page: number | null;
  section: string | null;
  start_character: number;
  end_character: number;
  text: string;
  relevance_score: number;
}

export interface KnowledgeStatus {
  corpus: {
    indexing_status: "not_indexed" | "indexed" | "empty" | "error";
    corpus_version: string | null;
    indexed_at: string | null;
    document_count: number;
    passage_count: number;
    source_document_count: number;
    documents: KnowledgeDocument[];
    warnings: string[];
  };
  retrieval: {
    status: "available" | "unavailable";
    diagnostic: string;
    method: string;
    minimum_cosine_similarity: number;
    chunk_characters: number;
    chunk_overlap_characters: number;
  };
  bert: {
    status: "configured" | "not_configured";
    label_set: string[];
    held_out_test_size: number;
    reason: string | null;
  };
  gpt_oss: {
    provider: string;
    model: string | null;
    configured: boolean;
    status: "configured" | "not_configured";
    diagnostic: string;
  };
}

export interface RetrievalResult {
  query: string;
  corpus_version: string;
  passages: RetrievedPassage[];
  empty_reason: string | null;
}

export interface KnowledgeAnswer {
  label: string;
  model: string;
  answer: string;
  citations: string[];
  insufficient_evidence: boolean;
}

export interface BertClassification {
  classifier: "BERT";
  predicted_label: string;
  confidence: number;
  confidence_method: string;
  label_set: string[];
  status: "inference";
}

export interface BertEvaluation {
  evaluation_split: string;
  test_example_count: number;
  label_set: string[];
  accuracy: number;
  macro_f1: number;
  confusion_matrix: number[][];
}
