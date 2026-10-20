import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import {
  buildKnowledgeIndex,
  classifyQuestion,
  evaluateBert,
  getKnowledgeAnswer,
  getKnowledgeStatus,
  retrievePassages,
  testKnowledgeConnection,
  uploadKnowledgeDocument,
} from "../services/knowledgeApi";
import { providerTestFailure } from "../services/providerApi";
import type {
  BertClassification,
  BertEvaluation,
  KnowledgeAnswer,
  KnowledgeStatus,
  RetrievedPassage,
} from "../types/knowledge";
import type { ProviderTestResult } from "../types/provider";

export function ThermalKnowledgeAssistant() {
  const [status, setStatus] = useState<KnowledgeStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);
  const [question, setQuestion] = useState("");
  const [passages, setPassages] = useState<RetrievedPassage[]>([]);
  const [answer, setAnswer] = useState<KnowledgeAnswer | null>(null);
  const [bertResult, setBertResult] = useState<BertClassification | null>(null);
  const [bertEvaluation, setBertEvaluation] = useState<BertEvaluation | null>(null);
  const [retrievalError, setRetrievalError] = useState<string | null>(null);
  const [answerError, setAnswerError] = useState<string | null>(null);
  const [bertError, setBertError] = useState<string | null>(null);
  const [connectionResult, setConnectionResult] =
    useState<ProviderTestResult | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [isRetrieving, setIsRetrieving] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [isEvaluatingBert, setIsEvaluatingBert] = useState(false);
  const [retrievalReady, setRetrievalReady] = useState<boolean | null>(null);
  const [copyMessage, setCopyMessage] = useState<string | null>(null);
  const [submittedQuestion, setSubmittedQuestion] = useState("");

  function updateQuestion(value: string) {
    setQuestion(value);
    setCopyMessage(null);
    if (value.trim() === submittedQuestion) return;
    setPassages([]);
    setAnswer(null);
    setBertResult(null);
    setBertEvaluation(null);
    setRetrievalError(null);
    setAnswerError(null);
    setBertError(null);
  }

  async function refreshStatus() {
    setIsLoadingStatus(true);
    setStatusError(null);
    try {
      setStatus(await getKnowledgeStatus());
    } catch (error) {
      setStatusError(errorMessage(error));
    } finally {
      setIsLoadingStatus(false);
    }
  }

  useEffect(() => {
    void refreshStatus();
  }, []);

  async function handleIndex() {
    setIsIndexing(true);
    setStatusError(null);
    setUploadMessage(null);
    try {
      const corpus = await buildKnowledgeIndex();
      setUploadMessage(
        `Index ${corpus.indexing_status}: ${corpus.document_count} documents and ${corpus.passage_count} passages.`,
      );
      await refreshStatus();
    } catch (error) {
      setStatusError(errorMessage(error));
    } finally {
      setIsIndexing(false);
    }
  }

  async function handleUpload(file: File | undefined) {
    if (!file) return;
    setUploadMessage(null);
    setStatusError(null);
    try {
      const result = await uploadKnowledgeDocument(file);
      setUploadMessage(
        `${result.file_name} uploaded (${formatBytes(result.size_bytes)}). Build the index to make it searchable.`,
      );
    } catch (error) {
      setStatusError(errorMessage(error));
    }
  }

  async function runQuestion() {
    const text = question.trim();
    if (!text) return;
    setSubmittedQuestion(text);
    setPassages([]);
    setAnswer(null);
    setBertResult(null);
    setBertEvaluation(null);
    setRetrievalError(null);
    setAnswerError(null);
    setBertError(null);
    setCopyMessage(null);
    setIsRetrieving(true);
    let classificationTask: Promise<void> = Promise.resolve();
    if (status?.bert.status === "configured") {
      classificationTask = classifyQuestion(text)
        .then(setBertResult)
        .catch((error: unknown) => setBertError(errorMessage(error)));
    }
    try {
      const result = await retrievePassages({ question: text });
      setRetrievalReady(true);
      setPassages(result.passages);
      if (result.passages.length === 0) {
        setAnswerError(
          result.empty_reason ?? "No useful source passages were found; no LLM answer was requested.",
        );
      } else {
        setIsAnswering(true);
        try {
          setAnswer(
            await getKnowledgeAnswer({
              question: text,
              passage_ids: result.passages.map((passage) => passage.passage_id),
            }),
          );
        } catch (error) {
          setAnswerError(errorMessage(error));
        } finally {
          setIsAnswering(false);
        }
      }
      await classificationTask;
    } catch (error) {
      setRetrievalReady(false);
      setRetrievalError(errorMessage(error));
      await classificationTask;
    } finally {
      setIsRetrieving(false);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await runQuestion();
  }

  async function handleConnectionTest() {
    setIsTestingConnection(true);
    setConnectionResult(null);
    try {
      const result = await testKnowledgeConnection();
      setConnectionResult(result);
    } catch (error) {
      setConnectionResult(
        providerTestFailure(
          status?.gpt_oss.provider ?? "GPT-OSS",
          status?.gpt_oss.model ?? null,
          error,
        ),
      );
    } finally {
      setIsTestingConnection(false);
    }
  }

  async function handleBertEvaluation() {
    setIsEvaluatingBert(true);
    setBertError(null);
    try {
      setBertEvaluation(await evaluateBert());
    } catch (error) {
      setBertError(errorMessage(error));
    } finally {
      setIsEvaluatingBert(false);
    }
  }

  const citedIds = new Set(answer?.citations ?? []);
  const corpusReady = status?.corpus.indexing_status === "indexed" &&
    status.corpus.document_count > 0 && status.corpus.passage_count > 0;

  const retrievalState = retrievalReady === true
    ? "Ready"
    : retrievalReady === false || status?.retrieval.status === "unavailable" || statusError
      ? "Unavailable"
      : status?.retrieval.status === "available"
        ? "Available · not queried"
        : isLoadingStatus
          ? "Checking…"
          : "Unavailable";
  const generationState = connectionResult?.status === "connected"
    ? "Ready"
    : status?.gpt_oss.configured
      ? "Configured"
      : statusError
        ? "Unavailable"
        : isLoadingStatus
          ? "Checking…"
          : "Unavailable";

  async function copyAnswer() {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer.answer);
      setCopyMessage("Answer copied.");
    } catch {
      setCopyMessage("Clipboard access is unavailable in this browser.");
    }
  }

  return (
    <section className="knowledge-assistant knowledge-assistant--compact">
      <header className="page-heading knowledge-page-heading">
        <div>
          <h1>Thermal Knowledge Assistant</h1>
          <p className="intro__copy">Ask thermal engineering questions and explore evidence from indexed technical literature.</p>
        </div>
        <span className={`status-pill ${corpusReady ? "status-pill--connected" : "status-pill--unavailable"}`} role="status">
          {isLoadingStatus ? "Checking knowledge base…" : corpusReady ? "Knowledge base ready" : statusError ? "Knowledge base unavailable" : "Knowledge base not ready"}
        </span>
      </header>

      <div className="knowledge-status-row" aria-label="System status">
        <StatusItem label="Knowledge base" value={isLoadingStatus ? "Checking…" : corpusReady ? "Ready" : "Unavailable"} ready={corpusReady} />
        <StatusItem label="Retrieval service" value={retrievalState} ready={retrievalReady === true} neutral={retrievalReady === null && status?.retrieval.status === "available"} />
        <StatusItem label="Answer generation" value={generationState} ready={generationState === "Ready"} neutral={generationState === "Configured" || generationState === "Checking…"} />
      </div>
      {statusError && <p className="message message--error" role="alert">{statusError}</p>}

      <Disclosure id="knowledge-system-details" title="System health and model details">
        <div className="knowledge-system-actions">
          <button className="secondary-button" disabled={isIndexing} onClick={() => void handleIndex()} type="button">
            {isIndexing ? "Building index…" : "Build / refresh index"}
          </button>
          <label className="upload-button">
            Add document
            <input accept=".pdf,.txt,.md,.markdown" aria-label="Add PDF, text, or Markdown document" onChange={(event) => {
              void handleUpload(event.currentTarget.files?.[0]);
              event.currentTarget.value = "";
            }} type="file" />
          </label>
          <button className="secondary-button" disabled={isTestingConnection} onClick={() => void handleConnectionTest()} type="button">
            {isTestingConnection ? "Testing connection…" : "Test answer-generation connection"}
          </button>
          {statusError && <button className="text-button" onClick={() => void refreshStatus()} type="button">Retry status</button>}
        </div>
        {uploadMessage && <p className="message message--notice" role="status">{uploadMessage}</p>}
        {status?.corpus.warnings.map((warning) => <p className="message message--notice" key={warning}>{warning}</p>)}
        {status ? (
          <>
            <dl className="knowledge-stats">
              <div><dt>Index status</dt><dd>{status.corpus.indexing_status.replace("_", " ")}</dd></div>
              <div><dt>Documents</dt><dd>{status.corpus.document_count}</dd></div>
              <div><dt>Source documents found</dt><dd>{status.corpus.source_document_count}</dd></div>
              <div><dt>Indexed passages</dt><dd>{status.corpus.passage_count}</dd></div>
              <div><dt>Corpus version</dt><dd>{status.corpus.corpus_version ?? "Not indexed"}</dd></div>
              <div><dt>Retrieval method</dt><dd>{status.retrieval.method}</dd></div>
              <div><dt>Retrieval status</dt><dd>{status.retrieval.status}</dd></div>
              <div><dt>Minimum similarity</dt><dd>{status.retrieval.minimum_cosine_similarity}</dd></div>
              <div><dt>BERT classifier</dt><dd>{status.bert.status === "configured" ? status.bert.label_set.join(", ") : "Not configured"}</dd></div>
              <div><dt>Answer model</dt><dd>{status.gpt_oss.model ?? "Not configured"}</dd></div>
            </dl>
            {status.corpus.documents.length > 0 && (
              <ul className="knowledge-documents">
                {status.corpus.documents.map((document) => (
                  <li key={document.document_id}>
                    <strong>{document.title}</strong>
                    <span>{document.file_name} · {document.chunk_count} passages</span>
                    {document.page_count !== null && <span>{document.page_count} pages</span>}
                    {document.source_url && <a href={document.source_url} rel="noreferrer" target="_blank">Original source</a>}
                  </li>
                ))}
              </ul>
            )}
            {status.retrieval.status === "unavailable" && <p className="message message--notice" role="status">{status.retrieval.diagnostic}</p>}
            {status.bert.reason && <p className="muted">{status.bert.reason}</p>}
            {status.bert.status === "configured" && (
              <div className="knowledge-system-subsection">
                <p className="muted">Verified labels: {status.bert.label_set.join(", ")} · held-out examples: {status.bert.held_out_test_size}</p>
                <button className="text-button" disabled={isEvaluatingBert} onClick={() => void handleBertEvaluation()} type="button">
                  {isEvaluatingBert ? "Evaluating…" : "Evaluate held-out classifier"}
                </button>
                {bertEvaluation && <p className="muted">n={bertEvaluation.test_example_count} · accuracy {bertEvaluation.accuracy.toFixed(3)} · macro-F1 {bertEvaluation.macro_f1.toFixed(3)}</p>}
                {bertError && <p className="message message--error" role="alert">{bertError}</p>}
              </div>
            )}
          </>
        ) : isLoadingStatus ? <p className="muted" role="status">Checking system status…</p> : null}
        {connectionResult && (
          <p className={`provider-test provider-test--${connectionResult.status}`} role={connectionResult.status === "connected" ? "status" : "alert"}>
            {connectionResult.provider}: {connectionResult.status.replace("_", " ")} · model {connectionResult.model ?? "not configured"} · {connectionResult.diagnostic}
          </p>
        )}
        {status?.gpt_oss.diagnostic && <p className="muted">Generation service: {status.gpt_oss.diagnostic}</p>}
        <p className="muted">“Configured” reflects backend configuration only. “Ready” is shown only after a successful live connection test; retrieval availability reflects the loaded index, while “Ready” follows a successful retrieval request.</p>
      </Disclosure>

      <section className="knowledge-process" aria-label="Question processing pipeline">
        <p className="knowledge-process__steps">Question <span>→</span> Find Evidence <span>→</span> Generate Answer <span>→</span> Verify Citations</p>
        <Disclosure id="knowledge-process-details" title="How the assistant works">
          <ul className="knowledge-detail-list">
            <li>A configured BERT/SciBERT classifier can categorize the question; classification is optional and does not determine the answer.</li>
            <li>Retrieval searches indexed passages using the method reported by the backend ({status?.retrieval.method ?? "method unavailable"}).</li>
            <li>Only retrieved passage identifiers are sent to the answer service. If retrieval returns no evidence, answer generation is skipped.</li>
            <li>GPT-OSS generates an explanation from the supplied passages. It does not run CFD, independently verify claims, or establish engineering safety.</li>
            <li>Citation IDs are checked against the retrieved passages and link to their matching source cards.</li>
          </ul>
        </Disclosure>
      </section>

      <section className="panel knowledge-question-panel" aria-labelledby="question-title">
        <h2 id="question-title">Ask a thermal engineering question</h2>
        <form onSubmit={(event) => void handleAsk(event)}>
          <label className="form-label" htmlFor="knowledge-question">Your question</label>
          <textarea id="knowledge-question" aria-describedby="knowledge-character-count" disabled={isRetrieving || isAnswering} maxLength={1000} onChange={(event) => {
            updateQuestion(event.currentTarget.value);
          }} onKeyDown={(event) => {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }} placeholder="Ask a focused question about your indexed thermal engineering sources…" rows={4} value={question} />
          <div className="knowledge-question__footer">
            <span className="muted" id="knowledge-character-count">{question.length}/1,000 characters · Ctrl/⌘ + Enter to submit</span>
            <button className="primary-button" disabled={isRetrieving || isAnswering || !question.trim()} type="submit">
              {isRetrieving ? "Finding evidence…" : isAnswering ? "Generating answer…" : "Submit Question"}
            </button>
          </div>
        </form>
        <div className="knowledge-examples" aria-label="Example questions">
          <span>Try:</span>
          {[
            "How does coolant flow remove heat from battery cells?",
            "Which operating conditions affect thermal runaway risk?",
            "What factors influence pressure drop in a cooling channel?",
            "How does cell spacing affect pack temperature?",
          ].map((example) => <button key={example} disabled={isRetrieving || isAnswering} onClick={() => updateQuestion(example)} type="button">{example}</button>)}
        </div>
        <Disclosure id="knowledge-more-examples" title="More example questions">
          <div className="knowledge-more-examples">
            {[
              "What evidence describes the effect of inlet temperature on cooling?",
              "How do flow rate and heat transfer relate in the indexed sources?",
              "What cooling constraints are discussed for transient operation?",
            ].map((example) => <button key={example} disabled={isRetrieving || isAnswering} onClick={() => updateQuestion(example)} type="button">{example}</button>)}
          </div>
        </Disclosure>
        {retrievalError && <p className="message message--error" role="alert">{retrievalError}</p>}
        {answerError && (
          passages.length === 0
            ? <p className="message message--notice" role="status">Insufficient evidence. {answerError}</p>
            : <p className="message message--error" role="alert">Answer unavailable. {answerError}</p>
        )}
        {isRetrieving && <p className="muted" role="status">Searching the indexed corpus for matching passages…</p>}
        {isAnswering && <p className="muted" role="status">Generating an evidence-limited answer…</p>}
      </section>

      {answer && (
        <section className="panel knowledge-answer knowledge-answer--compact" aria-labelledby="answer-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">SOURCE-GROUNDED RESPONSE</p>
              <h2 id="answer-title">Answer</h2>
            </div>
            <span className={`status-pill ${answer.citations.length ? "status-pill--connected" : "status-pill--unavailable"}`}>
              {passages.length > 0 ? `${passages.length} evidence passages` : "No evidence"}
              {" · "}{answer.citations.length ? `${answer.citations.length} citations` : "No citations"}
            </span>
          </div>
          {answer.insufficient_evidence && <p className="message message--notice">The model reports that retrieved passages do not fully answer this question.</p>}
          <p className="knowledge-answer__text">{answer.answer}</p>
          {answer.citations.length > 0 && (
            <p className="knowledge-citations">
              Cited evidence: {answer.citations.map((passageId, index) => {
                const passageIndex = passages.findIndex((passage) => passage.passage_id === passageId);
                return (
                  <span key={`${passageId}-${index}`}>
                    {index > 0 ? ", " : ""}
                    {passageIndex >= 0
                      ? <a href={`#source-${passageId}`}>[{passageIndex + 1}]</a>
                      : <span className="knowledge-unmatched-citation">Unmatched citation: {passageId}</span>}
                  </span>
                );
              })}
            </p>
          )}
          <div className="knowledge-answer__actions">
            <button className="secondary-button" onClick={() => void copyAnswer()} type="button">Copy Answer</button>
            <button className="secondary-button" disabled={isRetrieving || isAnswering} onClick={() => void runQuestion()} type="button">
              {isRetrieving || isAnswering ? "Regenerating…" : "Regenerate Answer"}
            </button>
            <Disclosure id="knowledge-more-actions" title="More actions">
              <button className="text-button" onClick={() => window.print()} type="button">Print / save as PDF</button>
            </Disclosure>
          </div>
          {copyMessage && <p className="muted" role="status">{copyMessage}</p>}
          <Disclosure id="knowledge-answer-details" title="Answer details and provenance">
            <dl className="knowledge-detail-grid">
              <div><dt>Answer label</dt><dd>{answer.label}</dd></div>
              <div><dt>Model</dt><dd>{answer.model}</dd></div>
              <div><dt>Retrieved passages</dt><dd>{passages.length}</dd></div>
              <div><dt>Returned citation IDs</dt><dd>{answer.citations.length}</dd></div>
              <div><dt>Matched citations</dt><dd>{answer.citations.filter((id) => passages.some((passage) => passage.passage_id === id)).length}</dd></div>
              {bertResult && <div><dt>Question classification</dt><dd>{bertResult.predicted_label} · {(bertResult.confidence * 100).toFixed(1)}% ({bertResult.confidence_method})</dd></div>}
              {bertResult && <div><dt>Classifier</dt><dd>{bertResult.classifier}</dd></div>}
              <div><dt>Retrieval method</dt><dd>{status?.retrieval.method ?? "Unavailable"}</dd></div>
            </dl>
            {bertResult && <p className="muted">Classifier labels: {bertResult.label_set.join(", ")}</p>}
            <p className="muted">Inference latency and generation settings are not returned by the current API.</p>
          </Disclosure>
        </section>
      )}

      <section className="panel knowledge-evidence" aria-labelledby="sources-title">
        <div className="section-heading">
          <div>
            <h2 id="sources-title">Retrieved evidence</h2>
            <p className="muted">Similarity is a retrieval score, not a probability or model confidence.</p>
          </div>
          {passages.length > 0 && <span className="status-pill">{passages.length} source{passages.length === 1 ? "" : "s"}</span>}
        </div>
        {passages.length === 0 ? (
          <p className="muted">{isRetrieving ? "Searching the indexed corpus…" : "Sources will appear here after you submit a question."}</p>
        ) : (
          <ol className="retrieved-passages">
            {passages.map((passage, index) => (
              <li id={`source-${passage.passage_id}`} key={passage.passage_id}>
                <article className="source-card source-card--compact">
                  <div className="source-card__heading">
                    <div>
                      <p className="eyebrow">[{index + 1}] · {passage.passage_id}</p>
                      <h3>{passage.title}</h3>
                    </div>
                    {citedIds.has(passage.passage_id) && <span className="status-pill status-pill--connected">Cited</span>}
                  </div>
                  <p className="source-card__summary">
                    {passage.section ? `${passage.section} · ` : ""}{passage.page !== null ? `Page ${passage.page} · ` : ""}
                    Similarity {passage.relevance_score.toFixed(3)}
                  </p>
                  <p className="source-card__excerpt">{passage.text.length > 260 ? `${passage.text.slice(0, 260).trimEnd()}…` : passage.text}</p>
                  <Disclosure id={`knowledge-source-details-${index}`} title="Source details">
                    <p className="muted"><strong>Full title:</strong> {passage.title}</p>
                    <p className="muted"><strong>File:</strong> {passage.file_name}</p>
                    {passage.section && <p className="muted"><strong>Section:</strong> {passage.section}</p>}
                    {passage.page !== null && <p className="muted"><strong>Page:</strong> {passage.page}</p>}
                    <p className="muted"><strong>Passage ID:</strong> {passage.passage_id}</p>
                    <p className="muted"><strong>Character range:</strong> {passage.start_character}–{passage.end_character}</p>
                    <p className="muted"><strong>Cosine similarity:</strong> {passage.relevance_score.toFixed(3)}</p>
                    <blockquote>{passage.text}</blockquote>
                    {passage.source_url && <a href={passage.source_url} rel="noreferrer" target="_blank">Open original source</a>}
                  </Disclosure>
                </article>
              </li>
            ))}
          </ol>
        )}
      </section>

      <section className="knowledge-tips">
        <Disclosure id="knowledge-tips-details" title="Tips for better answers">
          <ul className="knowledge-detail-list">
            <li>Ask one focused question at a time.</li>
            <li>Include exact engineering values, units, and operating conditions when available.</li>
            <li>Inspect the retrieved passages before applying a recommendation.</li>
            <li>Add relevant technical manuals or papers using the supported document uploader, then rebuild the index.</li>
          </ul>
        </Disclosure>
      </section>
    </section>
  );
}

function Disclosure({ children, id, title }: { children: ReactNode; id: string; title: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <section className="knowledge-disclosure">
      <h3>
        <button aria-controls={id} aria-expanded={expanded} className="knowledge-disclosure__toggle" onClick={() => setExpanded((value) => !value)} type="button">
          <span>{title}</span><span aria-hidden="true" className="knowledge-disclosure__chevron">⌄</span>
        </button>
      </h3>
      <div className="knowledge-disclosure__content" hidden={!expanded} id={id}>{children}</div>
    </section>
  );
}

function StatusItem({ label, value, ready, neutral = false }: { label: string; value: string; ready: boolean; neutral?: boolean }) {
  return (
    <div className="knowledge-status-item">
      <span>{label}</span>
      <strong className={ready ? "is-ready" : neutral ? "is-neutral" : "is-unavailable"}>{value}</strong>
    </div>
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}

function formatBytes(value: number): string {
  return `${(value / 1024).toFixed(1)} KiB`;
}
