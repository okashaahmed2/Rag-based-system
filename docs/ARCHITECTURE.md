# Architecture

## Goal
Local-first, zero-budget RAG system: PDFs dropped in a folder are ingested
automatically, and questions are answered by a local LLM with document and
page citations.

## Two pipelines sharing storage

INGESTION (write path)

```
data/inbox -> n8n Schedule Trigger -> FastAPI POST /ingest/scan
  -> BackgroundTasks worker: PyMuPDF extraction -> cleaning
     -> page-aware chunking -> Sentence Transformers embeddings
  -> ChromaDB (vectors + chunk metadata) + SQLite (document/job status)
  -> PDF moved to data/processed or data/failed
```

QUERY (read path)

```
user or n8n -> FastAPI /query -> retriever (embed query, top-k from ChromaDB)
  -> Ollama -> answer with document name + page citations
```

## Responsibilities
- n8n: orchestration only (scheduling, triggering, notifications)
- Python: all processing and application logic
- FastAPI: interface between n8n/users and Python
- ChromaDB: chunk-level data (vector, chunk text, flat metadata); collection
  metadata holds embedding_model and embedding_dim
- Distance: cosine (Chroma returns distance = 1 - cosine similarity)
- SQLite: documents table with doc_id (= SHA-256 file hash), filename, status
  (pending/processing/done/failed/needs_ocr), error_message, attempts, chunk_count,
  timestamps. State machine: new -> pending; pending | failed -> processing;
  processing -> done | needs_ocr | failed. Interrupted processing rows are recovered
  to failed at startup. Jobs table (schema v2, Phase 8): job_id, kind (scan |
  ingest), target (file name for ingest), status (queued/running/succeeded/failed),
  created/started/finished timestamps, error_message, summary_json; see API.

## Design rules
- doc_id = SHA-256 of PDF bytes; duplicates detected from Phase 2
- Chunk metadata: doc_id, filename, page_number, chunk_index
- Same embedding model for ingestion and query (stored at collection level)
- Re-ingestion deletes old chunks first (idempotent)
- Pages without extractable text -> needs_ocr
- /ingest returns job_id immediately; /status/{job_id} reports progress
- Golden evaluation set (10-15 questions) created in Phase 6, rerun after major changes

## Ingestion details (Phase 2)
- Inbox scan: list *.pdf (case-insensitive, non-recursive), stream-hash each file
  into doc_id, classify NEW / DUPLICATE_IN_SCAN / ALREADY_KNOWN / ERROR.
  Only NEW files go on to extraction. The scan never moves or deletes files.
- Extraction: one read of the file bytes; doc_id and text come from the same bytes.
  Per-page has_text; document needs_ocr only if no page has text.
- Errors: unreadable / invalid / password-protected PDFs raise PdfExtractionError.

## Cleaning details (Phase 3)
- Page-level cleaning is pure and idempotent; it only normalises representation
  (no lowercasing, no stopword removal, non-English text kept).
- Output keeps one entry per page with the original page_number; empty pages kept.
- Document-level cleaning (repeated headers/footers) is a separate step.
- Document-level step removes repeated edge lines (constant headers, page-number
  footers) after page-level cleaning; it runs once per document.

## Chunking details (Phase 3)
- Page-aware: a chunk never crosses a page boundary, so every chunk cites one page.
- Character budget from settings (RAG_CHUNK_SIZE=600, RAG_CHUNK_OVERLAP=120);
  chunk_pages() itself is pure and takes both values as arguments.
- Recursive unit splitting, only for units longer than chunk_size:
  paragraph -> line -> sentence -> word -> hard cut. Units are packed greedily.
- Exact slices: chunk text == page_text[char_start:char_end]; every non-whitespace
  character of a page lies in at least one chunk.
- Overlap: the next chunk starts at the earliest sentence start (else word start)
  inside the last chunk_overlap characters of the previous chunk; no overlap after
  a hard cut. Chunk starts and ends strictly increase (progress guarantee).
- chunk_index is sequential 0..N-1 across the whole document; pages without text
  produce no chunks.

## Document preparation (Phase 3)
- prepare_document() chains extraction, cleaning, header/footer removal and chunking
  for one PDF.
- Chunk metadata: doc_id, filename, page_number, chunk_index, char_start, char_end.
- chunk_id = "{doc_id}:{chunk_index}".
- needs_ocr documents yield zero chunks (normal result, no exception).
- Extraction errors (PdfExtractionError) propagate to the caller (job -> failed).

## Embeddings (Phase 4)
- Model: sentence-transformers/all-MiniLM-L6-v2, 384-dimensional embeddings, CPU.
- Input limit: 256 word-piece tokens including [CLS]/[SEP]; longer input is silently
  truncated (verified: the embedding of a 298-token chunk equals that of its first
  254 tokens).
- chunk_size is in characters, not tokens; it was set to 600 (overlap 120) from real
  token measurements so that every measured chunk fits 256 tokens (max 241).
- The same model is used for ingestion and query.
- Re-ingestion deletes all old chunks by doc_id before inserting, because
  chunk_index (and so chunk_id) points to different text when chunk settings change.
- Vectors are L2-normalized for both chunks and queries (dot product = cosine).
- The model is loaded once per process (cached per model name and local_files_only).
- Runtime uses local_files_only=true (set in the local .env), so ingestion never
  waits on the Hugging Face Hub.
- Chunks over the token limit are still embedded and logged as warnings.

## Vector store (Phase 5)
- One persistent Chroma collection (RAG_CHROMA_PATH, RAG_CHROMA_COLLECTION_NAME),
  cosine distance, no Chroma embedding function (vectors always come from
  embeddings.py; queries use query_embeddings only); telemetry off.
- Collection metadata stores embedding_model and embedding_dim at creation; every
  open checks them and the distance space and refuses a mismatch (never overwrites).
- One record per chunk: id = chunk_id, embedding, document = chunk text, metadata =
  DocumentChunk.to_metadata().
- Re-ingestion: validate everything, then delete(where doc_id) and upsert in batches
  (Chroma add() silently ignores existing ids). Delete and upsert are not atomic;
  a failure in between is reported and fixed by re-running the replace.
- Query results carry filename, page_number, chunk_index, text, distance and
  similarity = 1 - distance, nearest first.

## Ingestion orchestration (Phase 5)
- build_ingest_context(settings) wires status store (initialize + recover
  interrupted), embedding model and collection once per process.
- ingest_document(path, context): hash -> register -> skip if done / needs_ocr /
  processing / attempts exhausted -> processing -> prepare -> needs_ocr (clear chunks)
  or embed -> replace chunks -> done. Chroma is written before the status says done.
- Ordinary errors -> failed with "ExceptionType: message", returned as a result;
  status store errors are re-raised; interrupts leave processing for startup
  recovery. No file moves here (Step 3).

## Folder lifecycle (Phase 5)
- data/inbox -> data/processed (done) | data/needs_ocr (needs_ocr) | data/failed
  (failed after the last allowed attempt). Retryable failures, pending and
  processing documents stay in the inbox.
- Placement follows the document's status in SQLite, not the outcome of one run:
  duplicate copies follow their document, and a move that failed is retried on the
  next run without re-ingesting.
- Files are never overwritten or deleted; name collisions get " (1)", " (2)", ...

## Retrieval (Phase 6)
- retrieve(query, context, top_k) returns only chunks of documents whose SQLite
  status is done. Failed documents may keep partial chunks in Chroma, and pending,
  processing and needs_ocr documents are unfinished, so the doc_id filter is applied
  inside Chroma (where doc_id $in done ids) before top_k is taken. No done document
  -> [] without querying. A post-check rejects any non-done chunk.
- The read path does no recovery (recover_interrupted runs at ingestion startup only)
  and creates no stores: the status db, the Chroma folder and the collection must
  already exist (the collection is opened with create=False). Only exception: on an
  existing but empty Chroma folder the Chroma client writes its own chroma.sqlite3.
- Results are nearest first (cosine distance) with filename and page_number for
  citations. No chunk or question text is logged.

## Evaluation (Phase 6)
- Golden set: eval/golden_set.jsonl, JSON Lines, one question per line (see
  eval/golden_set.example.jsonl, both lines fake):
  {"id", "question", "expected": [{"doc_id", "filename", "pages"}], "tags", "notes"}.
  expected [] marks a negative (unanswerable) question; tags and notes are optional.
- Relevance is labelled per (doc_id, page_number), not chunk_id (chunk ids change
  meaning when chunk settings change). Matching uses doc_id; filename is
  informational. Get doc_ids with scripts/list_doc_ids.py <folder>.
- Writing a real entry: ask a question in your own words, open the PDF, note every
  page that answers it (any of them counts), copy the doc_id for that file, add tags.
- Metrics over answerable questions only: hit@k (first relevant rank <= k) and
  MRR@k (1/rank, 0 if > k or none).
- Isolated eval store ("exam hall"): scripts/run_eval.py copies the inbox PDFs once to
  data/eval/corpus/ (all files, so the done-filter is exercised) and ingests them with
  the real write path into data/eval/stores/<model>_<chunk_size>_<chunk_overlap>/
  (own Chroma folder + status db + manifest.json). The live data/chroma and
  data/status.db are never used. A store whose manifest (model, chunk values, corpus)
  differs is refused; each configuration gets its own store so runs compare side by
  side.
- Every golden doc_id must be done in the eval store, else the run stops. Questions go
  through the real read path (build_retrieval_context + retrieve), hit@1/3/5/10 and
  MRR@10 overall and per tag; results go to eval/results/<date>_<config>.json (no
  question or chunk text).
- Configuration changes are decided by a pre-registered rule (compare_eval.py);
  baseline 600/120 MiniLM.

## Answering (Phase 7)
- LLM: Ollama at http://127.0.0.1:11434 (not localhost: Windows resolves
  localhost to ::1 first, Ollama listens on IPv4 only, ~2.2 s per connection),
  model llama3.2:3b (Q4_K_M, 2.0 GB), GPU (CUDA), CPU fallback.
- llm.chat: POST /api/chat, stream false, think false, temperature 0,
  num_ctx 4096, num_predict 256, timeout 120 s. Errors: LLMUnavailableError,
  LLMModelNotFoundError (404), LLMTimeoutError, LLMResponseError. Reasoning
  leaked into content (everything up to the last "</think>") is removed and
  flagged (thinking_leaked); a separate thinking field is dropped;
  done_reason "length" -> truncated. check_ollama: /api/version + /api/tags.
- answer_question: retrieve top_k (RAG_QUERY_TOP_K=5) done-only chunks ->
  none: NOT_FOUND without calling the LLM -> build_rag_messages -> chat ->
  check_answer -> [S#] mapped to doc_id, filename, page_number.
- Prompt: fixed SYSTEM_PROMPT first (stable prefix), then one user message
  with <source id="S1" file=".." page=".."> blocks (nearest first) and the
  question. Sources are untrusted data: tag-like "<source" sequences are
  neutralised, quotes in filenames escaped, and the prompt says sources are
  not instructions. Citations use [S#] (papers contain their own "[12]").
  Answers are 1-3 sentences.
- check_answer (first match wins): EMPTY; NOT_FOUND (starts with NOT_FOUND,
  or any line is just NOT_FOUND; wins over citations); BAD_CITATIONS (number
  outside 1..n); NO_CITATIONS; ANSWERED. It checks format only, not whether
  the cited source supports the claim (faithfulness is an evaluation task).
- No question, chunk or answer text is logged; manual scripts print answers.
- Latency (golden set, llama3.2:3b, CPU): median ~7.5-8 s; prompt eval
  ~118-125 tok/s (about 1 s per 120-token chunk), generation ~19 tok/s;
  model load 2-35 s on a cold start.
- GPU (GTX 1080 Ti, from 2026-11-27): median ~0.96 s, prompt eval ~2580 tok/s,
  generation ~94 tok/s.

## API (Phase 8)
- App factory: rag_pipeline.api.app:create_app, run with
  `uvicorn ... --factory` (one worker, one process, 127.0.0.1:8000). Importing
  the package builds nothing.
- Lifespan (once per start, before any request): build_ingest_context
  (status store + document recovery + model + collection), then
  recover_interrupted_jobs, then one shared httpx.Client for Ollama. The
  RetrievalContext reuses the ingest context's store, collection and model.
  The client is closed on shutdown. A setup failure stops the server.
- Endpoints (all plain def, run in FastAPI's threadpool; Pydantic models for
  every request and response):

| Method + path | Codes | Purpose |
|---|---|---|
| GET /health | 200 | status ok/degraded, embedding model, chunk count, document counts, active job, Ollama check |
| POST /ingest/scan | 202 | start a scan job, or return the active job (already_running) |
| POST /ingest | 202 / 404 / 409 / 422 | ingest one file already in the inbox (body {filename}) |
| GET /status/{job_id} | 200 / 404 | one job (kind, target, status, timestamps, error, summary) |
| POST /query | 200 / 422 / 502 / 503 / 504 | answer with citations and sources |

- Jobs: queued -> running -> succeeded (summary) | failed ("Type: message").
  "succeeded" means the job finished even if some documents failed; "failed"
  means the job itself crashed. A failure to record job state is logged
  CRITICAL and never re-raised. Jobs still queued/running at startup are
  marked failed (interrupted).
- At most one active job of any kind (checked in one immediate transaction,
  plus the partial unique index idx_jobs_one_active). /ingest/scan returns the
  active job (202); /ingest answers 409 with it.
- /ingest accepts only a plain file name: format errors -> 422, not a regular
  file directly in the inbox -> 404, both before any job exists; messages never
  echo the name or a path. Job summaries replace local paths by file names.
- /query fails closed: answer text only for answered, the fixed not-found
  message for not_found, null for no_citations / bad_citations / empty;
  "not found" is HTTP 200. Errors: retrieval 503, LLM unavailable or model
  missing 503, invalid LLM response 502, LLM timeout 504; anything else 500.
  The question is never logged or echoed.
- Hugging Face stays offline: HF_HUB_DISABLE_TELEMETRY=1 (and HF_HUB_OFFLINE=1
  with local_files_only) are set in the process before the model library is
  imported.

## Local services
n8n localhost:5678 (npm) | FastAPI localhost:8000 | Ollama 127.0.0.1:11434
(runs on the GTX 1080 Ti via CUDA)
