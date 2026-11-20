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
  to failed at startup. Jobs table in Phase 8.

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

## Local services
n8n localhost:5678 (npm) | FastAPI localhost:8000 | Ollama localhost:11434
