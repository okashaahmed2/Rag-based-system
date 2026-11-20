# Project State

## Current position
- Phase 1: COMPLETE
- Phase 2 (PDF ingestion: PyMuPDF extraction, SHA-256 doc_id, needs_ocr, duplicate detection): COMPLETE
- Phase 3 Step 1 (page-level text cleaning): COMPLETE
- Phase 3 Step 2 (repeated header/footer removal): COMPLETE
- Phase 3 Step 3 (page-aware chunking): COMPLETE
- Phase 3 Step 4 (chunk metadata + prepare_document): COMPLETE
- Phase 3 (cleaning, header/footer removal, chunking, metadata): COMPLETE
- Phase 4 Step 1 (Sentence Transformers install, real token measurement, chunk_size 600): COMPLETE
- Phase 4 Step 2 (embeddings module): COMPLETE
- Phase 4 (embeddings): COMPLETE (commit f77f604)
- Phase 5 Step 1a (ChromaDB install + spike): COMPLETE (commit 07f9e3b)
- Phase 5 Step 1b (vector_store.py + tests + real smoke check): COMPLETE (commit 2116068)
- Phase 5 Step 1c (SQLite document status store + tests): COMPLETE (commit 18102d7)
- Phase 5 Step 2 (ingestion orchestration for one document + tests + smoke check):
  COMPLETE (commit df9450d)
- Phase 5 Step 3 (ingest_inbox: scan -> ingest -> move by status + tests + smoke):
  COMPLETE
- Phase 5 (ChromaDB + SQLite + ingestion orchestration + inbox lifecycle): COMPLETE
- Next: Phase 5 complete -> Phase 6 Step 1: retrieval prototype + golden evaluation set

## Verified (with evidence)
- Phase 1: Python 3.11 venv, Git/GitHub, pyproject editable install, pydantic-settings config, logging
- extract_pdf(): page-by-page raw text, 1-based page numbers, doc_id = SHA-256 of bytes
- needs_ocr = True only when NO page has text; blank pages listed in pages_without_text
- Verified on real files: typed notes (36/36 pages text), generated scanned PDF (needs_ocr True),
  partly blank PDF ([2]), password-protected and fake PDF raise PdfExtractionError
- compute_file_doc_id(): streaming SHA-256, equal to in-memory hash
- scan_inbox(): NEW / DUPLICATE_IN_SCAN / ALREADY_KNOWN / ERROR; renamed copy detected as duplicate
- clean_page_text(): pure, idempotent; NFC (not NFKC); line endings; ligatures;
  Zs spaces -> " "; removes U+00AD, U+200B, U+FEFF, C0/C1 controls, DEL;
  U+2028/U+0085 -> \n, U+2029 -> \n\n; whitespace collapse; lowercase-lowercase
  hyphen join; 3+ newlines -> 2
- clean_pages(): same count, order and page_numbers; empty pages kept
- Real file (36-page notes): pages 36/36, numbers unchanged, chars
  19,581 -> 16,814, idempotent on all pages; Roman Urdu, math symbols, emoji intact
- remove_repeated_headers_footers(): EXACT keys (constant headers) and PAGE_NUMBER keys
  (page-number-like lines only, via PAGE_NUMBER_KEY_PATTERN); edge region = first/last
  2 non-empty lines; threshold >= 3 text pages and >= 50% of text pages; removal by
  line index only in the matching region; empty pages kept; page_numbers preserved
- Synthetic 8-page PDF: header + "Page N of 8" removed, all body kept, header/footer
  extracted as first/last lines. All inbox PDFs: nothing wrongly removed
- chunk_pages(): chunks never cross pages, len <= chunk_size, exact slices
  (text == page_text[char_start:char_end]), full coverage of non-whitespace text,
  sequential chunk_index; overlap starts at the earliest sentence start (else word
  start) inside the last chunk_overlap characters of the previous chunk
- Settings RAG_CHUNK_SIZE=600 (was 800 until Phase 4 Step 1), RAG_CHUNK_OVERLAP=120
  (characters) with validation (size >= 100, 0 <= overlap < size)
- Real data, two-column research paper: 4 pages, 19 chunks, 15/15 same-page pairs
  overlap 82-120 chars, column-by-column reading order, 32 hyphen joins
  (3 spot-checked correct), header "preprint. under review." and page numbers removed
- Real data, lecture slides: 15 slides -> 15 chunks, course header removed from
  14 slides, slide titles kept
- Hyphen joins on real data (research paper, see above)
- Header/footer removal on real PDFs (research paper header + page numbers,
  lecture course header)
- prepare_document(): extract -> clean -> header/footer removal -> chunk ->
  DocumentChunk(chunk_id=f"{doc_id}:{chunk_index}", doc_id, filename basename,
  page_number, chunk_index, text, char_start, char_end); to_metadata() flat str/int,
  no text; needs_ocr returns 0 chunks without an exception; PdfExtractionError propagates
- Real files: research paper 19 chunks, slides 15 chunks, scanned needs_ocr True with
  0 chunks, invalid and password PDFs raise PdfExtractionError, renamed copy has the
  same doc_id and chunk_ids
- Lecture07 slides 8/13 human check: both fine (text complete, readable order)
- all-MiniLM-L6-v2 loads on CPU (torch +cpu), 384-dim, max_seq_length 256;
  model README says trained at 128 tokens; warm load ~0.4 s, encode ~0.03 s
- Silent truncation proven: chunk over 256 tokens -> cos(full, first 254 tokens) =
  1.000000, max diff 0.0; cos(full, tail) = 0.40
- Token report at 800/120: 110 chunks, max 298, 2 over 256, 51 over 128
- Token report at 600/120: 129 chunks, median 116, p95 177, max 241, 0 over 256,
  48 over 128
- embeddings.py: load_embedding_model (lazy import, CPU, lru_cache per
  (name, local_files_only), errors -> EmbeddingError), count_tokens, embed_chunks
  (normalized float32 (n, 384), order kept, over-limit WARNING + over_limit_chunk_ids,
  empty input skips encode), embed_query (same model + normalization)
- Importing rag_pipeline.embeddings loads neither sentence_transformers nor torch
  (subprocess test)
- Real smoke check: 129 chunks -> (129, 384), norms 1.000000, 0 over limit,
  embed 5.11 s; sanity query "What is a decision tree?" -> top-1 Lecture07 page 4
  (0.838), next 0.580
- local_files_only=True loads from cache with network blocked; flag off with network
  blocked stalled 110 s before cache fallback; uncached model with flag on ->
  EmbeddingError
- Local .env with one line RAG_EMBEDDING_LOCAL_FILES_ONLY=true, git-ignored;
  Settings read True and keep chunk_size 600 from the code default
- chromadb 1.5.9 installed (50 new packages, no upgrades/downgrades, pip check clean,
  +221.0 MB site-packages); pinned >=1.5,<2; editable reinstall changed nothing
- Chroma spike (scripts/chroma_spike.py, temp dir, fake normalized 384-dim vectors):
  PersistentClient with anonymized_telemetry=False; cosine set via
  configuration={"hnsw": {"space": "cosine"}} and read back as "cosine";
  collection metadata embedding_model/embedding_dim read back after reopen;
  embedding_function=None -> query_texts raises ValueError, ~/.cache/chroma not
  created; add() with ids/embeddings/documents/flat metadata round-trips;
  query_embeddings distance == 1 - dot (max diff 6e-08); upsert-only leaves 2
  orphans, delete(where doc_id) + add leaves exactly 3, other doc untouched;
  count/metadata/cosine persist after reopen (same process after
  clear_system_cache, and fresh process); get_max_batch_size() = 5461
- RSS: 34.3 MB before import, 71.0 MB after import chromadb (0.93 s), 91.5 MB after
  the spike
- Store folder cannot be deleted by the process that opened it (PermissionError,
  WinError 32 on HNSW data_level0.bin, and on chroma.sqlite3 in a second run, even
  after clear_system_cache); deletable
  after process exit
- Chroma add() with an existing id is silently ignored (no exception, no warning,
  old record kept) — spike check l
- Chroma 1.5.9 probes: missing collection -> NotFoundError; top_k > count returns all
  without warning; empty collection query -> no results; wrong dimension ->
  InvalidArgumentError; collection names need 3-512 chars [a-zA-Z0-9._-]
- vector_store.py (tests, real Chroma in tmp_path, fake vectors): new collection saves
  model/dim metadata + cosine; reopen with same settings keeps data (also in a fresh
  subprocess); different model or dim -> VectorStoreError and saved metadata
  unchanged; missing metadata or l2 space -> VectorStoreError; replace 5 then 3 ->
  exactly ids :0..:2 with new texts, other doc untouched; 0 chunks -> doc emptied;
  replace twice -> identical content; 9 invalid inputs -> VectorStoreError with no
  delete/upsert call and store unchanged; max_batch_size=2 -> upserts of [2, 2, 1];
  batch capped by client limit; metadata round-trip == to_metadata(); query sorted
  by distance, similarity == 1 - distance == dot (1e-5), int page/chunk_index,
  top_k > count returns all, empty -> [], wrong shape / bad top_k / inf ->
  VectorStoreError; no chunk text in logs
- Settings RAG_CHROMA_PATH (default data/chroma, resolved under project root) and
  RAG_CHROMA_COLLECTION_NAME (default chunks, stripped, non-empty); data/chroma
  covered by the existing .gitignore rule data/chroma/
- Real smoke check (scripts/vector_store_smoke.py, temp store, real model with
  local_files_only from .env): 129 chunks submitted incl. the renamed copy, 86
  stored (copy adds none: same doc_id and ids); research paper 24, Lecture07 17,
  partly blank 2, scanned needs_ocr 0, SVM 43 (+ copy 43); invalid and password PDFs
  skipped with PdfExtractionError; "What is a decision tree?" -> top-1 Lecture07
  page 4, similarity 0.838088 (Phase 4: 0.838), then pages 5/3/2/14 at 0.580-0.560;
  re-run replace of SVM: 43 -> 43, total 86 -> 86
- status_db.py (63 tests, real SQLite file in tmp_path, fake clock): initialize
  creates folder/table/index, user_version 1, twice harmless, version 2 -> error;
  constructor touches nothing; use before initialize -> StatusStoreError and no
  file created (read-write URI open); pending -> processing -> done (chunk_count,
  updated_at increases, created_at kept), needs_ocr (0), failed (message stripped),
  retry clears error and attempts 2; all 12 invalid state/action pairs and 4 unknown
  doc_id actions raise and leave the row unchanged; register of an existing doc
  returns it unchanged; known_doc_ids matrix for max_attempts 1/2/3; recover_interrupted
  changes only processing rows (count 3, attempts kept, idempotent), then retryable
  or known by attempts; new StatusStore on the same file sees the data; CHECK
  constraints reject bad status / negative attempts / negative chunk_count via raw
  SQL; quotes, SQL-injection-like text, Roman Urdu, emoji and Urdu script round-trip;
  message > 2000 chars trimmed, empty rejected; naive clock rejected; non-UTC clock
  stored as UTC ("2026-11-20T12:00:00.000000+00:00"); db file deletable in the same
  process after use incl. a failed transition (connections closed on Windows);
  scan_inbox(known_doc_ids=store.known_doc_ids(3)) -> done and 3-attempt failed
  ALREADY_KNOWN, 2-attempt failed and unregistered NEW
- Settings RAG_STATUS_DB_PATH (default data/status.db, resolved under project root;
  ignored by the existing .gitignore rule *.db) and RAG_MAX_INGEST_ATTEMPTS
  (default 3, >= 1)
- SQLite library 3.45.1 (Python 3.11.9 stdlib sqlite3)
- prepare_document can return 0 chunks with needs_ocr False: a PDF whose pages hold
  only "Page N of 3" footers or one repeated header (3 pages) has text but all of it
  is removed by header/footer removal -> ingestion marks it failed "no chunks
  produced" (test_text_without_chunks_fails)
- ingestion.py (31 tests, tmp_path SQLite + Chroma, stub model, real PyMuPDF PDFs):
  happy path DONE with chunk_count == Chroma count == SQLite chunk_count, attempts 1;
  needs_ocr -> NEEDS_OCR, 0 chunks, 3 stale chunks cleared; invalid and password
  PDFs -> FAILED "PdfExtractionError", attempts 1, no chunks; done / needs_ocr docs
  and a renamed copy -> SKIPPED "already ingested" with prepare/embed/replace never
  called; processing -> SKIPPED "in progress"; retry after failure -> DONE attempts 2,
  error cleared; 3 failures -> SKIPPED "attempts exhausted", no work; injected
  Chroma upsert failure after delete (simulated fault: patched Collection.upsert
  raising InternalError) -> FAILED (VectorStoreError), doc has 0 chunks,
  normal retry -> DONE with exact ids :0..:n-1 (no orphans from n+4 stale chunks),
  other doc untouched; KeyboardInterrupt in replace propagates, status stays
  processing, recover_interrupted -> failed, next ingest DONE attempts 2; embedding
  failure -> FAILED and existing chunks unchanged; doc_id mismatch -> FAILED "file
  changed during processing"; missing file -> FAILED, doc_id None, nothing
  registered; StatusStoreError from register/mark_processing/mark_done/mark_failed
  re-raised with a CRITICAL log; no chunk text in logs; no file moved or changed;
  build_ingest_context runs recovery, uses the model's dimension (12) and the
  settings model name, rejects a collection saved with another model, reports model
  and status store failures as IngestionSetupError
- Real smoke check (scripts/ingestion_smoke.py, temp Chroma + SQLite, real model,
  local_files_only from .env, 600/120): run 1 research paper 24, Lecture07 17,
  partly blank 2, SVM 43 DONE; scanned NEEDS_OCR 0; svm_copy_renamed SKIPPED
  "already ingested"; invalid and password FAILED (PdfExtractionError), attempts 1;
  Chroma 86, SQLite 7 rows. Run 2: all SKIPPED except the 2 FAILED again with
  attempts 2; Chroma 86, SQLite 7. Query "What is a decision tree?" -> top-1
  Lecture07 page 4, similarity 0.838088. Inbox files unchanged (names, sizes, mtimes);
  data/chroma and data/status.db never created
- Expected ingest failures (invalid / password PDF, no chunks) log one ERROR line
  without traceback; an unexpected RuntimeError logs the traceback (tests)
- inbox_ingestion.py (28 tests, tmp folders, tmp SQLite + Chroma, stub model):
  destination_for matrix (done -> processed, needs_ocr -> needs_ocr, failed at/over
  max -> failed, failed below max / pending / processing / unregistered -> None);
  safe_move: free name, collisions -> " (1)" / " (2)" with existing files
  byte-identical, FileExistsError race -> next name, Windows os.rename refuses to
  overwrite (FileExistsError), PermissionError propagates with the source kept;
  mixed inbox run 1 -> text + renamed copy in processed, scanned in needs_ocr,
  invalid + password stay (attempts 1), Chroma holds the text doc's chunks once;
  runs 2 and 3 -> invalid + password move to failed after attempt 3, run 4 ingests
  nothing (spy); duplicate in the same scan ingested once, both in processed; later
  copy of a done doc moved without ingestion; injected PermissionError on the move
  -> file stays, status done, reported, next run moves it without re-ingestion;
  processing record's file stays; scan ERROR file stays and is reported; collision
  with an existing different file in processed -> "a (1).pdf", old file untouched;
  file-content multiset across the four folders identical before/after every run;
  empty inbox -> empty report; files moved into a subfolder of the inbox are not
  rescanned; a destination equal to the inbox is rejected
- Settings RAG_NEEDS_OCR_DIR (default data/needs_ocr, resolved under project root);
  .gitignore rule data/needs_ocr/* added (it was not ignored)
- Real smoke check (scripts/inbox_smoke.py, COPY of data/inbox in a temp dir, temp
  folders/Chroma/SQLite, real model): run 1 processed = research paper, Lecture07,
  partly blank, SVM, svm_copy_renamed; needs_ocr = scanned; inbox = invalid +
  password (attempts 1); Chroma 86. Run 2: same folders, attempts 2. Run 3: invalid
  + password in failed (attempts 3), inbox empty, Chroma 86, SQLite 7. Query top-1
  Lecture07 page 4, 0.838088. data/inbox names, sizes and mtimes identical before
  and after
- 539 pytest tests passing

## Not yet verified / incomplete
- Real log rotation at ~1 MB
- ERROR status on a real locked/unreadable file (only simulated)
- Memory use when hashing very large files
- OCR itself (Phase 13)
- Ligatures and invisible-character rules on real PDF data (no real PDF contained
  any yet; unit tests only)
- Retrieval quality at 600 vs 800 (Phase 6 golden set)
- Denser PDFs may exceed 256 tokens at 600 chars (worst ~2.4 chars/token seen)
- HF model download on a fresh machine / behind a proxy
- Over-limit WARNING on real data (stub only; no real chunk > 256)
- mypy / Protocol conformance of the real SentenceTransformer
- Embedding memory use and batch sizes other than 32
- local_files_only on a machine with no cache (only an uncached model name tested)
- Real model load through Settings with the new .env (flag proven at function level)
- Chroma batches larger than get_max_batch_size() = 5461 (batching logic tested with
  small limits only)
- Concurrent writers to the same store (single process assumed)
- SQLite under concurrent access (busy timeout 5 s set, never exercised)
- Status store on the real data/status.db (tests use tmp_path only)
- A possible figure/table area on research paper page 3 (line-level column switches)

## Files and responsibilities
- AGENTS.md / CLAUDE.md: agent rules (CLAUDE.md imports AGENTS.md)
- pyproject.toml: package metadata and dependencies
- src/rag_pipeline/config.py: Settings + get_settings()
- src/rag_pipeline/logging_config.py: setup_logging()
- src/rag_pipeline/extraction.py: extract_pdf, compute_doc_id, compute_file_doc_id, PdfExtractionError
- src/rag_pipeline/inbox.py: scan_inbox, ScanStatus, InboxScanResult, InboxScanError
- src/rag_pipeline/cleaning.py: clean_page_text, clean_pages, CleanedPage;
  remove_repeated_headers_footers, HeaderFooterResult, RepeatedKey, KeyKind
- src/rag_pipeline/chunking.py: chunk_pages, TextChunk
- src/rag_pipeline/config.py also holds chunk_size / chunk_overlap (MIN_CHUNK_SIZE = 100)
- src/rag_pipeline/preparation.py: prepare_document, PreparedDocument, DocumentChunk
- src/rag_pipeline/embeddings.py: load_embedding_model, count_tokens, embed_chunks,
  embed_query, ChunkEmbeddings, EmbeddingModel, EmbeddingError
- src/rag_pipeline/config.py also holds embedding_model_name, embedding_batch_size,
  embedding_local_files_only, chroma_path, chroma_collection_name, status_db_path,
  max_ingest_attempts
- src/rag_pipeline/vector_store.py: open_collection, replace_document_chunks,
  count_document_chunks, query_by_embedding, RetrievedChunk, VectorStoreError
- src/rag_pipeline/status_db.py: StatusStore (initialize, register_document,
  mark_processing/done/needs_ocr/failed, get_document, list_documents,
  known_doc_ids, recover_interrupted), DocumentRecord, DocumentStatus, StatusStoreError
- src/rag_pipeline/ingestion.py: build_ingest_context, ingest_document, IngestContext,
  IngestResult, IngestOutcome, IngestionError, IngestionSetupError
- src/rag_pipeline/inbox_ingestion.py: ingest_inbox, destination_for, safe_move,
  DestinationFolders, InboxRunReport, FileMove, LeftInInbox, MoveFailure
- src/rag_pipeline/config.py also holds needs_ocr_dir
- tests/: conftest.py, test_config.py, test_logging_config.py, test_extraction.py, test_inbox.py,
  test_cleaning.py, test_headers_footers.py, test_chunking.py, test_preparation.py,
  test_embeddings.py, test_vector_store.py, test_status_db.py, test_ingestion.py,
  test_inbox_ingestion.py
- .env.example: settings template
- .env (local, git-ignored): only overrides (RAG_EMBEDDING_LOCAL_FILES_ONLY=true)
- scripts/measure_chunk_tokens.py: one-off token measurement (reused in Phase 6)
- scripts/chroma_spike.py: Chroma API verification spike (temp dir, fake
  vectors; rerun after chromadb upgrades; k = known Windows lock limitation)
- scripts/vector_store_smoke.py: real end-to-end store/query check on the inbox PDFs
  (temp store, never data/chroma)
- scripts/ingestion_smoke.py: real ingest_document run (twice) on the inbox PDFs plus
  query (temp Chroma + SQLite, never data/chroma or data/status.db; no file moves)
- scripts/inbox_smoke.py: real ingest_inbox run (3 times) on a COPY of data/inbox in a
  temp dir, plus query; proves data/inbox unchanged

## Key decisions
- Python 3.11 only; n8n 2.14.2 via npm -> FastAPI URL from n8n: http://localhost:8000
- PyMuPDF 2.x kept (bundles layout + 4llm, pulls numpy and onnxruntime); AGPL license noted
- Document needs_ocr only when no page has text (blank pages do not block a document)
- Duplicate detection by hashing BEFORE extraction; scan never moves or deletes files
- Duplicate "original" = first file in case-insensitive filename order
- known_doc_ids parameter is the hook for the SQLite lookup in Phase 5
- Settings precedence: environment variable > .env > code default
- NFC not NFKC (NFKC turns x² into x2)
- NFC runs after character removals (required for idempotency)
- ZWJ/ZWNJ kept (emoji sequences; ZWNJ meaningful in Urdu/Persian)
- Indentation and space runs collapsed; ASCII-art alignment lost (accepted,
  raw text still available from extraction)
- Hyphen join only when lowercase on both sides; compound words like
  self-esteem may be wrongly joined (accepted)
- CleanedPage.has_text is a computed property
- Header/footer removal is document-level, kept as a separate step
- Two key kinds (EXACT / PAGE_NUMBER) to avoid deleting slide titles like "Step 1";
  trade-offs: varying "Chapter N" headers kept, a lone "3.2" or year on >= 50% of
  edge lines may be removed; function is designed to run once (not idempotent)
- Chunk size/overlap in characters (originally 800/120, ~200 tokens estimated;
  replaced in Phase 4 Step 1 by 600/120 from real token measurements, see below)
- Page-aware chunks (never cross pages) for clean page citations
- Recursive unit splitting: paragraph -> line -> sentence -> word -> hard cut
- Sentence/word-aligned overlap inside the last chunk_overlap characters
- chunk_pages() allows chunk_size >= 1 (small test sizes); Settings require >= 100
- filename stored as basename only (files move inbox -> processed; doc_id is the identity)
- Deterministic chunk_id "{doc_id}:{chunk_index}" for idempotent upsert in Phase 5
- Chunk metadata is flat primitives (str/int) for the vector store
- needs_ocr is a normal result (0 chunks), not an error
- pages_without_text means "no text at extraction" (OCR candidates), not
  "empty after cleaning"
- Chunk settings recorded in PreparedDocument for reproducibility
- Embedding model all-MiniLM-L6-v2 (English docs and questions)
- chunk_size 600 / overlap 120 chars so all real chunks fit 256 tokens; final tuning
  on the golden set in Phase 6
- A chunk_size cap only affects the long tail of the token distribution, not the median
- Changing chunk settings changes what each chunk_id points to -> re-ingest must
  delete old chunks by doc_id before inserting (no orphan chunks)
- sentence-transformers pinned >=6.3,<7
- Embedding model passed in as a parameter (dependency injection, stub in tests)
- Lazy import of sentence_transformers keeps tests fast and download-free
- normalize_embeddings=True for both chunks and queries (scores comparable,
  threshold-ready for Phase 7)
- local_files_only passed to the SentenceTransformer constructor (not HF_HUB_OFFLINE,
  so it affects only this load); code default False (fresh machines can download),
  local .env sets true
- .env holds only overrides, never a copy of .env.example
- Over-limit chunks are embedded and warned, not split
- embed_chunks takes model_name from the caller because a loaded model does not keep
  its name
- numpy declared as a dependency because it is imported directly
- open_collection uses get_collection first and create_collection only on
  NotFoundError (no get_or_create_collection with metadata, which could hide a model
  mismatch); a mismatch raises, saved metadata is never overwritten
- replace_document_chunks validates everything before delete, then deletes by doc_id
  and upserts (add() would silently ignore existing ids)
- embeddings=None allowed only for zero chunks (needs_ocr documents need no model)
- Batch size = min(max_batch_size, client limit); client limit read from
  collection._client (private Chroma attribute, covered by tests)
- The legacy-embedding-function DeprecationWarning is suppressed only locally around
  the one configuration read in open_collection
- SQLite status: no file_hash column (doc_id IS the SHA-256 file hash); documents
  table only, jobs table in Phase 8
- State machine: new -> pending; pending | failed -> processing (attempts + 1, error
  cleared); processing -> done (chunk_count) | needs_ocr (0) | failed (error, max
  2000 chars); anything else raises and changes nothing
- Each transition is one literal conditional UPDATE (WHERE doc_id = ? AND status IN
  (...)) inside BEGIN IMMEDIATE; rowcount 0 -> error naming the current status
- known_doc_ids(max_attempts) = done, needs_ocr, processing, failed with attempts >=
  max; pending and failed below max are retried
- processing is skipped at runtime and recovered at startup: recover_interrupted()
  turns every processing row into failed ("interrupted: ..."), attempts unchanged
- register_document never resets an existing row (a renamed copy keeps the first
  filename and status)
- One short-lived connection per call, closed with contextlib.closing (Windows file
  locks); non-initialize calls open the file read-write via URI so a missing database
  is an error, not a silently created empty file
- Timestamps: UTC ISO 8601 with offset and microseconds; clock injectable for tests
- ingest_document catches Exception, never BaseException: ordinary errors end as
  failed; interrupts propagate and leave processing for startup recovery
- One result per document (IngestResult), not an exception, so a caller can continue
  with the next file; only StatusStoreError is re-raised (CRITICAL; the register is
  broken and continuing is unsafe)
- build_ingest_context is the only place that wires store, model and collection from
  settings; it runs recover_interrupted once and opens the collection with the
  loaded model's dimension and settings.embedding_model_name
- 0 chunks without needs_ocr -> failed "no chunks produced" (not done with 0 chunks)
- Skip before any work: done / needs_ocr ("already ingested"), processing ("in
  progress"), failed with attempts >= max ("attempts exhausted")
- Fixed order: Chroma replace before mark_done; a hashing failure registers nothing
  (no doc_id exists)
- The second hash inside prepare_document is compared with the registered doc_id
  (file changed during processing -> failed)
- prepare_document gets Settings.model_construct(chunk_size, chunk_overlap) (no env
  read; values validated by IngestContext)
- File placement follows the document's status, decided only by destination_for:
  done -> processed_dir; needs_ocr -> needs_ocr_dir; failed with attempts >= max ->
  failed_dir; failed below max, pending, processing -> stay in the inbox
- A duplicate copy follows its document's status (copy of a done doc -> processed)
- Name collision -> "stem (1).pdf", "stem (2).pdf", ...; never overwrite, never delete
  (check + rename that raises FileExistsError, next name on a race)
- A failed move is logged WARNING, reported, and leaves the status unchanged; the
  next run retries it (placement follows status), without re-ingestion
- ingest_inbox: NEW files ingested first, then every file with a doc_id placed;
  scan ERROR files stay and are reported; StatusStoreError propagates

## Open questions
- Questions will be asked in English; document types are lecture slides and
  research papers (decided).
- Roman Urdu notes: still to be checked with the Phase 6 golden set
  (all-MiniLM-L6-v2 vs a multilingual model).

## Verified versions
Python 3.11.9 | pip 26.3 | setuptools 84.2.1 (upgraded from 65.5.0 by the
sentence-transformers install) | pydantic-settings 2.16.2 |
pydantic 2.14.3 | pytest 9.2.0 | pymupdf 2.1 | numpy 2.4.6 (declared dependency
since Phase 4 Step 2b) | onnxruntime 1.31.1 |
sentence-transformers 6.3.0 | torch 2.15.0+cpu | transformers 5.23.0 |
tokenizers 0.23.3 | huggingface-hub 2.4.0 | chromadb 1.5.9 |
git 2.55.0 | node 24.14.1 | npm 11.11.0 | n8n 2.14.2

## Hardware (verified)
Intel Core i5-14400F (10C/16T), 15.8 GB RAM (6.6 GB free at check),
AMD Radeon HD 7000 (CPU only), Windows 11, PowerShell 5.1

## Risks
- Windows console (cp1252) cannot print some characters (e.g. emoji) from extracted text;
  app.log is UTF-8 and unaffected. Revisit when text is logged to console.
- Free RAM 6.6 GB at check time; review before choosing the Ollama model (Phase 7)
- Test PDFs in data/inbox are git-ignored (local only)
- Line endings: git LF->CRLF warnings on Windows; add .gitattributes in Phase 14.
- Only 15-token margin at 600 (max 241) -> token guard needed in Phase 4 Step 2 (done)
- HF symlink warning on Windows (degraded cache, harmless)
- get_sentence_embedding_dimension renamed to get_embedding_dimension
  (old name gives a FutureWarning)
- With local_files_only=false and no network, a model load can stall ~2 min
- Windows: the Chroma store folder stays locked by the process that opened it (seen on
  data_level0.bin and chroma.sqlite3, even after clear_system_cache) -> no rmtree of
  the store in tests
- DeprecationWarning "legacy embedding function config" whenever
  collection.configuration is read on a collection without an embedding function
- Chroma caches the system per path in-process (SharedSystemClient), so a
  same-process reopen is not a persistence proof (clear the cache or use a fresh
  process)
- Chroma attaches its downloading DefaultEmbeddingFunction unless
  embedding_function=None is passed on EVERY create/get of a collection
- chromadb adds 50 transitive packages (+221 MB, incl. kubernetes, grpcio,
  uvicorn 0.54.0); Phase 8 must still declare uvicorn/fastapi explicitly
- Delete and upsert in replace_document_chunks are not atomic: a crash in between
  leaves a document with no or partial chunks -> SQLite status must stay
  processing/failed until the replace returns, and a re-run fixes it
- vector_store reads the batch limit through the private collection._client; a
  chromadb upgrade may break it (test_batch_size_is_capped_by_client_limit catches it)
- recover_interrupted assumes a single process: running it while another process is
  ingesting would mark that live work failed
- A document stuck in processing is invisible to scans until the next startup
  recovery
- A failed document may keep no or partial chunks in Chroma (upsert failure after
  delete); queries can still return them until a retry succeeds -> consider
  filtering query results to done doc_ids (Phase 6/7)
- Expected failures are logged once by extraction and once by ingestion (one line,
  no traceback)
- ingestion passes Settings.model_construct(chunk_size, chunk_overlap) to
  prepare_document (validation skipped; any other Settings field would silently use
  its default) -> refactor prepare_document to take chunk values (Phase 11)
- An IngestContext built by hand could pair a model with the wrong name (only
  build_ingest_context guarantees they match)

- safe_move needs the destination on the same volume as the inbox (os.rename); a
  cross-volume folder setting makes every move fail (reported, file stays)
- A file that stays in the inbox after its last allowed attempt can only happen if the
  move keeps failing; it is retried every run (WARNING each time)
- Duplicate copies pile up in processed/ (no dedup of files, by design: never delete)
- safe_move on non-Windows uses hard link + unlink (untested); a crash between them
  leaves the file in both places -> next run moves the inbox copy as 'name (1).pdf'
  (no loss, one extra copy). Test on Linux before Docker (Phase 13).

## Next precise task
Phase 6 Step 1 — retrieval prototype + golden evaluation set (no LLM): filter query
results to done doc_ids; 10-15 questions with expected document/page; hit@k and MRR.
