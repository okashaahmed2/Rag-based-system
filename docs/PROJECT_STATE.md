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
- Phase 5 Step 1a (ChromaDB install + spike): COMPLETE (pending commit)
- Next: Phase 5 Step 1b — vector_store.py

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
- 356 pytest tests passing

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
- Chroma with real embeddings (spike used fake normalized vectors)
- Chroma batches larger than get_max_batch_size() = 5461
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
  embedding_local_files_only
- tests/: conftest.py, test_config.py, test_logging_config.py, test_extraction.py, test_inbox.py,
  test_cleaning.py, test_headers_footers.py, test_chunking.py, test_preparation.py,
  test_embeddings.py
- .env.example: settings template
- .env (local, git-ignored): only overrides (RAG_EMBEDDING_LOCAL_FILES_ONLY=true)
- scripts/measure_chunk_tokens.py: one-off token measurement (reused in Phase 6)
- scripts/chroma_spike.py: Chroma API verification spike (temp dir, fake
  vectors; rerun after chromadb upgrades; k = known Windows lock limitation)

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
- A wrong model_name passed to embed_chunks would mislabel the collection -> Phase 5
  must load the model and pass settings.embedding_model_name in one place
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

## Next precise task
Phase 5 Step 1b — src/rag_pipeline/vector_store.py (open/create collection with cosine
+ model metadata check, replace_document_chunks = delete by doc_id then insert, query
by embedding) with tests.
