# Development Log

## 2026-11-12 — Phase 1
- Step 0: read-only environment audit. Found Python conflict (python=3.11.9,
  py/python3=3.14.4), n8n 2.14.2 via npm, no Ollama, no Docker, CPU-only machine.
- Step 1: git init (main), .gitignore, folder skeleton, Python 3.11 venv,
  AGENTS.md + CLAUDE.md. Commit 1dad7db, pushed using GitHub noreply email.
- Step 2: pyproject.toml, editable install, config.py (pydantic-settings),
  logging_config.py, .env.example, 19 tests passing. Commit 9b97540, pushed.
  Issue: pip SSL verification failure (likely antivirus/proxy HTTPS inspection);
  worked around once with truststore.
- Step 2b: upgraded venv pip 24.0 -> 26.3; PyPI reachable without workaround.
- Manual check: log line appears on console and in logs/app.log.

## 2026-11-13 — Phase 2
- Step 1: extraction.py with PyMuPDF 2.x, SHA-256 doc_id, per-page has_text.
  (Claude Code started this step on its own after a resume prompt was pasted into
  Claude Code by mistake; work was kept and reviewed.)
- Review fix: needs_ocr changed from "any blank page" to "no page has text";
  pages_without_text added. Verified on 6 real files (scanned, partly blank,
  password-protected, fake, typed notes, renamed copy).
- Found: notes contain Roman Urdu; console cp1252 cannot print emoji.
- Step 2: compute_file_doc_id (streaming) and inbox.py scan_inbox with duplicate
  detection; renamed copy flagged DUPLICATE_IN_SCAN. 54 tests passing.
- Lesson: resume prompts go to claude.ai chat, never to Claude Code.

## 2026-11-14 — Phase 3
- Step 1: cleaning.py with clean_page_text (pure, idempotent) and clean_pages
  (CleanedPage, page numbers preserved). 148 tests passing.
- Review: Claude Code moved NFC after removals to keep idempotency
  (zero-width char between letter and combining accent). A wrong test
  expectation was fixed in the test, not the code.
- Real-file check: ASCII-art diagrams lose alignment (accepted trade-off).
- Fix: keep ZWJ/ZWNJ; handle U+2028/U+2029/U+0085; remove DEL and C1
  controls. 161 tests passing.

## 2026-11-18 — Phase 3
- Step 2: remove_repeated_headers_footers (document-level, edge lines repeated
  on >= 3 and >= 50% of text pages); HeaderFooterResult reports removed keys.
- Review: digit normalisation deleted short-page content (e.g. slide titles
  "Step 1"/"Step 2") -> split into EXACT and PAGE_NUMBER keys; only
  page-number-like lines are compared with digits normalised.
- Incident: a shell heredoc turned "\n" into real newlines in source files;
  fixed with exact edits; rule added to AGENTS.md (Edit tool only).
- Invisible literal characters in cleaning.py and tests replaced with escapes
  (string values verified identical to before).
- En/em dash page-number footers added. 224 tests passing.

## 2026-11-19 — Phase 3
- Step 3: chunking.py with chunk_pages (page-aware, character budget, recursive
  unit splitting, exact slices) and TextChunk; RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP
  settings with validation.
- Review: overlap was silently 0 on long paragraphs (overlap built only from whole
  trailing units; tests only checked overlap <= chunk_overlap) -> overlap now
  starts at the earliest sentence start, else word start, inside the last
  chunk_overlap characters.
- A conditional assert made one test vacuous; rewritten with a deterministic input.
- A scratchpad script named inspect.py shadowed the stdlib module and crashed
  pydantic imports; renamed.
- A real two-column research paper and lecture slides were added to the inbox and
  checked (overlap, reading order, header removal, hyphen joins).
- 314 tests passing.
- Step 4: preparation.py with prepare_document, PreparedDocument and DocumentChunk
  (chunk_id = "{doc_id}:{chunk_index}", flat to_metadata()).
- Real-file check on all inbox PDFs (chunks, needs_ocr, PdfExtractionError, renamed
  copy with identical doc_id and chunk_ids).
- Human check of Lecture07 slides 8/13: both fine (text complete, readable order).
- Phase 3 complete. 329 tests passing. Commit 10f2a29.

## 2026-11-19 — Phase 4
- Step 1: installed sentence-transformers 6.3.0 (pulls torch 2.15.0+cpu,
  transformers 5.23.0, tokenizers 0.23.3, huggingface-hub 2.4.0; setuptools upgraded
  65.5.0 -> 84.2.1). No SSL problems. all-MiniLM-L6-v2 loads on CPU: 384 dims,
  max_seq_length 256 (model README: "input text longer than 256 word pieces is
  truncated"; trained at 128 tokens). Model cache ~87 MB; HF symlink warning on
  Windows (degraded cache, harmless). Added scripts/measure_chunk_tokens.py
  (real tokenizer, special tokens included, no truncation). Token report at
  800/120: 110 chunks, median 122, p95 214, max 298, 2 chunks over 256 (both in the
  research paper, table-heavy text), 51 over 128.
- Step 1b: truncation proven on the 298-token chunk: cos(full, first 254 tokens) =
  1.000000 with max diff 0.0, cos(full, tail) = 0.40 -> the tail is silently ignored.
  chunk_size default 800 -> 600 (overlap 120 unchanged) in config.py and
  .env.example; one default test updated. At 600/120: 129 chunks, median 116,
  p95 177, max 241, 0 over 256, 48 over 128. Script now also reports "over 128".
- Lesson: a chunk_size cap changes only the long tail of the token distribution
  (max 298 -> 241, p95 214 -> 177), while the median barely moves (122 -> 116) and
  most chunks over 128 stay over it (51 -> 48).
- Step 1c: pinned sentence-transformers>=6.3,<7 (reinstall changed no packages),
  docs updated, committed 8e2bb22 and pushed. 329 tests passing.
- Step 2: embeddings.py with load_embedding_model (lazy import, CPU, lru_cache per
  (name, local_files_only)), count_tokens, embed_chunks (normalized float32, token
  guard with WARNING + over_limit_chunk_ids) and embed_query; Settings
  embedding_model_name / embedding_batch_size (>= 1) / embedding_local_files_only.
  27 new tests with a stub model and a fake sentence_transformers module (no
  download). Real smoke check: 129 chunks -> (129, 384), norms 1.000000, 0 over the
  limit, embed 5.11 s; sanity query "What is a decision tree?" -> top-1 Lecture07
  page 4 (0.838), next 0.580. local_files_only check with a dead proxy: flag on
  loads from cache with no network attempt; flag off retries the Hub and stalls
  110 s before falling back to the cache; uncached model with flag on ->
  EmbeddingError.

## 2026-11-20 — Phase 4
- Step 2b: numpy>=2,<3 declared (imported directly; reinstall changed no packages);
  local .env with one override (RAG_EMBEDDING_LOCAL_FILES_ONLY=true); docs updated,
  committed f77f604 and pushed.
- Phase 4 complete. 356 tests passing.

## 2026-11-20 — Phase 5
- Step 1a: installed chromadb 1.5.9 (Requires-Python >=3.9), pinned >=1.5,<2. Dry-run
  and install: 50 new packages (incl. grpcio, kubernetes, opentelemetry-*, uvicorn,
  httpx, aiohttp), no upgrades or downgrades; pip check clean; editable reinstall
  changed no packages. site-packages 1,205.9 MB -> 1,426.8 MB (+221.0 MB). No SSL
  problems. Commit 07f9e3b, pushed.
- Spike scripts/chroma_spike.py (temp dir, fake 384-dim vectors, no model):
  cosine via configuration={"hnsw": {"space": "cosine"}} read back as "cosine";
  legacy metadata {"hnsw:space": "cosine"} also works without a warning but leaves
  the key in collection metadata. Collection metadata embedding_model/embedding_dim
  persists. embedding_function=None: query_texts raises ValueError, ~/.cache/chroma
  never created. Chroma distance = 1 - dot for normalized vectors (max diff 6e-08).
  Orphans: upsert of 3 over 5 leaves 2; delete(where doc_id) + add leaves exactly 3,
  other doc untouched. Persistence verified in-process (after clear_system_cache)
  and in a fresh process. get_max_batch_size() = 5461. RSS 34.3 MB -> 71.0 MB after
  import (0.93 s) -> 91.5 MB after the spike.
- Found: rmtree of the store in the same process raises PermissionError (WinError 32,
  HNSW data_level0.bin locked) even after dropping the client and
  clear_system_cache(); deletion works after the process exits.
- Found: reading collection.configuration with no embedding function emits
  DeprecationWarning "legacy embedding function config".
- 356 tests passing.
- Step 1a-fix: spike check k reclassified as KNOWN_LIMITATION (excluded from the
  failure count); spike now exits 0 with all other checks VERIFIED. The lock was seen
  on data_level0.bin in one run and on chroma.sqlite3 in another, so the whole store
  folder is locked by the opening process. Repo docs/ had diverged from the
  claude.ai project copies (commit hashes, Step 2b date note, .env facts, Risks)
  because end-of-session doc updates were applied to the project but not to the
  repo; merged section by section, Step 1a facts kept. 356 tests passing.
- Lesson: End-of-session doc updates must be applied to BOTH the repo docs/ and the
  claude.ai project files.
- Step 1b: vector_store.py with open_collection (get first, create only if missing,
  cosine + embedding_model/embedding_dim metadata, saved model/dim/space checked on
  every open), replace_document_chunks (all validation before delete, then
  delete(where doc_id) + batched upsert), count_document_chunks, query_by_embedding
  (query_embeddings only, RetrievedChunk with similarity = 1 - distance). Settings
  RAG_CHROMA_PATH (data/chroma, git-ignored by the existing data/chroma/ rule) and
  RAG_CHROMA_COLLECTION_NAME (chunks, non-empty). 45 vector store + 5 config tests
  (real Chroma in tmp_path, fake vectors, fresh-subprocess persistence). 406 tests
  passing.
- Probed Chroma 1.5.9: missing collection -> NotFoundError; top_k > count returns all
  without warning; empty collection -> [[]]; wrong dimension -> InvalidArgumentError;
  n_results=0 -> TypeError; collection names need 3-512 chars [a-zA-Z0-9._-].
- Spike check l: add() with an existing id is silently ignored (no exception, no
  warning, old record kept, count unchanged) -> replace must use upsert after delete.
- Real smoke check (scripts/vector_store_smoke.py, temp store): 129 chunks submitted
  (incl. the renamed SVM copy), 86 stored (copy has the same doc_id/ids and adds
  none), needs_ocr doc 0 chunks, invalid and password PDFs skipped with
  PdfExtractionError. "What is a decision tree?" -> top-1 Lecture07 page 4,
  similarity 0.838088 (same as Phase 4). Re-run replace: 43 -> 43, total 86 -> 86.
- Found: the smoke check's first expectation (collection count 129) was wrong: 129
  counts the duplicate copy; the store correctly holds 86.
- Step 1b committed 2116068 and pushed. 406 tests passing.
- Step 1c: status_db.py with StatusStore (documents table, CHECK constraints, status
  index, user_version 1), state machine pending -> processing -> done / needs_ocr /
  failed with retry failed -> processing, each transition one literal conditional
  UPDATE in BEGIN IMMEDIATE, known_doc_ids(max_attempts) for scan_inbox,
  recover_interrupted() for startup. Settings RAG_STATUS_DB_PATH (data/status.db,
  covered by *.db) and RAG_MAX_INGEST_ATTEMPTS (3). 63 status_db + 5 config tests,
  incl. scan_inbox integration and deleting the db file in the same process.
  474 tests passing.
- Found: data/status.db-journal was not covered by .gitignore; added *.db-journal,
  *.db-wal, *.db-shm.
- Step 1c committed 18102d7 and pushed. 474 tests passing.
- Step 2: ingestion.py with build_ingest_context (only wiring place; startup recovery,
  model dimension -> collection) and ingest_document (hash -> register -> skip check
  -> processing -> prepare -> needs_ocr | embed -> replace -> done; Exception ->
  failed and returned; StatusStoreError re-raised; BaseException propagates). 31
  tests incl. fault injection (upsert failure after delete, KeyboardInterrupt).
  505 tests passing.
- Found: prepare_document returns 0 chunks with needs_ocr False when all text is
  repeated headers/footers (3 pages of "Page N of 3") -> treated as failed "no chunks
  produced".
- Real smoke check (scripts/ingestion_smoke.py, temp stores): run 1 matches the
  expected table (Chroma 86, SQLite 7 rows, copy SKIPPED), run 2 all SKIPPED except
  the invalid and password PDFs FAILED with attempts 2; top-1 Lecture07 page 4
  0.838088.
- Log fix: expected failures (PdfExtractionError, IngestionError) now log one ERROR
  line without traceback; other exceptions keep the traceback. 4 new tests, 509 tests
  passing.
- Step 2 committed df9450d and pushed.
- Step 3: inbox_ingestion.py with destination_for (placement follows status),
  safe_move (first free "stem (n)" name, rename that never overwrites, FileExistsError
  race handled) and ingest_inbox (scan with known ids -> ingest NEW -> place every
  file; move failures reported and healed on the next run). Setting RAG_NEEDS_OCR_DIR.
  28 + 2 tests incl. conservation of file contents across the four folders.
  539 tests passing.
- Found: data/needs_ocr/ was not git-ignored; added data/needs_ocr/* (+ .gitkeep
  exception, no .gitkeep created).
- Real smoke check (scripts/inbox_smoke.py, temp copy of data/inbox): run 1 -> 5 in
  processed, scanned in needs_ocr, invalid + password stay; run 3 -> both in failed,
  inbox empty; Chroma 86, SQLite 7; top-1 Lecture07 p4 0.838088; data/inbox unchanged.
- Step 3 committed 4dc8c15 and pushed.

## 2026-11-23 — Phase 6
- Step 1 probe (temp dir, fake vectors): Chroma 1.5.9 query with query_embeddings and
  where={"doc_id": {"$in": [...]}} works with several ids and with one; n_results
  larger than the matches returns all matches; no match -> []; an empty $in list
  raises builtins.ValueError (not ChromaError).
- Step 1: retrieval.py with build_retrieval_context (no recover_interrupted; missing
  status db / Chroma folder -> RetrievalError, nothing created) and retrieve (validate
  -> done ids -> [] if none -> embed -> Chroma pre-filter $in done ids -> post-check).
  vector_store.query_by_embedding got an optional doc_ids filter. evaluation.py with
  load_golden_set (JSON Lines, strict, line-numbered errors), first_relevant_rank
  (doc_id + page), hit_at_k, mrr_at_k. scripts/list_doc_ids.py and
  eval/golden_set.example.jsonl (2 fake lines). 34 retrieval + 7 vector store +
  69 evaluation + 4 list_doc_ids tests. 653 tests passing.
- No real evaluation run and no real-data retrieval yet (Step 2).
- Review fix: the read path no longer creates a missing collection.
  open_collection(..., create=False) raises CollectionNotFoundError (saved
  model/dim/space still checked); build_retrieval_context maps it to RetrievalError
  "...; run ingestion first". Found: on an existing empty Chroma folder the client
  still writes chroma.sqlite3 (no collection). 6 vector store + 2 retrieval tests.
  661 tests passing.
- Step 1 committed 8a641b4 and pushed.
- Step 2 prep: golden set drafted from extract_pdf page text + keyword search only (no
  embeddings/Chroma), longest word run shared with the source page <= 3; mentor
  cross-checked every doc_id, page and negative with pdftotext. Promoted to
  eval/golden_set.jsonl changing only notes (all other fields proven identical per
  id); draft deleted. 15 questions (13 answerable, 2 negative; 4 LACE, 4 Lecture07,
  5 Roman Urdu notes). Found: SVM,K-mean,KNN.pdf is the Roman Urdu notes, so the
  roman_urdu tag is confounded with one document. 661 tests passing.
- Step 2 prep committed e835fe9 and pushed.
- Step 2: evaluation.evaluate (any retrieve callable; per-question ranks, hit@1/3/5/10
  and MRR@10 overall and per tag, similarity summary) and scripts/run_eval.py (one-time
  inbox copy to data/eval/corpus, eval store data/eval/stores/<config>/ with manifest,
  golden doc_id done check, real read path, eval/results/<date>_<config>.json).
  .gitignore: data/eval/. 16 evaluation + 14 run_eval tests. 691 tests passing.
- Baseline 600/120 MiniLM (13 answerable): hit@1 0.692, hit@3 0.846, hit@5 1.000,
  hit@10 1.000, MRR@10 0.795; misses at 1: q01 (rank 3), q06 (4), q10 (4), q12 (2).
  Negatives top-1 0.253 / 0.218, answerable top-1 0.363-0.809. Second run: identical
  results, nothing embedded (done files SKIPPED; invalid + password retried, attempt 2).
- Found: the first printed summary named the duplicate svm_copy_renamed.pdf for
  SVM-notes hits (doc_id -> filename map from the corpus list, last copy won); the
  JSON (Chroma metadata) was correct. Fixed: summary uses the hit's filename, falling
  back to the first corpus file; regression test added.
- Step 2 committed 8bcba2b; baseline regenerated clean (dirty=false).
- Step 3a: decision rule pre-registered in PROJECT_STATE before any run (replace
  600/120 only if hit@5 stays 1.000, MRR@10 +>= 0.05, no answerable question leaves
  the top 5). run_eval dirty flag now ignores only untracked eval/results/ files
  (git status --porcelain=v1 -z --untracked-files=all; tested on a tmp git repo).
  scripts/compare_eval.py (metrics + deltas, rank changes, outside-top-5 flags,
  similarity, rule check; refuses different golden sets / ks). 721 tests passing.
  Committed 6d8a536 and pushed.
- Step 3a runs from the clean tree (commit 6d8a536, dirty false): token reports on the
  eval corpus 400/80 max 167 (0 over 256), 800/120 max 298 (2 over 256, = Phase 4).
  400/80: hit@1 0.615, hit@3 0.846, hit@5 1.000, MRR@10 0.753; 800/120: 0.615, 0.769,
  1.000, MRR@10 0.746; baseline 600/120: 0.692, 0.846, 1.000, MRR@10 0.795. Rule: both
  FAIL on MRR (-0.042 / -0.049) -> 600/120 stays. No question left the top 5.
- Found: compare_eval column labels are cut to 18/22 characters, so same-prefix configs
  read as "all-minilm-l6-v2_6/_4/_8" (cosmetic, not fixed). data/inbox unchanged,
  data/chroma and data/status.db absent.
- Step 3 close: 400/80 and 800/120 result files committed (unchanged, SHA-256 checked
  before/after). Step 3b (multilingual model) skipped by user decision: roman_urdu not
  worse than overall at baseline (MRR@10 0.750 vs 0.795, hit@5 1.000), English
  questions with English technical terms in the notes, paraphrase-multilingual-
  MiniLM-L12-v2 max_seq_length 128, multilingual-e5-small needs query/passage
  prefixes; keep all-MiniLM-L6-v2 with revisit triggers. compare_eval labels fixed
  (chunk part for a shared model, full names otherwise; 3 tests). Chunk settings stay
  600/120. 724 tests passing. Phase 6 complete.

## 2026-11-23 — Phase 7
- Step 0 (read-only): Ollama not installed; 15.8 GB RAM (5.99 GB free with
  Chrome + VS Code open), 2 x 8 GB DDR4-3200 on two controllers (dual
  channel), i5-14400F 6P+4E/16T; D: 388 GB free, C: 69 GB free; httpx 0.28.1
  present via chromadb; Phase 6 commit f1052c3.
- Ollama 0.40.17 installed by the user on D: (OllamaSetup.exe
  /DIR="D:\Ollama\app"; user variable OLLAMA_MODELS=D:\Ollama\models set
  before install); granite4.2:3b pulled (2.09 GB on D:, nothing on C:).
  The user's terminal was cmd, not PowerShell.
- Step 1 (measurement only): granite4.2:3b 3.7B Q4_K_M, loaded 2.7 GB,
  100% CPU, context 4096 by default; prompt eval ~118 tok/s (748 tokens ->
  6.3 s), generation 15-17 tok/s, cold load 2.6 s; an identical prompt was
  served from Ollama's cache. Without "think" the reasoning appeared inside
  "response"; "think": false removed it on short prompts.
- Step 2: llm.py (httpx client: chat, check_ollama, LLMError family;
  stream false, think false, temperature + num_ctx from settings), 5 LLM
  settings, httpx>=0.28,<1 declared (no package changed). 86 MockTransport
  tests, 810 passing.
- Found: "localhost" resolves to ::1 first and Ollama listens on 127.0.0.1
  only -> ~2.2 s extra per new connection (3 + 3 runs: 2.20 s vs 0.15 s).
  Default changed to http://127.0.0.1:11434. Commit 79db4b0 (811 tests).
- Step 3a: answering.py: fixed SYSTEM_PROMPT, build_rag_messages
  (<source id="S1" file=".." page=".."> blocks, tag neutralisation, quotes in
  filenames escaped), check_answer (ANSWERED / NOT_FOUND / NO_CITATIONS /
  BAD_CITATIONS / EMPTY). [S#] chosen because the LACE paper contains its own
  "[12]" references. 68 tests, 879 passing. Commit 6566a45.
- Step 3b: answer_question (retrieve -> no chunks: NOT_FOUND without the LLM
  -> messages -> chat -> check -> [S#] mapped to doc_id/file/page),
  RAG_QUERY_TOP_K=5, scripts/rag_smoke.py (golden set, eval store read path).
- First golden run (granite): q01 and q12 timed out at 120 s; reasoning leaked
  into content ending with "</think>" despite think:false (q13: 1593 tokens,
  109 s); the format check passed the leaked text because it contained [S#];
  negatives explained first and ended with a NOT_FOUND line -> NO_CITATIONS;
  q03 cited all 5 sources in a non-answer (citation_hit misleading).
- Diagnosis (scripts/llm_think_probe.py, 3 questions x 4 variants): granite's
  template defaults enable_thinking=True and ends the prompt with "<think>";
  think:false only turns off Ollama's parser, so reasoning lands in content
  (732-848 tokens). Without "think" the reasoning goes to message.thinking but
  is still generated (q02 used all 2048 tokens, empty answer). llama3.2:3b:
  3-38 tokens, ~2 s generation, correct statuses, accepts think:false.
- Fixes: "</think>" strip in llm.py (thinking_leaked flag), NOT_FOUND line
  rule, RAG_LLM_MAX_ANSWER_TOKENS=256 (num_predict), flushed script output.
- llama3.2:3b with "at most 5 sentences": rule conditions 1-4 passed (11/13),
  but review found padding errors (q04 "4-6 times faster" = embedding calls,
  q10 "fit more tightly", q03 invented embedding model).
- Rule 5 changed to "1 to 3 short sentences. Do not add anything that is not
  stated in the sources.": padding errors gone, q03 became NOT_FOUND;
  10/13 ANSWERED with citation hit (condition 3 FAIL), 0 errors, negatives
  2/2, median 7.9 s; identical answers on re-run (temperature 0). All misses
  had an expected page retrieved: q01 (Table 1 flattened by cleaning),
  q03 (answer chunk possibly not retrieved; unverified), q12 (cites p.20,
  which has no K-Means++; expected pp. 34/36 were retrieved).
- Found: a query-only run changes the bytes of the eval store's Chroma files
  (chroma.sqlite3, data_level0.bin, length.bin); status.db and manifest.json
  unchanged.
- Decision: default llama3.2:3b by mentor judgment (pre-registered rule
  failed on condition 3; granite failed conditions 1 and 4). Commit ff40009.
  948 tests passing. Phase 7 complete.

## 2026-11-23 — Phase 8
- Step 0 (read-only): port 8000 free; uvicorn 0.54.0 already present via
  chromadb; plan for /ingest/scan, /status/{job_id}, /query with a jobs table,
  BackgroundTasks and one shared httpx client. No installs.

## 2026-11-26 — Phase 8
- Step 1: fastapi 0.165.0 + starlette 1.10.0 installed (nothing else changed);
  fastapi>=0.165,<0.166 and uvicorn>=0.54,<1 declared. api/app.py: create_app
  factory (run with --factory), lifespan builds the ingest context once and
  derives the RetrievalContext from it (one Chroma client, one model), one
  shared httpx.Client closed on shutdown, GET /health (plain def, Ollama
  problems -> 200 "degraded"). configure_logging parameter so tests keep
  pytest's logging. Routes registered with app.add_api_route instead of an
  APIRouter (FastAPI 0.165 _IncludedRouter observation by Claude Code, not
  independently verified). RAG_API_HOST / RAG_API_PORT settings. Commit 59d53ad.
- User's manual test from their own terminal: server up in 7.39 s, /docs
  opened, Ctrl+C logged "API stopped: HTTP client closed"; a /health call failed
  only because it was sent after the server had stopped.
- Step 1c: diagnosis runs showed huggingface_hub 2.4.0 sending GET
  {ENDPOINT}/api/agent-harnesses on every model load even with
  local_files_only=True (build_hf_headers -> detect_agent -> _fetch_registry;
  cached 24 h in ~/.cache/huggingface/.agent_harnesses.json; independent of
  agent env vars). Fix: apply_hf_runtime_env (setdefault
  HF_HUB_DISABLE_TELEMETRY=1, and HF_HUB_OFFLINE=1 with local_files_only),
  called in load_embedding_model before the sentence_transformers import ->
  no Hub request, cache file not recreated. .env values are not exported to
  os.environ, so the variables are set in the process. Commit e4f7ec3.
- Step 2: jobs table and schema v2 migration in status_db.py (_MIGRATIONS);
  documents rows byte-identical after v1 -> v2 (tests + eval-store copy, 7
  rows, original hash unchanged). create_job_or_get_active allows at most one
  queued/running job of any kind; 50 rounds x 2 threads gave exactly one job
  per round. Commit 0b5ae53.

## 2026-11-27 — Phase 8
- The Claude Code session that built Step 3 was deleted before it reported.
  A new session ran a read-only resume check: Step 2 was already committed
  and pushed, including idx_jobs_one_active (the partial unique index); Step 3
  code was complete but uncommitted; 1106 tests passing. Nothing was lost.
- Step 3 verified against its spec (9 items present): lifespan job recovery,
  POST /ingest/scan (202, active job returned with already_running), GET
  /status/{job_id} (404 for unknown or malformed ids), run_job, summary_from_
  report, typed /health documents + active_job. Real server on scratch copies:
  double POST -> same job_id, already_running true; job ~6 s; /status median
  ~5 ms, max 17-29 ms while running.
- Found: per-file error strings in job summaries held absolute paths (e.g.
  PdfExtractionError messages). Fixed in jobs.py only: the file's full path,
  resolved path and absolute parent folder are replaced by the file name before
  trimming (5 tests, 1111 passing). Real check: no scratch path and no ":\" in
  /status JSON. Logs and the documents table keep the full text.
- Mentor's expectation for the second scan (done files skipped) was wrong: the
  first scan had already moved them to processed, so the second scan saw only
  the invalid and password PDFs, retried at attempt 2. Commit d5316b1.
- Step 4: POST /ingest {filename} for a file already in the inbox:
  validate_inbox_filename (format -> 422, not a regular file directly in the
  inbox -> 404, both before any job is created), 409 with the active job,
  ingest_one_file (ingest_document + placement by status, missing file ->
  failed result), run_job generalised via _run_work without changing the scan
  path. 91 tests, 1202 passing. Real check: bad names 422 with no job row,
  same name after success 404, POST /ingest right after a scan -> 409 with the
  scan's job_id, not_a_pdf_test.pdf failed at attempt 2 and stayed in the
  inbox. Commit 87dde5a.
- Step 5: POST /query (QueryRequest stripped 1..1000 chars, top_k strict int
  1..20; fail-closed answer; citations only for answered; sources without chunk
  text; timings null when the LLM was not called). Errors: RetrievalError 503,
  LLMUnavailable/ModelNotFound 503, LLMResponseError 502, LLMTimeoutError 504;
  the ValueError -> 422 mapping was removed before the commit (a ValueError can
  only be a bug -> 500). 1237 passing. Commit 0ee831e.
- Step 5 real check: before ingestion 200 not_found without the LLM; 3 queries
  during a scan embedded concurrently with ingestion without errors; Ollama on
  a closed port -> 503 "LLM server unavailable", /health degraded. Every real
  answer failed: Ollama /api/chat returned HTTP 500 "llama-server process has
  terminated: exit status 0xc0000409" (mapped to 502), for llama3.2:3b and
  granite4.2:3b, also outside the API. Cause: the GTX 1080 Ti had been
  installed, Ollama (auto-updated 0.40.17 -> 0.40.18) started using CUDA, and
  driver 566.36 is below Ollama's 570 minimum ("PTX was compiled with an
  unsupported toolchain"). Fixed for now with the user variable
  CUDA_VISIBLE_DEVICES=-1; it only took effect after Ollama was fully
  restarted.
- Step 6 (manual end-to-end from the user's terminal, temp copy of data/inbox,
  llama3.2:3b on CPU): startup 5.92 s; scan 2.05 s (done 4, failed 2, needs_ocr
  1; 86 chunks); q06 answered "Outlook ... 0.246" citing Lecture07 p.9
  (expected) + p.5 in 6.0 s; q14 (negative) not_found 4.4 s; q03 not_found
  6.9 s (same miss as Phase 7). The ~25-49 s startups seen earlier happened only
  inside Claude Code processes (cause unknown). Phase 8 complete.

## 2026-11-27 — GPU step
- G0 (read-only): CPU baseline on Ollama 0.40.18: 10/13, 2/2, median 7.85 s
  (matches Phase 7); the second back-to-back run hit Ollama's prompt cache
  (prompt eval ~0.06 s), so only uncached runs are compared.
- Driver 566.36 -> 582.78 (R580 GeForce Security Update, clean install).
- First G2 attempt stopped correctly: CUDA_VISIBLE_DEVICES was removed but the
  Ollama server started before the change still ran on CPU; fixed by a reboot.
- G2: library=CUDA, 29/29 layers on GPU, 100% GPU; median 7.85 s -> 0.96 s,
  prompt eval ~17x, generation ~5x; statuses and citation hits identical to CPU.
  Pre-registered rule 4/4 PASS -> GPU kept.

## 2026-11-27 — Phase 9
- Step 0 (read-only): n8n 2.14.2 via npm (C:\Users\xt\AppData\Roaming\npm);
  %USERPROFILE%\.n8n already existed since 2026-04-06 (own database.sqlite,
  practice workflows only). Installed defaults: port 5678 listen ::, task-runner
  broker 127.0.0.1:5679, NODES_EXCLUDE executeCommand + localFileTrigger, file
  access ~/.n8n-files; `n8n` == `n8n start`. The n8n bin loads dotenv from the
  cwd and disables autoSelectFamily, and Node resolves localhost ::1 first
  (verbatim). The live inbox's 8 PDFs are fully backed up (SHA-256) in
  data/eval/corpus.
- Step 1: scripts/run_api_staging.py prepare (timestamped data/staging/<run>/,
  copy2, never overwrite or delete) and serve (RAG_* overrides, safety check
  exit 2, --print-only); 33 tests. Commit c81496d. n8n probe:
  http://localhost:8000/health -> "The service refused the connection";
  http://127.0.0.1:8000/health -> 200. Staging startup 8.44 s; stores created
  only under the run folder.
- Step 2 (manual poll workflow on staging): run 1 job 4.80 s: done 4 (86
  chunks), needs_ocr 1, failed 2 left in the inbox, duplicate moved without
  ingestion, 2 polls. An extra scan (likely "Execute step" on the POST node)
  gave attempt 2; the next run (attempt 3) moved both bad PDFs to failed/.
  Empty inbox -> job succeeded. A finished job gives a new job_id with
  already_running false.

## 2026-11-28 — Phase 9
- Step 3: workflow as code n8n/workflows/rag_ingest_scheduled.json: Schedule
  every 5 min + Manual run -> Start scan POST /ingest/scan (timeout 10000,
  retry 3 x 5000 ms) -> Already running? (true -> Skip) -> Wait 3s -> Check
  status GET /status/{{ $('Start scan').first().json.job_id }} (same retry) ->
  Still running? (queued/running AND $runIndex < 40 -> loop) -> Job succeeded?
  -> Done | Scan not succeeded (Stop and Error). typeVersions from the
  installed n8n files: httpRequest 4.4, wait 1.1, if 2.3, noOp 1, manualTrigger
  1, scheduleTrigger 1.3, stopAndError 1. tests/test_n8n_workflow.py (17
  tests), n8n/README.md, .gitignore data/test_pdfs/ and n8n/reference/.
  Import accepted (11 nodes); the live export differs from the repo file only
  by omitted default parameters and n8n default settings.
- Staging tests in n8n: happy path Done with failed documents (no alarm);
  Skip branch on already_running reported by the user (no screenshot); guard
  set to 0 -> "Scan job ... ended with status 'running'"; API stopped -> Start
  scan error "The service refused the connection"; one Start scan took
  10.189 s and succeeded (retry, likely while the server was starting;
  unverified). The imported settings.timezone was not applied
  (America/New_York) -> Asia/Karachi set in the UI, verified in the trigger
  output. Real-console Ctrl+C: "API stopped: HTTP client closed" and the
  launcher's "Ctrl+C received, waiting for the server to stop".
- Go-live: the user moved 5 test PDFs to data/test_pdfs/ (sizes unchanged);
  live inbox LACE, Lecture07, SVM notes + the new
  Phase8_Concepts_and_Interview_QA.pdf. The live API created data/status.db
  and data/chroma; startup 22.61 s (~21 s before the model load line; cause
  unknown). Workflow published; schedule tick 16:15:19 -> job succeeded in
  7.40 s, 3 polls, done 4: LACE 24, Lecture07 17, Phase8 63, SVM 43 = 147
  chunks, all 4 moved to data/processed.
- /query on the live store: Outlook question -> "0.246" citing Lecture07 p.9
  (first call 33.8 s incl. Ollama GPU cold load 32.57 s); "Why are the FastAPI
  endpoint handlers plain def instead of async def?" -> Phase8 p.11
  (similarity 0.63), 1.57 s; a compound RAG + SVM question cited SVM notes
  p.32 and LACE p.1 (retrieval split across topics; citation pages not
  manually checked). 1287 tests passing. Phase 9 complete.
- Lessons: "Execute step" on a POST node has real side effects (it sent an
  extra scan). A file setting is not runtime behaviour: the imported timezone
  was ignored until set in the UI.

## 2026-11-28 — Phase 10
- Step 0 (read-only design check + live probe, Ollama 0.40.19, llama3.2:3b): a
  dict format incl. $defs/$ref is accepted; maxLength in the grammar silently
  truncated "stochastic gradient descent" to "stochastic gradient";
  done_reason length gives broken JSON; ~70 completion tokens per item; the
  model invented an item from sources without definitions.
- P1 (Steps 1-2): structured_output.py (grammar_schema strips constraint
  keywords, call_structured fail-closed, no retry), llm.py keyword-only format
  / num_predict (old payloads byte-identical, proven by request bytes),
  RAG_LLM_MAX_EXTRACTION_TOKENS=768, pydantic>=2.14,<3 declared (pip freeze
  unchanged), vector_store PageChunk + get_page_chunks ($and filter verified on
  Chroma 1.5.9; get() order is not chunk_index order), glossary.py
  (extract_page_glossary + item checks). Commit b17bb7f.
- P1 fix: Pydantic validation is all-or-nothing, so one long item made the whole
  page INVALID_OUTPUT; models reduced to shape only, limits checked per item
  (TOO_MANY_ITEMS, BLANK_FIELD, FIELD_TOO_LONG, source pattern in
  SOURCE_OUT_OF_RANGE); prompt rule 2 asks for the shortest exact part (<= 200
  chars). Commit 54252e0.
- P2a: measurement rule pre-registered before any real glossary run (commit
  2ad223a); scripts/glossary_smoke.py pages / run on the eval store only
  (commit 6ab5dcc); dev page selection recorded before the run (commit b48abcf).
- Dev run 20261128-213349: FAIL on condition 2 (4/6); conditions 1, 3 (20%), 4,
  5 (warm median 1.04 s) passed. Labels by user + mentor; these pages became
  the DEV set.
- Diagnosis (4 dev pages): WRONG_SOURCE 4/6 of EVIDENCE_NOT_IN_SOURCE (exact
  text in another chunk of the page), 2 with characters changed by the model
  (". . ." -> "...", formula symbols); 3 TERM_NOT_IN_EVIDENCE with the term in
  the cited chunk. TEST set pre-registered before any fix (commit d7032f2).
- Fixes: prompt rule 2 "The evidence must contain the term itself.", rule 3
  "using only information from that source"; page-level source remap (WRONG_
  SOURCE >= 50%) with GlossaryResult.source_remapped (commit 91c9604); smoke
  script records source_remapped (commit f880245). 1493 tests passing.
- Test run 20261128-215441 (one attempt): FAIL. Condition 2 1/6 (2/6 even if
  Kernel counted), condition 3 3/4 = 75% wrong; conditions 1 (0/8), 4 (1/2) and
  5 (warm median 0.89 s, run 1) passed. SVM p.26 (negative) got 3 wrong items.
  3/16 pages differed between runs at temperature 0.
- Decision (pre-registered rule): Phase 10 PARTIAL. structured_output.py,
  glossary.py, get_page_chunks and glossary_smoke.py stay; no /extract endpoint.
- Lesson: valid JSON is not true data.
- Lesson: grammar constraints corrupt silently.
- Lesson: Pydantic validation is all-or-nothing.
- Lesson: a dev-set improvement did not transfer to new pages.
- Lesson: temperature 0 is not deterministic on GPU.
- Lesson: mechanical checks prove the evidence is real page text, not that it
  defines the term.

## 2026-11-28 — Planning (web UI + admin upload)
- Planning only: web UI (beige + white, query left, admin upload right) and
  single admin password decided. Upload + admin check go in Phase 12, the UI in
  Phase 13. Nothing implemented.
