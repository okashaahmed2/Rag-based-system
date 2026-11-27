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
  COMPLETE (commit 4dc8c15)
- Phase 5 (ChromaDB + SQLite + ingestion orchestration + inbox lifecycle): COMPLETE
  (commit 4dc8c15)
- Phase 6 Step 1 (retrieval with done-filter + evaluation primitives, no LLM, no real
  evaluation run): COMPLETE (commit 8a641b4)
- Phase 6 Step 2 prep (verified golden set, 15 questions): COMPLETE (commit e835fe9)
- Phase 6 Step 2 (isolated eval store + evaluation runner + first baseline run at
  600/120 MiniLM): COMPLETE (commit 8bcba2b; baseline result regenerated clean)
- Phase 6 Step 3a (dirty-flag fix + compare_eval): COMPLETE (commit 6d8a536); chunk-size
  comparison 400/80 and 800/120, rule outcome: keep 600/120
- Phase 6 Step 3b (multilingual embedding model comparison): SKIPPED by user decision
  (2026-11-23), see Key decisions
- Phase 6 Step 3 (3a results committed, 3b skipped, compare_eval labels): COMPLETE
- Phase 6 (retrieval with done-filter, golden set, isolated eval store, baseline and
  chunk-size comparison): COMPLETE (commit f1052c3)
- Phase 7 Step 0 (read-only audit) and Ollama install on D:: COMPLETE
- Phase 7 Step 1 (granite4.2:3b measurement): COMPLETE
- Phase 7 Step 2 (llm.py client + LLM settings, default 127.0.0.1): COMPLETE
  (commit 79db4b0)
- Phase 7 Step 3a (prompt builder + answer checker): COMPLETE (commit 6566a45)
- Phase 7 Step 3b (answer_question, model-proof answers, default
  llama3.2:3b): COMPLETE (commit ff40009)
- Phase 7 (local LLM, prompt with citations, NOT_FOUND behaviour): COMPLETE
- Phase 8 Step 0 (read-only FastAPI check): COMPLETE
- Phase 8 Step 1 (app factory, lifespan, shared httpx client, /health): COMPLETE
  (commit 59d53ad)
- Phase 8 Step 1c (HF model loading kept off the network): COMPLETE (commit e4f7ec3)
- Phase 8 Step 2 (jobs table, schema v2, single active job + partial unique index):
  COMPLETE (commit 0b5ae53)
- Phase 8 Step 3 (/ingest/scan, /status, background job runner, startup job
  recovery, no local paths in job summaries): COMPLETE (commit d5316b1)
- Phase 8 Step 4 (POST /ingest one inbox file, validated name, 409): COMPLETE
  (commit 87dde5a)
- Phase 8 Step 5 (POST /query, fail-closed answers, error mapping): COMPLETE
  (commit 0ee831e)
- Phase 8 Step 6 (manual end-to-end from the user's terminal): COMPLETE (2026-11-27)
- Phase 8 (FastAPI backend): COMPLETE
- GPU step (between Phase 8 and 9): COMPLETE (2026-11-27): G0 CPU baseline,
  NVIDIA driver 582.78, CUDA_VISIBLE_DEVICES removed, G2 pre-registered rule
  4/4 PASS
- Next: see Next precise task

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
- Chroma 1.5.9 probe (temp dir, 9 fake 4-dim vectors, docs A/B/C x 3):
  query(query_embeddings, where={"doc_id": {"$in": [...]}}) works; one-element list
  works; n_results > matches returns all matches without error ($in [B], n=50 -> 3);
  no match -> []; EMPTY list -> builtins.ValueError "Expected where operand value to
  be a non-empty list ..." (not a ChromaError); filtered order equals the unfiltered
  order minus the excluded documents
- retrieve(): only chunks of done documents (failed doc with the query's own vector
  excluded at top_k 1/3/50; pending, processing, failed, needs_ocr and unregistered
  doc_ids excluded); no done docs -> [] without embedding or querying; top_k >
  matches -> all matches; order by distance across documents; invalid query/top_k
  -> RetrievalError; recover_interrupted never called (spy; processing row stays
  processing); no chunk or query text in logs (tests only, tmp stores, stub model)
- build_retrieval_context(): missing status db or Chroma folder -> RetrievalError
  and nothing created; folder present but collection missing -> RetrievalError
  "Collection 'chunks' not found ...; run ingestion first" and a fresh client still
  lists no such collection; model/collection mismatch -> RetrievalError
- open_collection(create=False): missing folder or collection ->
  CollectionNotFoundError, nothing created; existing collection opens; saved model,
  dimension and space still checked; create=True (default) unchanged
- query_by_embedding(doc_ids=...): pre-filter in Chroma; empty / invalid doc_ids ->
  VectorStoreError
- evaluation.py: loader validation with line numbers (28 parametrized error cases), Roman Urdu +
  emoji round-trip, UTF-8 BOM accepted; worked example ranks [1, 3, None] ->
  hit@1 1/3, hit@5 2/3, MRR@5 (1 + 1/3)/3; rank > k is a miss
- scripts/list_doc_ids.py: renamed copy -> same doc_id; folder unchanged (tests only)
- Golden set eval/golden_set.jsonl verified 2026-11-23 (PyMuPDF draft + pdftotext
  cross-check of every doc_id, page and negative): 15 questions, 13 answerable,
  2 negative; 4 LACE paper, 4 Lecture07, 5 SVM/KNN/K-Means Roman Urdu notes; tags
  comparison 4, definition 2, fact 5, table 2, roman_urdu 5, negative 2;
  load_golden_set passes. SVM,K-mean,KNN.pdf IS the 36-page Roman Urdu notes
  (264 of 2405 words are hai/ka/ki/ke/nahi/aur/mein/ko); there is no separate
  English SVM document
- evaluate(): worked example [1, 3, None] -> hit@1 1/3, hit@3/5/10 2/3, MRR@10 4/9;
  per-tag split; tag without answerable questions -> None (not 0); negatives excluded
  from metrics but in the similarity summary; relevant page only at rank 7 -> miss at
  5, hit at 10; empty retrieve result; deterministic to_dict (tests, fake retriever)
- run_eval.py logic (tests, stub model): corpus copied once (copy2, inbox unchanged);
  manifest mismatch (chunk_size, chunk_overlap, model name, corpus) refused before
  any model load or store open; unreadable manifest refused; rebuild embeds nothing;
  golden doc_id not done -> error naming each doc_id and its question ids
- BASELINE (real run 2026-11-23, scripts/run_eval.py, store
  data/eval/stores/all-minilm-l6-v2_600_120, MiniLM dim 384, 600/120, corpus = copy of
  the 8 inbox PDFs; ingest done 4, failed 2, needs_ocr 1, skipped 1; 60.3 s):
  OVERALL (13 answerable) hit@1 0.692, hit@3 0.846, hit@5 1.000, hit@10 1.000,
  MRR@10 0.795. Per tag: comparison 4 -> 1.000/1.000/1.000/1.000 MRR 1.000;
  definition 2 -> 1.000 all, MRR 1.000; fact 5 -> 0.400/0.600/1.000/1.000 MRR 0.600;
  roman_urdu 5 -> 0.600/0.800/1.000/1.000 MRR 0.750; table 2 ->
  0.500/1.000/1.000/1.000 MRR 0.667; negative 2 -> n/a
- Baseline ranks: q01 3, q02 1, q03 1, q04 1, q05 1, q06 4, q07 1, q08 1, q09 1,
  q10 4, q11 1, q12 2, q13 1 (misses at 1: q01 top-1 LACE p1, q06 Lecture07 p6,
  q10 SVM notes p3, q12 SVM notes p17)
- Baseline similarity: answerable top-1 min 0.363 (q10), median 0.651, max 0.809
  (q04); negatives top-1 q14 0.253, q15 0.218 (both below every answerable top-1)
- Second run on the same store: results identical (compared with the saved file), 9.2 s;
  the 6 done/needs_ocr/duplicate files SKIPPED; invalid + password PDFs retried
  (attempt 2) and failed at extraction (no embedding)
- data/inbox names/sizes/mtimes identical before and after both runs; data/chroma
  and data/status.db not created; results file contains 0 of 15 question texts
- run_eval dirty flag (Step 3a): untracked eval/results/ file -> clean; untracked file
  elsewhere -> dirty; modified tracked file (incl. a tracked result file) -> dirty
  (tmp git repo tests). Real: the 800/120 run started with the untracked 400_80
  result present and recorded dirty false
- compare_eval.py: deltas, better/worse/same, OUT-TOP5 flags, similarity, rule check;
  different question ids/order, answerable flags or ks refused with exit 2 (tests)
- Token reports on the eval corpus (measure_chunk_tokens.py with RAG_INBOX_DIR /
  RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP env overrides, HF_HUB_OFFLINE=1; overall counts
  include the duplicate SVM copy, as the 600/120 figures did):
  400/80: 183 chunks, median 84, p95 143, max 167, 0 over 256, 13 over 128;
  800/120: 110 chunks, median 122, p95 214, max 298, 2 over 256 (LACE chunks 9 and 13,
  280 and 298 tokens, truncated at embedding), 51 over 128 (= Phase 4 figures);
  600/120 (Phase 4): 129 chunks, median 116, p95 177, max 241, 0 over 256, 48 over 128
- Step 3a runs (commit 6d8a536, dirty false, own stores all-minilm-l6-v2_400_80 and
  _800_120, each ingest done 4 / failed 2 / needs_ocr 1 / skipped 1):
  400/80: hit@1 0.615, hit@3 0.846, hit@5 1.000, hit@10 1.000, MRR@10 0.753 (-0.042);
  800/120: hit@1 0.615, hit@3 0.769, hit@5 1.000, hit@10 1.000, MRR@10 0.746 (-0.049)
  vs baseline 600/120 0.692 / 0.846 / 1.000 / 1.000 / 0.795
- Rank changes vs 600/120: 400/80 worse q01 3->4, q03 1->3, q06 4->5, better q10 4->2;
  800/120 worse q01 3->4, q03 1->2, q10 4->5; all other questions same; no answerable
  question outside the top 5 in any config
- Per tag (MRR@10, 600 / 400 / 800): fact 0.600 / 0.507 / 0.490; roman_urdu 0.750 /
  0.800 / 0.740; table 0.667 / 0.625 / 0.625; comparison and definition 1.000 in all
- Negatives top-1: 400/80 q14 0.289, q15 0.231 (higher than 600/120's 0.253 / 0.218);
  800/120 equal to 600/120. Lowest answerable top-1: 0.415 at 400/80, 0.363 otherwise
- DECISION RULE OUTCOME (mechanical, compare_eval): 400/80 FAIL (MRR -0.042),
  800/120 FAIL (MRR -0.049); hit@5 and the top-5 condition held for both ->
  600/120 stays; Settings defaults unchanged
- compare_eval.py labels: runs sharing a model show the chunk part ("600_120",
  "400_80", "800_120") plus a "(all runs use model ...)" note; different models show
  full, untruncated config names (tests); the real 3-file comparison re-run gives the
  same numbers and the result files' SHA-256 hashes are unchanged
- 724 pytest tests passing
- Ollama 0.40.17 on D:\Ollama\app, models in D:\Ollama\models (user
  OLLAMA_MODELS), nothing on C:; llama3.2:3b and granite4.2:3b pulled
- granite4.2:3b: loaded 2.7 GB, 100% CPU, context 4096 default; prompt eval
  ~118 tok/s, generation 15-17 tok/s; RAG prompts with think:false leak
  reasoning ending in "</think>" (template enable_thinking defaults True)
- localhost vs 127.0.0.1: wall - Ollama total = 2.20 s vs 0.15 s (3+3 runs);
  DNS order ::1 then 127.0.0.1; Ollama listens on 127.0.0.1:11434 only
- llm.py (MockTransport tests): exact request body; ns -> s; thinking field
  dropped; "</think>" strip (one/two blocks, unclosed "<think>" -> "");
  done_reason length -> truncated; error mapping; no prompt/answer text in logs
- answering.py (pure + fake retrieve/chat tests): message structure, S
  numbering, tag neutralisation, every check_answer branch incl. NOT_FOUND
  line rule; no chunks -> chat never called; [S#] -> file/page mapping
- Golden set, llama3.2:3b, top_k 5, rule 5 "1 to 3 short sentences":
  10/13 answerable ANSWERED with citation hit, negatives 2/2 NOT_FOUND,
  0 errors, 0 thinking leaks, 0 truncated, median wall 7.5-7.9 s, max prompt
  1330 tokens; identical answers on re-run (temperature 0)
- Mentor review of that run: 10/13 fully correct (granite after "</think>"
  also 10/13, with 2 timeouts); misses q01, q03 (NOT_FOUND) and q12 (wrong
  citation p.20)
- A query-only run changes the eval store's Chroma file bytes; status.db and
  manifest.json unchanged; retrieval results reproduced exactly
- 948 pytest tests passing
- fastapi 0.165.0 + starlette 1.10.0 installed (nothing else changed); uvicorn
  0.54.0 declared; port 8000 free (Phase 8 Step 0)
- Lifespan builds the ingest context once (document recovery once), then job
  recovery once; RetrievalContext derived from the ingest context (one Chroma
  client, one model); shared httpx.Client closed on shutdown (tests + real
  Ctrl+C: "API stopped: HTTP client closed")
- /health plain def; Ollama problems -> 200 "degraded"; fixed DocumentCounts +
  active_job
- HF: huggingface_hub 2.4.0 sent GET {ENDPOINT}/api/agent-harnesses on every
  model load even with local_files_only=True (build_hf_headers -> detect_agent
  -> _fetch_registry; cached 24 h in ~/.cache/huggingface/.agent_harnesses.json;
  independent of agent env vars). apply_hf_runtime_env (setdefault
  HF_HUB_DISABLE_TELEMETRY=1; HF_HUB_OFFLINE=1 with local_files_only), called in
  load_embedding_model before the sentence_transformers import -> no Hub
  request, cache file not recreated
- Schema v1 -> v2 keeps documents rows byte-identical (tests + eval-store copy,
  7 rows, original hash unchanged); single active job: 50 rounds x 2 threads,
  exactly one job per round (repeated); idx_jobs_one_active rejects a 2nd
  active row
- /ingest/scan: double POST -> same job_id, already_running true; scan after
  finish -> new job; /status latency during ingestion median ~5 ms, max 17-29 ms
- /ingest: bad names (../, C:\..., .txt, CON.pdf) -> 422 with no job row;
  missing -> 404; while a scan runs -> 409 with the scan's job_id; same name
  again after success -> 404 (file already moved to processed)
- /query: no done docs -> 200 not_found without LLM; Ollama on a closed port ->
  503 "LLM server unavailable"; 3 queries during a scan embedded concurrently
  with ingestion, no errors (light test, not a stress test)
- Step 6 end-to-end (user's terminal, temp copy of data/inbox, llama3.2:3b on
  CPU): startup 5.92 s; scan 2.05 s (done 4, failed 2, needs_ocr 1; 86 chunks);
  q06 answered "Outlook ... 0.246" citing Lecture07 p.9 (expected) + p.5, 6.0 s;
  q14 (negative) not_found 4.4 s; q03 not_found 6.9 s (same miss as Phase 7)
- G0 CPU baseline (Ollama 0.40.18, llama3.2:3b, 100% CPU, uncached run):
  10/13 answered+hit, negatives 2/2, 0 errors/truncated, median wall 7.85 s,
  max 10.94 s, prompt eval ~153 tok/s, generation ~18 tok/s, cold load 22.4 s
- G2 GPU (driver 582.78, server.log: library=CUDA compute=6.1, cuda_v12
  backend, 29/29 layers offloaded, ollama ps 100% GPU, ~2.9 GB VRAM): uncached
  run 10/13, 2/2, 0 errors/truncated, median wall 0.96 s, max 2.46 s (q01, first
  question incl. warm-up), prompt eval ~2580 tok/s, generation ~94 tok/s;
  no status or citation-hit change vs CPU (q04 43 vs 47 completion tokens)
- 1237 pytest tests passing

## Not yet verified / incomplete
- Real log rotation at ~1 MB
- ERROR status on a real locked/unreadable file (only simulated)
- Memory use when hashing very large files
- OCR itself (Phase 13)
- Ligatures and invisible-character rules on real PDF data (no real PDF contained
  any yet; unit tests only)
- Whether the 400/80 and 800/120 MRR differences are more than noise (13 answerable
  questions; a one-rank move on one question changes MRR by up to 0.04)
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
- retrieve() and build_retrieval_context() on real data (real model, real stores):
  tests use tmp stores and a stub model only; no real evaluation run yet (Step 2)
- scripts/list_doc_ids.py on the real PDFs (tests only)
- $in filter speed / limits with many done doc_ids (probe had 3 documents)
- Real Ollama 404 (model missing), timeout and server-down paths (mock tests only)
- Whether q03's answer chunk (LACE p.2 "Pipeline" paragraph) was among the
  retrieved chunks (page-level check only)
- Faithfulness of answers (cited source supports the claim): not measured
  automatically; manual review only
- llama3.2:3b on documents or questions beyond the 15-question golden set
- n8n -> FastAPI latency with "localhost" (Phase 9)
- GPU cold load time (only a 40.17 s first /api/generate including the load
  was seen)
- Ollama total duration per question (rag_smoke.py does not print it)
- Concurrent encode under load (only 3 queries during one scan tested)
- Real n8n calls (Phase 9)
- Cause of the ~25-49 s startups seen only inside Claude Code processes

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
  count_document_chunks, query_by_embedding, RetrievedChunk, VectorStoreError,
  CollectionNotFoundError
- src/rag_pipeline/status_db.py: StatusStore (initialize, register_document,
  mark_processing/done/needs_ocr/failed, get_document, list_documents,
  known_doc_ids, recover_interrupted), DocumentRecord, DocumentStatus, StatusStoreError
- src/rag_pipeline/ingestion.py: build_ingest_context, ingest_document, IngestContext,
  IngestResult, IngestOutcome, IngestionError, IngestionSetupError
- src/rag_pipeline/inbox_ingestion.py: ingest_inbox, destination_for, safe_move,
  DestinationFolders, InboxRunReport, FileMove, LeftInInbox, MoveFailure
- src/rag_pipeline/config.py also holds needs_ocr_dir
- src/rag_pipeline/retrieval.py: build_retrieval_context, retrieve, done_doc_ids,
  RetrievalContext, RetrievalError
- src/rag_pipeline/evaluation.py: load_golden_set, first_relevant_rank, hit_at_k,
  mrr_at_k, GoldenQuestion, ExpectedSource, GoldenSetError, DOC_ID_PATTERN
- eval/golden_set.example.jsonl: 2 FAKE example lines (answerable + negative)
- eval/golden_set.jsonl: verified golden evaluation set (15 questions)
- scripts/list_doc_ids.py: filename<TAB>doc_id for every *.pdf in a folder (read-only)
- src/rag_pipeline/evaluation.py also holds evaluate, EvaluationResult, QuestionResult,
  MetricSummary, SimilaritySummary, RetrievedHit, RetrieveFn, DEFAULT_KS, TOP_SHOWN
- scripts/run_eval.py: copy inbox -> data/eval/corpus (once), build/reuse
  data/eval/stores/<config>/ (chroma + status.db + manifest.json), check golden
  doc_ids done, score via build_retrieval_context + retrieve, write
  eval/results/<date>_<config>.json, print ASCII summary
- eval/results/2026-11-23_all-minilm-l6-v2_600_120.json: baseline result (tracked)
- scripts/compare_eval.py: BASE.json OTHER.json [...] -> metrics with deltas, rank
  changes, outside-top-5 flags, similarity, decision rule check (ASCII)
- data/eval/ (git-ignored): corpus copy + eval stores
- tests/: conftest.py, test_config.py, test_logging_config.py, test_extraction.py, test_inbox.py,
  test_cleaning.py, test_headers_footers.py, test_chunking.py, test_preparation.py,
  test_embeddings.py, test_vector_store.py, test_status_db.py, test_ingestion.py,
  test_inbox_ingestion.py, test_retrieval.py, test_evaluation.py, test_list_doc_ids.py
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
- src/rag_pipeline/llm.py: chat, check_ollama, build_chat_payload,
  parse_chat_response, strip_leaked_thinking, validate_messages, LLMResponse,
  LLMError, LLMUnavailableError, LLMModelNotFoundError, LLMTimeoutError,
  LLMResponseError
- src/rag_pipeline/answering.py: SYSTEM_PROMPT, NOT_FOUND_SENTINEL,
  NOT_FOUND_MESSAGE, build_rag_messages, build_user_message,
  neutralize_source_tags, find_citations, is_not_found, check_answer,
  AnswerStatus, CheckedAnswer, Citation, RagAnswer, sources_from_chunks,
  answer_question
- src/rag_pipeline/config.py also holds ollama_base_url, llm_model,
  llm_temperature, llm_num_ctx, llm_timeout_seconds, llm_max_answer_tokens,
  query_top_k
- scripts/llm_smoke.py: manual check_ollama + one chat (real Ollama)
- scripts/rag_smoke.py: manual golden-set RAG run on the eval store read path
  (sources with *EXPECTED*, LLM miss vs retrieval miss; no result file)
- scripts/llm_think_probe.py: diagnosis record of the granite thinking leak
- tests/test_llm.py, tests/test_answering.py, tests/test_answer_question.py
- src/rag_pipeline/api/__init__.py (builds nothing on import)
- src/rag_pipeline/api/app.py: create_app, lifespan, ApiState, models, routes
  /health, /ingest/scan, /ingest, /status/{job_id}, /query; QUERY_ERRORS
- src/rag_pipeline/api/jobs.py: _run_work, run_job, run_ingest_job,
  summary_from_report (local paths replaced by file names)
- src/rag_pipeline/api/validation.py: validate_inbox_filename,
  InboxFilenameError (422), InboxFileNotFoundError (404)
- src/rag_pipeline/inbox_ingestion.py also holds ingest_one_file (+ private _place,
  _destination_folders)
- src/rag_pipeline/status_db.py also holds jobs: JobKind, JobStatus, JobRecord,
  create_job_or_get_active, mark_job_*, get_job, get_active_job,
  recover_interrupted_jobs; schema v2 + _MIGRATIONS
- src/rag_pipeline/embeddings.py also holds apply_hf_runtime_env;
  src/rag_pipeline/config.py also holds api_host, api_port
- tests/test_api_app.py, tests/test_api_jobs.py, tests/test_api_ingest.py,
  tests/test_api_validation.py, tests/test_api_query.py,
  tests/test_inbox_one_file.py, tests/test_status_db_jobs.py

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
- Retrieval returns only chunks of done documents: pre-filter inside Chroma
  (where doc_id $in done ids, so top_k is taken among allowed chunks), plus a
  post-check that raises if a non-done chunk ever comes back (filter regression)
- No done documents -> [] before embedding or querying (Chroma rejects an empty $in)
- query_by_embedding got an optional keyword doc_ids (None = unfiltered, as before)
  instead of a second query function; done ids come from the existing
  StatusStore.list_documents(DONE) (no new StatusStore method, no schema change)
- The read path never calls recover_interrupted (ingestion startup only) and never
  creates stores: build_retrieval_context requires the status db file, the Chroma
  folder and the collection to exist (RetrievalError "run ingestion first"); it opens
  the collection with open_collection(..., create=False) (keyword, default True keeps
  the ingestion behaviour); CollectionNotFoundError subclasses VectorStoreError
- retrieve(query, context, top_k): top_k is an explicit argument (no new setting yet)
- Golden set relevance is labelled per (doc_id, page), never chunk_id; matching on
  doc_id only (filename informational); expected [] = negative question
- hit_at_k / mrr_at_k take ranks of answerable questions only and raise ValueError on
  empty input (undefined metric, no misleading 0.0); negatives are excluded by the
  caller
- Golden set loader accepts a UTF-8 BOM (Windows PowerShell 5.1 writes one)
- Evaluation runs only on an isolated eval store (data/eval/stores/<config>/, own
  Chroma + status db), never on data/chroma / data/status.db; the corpus is a one-time
  copy of the inbox incl. scanned/invalid/password/duplicate files so the done-filter
  is exercised
- Store name <config> = <model slug>_<chunk_size>_<chunk_overlap>
  (all-minilm-l6-v2_600_120); manifest.json (model, dim, chunk values, corpus
  filename+doc_id, created_at) is written before ingestion; a mismatch is refused,
  never rebuilt in place (a different dim is refused by open_collection)
- The eval store is filled with build_ingest_context + ingest_document (real write
  path, no file moves) and read with build_retrieval_context + retrieve (real read
  path); evaluate() takes any retrieve callable, so metric logic is tested without
  a model
- A golden doc_id that is not done in the eval store stops the run (exit 1)
- MRR is reported at max(ks) = 10; hit@k for k in 1, 3, 5, 10; per-tag metrics for a
  tag without answerable questions are n/a (None), not 0
- Result files hold ids, filenames, pages, ranks, similarities, manifest and git
  commit + dirty flag; no question or chunk text; eval/results/ is tracked,
  data/eval/ is ignored
- A same-day re-run overwrites the result file and reports whether the results are
  identical to the previous file
- Chunk settings stay 600/120 (Step 3a rule outcome: 400/80 and 800/120 both failed
  the MRR condition; the differences are within noise for 13 answerable questions,
  and 600 keeps every measured chunk <= 256 tokens)
- Step 3b (multilingual embedding model) SKIPPED by user decision (2026-11-23).
  Reasons: the roman_urdu tag is not worse than overall at baseline (hit@5 1.000,
  MRR@10 0.750 vs 0.795); questions are asked in English and the notes keep technical
  terms in English; paraphrase-multilingual-MiniLM-L12-v2 has max_seq_length 128
  (truncation risk at 600 chars); multilingual-e5-small needs "query: " / "passage: "
  prefixes (a code change). Revisit if Roman Urdu documents grow, users ask in Roman
  Urdu, or the roman_urdu tag drops in a later eval run
- PRE-REGISTERED DECISION RULE (recorded 2026-11-23, before any Step 3 run): 600/120
  is replaced only if a config keeps hit@5 = 1.000, improves MRR@10 by >= 0.05, and
  drops no answerable question out of the top 5. Otherwise 600/120 stays (it keeps
  every measured chunk <= 256 tokens). scripts/compare_eval.py checks it mechanically
- run_eval.py dirty flag: any modified/staged/untracked file means dirty, except
  UNTRACKED files under eval/results/ (earlier result files)
- Ollama HTTP via httpx (declared), not the ollama Python package
- Default base URL http://127.0.0.1:11434 (measured ~2.2 s per connection
  with localhost on this machine); localhost still accepted
- Every chat sends stream false, think false, temperature 0, num_ctx 4096,
  num_predict 256 (single source: build_chat_payload)
- think:false is not trusted: leaked reasoning up to the last "</think>" is
  stripped and flagged in the client (model-proof)
- Citation format [S#], sentinel NOT_FOUND (exact word, also as its own line;
  a NOT_FOUND line wins over citations); user-facing text
  "The answer was not found in the documents."
- No retrieved chunks -> NOT_FOUND without calling the LLM
- No similarity threshold before the LLM yet (negatives 0.22-0.25 vs lowest
  answerable 0.36 on 15 questions only; not validated)
- Prompt rule 5 "1 to 3 short sentences; add nothing not in the sources"
  (longer answers padded with unsupported claims)
- Default model llama3.2:3b BY JUDGMENT: pre-registered rule (0 errors,
  negatives 2/2, >= 11/13 ANSWERED+hit, median <= 15 s) FAILED condition 3
  (10/13); granite4.2:3b failed conditions 1 and 4. Model change -> rerun
  llm_smoke + rag_smoke + manual answer review
- Lesson: condition 3 counts an honest NOT_FOUND like a hallucination;
  Phase 11 metrics must separate correct / abstained / wrong
- API handlers are plain def (run in the threadpool); async def would block the
  event loop with the blocking pipeline calls
- One Uvicorn worker, one process, 127.0.0.1, run with --factory
- Ingest context built once in the lifespan; job recovery only at startup (in a
  request it would fail a live job)
- At most one active ingestion job of any kind (code + partial unique index);
  /ingest/scan returns the active job (202, already_running); /ingest -> 409
- Job "succeeded" = the job finished even if documents failed; "failed" = the job
  itself crashed; run_job catches Exception only; failure to record job state ->
  CRITICAL, never re-raised
- /ingest takes a plain file name already in the inbox (whitelist validation,
  resolved parent must be the inbox; no upload, no python-multipart); invalid
  requests never create a job; error messages never echo the name or a path
- /query answers fail closed: text only for answered; fixed message for
  not_found; null for no_citations/bad_citations/empty; "not found" is HTTP 200
- /query errors: RetrievalError 503, LLMUnavailable/ModelNotFound 503,
  LLMResponseError 502, LLMTimeoutError 504; ValueError NOT mapped (bug -> 500);
  question never logged or echoed
- Shared Ollama client passed via functools.partial(chat, client=...)
- HF env vars set in the process (apply_hf_runtime_env in load_embedding_model)
  before the model library import (.env is not exported to os.environ); with
  local_files_only this now also sets HF_HUB_OFFLINE=1 process-wide (the Phase 4
  note "not HF_HUB_OFFLINE" no longer holds alone)
- Routes via app.add_api_route (FastAPI 0.165 _IncludedRouter observation by
  Claude Code, not independently verified)
- GPU: Ollama runs llama3.2:3b on the GTX 1080 Ti via CUDA (cuda_v12 runner);
  driver 582.78 (R580 GeForce Security Update, chosen over Studio 581.57: newer,
  security fixes); embeddings stay on CPU (torch +cpu). Kept by the
  pre-registered rule (0 errors/truncated, negatives 2/2, answered+hit >= 10/13,
  median < 7.85 s): 4/4 PASS. Rollback: User variable CUDA_VISIBLE_DEVICES=-1
  + full Ollama restart.
- Lessons: commit after every verified step and update docs at session end (a
  Claude Code session was deleted on 2026-11-27; nothing lost); change one thing
  at a time (driver upgrade deferred); Phase 4's "no network" check missed the
  HF header path; Ollama auto-updates on Windows (0.40.17 -> 0.40.18); an
  environment-variable change needs a full Ollama server restart (reboot is
  simplest); Ollama's prompt cache makes back-to-back benchmark runs
  misleading, compare uncached run against uncached run

## Open questions
- Questions will be asked in English; document types are lecture slides and
  research papers (decided).
- Roman Urdu notes: RESOLVED (2026-11-23) — keep all-MiniLM-L6-v2 (baseline roman_urdu
  hit@1 0.600, hit@5 1.000, MRR@10 0.750; 5 questions from one document). Revisit if
  Roman Urdu documents grow, users ask in Roman Urdu, or the roman_urdu tag drops in a
  later eval run (Step 3b skipped, see Key decisions)
- Table-aware extraction (q01): keep table rows/columns through cleaning?
  (Phase 13 candidate)
- Automatic citation check (key terms/numbers of the answer present in the
  cited chunk, or an NLI model): Phase 11 candidate

## Verified versions
Python 3.11.9 | pip 26.3 | setuptools 84.2.1 (upgraded from 65.5.0 by the
sentence-transformers install) | pydantic-settings 2.16.2 |
pydantic 2.14.3 | pytest 9.2.0 | pymupdf 2.1 | numpy 2.4.6 (declared dependency
since Phase 4 Step 2b) | onnxruntime 1.31.1 |
sentence-transformers 6.3.0 | torch 2.15.0+cpu | transformers 5.23.0 |
tokenizers 0.23.3 | huggingface-hub 2.4.0 | chromadb 1.5.9 |
git 2.55.0 | node 24.14.1 | npm 11.11.0 | n8n 2.14.2 | httpx 0.28.1 |
Ollama 0.40.18 (auto-updated from 0.40.17) | llama3.2:3b | granite4.2:3b (rejected) |
fastapi 0.165.0 | starlette 1.10.0 | uvicorn 0.54.0 (declared >=0.54,<1) |
anyio 4.16.0 | httpx2 2.16.2 (via huggingface_hub) | NVIDIA driver 582.78 (was 566.36)

## Hardware (verified)
Intel Core i5-14400F (10C/16T), 15.8 GB RAM (2 x 8 GB DDR4-3200, dual channel; 5.99-6.82 GB free with
Chrome + VS Code open),
NVIDIA GeForce GTX 1080 Ti 11 GB (Pascal, compute 6.1), installed 2026-11-27,
driver 582.78 (R580, reports CUDA 13.0); LLM on GPU,
Windows 11, PowerShell 5.1

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
  delete); retrieve() excludes them (done-filter, Phase 6 Step 1), but raw
  query_by_embedding without doc_ids still returns them
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
- A document whose status changes between reading the done ids and the Chroma query
  can still appear in that one retrieve() call (no lock across SQLite and Chroma)
- The $in list grows with every done document; fine at this scale, unmeasured for
  many documents
- build_retrieval_context on an existing but EMPTY Chroma folder: no collection is
  created (RetrievalError), but the Chroma client writes its own chroma.sqlite3 into
  the folder (probe: before [], after ['chroma.sqlite3'])
- retrieval.py imports the ModelLoader type alias from ingestion.py
- roman_urdu tag is confounded with one document: all 5 roman_urdu questions come
  from SVM,K-mean,KNN.pdf, so a per-tag score cannot separate "Roman Urdu text"
  from "this document" (e.g. its formula/diagram extraction)
- Small golden set: 13 answerable questions, so one question moves hit@k by 0.077;
  per-tag groups have 2-5 questions (differences between configs may be noise)
- hit@5 = hit@10 = 1.000 at baseline: the set cannot show improvements at k >= 5,
  only at hit@1/hit@3/MRR
- Negative top-1 similarities (0.218-0.253) sit below the lowest answerable top-1
  (0.363) on this set only; not a validated refusal threshold
- The eval corpus is a snapshot: later inbox changes are not reflected (by design);
  changing it requires a new store (manifest refuses a changed corpus)
- Each eval run retries the invalid and password PDFs until they reach
  max_ingest_attempts (3), then they are skipped (ingestion behaviour, unchanged)
- The baseline result (commit 8bcba2b) was regenerated clean with the old untracked
  result file moved aside; fixed in Step 3a (untracked eval/results/ files ignored)
- measure_chunk_tokens.py loads SentenceTransformer without local_files_only (needs
  HF_HUB_OFFLINE=1 to guarantee no Hub contact) and writes logs/token_report.txt,
  overwritten per run
- llama3.2:3b can cite a retrieved source that does not hold the claim (q12);
  check_answer cannot detect it
- Tables are flattened by cleaning (one value per line), so table questions
  can fail at generation (q01)
- Chroma rewrites its files on a query-only open/close: eval stores are not
  byte-identical across runs (run evals on a copy in Phase 11)
- Uvicorn also binds 127.0.0.1 by default: n8n calling "localhost" may pay
  the same IPv6 fallback delay (measure in Phase 9)
- A 404 from /api/chat is always reported as "model not found" (a wrong base
  URL also gives 404)
- llm.chat opens a new HTTP connection per call (~0.15 s); Phase 8 can reuse
  one client
- Cold model load 2-35 s after Ollama unloads the model (keep_alive ~5 min)
- llama3.2:3b license: Llama 3.2 Community License ("Built with Llama" in the
  README, Phase 14)
- Ollama auto-update can change behaviour (0.40.18 started using CUDA)
- NVIDIA R580 is the last driver branch for Pascal (GTX 10); 590+ does not
  support the 1080 Ti
- CUDA 13 dropped pre-Turing GPUs; a future Ollama version may drop the
  CUDA 12 runner and fall back to CPU (check ollama ps after auto-updates;
  CPU path stays documented)
- OLLAMA_VULKAN is true in the server config; CUDA was chosen (check
  server.log library= after updates)
- Jobs table grows without cleanup; /health 500 if Chroma/SQLite fail; /health
  shows the Ollama URL/OS error text; RAG_API_HOST accepts 0.0.0.0 (Phase 12)
- Absolute paths still appear in logs and in the documents table error_message
- HF env vars process-wide, sticky, inherited by child processes
- uvicorn --reload re-runs startup recovery; v2 db unreadable by code <= e4f7ec3
- 422 has two shapes (string detail for bad names, list for malformed bodies)
- File replaced between /ingest validation and the job -> new bytes ingested

## Next precise task
Phase 9 Step 0: n8n basics + read-only check (n8n 2.14.2 via npm, port 5678,
Schedule Trigger -> POST /ingest/scan -> poll /status), live data paths
decision (data/inbox, data/chroma, data/status.db still unused).
