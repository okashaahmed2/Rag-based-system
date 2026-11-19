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
- Phase 3 complete. 329 tests passing.

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
  docs updated, committed and pushed. 329 tests passing.
