# Project State

## Current position
- Phase 1: COMPLETE
- Phase 2 (PDF ingestion: PyMuPDF extraction, SHA-256 doc_id, needs_ocr, duplicate detection): COMPLETE
- Phase 3 Step 1 (page-level text cleaning): COMPLETE
- Phase 3 Step 2 (repeated header/footer removal): COMPLETE
- Phase 3 Step 3 (page-aware chunking): COMPLETE
- Next: Phase 3, Step 4 — chunk metadata + prepare_document()

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
- Settings RAG_CHUNK_SIZE=800, RAG_CHUNK_OVERLAP=120 (characters) with validation
  (size >= 100, 0 <= overlap < size)
- Real data, two-column research paper: 4 pages, 19 chunks, 15/15 same-page pairs
  overlap 82-120 chars, column-by-column reading order, 32 hyphen joins
  (3 spot-checked correct), header "preprint. under review." and page numbers removed
- Real data, lecture slides: 15 slides -> 15 chunks, course header removed from
  14 slides, slide titles kept
- Hyphen joins on real data (research paper, see above)
- Header/footer removal on real PDFs (research paper header + page numbers,
  lecture course header)
- 314 pytest tests passing

## Not yet verified / incomplete
- Real log rotation at ~1 MB
- ERROR status on a real locked/unreadable file (only simulated)
- Memory use when hashing very large files
- OCR itself (Phase 13)
- Ligatures and invisible-character rules on real PDF data (no real PDF contained
  any yet; unit tests only)
- Chunk token counts <= embedding model limit (check with the real tokenizer in Phase 4)
- Extraction quality of diagram/table/code slides (Lecture07 slides 8 and 13 look suspect)
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
- tests/: conftest.py, test_config.py, test_logging_config.py, test_extraction.py, test_inbox.py,
  test_cleaning.py, test_headers_footers.py, test_chunking.py
- .env.example: settings template

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
- Chunk size/overlap in characters (800/120, ~200 tokens, under the 256 word-piece
  limit of all-MiniLM-L6-v2); tunable in Phase 6
- Page-aware chunks (never cross pages) for clean page citations
- Recursive unit splitting: paragraph -> line -> sentence -> word -> hard cut
- Sentence/word-aligned overlap inside the last chunk_overlap characters
- chunk_pages() allows chunk_size >= 1 (small test sizes); Settings require >= 100

## Open questions
- Questions will be asked in English; document types are lecture slides and
  research papers (decided).
- Roman Urdu notes: still to be checked with the Phase 6 golden set
  (all-MiniLM-L6-v2 vs a multilingual model).

## Verified versions
Python 3.11.9 | pip 26.3 | setuptools 65.5.0 | pydantic-settings 2.16.2 |
pydantic 2.14.3 | pytest 9.2.0 | pymupdf 2.1 | numpy 2.4.6 | onnxruntime 1.31.1 |
git 2.55.0 | node 24.14.1 | npm 11.11.0 | n8n 2.14.2

## Hardware (verified)
Intel Core i5-14400F (10C/16T), 15.8 GB RAM (6.6 GB free at check),
AMD Radeon HD 7000 (CPU only), Windows 11, PowerShell 5.1

## Risks
- Windows console (cp1252) cannot print some characters (e.g. emoji) from extracted text;
  app.log is UTF-8 and unaffected. Revisit when text is logged to console.
- Free RAM 6.6 GB at check time; review before choosing the Ollama model (Phase 7)
- Test PDFs in data/inbox are git-ignored (local only)

## Next precise task
Phase 3, Step 4: chunk metadata (DocumentChunk with doc_id, filename, page_number,
chunk_index) and prepare_document() running extraction -> cleaning -> header/footer
removal -> chunking for one PDF; tests.
