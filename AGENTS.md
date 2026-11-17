# AGENTS.md — Project rules for AI coding agents

This file is the single source of project rules. Read it fully before any task.

## Project overview

Autonomous RAG-Based Document Intelligence & Knowledge Extraction Pipeline: a
local-first, zero-budget system that ingests PDFs dropped into a folder, extracts
and chunks their text page by page, stores embeddings in ChromaDB and status in
SQLite, and answers natural-language questions with a local LLM (Ollama), citing
the document name and page numbers. n8n orchestrates; Python does all processing.

## Architecture: two pipelines sharing storage

**INGESTION**

```
data/inbox -> n8n (Schedule Trigger) -> FastAPI POST /ingest/scan
  -> BackgroundTasks worker:
       PyMuPDF text extraction -> cleaning -> page-aware chunking
       -> Sentence Transformers embeddings
  -> ChromaDB (vectors + chunk metadata) + SQLite (document/job status)
  -> file moved to data/processed (success) or data/failed (error)
```

**QUERY**

```
user or n8n -> FastAPI /query -> retriever (embed query, top-k from ChromaDB)
  -> Ollama -> answer with document name + page citations
```

## Fixed design decisions (do not change without explicit approval)

- n8n = orchestration only (scheduling, triggering, notifications). Python = all
  processing (extraction, cleaning, chunking, embedding, retrieval, generation).
- `doc_id` = SHA-256 hex digest of the raw PDF bytes.
- SQLite holds document/job status: `pending`, `processing`, `done`, `failed`,
  `needs_ocr`.
- ChromaDB holds vectors + chunk metadata only (no status, no business state).
- Chunk metadata: `doc_id`, `filename`, `page_number`, `chunk_index`.
- The same embedding model is used for ingestion and query.
- Re-ingestion is idempotent: delete all existing chunks for a `doc_id` before
  inserting new ones.
- Pages without extractable text are flagged; such documents get `needs_ocr`.

## Environment

- Windows 11 + Windows PowerShell 5.1.
- Python 3.11 only, always via `.venv\Scripts\python.exe` (e.g.
  `.venv\Scripts\python.exe -m pytest`). Never use bare `python`, `py` or
  `python3`: on this machine `py`/`python3` resolve to Python 3.14.
- n8n 2.14.2 runs via npm (not Docker) on `http://localhost:5678`.
- FastAPI will run on `http://localhost:8000`.
- Ollama will run on `http://localhost:11434` (not installed yet).

## Folder layout

```
src/rag_pipeline/     application package (src layout)
tests/                pytest tests
data/inbox/           new PDFs waiting for ingestion   (contents git-ignored)
data/processed/       successfully ingested PDFs       (contents git-ignored)
data/failed/          PDFs that failed ingestion       (contents git-ignored)
docs/                 project documentation
.venv/                Python 3.11 virtual environment  (git-ignored)
AGENTS.md             this file
CLAUDE.md             Claude Code entry point (imports this file)
```

Future local storage (`data/chroma/`, SQLite `*.db`, `logs/`) is git-ignored.

## Coding rules

- Modular code: small, single-purpose modules and functions.
- Type hints on all public functions and methods.
- Use `logging`, never `print`, in application code.
- Clear error handling: catch specific exceptions, log with context, never
  silently swallow errors.
- Every new piece of logic gets pytest tests.
- Configuration via `.env` + pydantic-settings (introduced in a later step).
  No hard-coded paths, ports or model names in business logic.
- No paid APIs or cloud services.
- No new dependency without explicit approval from the user.
- Edit source files with the Edit tool. Never rewrite source files with
  shell heredocs, sed or ad-hoc scripts (they can corrupt escape sequences).
- Write invisible or whitespace-like characters in source code as \uXXXX
  escapes, never as literal characters.

## Agent workflow rules

- Implement only the current task. Make no unrelated changes.
- Report every file created, modified or deleted.
- Report the commands actually run, with their real output.
- List any tests that failed or were not run.
- Never claim a feature works unless it was verified.
- Stop after the task. Never start the next step or phase on your own.
- Never commit unless explicitly asked.

## Security rules

- Never commit `.env` or print its values (variable names only).
- Never open, read or print anything in the n8n user folder (`~/.n8n`,
  `%USERPROFILE%\.n8n`): it holds the encryption key and credentials.
- Never commit PDFs, databases or vector stores.
