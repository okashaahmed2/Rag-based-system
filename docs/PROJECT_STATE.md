# Project State

## Current position
- Phase 1 (Environment, structure, Git, dependencies, config, logging): COMPLETE
- Next: Phase 2, Step 1 — PDF extraction with PyMuPDF, SHA-256 doc_id, needs_ocr detection

## Verified (with evidence)
- Python 3.11 venv at .venv (always called as .venv\Scripts\python.exe)
- Git repo on main, pushed to https://github.com/okashaahmed2/Rag-based-system
- pyproject.toml with editable install (pip install -e ".[dev]")
- Settings via pydantic-settings (RAG_ prefix, .env optional), 19 pytest tests passing
- Logging: console + rotating logs/app.log, idempotent; verified manually and by tests
- pip certificate issue fixed by upgrading venv pip to >= 24.2 (truststore default)

## Not yet verified / incomplete
- Real log rotation at ~1 MB (only handler settings tested)
- Ollama not installed (planned for Phase 7)

## Files and responsibilities
- AGENTS.md: rules for AI coding agents (single source of truth)
- CLAUDE.md: Claude Code entry point, imports AGENTS.md
- pyproject.toml: package metadata and dependencies
- src/rag_pipeline/config.py: Settings + get_settings()
- src/rag_pipeline/logging_config.py: setup_logging()
- tests/conftest.py: isolates RAG_ env vars and settings cache per test
- tests/test_config.py, tests/test_logging_config.py
- .env.example: documented settings template (real .env is git-ignored)

## Key decisions
- Python 3.11 only (machine also has 3.14 as py/python3 default)
- n8n 2.14.2 runs via npm (not Docker) -> FastAPI URL from n8n will be http://localhost:8000
- Documents are English only -> small English embedding model (e.g. all-MiniLM-L6-v2) is the Phase 4 candidate
- Settings precedence: environment variable > .env > code default
- Paths resolved against project root derived from package location, not cwd
- New dependencies only when a phase needs them

## Verified versions
Python 3.11.9 | pip 26.3 | setuptools 65.5.0 | pydantic-settings 2.16.2 |
pydantic 2.14.3 | pytest 9.2.0 | git 2.55.0 | node 24.14.1 | npm 11.11.0 | n8n 2.14.2

## Hardware (verified)
Intel Core i5-14400F (10 cores / 16 threads), 15.8 GB RAM (6.6 GB free at check),
AMD Radeon HD 7000 (no CUDA/ROCm -> CPU only), Windows 11, PowerShell 5.1

## Blockers / risks
- Free RAM was only 6.6 GB at check time; review before choosing the Ollama model (Phase 7)

## Next precise task
Phase 2, Step 1: add PyMuPDF; implement page-by-page text extraction,
SHA-256 doc_id from file bytes, and needs_ocr flag for pages without text; tests.
