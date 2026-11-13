# Tool Registry

| Tool | Purpose | First used | Required | Status | Version |
|---|---|---|---|---|---|
| Python | Application language | Phase 1 | Yes | Installed (venv) | 3.11.9 |
| pip | Package installer | Phase 1 | Yes | Upgraded in venv | 26.3 |
| Git + GitHub | Version control, backup, portfolio | Phase 1 | Yes | Working | 2.55.0 |
| pydantic-settings | Typed config from env/.env | Phase 1 | Yes | Installed | 2.16.2 |
| pytest | Automated tests | Phase 1 | Yes (dev) | Installed | 9.2.0 |
| logging | Logs (stdlib) | Phase 1 | Yes | Built in | - |
| n8n | Workflow orchestration | Phase 9 | Yes | Installed via npm, not running | 2.14.2 |
| Node.js / npm | Runtime for n8n | Phase 9 | Yes | Installed | 24.14.1 / 11.11.0 |
| PyMuPDF | PDF text extraction | Phase 2 | Yes | Installed (2.x, bundles layout/4llm) | 2.1 |
| numpy | Dependency of PyMuPDF 2.x (also needed for embeddings) | Phase 2 | Yes | Installed | 2.4.6 |
| onnxruntime | Dependency of PyMuPDF 2.x layout | Phase 2 | Indirect | Installed | 1.31.1 |
| Sentence Transformers | Embeddings | Phase 4 | Yes | Not installed | - |
| ChromaDB | Vector store | Phase 5 | Yes | Not installed | - |
| SQLite | Status tracking (stdlib) | Phase 5 | Yes | Built in | - |
| Ollama | Local LLM | Phase 7 | Yes | Not installed | - |
| FastAPI + Uvicorn | API backend | Phase 8 | Yes | Not installed | - |
| Tesseract OCR | OCR for scanned pages | Phase 13 | Optional | Not installed | - |
| Docker | Packaging | Phase 13 | Optional | Not installed | - |
