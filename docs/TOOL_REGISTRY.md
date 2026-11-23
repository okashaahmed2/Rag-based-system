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
| numpy | Dependency of PyMuPDF 2.x (also needed for embeddings) | Phase 2 | Yes (direct import in embeddings) | Installed | 2.4.6 |
| onnxruntime | Dependency of PyMuPDF 2.x layout | Phase 2 | Indirect | Installed | 1.31.1 |
| Sentence Transformers | Embeddings | Phase 4 | Yes | Installed | 6.3.0 (pinned >=6.3,<7) |
| PyTorch | CPU build, pulled by sentence-transformers | Phase 4 | Indirect | Installed | 2.15.0+cpu |
| transformers | Tokenizer/model code | Phase 4 | Indirect | Installed | 5.23.0 |
| huggingface-hub | Model download/cache | Phase 4 | Indirect | Installed | 2.4.0 |
| ChromaDB | Vector store | Phase 5 | Yes | Installed | 1.5.9 (pinned >=1.5,<2) |
| SQLite | Status tracking (stdlib) | Phase 5 | Yes | Built in | - |
| Ollama | Local LLM server | Phase 7 | Yes | Installed (D:\Ollama\app, models in D:\Ollama\models) | 0.40.17 |
| httpx | HTTP client for Ollama | Phase 7 | Yes | Installed (declared >=0.28,<1; was transitive via chromadb) | 0.28.1 |
| llama3.2:3b | Default answer model (Q4_K_M, 2.0 GB, Llama 3.2 Community License) | Phase 7 | Yes | Pulled | - |
| granite4.2:3b | Rejected candidate (thinking leak, 60-150 s answers; Apache 2.0, 2.2 GB) | Phase 7 | No | Pulled (can be removed) | - |
| FastAPI + Uvicorn | API backend | Phase 8 | Yes | Not installed (uvicorn 0.54.0 present only as a chromadb dependency) | - |
| Tesseract OCR | OCR for scanned pages | Phase 13 | Optional | Not installed | - |
| Docker | Packaging | Phase 13 | Optional | Not installed | - |
