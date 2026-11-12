# Architecture

## Goal
Local-first, zero-budget RAG system: PDFs dropped in a folder are ingested
automatically, and questions are answered by a local LLM with document and
page citations.

## Two pipelines sharing storage

INGESTION (write path)

```
data/inbox -> n8n Schedule Trigger -> FastAPI POST /ingest/scan
  -> BackgroundTasks worker: PyMuPDF extraction -> cleaning
     -> page-aware chunking -> Sentence Transformers embeddings
  -> ChromaDB (vectors + chunk metadata) + SQLite (document/job status)
  -> PDF moved to data/processed or data/failed
```

QUERY (read path)

```
user or n8n -> FastAPI /query -> retriever (embed query, top-k from ChromaDB)
  -> Ollama -> answer with document name + page citations
```

## Responsibilities
- n8n: orchestration only (scheduling, triggering, notifications)
- Python: all processing and application logic
- FastAPI: interface between n8n/users and Python
- ChromaDB: vectors and chunk metadata only
- SQLite: doc_id, filename, file_hash, status
  (pending/processing/done/failed/needs_ocr), error_message, attempts, timestamps

## Design rules
- doc_id = SHA-256 of PDF bytes; duplicates detected from Phase 2
- Chunk metadata: doc_id, filename, page_number, chunk_index
- Same embedding model for ingestion and query (stored at collection level)
- Re-ingestion deletes old chunks first (idempotent)
- Pages without extractable text -> needs_ocr
- /ingest returns job_id immediately; /status/{job_id} reports progress
- Golden evaluation set (10-15 questions) created in Phase 6, rerun after major changes

## Local services
n8n localhost:5678 (npm) | FastAPI localhost:8000 | Ollama localhost:11434
