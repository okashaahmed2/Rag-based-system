"""HTTP API (FastAPI) on top of the ingestion and query pipelines.

Importing this package builds nothing; the app is created by
:func:`rag_pipeline.api.app.create_app` (uvicorn ``--factory``).
"""
