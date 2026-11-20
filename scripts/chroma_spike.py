"""Throwaway spike: verify the ChromaDB behaviour Phase 5 Step 1b relies on.

Not application code and not part of the pytest suite. Uses a fresh temporary
directory (never ``data/``) and small fake L2-normalised 384-dim vectors, so
no embedding model is loaded. Every check logs its evidence and a
``[a] VERIFIED`` / ``NOT VERIFIED`` line.

Usage::

    .venv\\Scripts\\python.exe scripts\\chroma_spike.py
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import psutil

logger = logging.getLogger("chroma_spike")

DIM = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "chunks"
LEGACY_COLLECTION = "legacy_space"
TOLERANCE = 1e-5
CHROMA_CACHE = Path.home() / ".cache" / "chroma"
RESULTS: dict[str, bool] = {}
#: Checks with a documented, accepted limitation; excluded from the failure count.
KNOWN_LIMITATIONS: dict[str, str] = {}


def rss_mb() -> float:
    """Resident set size of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def record(key: str, ok: bool, what: str) -> None:
    """Log and remember the result of one check."""
    RESULTS[key] = ok
    logger.info("[%s] %s: %s", key, "VERIFIED" if ok else "NOT VERIFIED", what)


def cache_snapshot() -> tuple[bool, int]:
    """(exists, number of files) of the Chroma model cache."""
    if not CHROMA_CACHE.exists():
        return False, 0
    return True, sum(1 for p in CHROMA_CACHE.rglob("*") if p.is_file())


def fake_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """``n`` random L2-normalised float32 vectors of length DIM."""
    vectors = rng.standard_normal((n, DIM)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def chunk_records(
    rng: np.random.Generator, doc_id: str, filename: str, n: int
) -> dict[str, Any]:
    """ids/embeddings/documents/metadatas for ``n`` chunks of one document."""
    return {
        "ids": [f"{doc_id}:{i}" for i in range(n)],
        "embeddings": fake_vectors(rng, n),
        "documents": [f"chunk {i} text of {filename}" for i in range(n)],
        "metadatas": [
            {
                "doc_id": doc_id,
                "filename": filename,
                "page_number": i + 1,
                "chunk_index": i,
                "char_start": i * 100,
                "char_end": i * 100 + 90,
            }
            for i in range(n)
        ],
    }


def count_doc(collection: Any, doc_id: str) -> list[str]:
    """Sorted ids of all chunks stored for ``doc_id``."""
    return sorted(collection.get(where={"doc_id": doc_id}, include=[])["ids"])


def space_of(collection: Any) -> Any:
    """The HNSW distance space read back from a collection's configuration."""
    hnsw = collection.configuration.get("hnsw") or {}
    return hnsw.get("space")


def main() -> int:
    """Run all checks; return 0 if every check was verified."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    warnings.simplefilter("always")
    rng = np.random.default_rng(42)

    cache_before = cache_snapshot()
    logger.info("~/.cache/chroma before: exists=%s files=%d", *cache_before)
    logger.info("RSS before import chromadb: %.1f MB", rss_mb())

    started = time.perf_counter()
    import chromadb
    from chromadb.api.shared_system_client import SharedSystemClient
    from chromadb.config import Settings

    logger.info(
        "import chromadb %s: %.2f s, RSS after import: %.1f MB",
        chromadb.__version__,
        time.perf_counter() - started,
        rss_mb(),
    )

    root = Path(tempfile.mkdtemp(prefix="chroma_spike_"))
    path = root / "store"
    logger.info("Temp store: %s", path)
    caught: list[warnings.WarningMessage] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # a. PersistentClient with telemetry off
        settings = Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=str(path), settings=settings)
        client_settings = client.get_settings()
        record(
            "a",
            client_settings.anonymized_telemetry is False,
            f"PersistentClient created, anonymized_telemetry="
            f"{client_settings.anonymized_telemetry}, otel endpoint="
            f"{client_settings.chroma_otel_collection_endpoint!r}",
        )

        # b + c + d. Collection: cosine via configuration, metadata, no EF
        collection_metadata = {"embedding_model": MODEL_NAME, "embedding_dim": DIM}
        collection = client.create_collection(
            name=COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=collection_metadata,
            embedding_function=None,
        )
        record(
            "b",
            space_of(collection) == "cosine",
            f"configuration={{'hnsw': {{'space': 'cosine'}}}} -> read back "
            f"space={space_of(collection)!r}; full configuration_json="
            f"{collection.configuration_json}",
        )

        # b (comparison): legacy "hnsw:space" metadata on a second collection
        with warnings.catch_warnings(record=True) as legacy_warns:
            warnings.simplefilter("always")
            legacy = client.create_collection(
                name=LEGACY_COLLECTION,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        logger.info(
            "[b] legacy metadata {'hnsw:space': 'cosine'} -> read back space=%r, "
            "metadata=%r, warnings=%r",
            space_of(legacy),
            legacy.metadata,
            [f"{w.category.__name__}: {w.message}" for w in legacy_warns],
        )
        client.delete_collection(LEGACY_COLLECTION)

        record(
            "c (create)",
            collection.metadata == collection_metadata,
            f"metadata at creation read back: {collection.metadata}",
        )
        ef_attr = getattr(collection, "_embedding_function", "missing")
        try:
            collection.query(query_texts=["probe"], n_results=1)
            query_texts_result = "query_texts ACCEPTED (an embedding function exists!)"
            query_texts_rejected = False
        except Exception as exc:  # noqa: BLE001 - we only report the type
            query_texts_result = f"query_texts rejected: {type(exc).__name__}: {exc}"
            query_texts_rejected = True
        logger.info("[d] collection._embedding_function=%r; %s", ef_attr, query_texts_result)

        # e. add() with ids, embeddings, documents, flat metadatas
        doc_a, doc_b = "a" * 64, "b" * 64
        rec_a = chunk_records(rng, doc_a, "doc_a.pdf", 5)
        rec_b = chunk_records(rng, doc_b, "doc_b.pdf", 2)
        collection.add(**rec_a)
        collection.add(**rec_b)
        got = collection.get(ids=[f"{doc_a}:3"], include=["documents", "metadatas", "embeddings"])
        stored_vec = np.asarray(got["embeddings"][0], dtype=np.float32)
        record(
            "e",
            collection.count() == 7
            and got["metadatas"][0] == rec_a["metadatas"][3]
            and got["documents"][0] == rec_a["documents"][3]
            and np.allclose(stored_vec, rec_a["embeddings"][3], atol=1e-7),
            f"count={collection.count()}; {doc_a[:8]}..:3 metadata={got['metadatas'][0]}, "
            f"document={got['documents'][0]!r}, embedding equal="
            f"{np.allclose(stored_vec, rec_a['embeddings'][3], atol=1e-7)}",
        )

        # f. query() with query_embeddings; distance == 1 - dot(q, v)
        query = fake_vectors(rng, 1)[0]
        result = collection.query(
            query_embeddings=[query],
            n_results=7,
            include=["distances", "metadatas", "embeddings"],
        )
        max_err = 0.0
        for chunk_id, dist, emb in zip(
            result["ids"][0], result["distances"][0], result["embeddings"][0]
        ):
            expected = 1.0 - float(np.dot(query, np.asarray(emb, dtype=np.float32)))
            err = abs(dist - expected)
            max_err = max(max_err, err)
            logger.info(
                "[f]   %s..%s distance=%.7f 1-dot=%.7f |diff|=%.2e",
                chunk_id[:4],
                chunk_id[-2:],
                dist,
                expected,
                err,
            )
        sorted_ok = result["distances"][0] == sorted(result["distances"][0])
        record(
            "f",
            max_err <= TOLERANCE and sorted_ok,
            f"query_embeddings only; 7 results ascending={sorted_ok}; "
            f"max |distance - (1 - dot)| = {max_err:.2e} (tolerance {TOLERANCE})",
        )

        # g. orphan demonstration
        logger.info("[g] doc A before: %d chunks, doc B: %d", len(count_doc(collection, doc_a)), len(count_doc(collection, doc_b)))
        rec_a3 = chunk_records(rng, doc_a, "doc_a.pdf", 3)
        collection.upsert(**rec_a3)
        after_upsert = count_doc(collection, doc_a)
        orphans = [i for i in after_upsert if i not in rec_a3["ids"]]
        logger.info(
            "[g] upsert of 3 chunks only -> doc A has %d chunks, orphans=%s",
            len(after_upsert),
            [o[-2:] for o in orphans],
        )
        collection.delete(where={"doc_id": doc_a})
        after_delete = count_doc(collection, doc_a)
        b_after_delete = count_doc(collection, doc_b)
        collection.add(**rec_a3)
        after_readd = count_doc(collection, doc_a)
        logger.info(
            "[g] delete(where doc_id=A) -> A=%d, B=%d; add 3 -> A=%d ids=%s, B=%d, total=%d",
            len(after_delete),
            len(b_after_delete),
            len(after_readd),
            [i[-2:] for i in after_readd],
            len(count_doc(collection, doc_b)),
            collection.count(),
        )
        record(
            "g",
            len(after_upsert) == 5
            and len(orphans) == 2
            and after_delete == []
            and after_readd == sorted(rec_a3["ids"])
            and b_after_delete == sorted(rec_b["ids"])
            and collection.count() == 5,
            "upsert-only leaves 2 orphans; delete(where)+add leaves exactly 3; doc B untouched (2)",
        )

        # l. add() with an id that already exists
        existing_id = f"{doc_a}:0"
        before = collection.get(ids=[existing_id], include=["documents", "metadatas", "embeddings"])
        count_before = collection.count()
        new_vector = fake_vectors(rng, 1)
        with warnings.catch_warnings(record=True) as add_warns:
            warnings.simplefilter("always")
            try:
                collection.add(
                    ids=[existing_id],
                    embeddings=new_vector,
                    documents=["REPLACED text"],
                    metadatas=[{**rec_a3["metadatas"][0], "filename": "replaced.pdf"}],
                )
                add_error = None
            except Exception as exc:  # noqa: BLE001 - we only report the type
                add_error = f"{type(exc).__name__}: {exc}"
        after = collection.get(ids=[existing_id], include=["documents", "metadatas", "embeddings"])
        unchanged = (
            after["documents"] == before["documents"]
            and after["metadatas"] == before["metadatas"]
            and np.allclose(after["embeddings"][0], before["embeddings"][0])
        )
        if add_error is not None:
            behaviour = f"raises {add_error}"
        elif add_warns:
            behaviour = f"warns {[str(w.message) for w in add_warns]}"
        elif unchanged:
            behaviour = "silently ignored (no exception, no warning, old record kept)"
        else:
            behaviour = "silently updated"
        record(
            "l",
            add_error is None and not add_warns and unchanged and collection.count() == count_before,
            f"add() with existing id: {behaviour}; document {before['documents']} -> "
            f"{after['documents']}, filename {before['metadatas'][0]['filename']!r} -> "
            f"{after['metadatas'][0]['filename']!r}, count {count_before} -> "
            f"{collection.count()} (so replace must delete + upsert, never add)",
        )

        # i. max batch size
        max_batch = client.get_max_batch_size()
        record("i", isinstance(max_batch, int) and max_batch > 0, f"client.get_max_batch_size()={max_batch}")

        # h. close / reopen: drop the client and the shared system cache
        del collection, legacy, client
        gc.collect()
        SharedSystemClient.clear_system_cache()
        client2 = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        reopened = client2.get_collection(COLLECTION, embedding_function=None)
        in_process_ok = (
            reopened.count() == 5
            and reopened.metadata == collection_metadata
            and space_of(reopened) == "cosine"
        )
        logger.info(
            "[h] in-process reopen: count=%d metadata=%s space=%r",
            reopened.count(),
            reopened.metadata,
            space_of(reopened),
        )
        record(
            "c (reopen)",
            reopened.metadata == collection_metadata,
            f"metadata after reopen: {reopened.metadata}",
        )
        del reopened, client2
        gc.collect()
        SharedSystemClient.clear_system_cache()

    # h (fresh process): nothing shared in memory
    probe = (
        "import chromadb, sys;"
        "from chromadb.config import Settings;"
        "c = chromadb.PersistentClient(path=sys.argv[1], settings=Settings(anonymized_telemetry=False));"
        "col = c.get_collection('chunks', embedding_function=None);"
        "print(col.count(), col.metadata, (col.configuration.get('hnsw') or {}).get('space'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    fresh_out = proc.stdout.strip()
    logger.info("[h] fresh-process reopen: rc=%d stdout=%r stderr=%r", proc.returncode, fresh_out, proc.stderr.strip()[-300:])
    record(
        "h",
        in_process_ok and proc.returncode == 0 and fresh_out.startswith("5 ") and "cosine" in fresh_out and MODEL_NAME in fresh_out,
        "count=5, metadata and cosine persisted (in-process reopen + fresh process)",
    )

    # d. no model download
    cache_after = cache_snapshot()
    logger.info("~/.cache/chroma after: exists=%s files=%d", *cache_after)
    record(
        "d",
        query_texts_rejected and cache_after == cache_before,
        f"embedding_function=None; {query_texts_result}; ~/.cache/chroma "
        f"before={cache_before} after={cache_after}",
    )

    # j. memory
    logger.info("[j] RSS after spike: %.1f MB", rss_mb())
    RESULTS["j"] = True

    # k. temp directory cleanup on Windows
    try:
        shutil.rmtree(root)
        record("k", not root.exists(), f"shutil.rmtree({root}) succeeded, no PermissionError")
    except PermissionError as exc:
        KNOWN_LIMITATIONS["k"] = f"shutil.rmtree raised PermissionError: {exc}"
        logger.info(
            "[k] KNOWN_LIMITATION: %s (the opening process keeps the store locked)",
            KNOWN_LIMITATIONS["k"],
        )

    logger.info("Warnings during the spike (%d):", len(caught))
    for w in caught:
        logger.info("  %s: %s (%s:%s)", w.category.__name__, w.message, Path(w.filename).name, w.lineno)

    failed = [key for key, ok in RESULTS.items() if not ok]
    logger.info(
        "Summary: %d checks recorded, not verified: %s; known limitations: %s",
        len(RESULTS),
        failed or "none",
        sorted(KNOWN_LIMITATIONS) or "none",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
