"""Manual RAG run on the golden set against the REAL eval store and Ollama.

Not run by pytest. Read path only: ``build_retrieval_context`` (which opens
the collection with ``create=False`` and never creates a missing store) +
``answer_question`` for each golden question, on the existing eval store
``data/eval/stores/all-minilm-l6-v2_600_120`` (pointed at the same way as
``scripts/run_eval.py``: settings overrides for ``chroma_path`` and
``status_db_path``). Nothing is ingested, nothing is written, no result file
is created; ``data/chroma`` and ``data/status.db`` are never used.

Per question it prints status, sources cited, whether a cited (doc_id, page)
is one of the expected pages, token counts, timings and the answer text
(printing answers is acceptable in this manual script only), then a summary.

Usage::

    .venv\\Scripts\\python.exe scripts\\rag_smoke.py
"""

from __future__ import annotations

import logging
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from rag_pipeline.answering import AnswerStatus, RagAnswer, answer_question
from rag_pipeline.config import Settings
from rag_pipeline.evaluation import GoldenQuestion, load_golden_set
from rag_pipeline.llm import LLMError, check_ollama
from rag_pipeline.retrieval import RetrievalError, build_retrieval_context

#: Existing eval store (relative to the project root), read only.
EVAL_STORE = Path("data/eval/stores/all-minilm-l6-v2_600_120")
GOLDEN_SET = Path("eval/golden_set.jsonl")


def out(line: str = "") -> None:
    print(line, flush=True)


def expected_pages(question: GoldenQuestion) -> set[tuple[str, int]]:
    """Every expected ``(doc_id, page)`` of ``question`` (empty for negatives)."""
    return {(s.doc_id, page) for s in question.expected for page in s.pages}


def citation_hit(question: GoldenQuestion, answer: RagAnswer) -> bool | None:
    """Any cited (doc_id, page) among the expected pages; ``None`` for negatives."""
    if not question.is_answerable:
        return None
    expected = expected_pages(question)
    return any((c.doc_id, c.page_number) in expected for c in answer.citations)


def expected_retrieved(question: GoldenQuestion, answer: RagAnswer) -> bool:
    """Whether any retrieved source is an expected (doc_id, page)."""
    expected = expected_pages(question)
    return any((s.doc_id, s.page_number) in expected for s in answer.sources)


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    base = Settings()
    store = base.project_root / EVAL_STORE
    settings = Settings(chroma_path=store / "chroma", status_db_path=store / "status.db")
    out(f"MODEL {settings.llm_model}")
    out(
        f"Store {EVAL_STORE.as_posix()} | Ollama {settings.ollama_base_url} | "
        f"top_k {settings.query_top_k} | num_ctx {settings.llm_num_ctx} | "
        f"max_answer_tokens {settings.llm_max_answer_tokens} | "
        f"timeout {settings.llm_timeout_seconds} s"
    )

    try:
        info = check_ollama(settings)
    except LLMError as exc:
        out(f"ERROR check_ollama: {type(exc).__name__}: {exc}")
        return 1
    out(f"Ollama {info['version']}, model present: {info['model_present']}")

    try:
        questions = load_golden_set(base.project_root / GOLDEN_SET)
        context = build_retrieval_context(settings)
    except (RetrievalError, ValueError) as exc:
        out(f"ERROR setup: {type(exc).__name__}: {exc}")
        return 1

    rows: list[tuple[GoldenQuestion, RagAnswer, float]] = []
    errors = 0
    for question in questions:
        started = time.perf_counter()
        try:
            answer = answer_question(question.question, context, settings)
        except (LLMError, RetrievalError, ValueError) as exc:
            errors += 1
            out(f"\n=== {question.id}: ERROR {type(exc).__name__}: {exc}")
            continue
        wall = time.perf_counter() - started
        rows.append((question, answer, wall))
        hit = citation_hit(question, answer)
        cited = ", ".join(f"S{c.source_number} {c.filename} p.{c.page_number}" for c in answer.citations)
        out(f"\n=== {question.id} ({'answerable' if question.is_answerable else 'negative'})")
        out(
            f"status={answer.status.value} llm_called={answer.llm_called} "
            f"citation_hit={'n/a' if hit is None else hit} invalid={list(answer.invalid_citations)} "
            f"truncated={answer.truncated} thinking_leaked={answer.thinking_leaked}"
        )
        out(f"cited: {cited or '-'}")
        expected = expected_pages(question)
        for s in answer.sources:
            mark = " *EXPECTED*" if (s.doc_id, s.page_number) in expected else ""
            out(f"  S{s.source_number} {s.filename} p.{s.page_number} sim={s.similarity:.3f}{mark}")
        out(
            f"prompt_tokens={answer.prompt_tokens} completion_tokens={answer.completion_tokens} "
            f"load={answer.load_seconds:.2f}s prompt_eval={answer.prompt_eval_seconds:.2f}s "
            f"eval={answer.eval_seconds:.2f}s wall={wall:.2f}s"
        )
        out(f"answer: {answer.answer_text}")

    out("\n=== SUMMARY")
    statuses = Counter(a.status.value for _, a, _ in rows)
    out("status counts: " + ", ".join(f"{s.value} {statuses.get(s.value, 0)}" for s in AnswerStatus))
    answerable = [(q, a) for q, a, _ in rows if q.is_answerable]
    hits = [q.id for q, a in answerable if citation_hit(q, a)]
    out(f"answerable with citation_hit: {len(hits)}/{len(answerable)} ({', '.join(hits)})")
    answered_hits = [
        q.id for q, a in answerable if a.status is AnswerStatus.ANSWERED and citation_hit(q, a)
    ]
    n_answerable = sum(1 for q in questions if q.is_answerable)
    out(
        f"answerable ANSWERED with citation_hit: {len(answered_hits)}/{n_answerable} "
        f"({', '.join(answered_hits)})"
    )
    misses = [
        f"{q.id}={a.status.value} "
        f"({'LLM miss: expected page retrieved' if expected_retrieved(q, a) else 'retrieval miss: no expected page retrieved'})"
        for q, a in answerable
        if q.id not in answered_hits
    ]
    out(f"answerable not ANSWERED+hit: {'; '.join(misses) or '-'}")
    negatives = [(q, a) for q, a, _ in rows if not q.is_answerable]
    neg_ok = [q.id for q, a in negatives if a.status is AnswerStatus.NOT_FOUND]
    out(
        f"negatives answered NOT_FOUND: {len(neg_ok)}/{len(negatives)} "
        f"({', '.join(f'{q.id}={a.status.value}' for q, a in negatives)})"
    )
    walls = [w for _, _, w in rows]
    if walls:
        out(f"wall seconds: median {statistics.median(walls):.2f}, max {max(walls):.2f}")
        out(f"max prompt_tokens: {max(a.prompt_tokens for _, a, _ in rows)}")
    out(f"thinking_leaked: {sum(a.thinking_leaked for _, a, _ in rows)}/{len(rows)}")
    out(f"truncated: {sum(a.truncated for _, a, _ in rows)}/{len(rows)}")
    out(f"errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
