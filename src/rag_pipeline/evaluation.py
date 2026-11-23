"""Golden evaluation set and retrieval metrics (pure: no model, no Chroma).

Golden set format: JSON Lines (UTF-8), one question per line::

    {"id": "q01", "question": "...",
     "expected": [{"doc_id": "<sha256>", "filename": "Lecture07.pdf", "pages": [4, 5]}],
     "tags": ["definition"], "notes": "optional"}

``expected == []`` marks a negative (unanswerable) question. Blank lines are
allowed; ``tags`` and ``notes`` are optional; any other key is rejected.

Relevance is labelled per ``(doc_id, page_number)``, not per chunk id: chunk
ids point to different text when the chunk settings change, pages do not.
Matching uses ``doc_id`` only; ``filename`` is informational (a file is
renamed when it is moved or collides with another name).

Metrics are computed over the ranks of answerable questions only; the caller
leaves negative questions out (their rank is always ``None`` and would
otherwise count as a miss).

:func:`evaluate` runs a golden set through any retrieve callable and returns
per-question ranks plus hit@k / MRR@k overall and per tag. It holds ids,
filenames, pages, ranks and similarities only, never question or chunk text.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: A ``doc_id`` is the lowercase SHA-256 hex digest of the PDF bytes.
DOC_ID_PATTERN = re.compile(r"[0-9a-f]{64}")

_QUESTION_REQUIRED = frozenset({"id", "question", "expected"})
_QUESTION_ALLOWED = _QUESTION_REQUIRED | {"tags", "notes"}
_EXPECTED_KEYS = frozenset({"doc_id", "filename", "pages"})


class GoldenSetError(ValueError):
    """Raised when a golden set file is unreadable or invalid.

    Attributes:
        line_number: 1-based line of the problem, or ``None`` for the whole file.
    """

    def __init__(self, message: str, line_number: int | None = None) -> None:
        self.line_number = line_number
        super().__init__(message if line_number is None else f"line {line_number}: {message}")


@dataclass(frozen=True)
class ExpectedSource:
    """Pages of one document that answer a question.

    Attributes:
        doc_id: SHA-256 hex digest of the PDF (the match key).
        filename: File name at labelling time (informational only).
        pages: 1-based page numbers, any of which counts as relevant.
    """

    doc_id: str
    filename: str
    pages: tuple[int, ...]


@dataclass(frozen=True)
class GoldenQuestion:
    """One golden set entry.

    Attributes:
        id: Unique question id.
        question: Question text.
        expected: Relevant sources; empty for a negative question.
        tags: Free labels (e.g. ``definition``).
        notes: Optional free text.
        line_number: Line of the entry in the file.
    """

    id: str
    question: str
    expected: tuple[ExpectedSource, ...]
    tags: tuple[str, ...]
    notes: str | None
    line_number: int

    @property
    def is_answerable(self) -> bool:
        """``False`` for a negative question (``expected == []``)."""
        return bool(self.expected)


def _parse_expected(raw: Any, line: int) -> tuple[ExpectedSource, ...]:
    if not isinstance(raw, list):
        raise GoldenSetError("'expected' must be a list", line)
    sources: list[ExpectedSource] = []
    for position, item in enumerate(raw, start=1):
        where = f"expected[{position}]"
        if not isinstance(item, dict):
            raise GoldenSetError(f"{where} must be an object", line)
        unknown = sorted(set(item) - _EXPECTED_KEYS)
        if unknown:
            raise GoldenSetError(f"{where} has unknown key(s): {', '.join(unknown)}", line)
        missing = sorted(_EXPECTED_KEYS - set(item))
        if missing:
            raise GoldenSetError(f"{where} lacks key(s): {', '.join(missing)}", line)
        doc_id, filename, pages = item["doc_id"], item["filename"], item["pages"]
        if not isinstance(doc_id, str) or not DOC_ID_PATTERN.fullmatch(doc_id):
            raise GoldenSetError(f"{where}.doc_id must be 64 lowercase hex characters", line)
        if not isinstance(filename, str) or not filename.strip():
            raise GoldenSetError(f"{where}.filename must be a non-empty string", line)
        if not isinstance(pages, list) or not pages:
            raise GoldenSetError(f"{where}.pages must be a non-empty list", line)
        if not all(type(p) is int and p >= 1 for p in pages):
            raise GoldenSetError(f"{where}.pages must hold integers >= 1", line)
        if len(set(pages)) != len(pages):
            raise GoldenSetError(f"{where}.pages has duplicates", line)
        sources.append(ExpectedSource(doc_id=doc_id, filename=filename, pages=tuple(pages)))
    doc_ids = [s.doc_id for s in sources]
    if len(set(doc_ids)) != len(doc_ids):
        raise GoldenSetError("'expected' lists the same doc_id twice (merge its pages)", line)
    return tuple(sources)


def _parse_question(raw: Any, line: int) -> GoldenQuestion:
    if not isinstance(raw, dict):
        raise GoldenSetError("each line must be a JSON object", line)
    unknown = sorted(set(raw) - _QUESTION_ALLOWED)
    if unknown:
        raise GoldenSetError(f"unknown key(s): {', '.join(unknown)}", line)
    missing = sorted(_QUESTION_REQUIRED - set(raw))
    if missing:
        raise GoldenSetError(f"missing key(s): {', '.join(missing)}", line)
    qid, question = raw["id"], raw["question"]
    if not isinstance(qid, str) or not qid.strip():
        raise GoldenSetError("'id' must be a non-empty string", line)
    if not isinstance(question, str) or not question.strip():
        raise GoldenSetError("'question' must be a non-empty string", line)
    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise GoldenSetError("'tags' must be a list of strings", line)
    notes = raw.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise GoldenSetError("'notes' must be a string", line)
    return GoldenQuestion(
        id=qid,
        question=question,
        expected=_parse_expected(raw["expected"], line),
        tags=tuple(tags),
        notes=notes,
        line_number=line,
    )


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Read and validate a JSON Lines golden set, in file order.

    The file is read as UTF-8 (a leading byte order mark is accepted, as
    written by Windows PowerShell 5.1). Blank lines are skipped. Every error
    names its line.

    Raises:
        GoldenSetError: the file cannot be read or decoded, a line is not
            valid JSON or breaks the format, an id repeats, or the file holds
            no question.
    """
    try:
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise GoldenSetError(f"cannot read golden set {path}: {exc}") from exc

    questions: list[GoldenQuestion] = []
    first_line_of: dict[str, int] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"invalid JSON: {exc.msg}", line_number) from exc
        question = _parse_question(raw, line_number)
        if question.id in first_line_of:
            raise GoldenSetError(
                f"duplicate id {question.id!r} (first on line {first_line_of[question.id]})",
                line_number,
            )
        first_line_of[question.id] = line_number
        questions.append(question)
    if not questions:
        raise GoldenSetError(f"golden set {path} holds no question")
    return questions


def first_relevant_rank(
    results: Sequence[tuple[str, int]], expected: Sequence[ExpectedSource]
) -> int | None:
    """1-based rank of the first result on an expected page, else ``None``.

    ``results`` are ``(doc_id, page_number)`` pairs, nearest first. A result
    is relevant when its ``doc_id`` equals an expected ``doc_id`` and its page
    is one of that source's pages. A negative question (no expected source)
    always gives ``None``.
    """
    relevant = {(source.doc_id, page) for source in expected for page in source.pages}
    for rank, (doc_id, page_number) in enumerate(results, start=1):
        if (doc_id, page_number) in relevant:
            return rank
    return None


def _check_ranks(ranks: Sequence[int | None], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be an int >= 1, got {k!r}")
    if not ranks:
        raise ValueError("no ranks given: metrics need at least one answerable question")
    if not all(r is None or (type(r) is int and r >= 1) for r in ranks):
        raise ValueError("ranks must be ints >= 1 or None")


def hit_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of questions whose first relevant rank is ``<= k``.

    ``ranks`` holds one entry per answerable question (``None`` = no relevant
    result). A rank above ``k`` is a miss.

    Raises:
        ValueError: ``ranks`` is empty (no answerable question: the metric is
            undefined, so no misleading 0.0 is returned), ``k < 1``, or a rank
            is not ``None`` or an int >= 1.
    """
    _check_ranks(ranks, k)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Mean reciprocal rank: mean of ``1 / rank`` (0 if ``None`` or ``> k``).

    Raises:
        ValueError: same conditions as :func:`hit_at_k`.
    """
    _check_ranks(ranks, k)
    total = sum(1.0 / r for r in ranks if r is not None and r <= k)
    return total / len(ranks)


# --- evaluation runner -----------------------------------------------------------

#: Cut-offs evaluated by default; retrieval asks for ``max(ks)`` results.
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
#: Number of top results kept per question in the result.
TOP_SHOWN = 5


@dataclass(frozen=True)
class RetrievedHit:
    """One retrieved result as the evaluation sees it (no text).

    Attributes:
        doc_id: Document of the result.
        page_number: 1-based page of the result.
        similarity: Cosine similarity to the question.
        filename: File name, if the retriever knows it.
    """

    doc_id: str
    page_number: int
    similarity: float
    filename: str | None = None


#: ``retrieve(question, top_k)`` -> results nearest first, at most ``top_k``.
RetrieveFn = Callable[[str, int], Sequence[RetrievedHit]]


@dataclass(frozen=True)
class QuestionResult:
    """Outcome for one golden question.

    Attributes:
        id: Question id.
        tags: Question tags.
        answerable: ``False`` for a negative question.
        first_relevant_rank: 1-based rank of the first relevant result within
            ``max(ks)`` results, or ``None`` (always ``None`` for negatives).
        top: The first :data:`TOP_SHOWN` results.
        top1_similarity: Similarity of the first result, ``None`` if there was none.
    """

    id: str
    tags: tuple[str, ...]
    answerable: bool
    first_relevant_rank: int | None
    top: tuple[RetrievedHit, ...]
    top1_similarity: float | None


@dataclass(frozen=True)
class MetricSummary:
    """hit@k and MRR for a group of questions.

    ``hit_at`` and ``mrr`` are ``None`` when the group has no answerable
    question (the metrics are undefined there, not 0).
    """

    questions: int
    answerable: int
    hit_at: dict[int, float] | None
    mrr: float | None


@dataclass(frozen=True)
class SimilaritySummary:
    """Top-1 similarities of answerable and negative questions.

    Attributes:
        answerable_top1_min / _median / _max: Over answerable questions with
            at least one result; ``None`` if there is none.
        answerable_without_results: Answerable questions that got no result.
        negative_top1: ``(question id, top-1 similarity or None)`` per negative.
    """

    answerable_top1_min: float | None
    answerable_top1_median: float | None
    answerable_top1_max: float | None
    answerable_without_results: int
    negative_top1: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Everything :func:`evaluate` measured, in golden-set order."""

    ks: tuple[int, ...]
    mrr_k: int
    questions: tuple[QuestionResult, ...]
    overall: MetricSummary
    per_tag: dict[str, MetricSummary]
    similarity: SimilaritySummary

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form (string keys, deterministic order)."""

        def metrics(summary: MetricSummary) -> dict[str, Any]:
            return {
                "questions": summary.questions,
                "answerable": summary.answerable,
                "hit_at": None
                if summary.hit_at is None
                else {str(k): v for k, v in summary.hit_at.items()},
                f"mrr_at_{self.mrr_k}": summary.mrr,
            }

        sim = self.similarity
        return {
            "ks": list(self.ks),
            "mrr_k": self.mrr_k,
            "overall": metrics(self.overall),
            "per_tag": {tag: metrics(s) for tag, s in self.per_tag.items()},
            "similarity": {
                "answerable_top1": {
                    "min": sim.answerable_top1_min,
                    "median": sim.answerable_top1_median,
                    "max": sim.answerable_top1_max,
                    "without_results": sim.answerable_without_results,
                },
                "negative_top1": {qid: value for qid, value in sim.negative_top1},
            },
            "questions": [
                {
                    "id": q.id,
                    "tags": list(q.tags),
                    "answerable": q.answerable,
                    "first_relevant_rank": q.first_relevant_rank,
                    "top1_similarity": q.top1_similarity,
                    "top": [
                        {
                            "doc_id": hit.doc_id,
                            "filename": hit.filename,
                            "page": hit.page_number,
                            "similarity": hit.similarity,
                        }
                        for hit in q.top
                    ],
                }
                for q in self.questions
            ],
        }


def _check_ks(ks: Sequence[int]) -> tuple[int, ...]:
    if not ks:
        raise ValueError("ks must not be empty")
    if not all(type(k) is int and k >= 1 for k in ks):
        raise ValueError(f"ks must be ints >= 1, got {list(ks)!r}")
    if len(set(ks)) != len(ks):
        raise ValueError(f"ks must be unique, got {list(ks)!r}")
    return tuple(sorted(ks))


def _summarise(results: Sequence[QuestionResult], ks: tuple[int, ...]) -> MetricSummary:
    """Metrics over the answerable questions in ``results`` (``None`` if there are none)."""
    ranks = [r.first_relevant_rank for r in results if r.answerable]
    if not ranks:
        return MetricSummary(len(results), 0, None, None)
    return MetricSummary(
        questions=len(results),
        answerable=len(ranks),
        hit_at={k: hit_at_k(ranks, k) for k in ks},
        mrr=mrr_at_k(ranks, max(ks)),
    )


def _similarity_summary(results: Sequence[QuestionResult]) -> SimilaritySummary:
    answerable = [r for r in results if r.answerable]
    values = [r.top1_similarity for r in answerable if r.top1_similarity is not None]
    return SimilaritySummary(
        answerable_top1_min=min(values) if values else None,
        answerable_top1_median=statistics.median(values) if values else None,
        answerable_top1_max=max(values) if values else None,
        answerable_without_results=len(answerable) - len(values),
        negative_top1=tuple((r.id, r.top1_similarity) for r in results if not r.answerable),
    )


def evaluate(
    questions: Sequence[GoldenQuestion],
    retrieve: RetrieveFn,
    ks: Sequence[int] = DEFAULT_KS,
) -> EvaluationResult:
    """Run every question through ``retrieve`` and score the ranked results.

    ``retrieve(question, max(ks))`` is called once per question, in order.
    The rank of the first result on an expected ``(doc_id, page)`` decides
    hit@k for every ``k`` and MRR@``max(ks)``; metrics cover answerable
    questions only (negatives keep their top-1 similarity in the summary).
    ``per_tag`` holds every tag in alphabetical order; a tag without
    answerable questions has ``hit_at``/``mrr`` ``None``.

    Raises:
        ValueError: ``questions`` is empty, ``ks`` is invalid, or
            ``retrieve`` returns more than ``max(ks)`` results.
    """
    if not questions:
        raise ValueError("questions must not be empty")
    sorted_ks = _check_ks(ks)
    depth = max(sorted_ks)

    results: list[QuestionResult] = []
    for question in questions:
        hits = list(retrieve(question.question, depth))
        if len(hits) > depth:
            raise ValueError(
                f"retrieve returned {len(hits)} results for {question.id}, asked for {depth}"
            )
        rank = first_relevant_rank([(h.doc_id, h.page_number) for h in hits], question.expected)
        results.append(
            QuestionResult(
                id=question.id,
                tags=question.tags,
                answerable=question.is_answerable,
                first_relevant_rank=rank,
                top=tuple(hits[:TOP_SHOWN]),
                top1_similarity=hits[0].similarity if hits else None,
            )
        )

    tags = sorted({tag for r in results for tag in r.tags})
    return EvaluationResult(
        ks=sorted_ks,
        mrr_k=depth,
        questions=tuple(results),
        overall=_summarise(results, sorted_ks),
        per_tag={tag: _summarise([r for r in results if tag in r.tags], sorted_ks) for tag in tags},
        similarity=_similarity_summary(results),
    )
