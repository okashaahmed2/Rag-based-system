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
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
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
