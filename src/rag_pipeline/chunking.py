"""Page-aware chunking of cleaned page text.

The public entry point is :func:`chunk_pages`, which splits the pages
produced by :mod:`rag_pipeline.cleaning` into :class:`TextChunk` objects.
Chunks never cross a page boundary, never exceed ``chunk_size`` characters,
end on the most natural boundary available (paragraph, line, sentence, word)
and overlap their predecessor on the same page by at most ``chunk_overlap``
characters.

Overlap rule: the next chunk on a page starts inside the overlap window, the
last ``chunk_overlap`` characters of the previous chunk, at the earliest
sentence start in the window (right after ".", "?" or "!" plus whitespace,
or after a line break); if there is none, at the earliest word start (right
after whitespace); if there is none either (e.g. inside a hard-cut run), the
next chunk starts at the next unit without overlap. A start is only
accepted if the next chunk starts after the previous chunk's start and ends
after its end, which guarantees progress.

Document metadata (``doc_id``, ``filename``) is attached in a later step.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from rag_pipeline.cleaning import CleanedPage

logger = logging.getLogger(__name__)

#: Separators tried in order of preference when a unit is longer than the
#: chunk size. Each pattern matches the separator itself; the text between
#: matches becomes the sub-units (whitespace around them is excluded).
#: Sentence ends are ".", "?" or "!" followed by whitespace; the punctuation
#: stays with the sentence because only the whitespace is matched.
SPLIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\s*\n"),  # paragraphs
    re.compile(r"\n"),  # lines
    re.compile(r"(?<=[.?!])\s+"),  # sentences
    re.compile(r"\s+"),  # words
)

_Span = tuple[int, int]


@dataclass(frozen=True)
class TextChunk:
    """A contiguous piece of one page's cleaned text.

    Attributes:
        chunk_index: Sequential index 0..N-1 across the whole document, in
            input page order.
        page_number: Page the chunk comes from (copied from the page).
        text: Exactly ``page_text[char_start:char_end]``.
        char_start: Start offset into the page's cleaned text (inclusive).
        char_end: End offset into the page's cleaned text (exclusive).
    """

    chunk_index: int
    page_number: int
    text: str
    char_start: int
    char_end: int


def _require_int(name: str, value: object) -> None:
    """Raise ``TypeError`` unless ``value`` is an ``int`` (``bool`` rejected)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


def _validate_sizes(chunk_size: int, chunk_overlap: int) -> None:
    """Raise if the sizes are not ints with ``1 <= size`` and ``0 <= overlap < size``."""
    _require_int("chunk_size", chunk_size)
    _require_int("chunk_overlap", chunk_overlap)
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), "
            f"got {chunk_overlap}"
        )


def _strip_span(text: str, start: int, end: int) -> _Span | None:
    """Shrink ``[start, end)`` to exclude surrounding whitespace; None if empty."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split_by_pattern(
    text: str, start: int, end: int, pattern: re.Pattern[str]
) -> list[_Span]:
    """Split ``[start, end)`` at every match of ``pattern``; drop empty parts."""
    parts: list[_Span] = []
    cursor = start
    for match in pattern.finditer(text, start, end):
        span = _strip_span(text, cursor, match.start())
        if span is not None:
            parts.append(span)
        cursor = match.end()
    span = _strip_span(text, cursor, end)
    if span is not None:
        parts.append(span)
    return parts


def _hard_cut(start: int, end: int, chunk_size: int) -> list[_Span]:
    """Cut ``[start, end)`` into consecutive pieces of at most ``chunk_size``."""
    return [(pos, min(pos + chunk_size, end)) for pos in range(start, end, chunk_size)]


def _split_units(
    text: str, start: int, end: int, chunk_size: int, level: int = 0
) -> list[_Span]:
    """Recursively split a stripped span into units of at most ``chunk_size``.

    Only a unit longer than ``chunk_size`` is split further, first by the
    separator at ``level``, falling through to the next separator when the
    current one does not occur, and finally by a hard cut.
    """
    if end - start <= chunk_size:
        return [(start, end)]
    for next_level in range(level, len(SPLIT_PATTERNS)):
        parts = _split_by_pattern(text, start, end, SPLIT_PATTERNS[next_level])
        if len(parts) > 1:
            units: list[_Span] = []
            for part_start, part_end in parts:
                units.extend(
                    _split_units(text, part_start, part_end, chunk_size, next_level + 1)
                )
            return units
    return _hard_cut(start, end, chunk_size)


def _is_word_start(text: str, pos: int) -> bool:
    """``True`` if ``pos`` holds a non-whitespace character after whitespace."""
    return pos > 0 and not text[pos].isspace() and text[pos - 1].isspace()


def _is_sentence_start(text: str, pos: int) -> bool:
    """``True`` for a word start after a line break or after ".", "?" or "!".

    The whitespace run before ``pos`` must contain a newline, or the last
    non-whitespace character before it must be sentence-ending punctuation.
    """
    if not _is_word_start(text, pos):
        return False
    before = pos - 1
    while before >= 0 and text[before].isspace():
        if text[before] == "\n":
            return True
        before -= 1
    return before >= 0 and text[before] in ".?!"


def _overlap_start(
    text: str,
    chunk_start: int,
    chunk_end: int,
    next_unit_end: int,
    chunk_size: int,
    chunk_overlap: int,
) -> int | None:
    """Return where the next chunk starts inside the overlap window, or None.

    Candidates lie in the last ``chunk_overlap`` characters of the previous
    chunk ``[chunk_start, chunk_end)``. A candidate must be after
    ``chunk_start`` and close enough to ``next_unit_end`` that the next
    chunk still includes the next unit within ``chunk_size``, so it ends
    after ``chunk_end``. The earliest valid sentence start wins, then the
    earliest valid word start; ``None`` means no overlap.
    """
    lowest = max(chunk_start + 1, chunk_end - chunk_overlap, next_unit_end - chunk_size)
    window = range(lowest, chunk_end)
    for is_start in (_is_sentence_start, _is_word_start):
        for pos in window:
            if is_start(text, pos):
                return pos
    return None


def _pack_units(
    text: str, units: Sequence[_Span], chunk_size: int, chunk_overlap: int
) -> list[_Span]:
    """Greedily pack units into chunk spans; overlap starts may split a unit.

    A chunk starting at ``start`` inside unit ``first`` begins with the
    partial unit ``[start, unit_end)``. That partial unit is never longer
    than the whole unit, which is at most ``chunk_size``, so it never needs
    further splitting.
    """
    unit_starts = [unit_start for unit_start, _ in units]
    spans: list[_Span] = []
    first, start = 0, units[0][0]
    while True:
        last = first
        while last + 1 < len(units) and units[last + 1][1] - start <= chunk_size:
            last += 1
        end = units[last][1]
        spans.append((start, end))
        if last + 1 >= len(units):
            return spans
        overlap = _overlap_start(
            text, start, end, units[last + 1][1], chunk_size, chunk_overlap
        )
        if overlap is None:
            first, start = last + 1, units[last + 1][0]
        else:
            first, start = bisect_right(unit_starts, overlap) - 1, overlap


def _chunk_page_spans(text: str, chunk_size: int, chunk_overlap: int) -> list[_Span]:
    """Return the chunk spans of one page's text (empty list if no text)."""
    page_span = _strip_span(text, 0, len(text))
    if page_span is None:
        return []
    units = _split_units(text, page_span[0], page_span[1], chunk_size)
    return _pack_units(text, units, chunk_size, chunk_overlap)


def chunk_pages(
    pages: Sequence[CleanedPage], chunk_size: int, chunk_overlap: int
) -> list[TextChunk]:
    """Split cleaned pages into page-aware, size-bounded, overlapping chunks.

    Algorithm, applied to every page independently:

    1. The page text is split into units, recursively by preference:
       paragraphs ("\\n\\n") -> lines ("\\n") -> sentences (".", "?" or
       "!" followed by whitespace) -> words (whitespace) -> hard cut every
       ``chunk_size`` characters. Only a unit longer than ``chunk_size`` is
       split further. Units are ``(start, end)`` spans into the page text
       that exclude surrounding whitespace, so no non-whitespace character
       is ever dropped.
    2. Consecutive units are packed greedily while the span from the first
       unit's start to the last unit's end is ``<= chunk_size``.
    3. Chunk text is ``page_text[start:end]`` exactly; separators inside a
       chunk are kept.
    4. Overlap: the next chunk on the same page starts inside the last
       ``chunk_overlap`` characters of the previous chunk, at the earliest
       sentence start there (after ".?!" + whitespace, or after a line
       break), else the earliest word start (after whitespace), else
       nowhere (no overlap, e.g. inside a hard-cut run). The chosen start
       must lie after the previous chunk's start and let the next chunk end
       after the previous chunk's end; otherwise a later word start in the
       window, or no overlap, is used. Both chunk starts and chunk ends
       therefore strictly increase, so the loop always terminates. The
       partial unit at the start of a chunk is shorter than its unit, so it
       never exceeds ``chunk_size``.
    5. Pages without text produce no chunks; ``chunk_index`` stays
       sequential across them.

    Pure and deterministic: no settings are read, no I/O is done and the
    input is not modified. Logs one INFO summary line; text is never logged.

    Known trade-offs:

    - Sentence splitting is punctuation-based, so abbreviations ("e.g.",
      "Fig. 3") may end a unit early. No text is lost or duplicated by this;
      it only moves a possible chunk boundary or overlap start.
    - A hard cut counts code points, so it may separate a base letter from a
      combining mark or split an emoji ZWJ sequence. It only happens for a
      whitespace-free run longer than ``chunk_size``.

    Raises:
        TypeError: if ``chunk_size`` or ``chunk_overlap`` is not an ``int``.
        ValueError: if ``chunk_size < 1`` or not ``0 <= chunk_overlap <
            chunk_size``.
    """
    _validate_sizes(chunk_size, chunk_overlap)
    chunks: list[TextChunk] = []
    for page in pages:
        for start, end in _chunk_page_spans(page.text, chunk_size, chunk_overlap):
            chunks.append(
                TextChunk(
                    chunk_index=len(chunks),
                    page_number=page.page_number,
                    text=page.text[start:end],
                    char_start=start,
                    char_end=end,
                )
            )
    lengths = [len(chunk.text) for chunk in chunks]
    logger.info(
        "Chunked %d page(s) into %d chunk(s); chunk length min/avg/max = %d/%.1f/%d",
        len(pages),
        len(chunks),
        min(lengths, default=0),
        sum(lengths) / len(lengths) if lengths else 0.0,
        max(lengths, default=0),
    )
    return chunks
