"""Page-level text cleaning for extracted PDF text.

The public entry points are :func:`clean_page_text`, a pure, idempotent
function that cleans the text of one page, and :func:`clean_pages`, which
cleans every page returned by :func:`rag_pipeline.extraction.extract_pdf`
while keeping page count, order and page numbers unchanged.

Cleaning only normalises representation (line endings, invisible
characters, ligatures, whitespace, hyphenated line breaks). It never changes
case, removes punctuation or drops non-ASCII text, so formulas, Roman Urdu,
Urdu script and emoji survive unchanged.

:func:`remove_repeated_headers_footers` is the document-level step that runs
on the output of :func:`clean_pages` and removes lines repeated at the top or
bottom of many pages. Chunking happens in a later step.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rag_pipeline.extraction import PageText

logger = logging.getLogger(__name__)

# --- Character tables ---------------------------------------------------------

#: Typographic ligatures expanded to their plain letters.
LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

#: Unicode "Space Separator" (category Zs) characters other than U+0020,
#: including the non-breaking space U+00A0. All are replaced by a plain space.
#: Hard-coded to avoid scanning the whole code space at import time; a test
#: checks it against ``unicodedata``.
SPACE_SEPARATORS: frozenset[str] = frozenset(
    "\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u202f\u205f\u3000"
)

#: Invisible characters that are deleted outright. ZWNJ (U+200C) and ZWJ
#: (U+200D) are deliberately kept: ZWNJ is meaningful in Urdu/Persian script
#: and ZWJ joins emoji sequences.
INVISIBLE_CHARS: frozenset[str] = frozenset(
    "\u00ad"  # soft hyphen
    "\u200b"  # zero-width space
    "\ufeff"  # byte order mark / zero-width no-break space
)

#: Characters that start a new line besides "\n" (after "\r\n" is handled):
#: CR, form feed, vertical tab, NEL (U+0085) and LINE SEPARATOR (U+2028).
LINE_BREAK_CHARS: frozenset[str] = frozenset("\r\f\v\u0085\u2028")

#: Characters that start a new paragraph: PARAGRAPH SEPARATOR (U+2029).
PARAGRAPH_BREAK_CHARS: frozenset[str] = frozenset("\u2029")

#: Control characters that are deleted: C0 controls (U+0000..U+001F), DEL
#: (U+007F) and C1 controls (U+0080..U+009F). "\n" and "\t" are kept and the
#: line-break controls above are turned into "\n" instead.
REMOVED_CONTROL_CHARS: frozenset[str] = frozenset(
    chr(code)
    for code in (*range(0x20), 0x7F, *range(0x80, 0xA0))
    if chr(code) not in "\n\t" and chr(code) not in LINE_BREAK_CHARS
)


def _build_translation_table() -> dict[int, str | None]:
    """Build the ``str.translate`` table for all single-character rules."""
    table: dict[int, str | None] = {}
    table.update({ord(char): "\n" for char in LINE_BREAK_CHARS})
    table.update({ord(char): "\n\n" for char in PARAGRAPH_BREAK_CHARS})
    table.update({ord(char): None for char in REMOVED_CONTROL_CHARS})
    table.update({ord(char): None for char in INVISIBLE_CHARS})
    table.update({ord(char): " " for char in SPACE_SEPARATORS})
    table.update({ord(char): plain for char, plain in LIGATURES.items()})
    return table


_TRANSLATION_TABLE = _build_translation_table()

# --- Regular expressions ------------------------------------------------------

#: Runs of spaces and tabs (after Zs characters became plain spaces).
_HORIZONTAL_WS_RUN = re.compile(r"[ \t]+")

#: A newline together with the (already collapsed) spaces around it.
_SPACES_AROUND_NEWLINE = re.compile(r" *\n *")

#: "-" + single newline, with the character before and after captured inside
#: zero-width lookarounds so that consecutive breaks ("a-\nb-\nc") are all
#: found in one pass. Whether to join is decided in :func:`_join_or_keep`
#: with ``str.islower`` because ``re`` has no Unicode lowercase class.
_HYPHENATED_BREAK = re.compile(r"(?<=(.))-\n(?=(.))")

#: Three or more newlines (lines are already stripped, so no spaces between).
_EXCESS_NEWLINES = re.compile(r"\n{3,}")


# --- Helpers ------------------------------------------------------------------


def _normalize_line_endings(text: str) -> str:
    """Turn "\\r\\n" into "\\n"; other line breaks go via the table."""
    return text.replace("\r\n", "\n")


def _translate_characters(text: str) -> str:
    """Apply line/paragraph-break, control, invisible, space and ligature rules."""
    return text.translate(_TRANSLATION_TABLE)


def _normalize_unicode(text: str) -> str:
    """Apply NFC (canonical) normalisation only; NFKC would turn "x²" into "x2"."""
    return unicodedata.normalize("NFC", text)


def _normalize_horizontal_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs to one space and strip them around newlines."""
    text = _HORIZONTAL_WS_RUN.sub(" ", text)
    return _SPACES_AROUND_NEWLINE.sub("\n", text)


def _join_or_keep(match: re.Match[str]) -> str:
    """Remove "-\\n" only between two lowercase letters; otherwise keep it."""
    before, after = match.group(1), match.group(2)
    if before.islower() and after.islower():
        return ""
    return match.group(0)


def _join_hyphenated_line_breaks(text: str) -> str:
    """Join words split across lines, e.g. "informa-\\ntion" -> "information"."""
    return _HYPHENATED_BREAK.sub(_join_or_keep, text)


def _collapse_blank_lines(text: str) -> str:
    """Reduce three or more consecutive newlines to one paragraph break."""
    return _EXCESS_NEWLINES.sub("\n\n", text)


def _collapse_and_strip(text: str) -> str:
    """Final step shared by page and document cleaning: collapse, then strip."""
    return _collapse_blank_lines(text).strip()


# --- Public API ---------------------------------------------------------------


def clean_page_text(text: str) -> str:
    """Return a cleaned copy of one page's text.

    Pure, deterministic and idempotent:
    ``clean_page_text(clean_page_text(x)) == clean_page_text(x)``.

    Rules, applied in this order:

    1. Line endings: "\\r\\n" becomes "\\n".
    2. Character rules (one ``str.translate`` pass):
       - lone "\\r", form feed, vertical tab, NEL (U+0085) and LINE
         SEPARATOR (U+2028) become "\\n"; PARAGRAPH SEPARATOR (U+2029)
         becomes "\\n\\n";
       - ligatures are expanded (U+FB00..U+FB06);
       - NBSP and every other Unicode space separator (Zs) becomes " ";
       - soft hyphen, zero-width space, BOM, C0 controls other than "\\n"
         and "\\t", DEL (U+007F) and C1 controls (U+0080..U+009F) are
         removed. ZWNJ (U+200C) and ZWJ (U+200D) are kept.
    3. Unicode NFC normalisation (never NFKC/NFKD). It runs *after* the
       removals because deleting an invisible character can bring a base
       letter next to a combining mark (``"e\\u200b\\u0301"``); composing
       afterwards keeps the function idempotent.
    4. Horizontal whitespace: runs of spaces/tabs become one space; spaces
       at the start and end of every line are removed.
    5. Hyphenated line breaks: "-\\n" is removed only when the character
       before "-" and the first character of the next line are both
       lowercase letters (Unicode-aware). "COVID-\\n19", "Data-\\nScience"
       and "- item" bullets are left unchanged.
    6. Three or more consecutive newlines become exactly two. Single and
       double newlines are kept.
    7. Leading/trailing whitespace of the whole page is stripped, so
       whitespace-only input returns "".

    Case, punctuation and non-ASCII text are never changed.

    Known trade-offs:

    - Leading indentation and internal runs of spaces are collapsed, so
      ASCII-art diagrams lose their alignment. The text itself is kept, and
      the raw text remains available from extraction.
    - Compound words broken at their own hyphen with lowercase letters on
      both sides are joined (e.g. "self-\\nesteem" -> "selfesteem").

    Raises:
        TypeError: if ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    text = _normalize_line_endings(text)
    text = _translate_characters(text)
    text = _normalize_unicode(text)
    text = _normalize_horizontal_whitespace(text)
    text = _join_hyphenated_line_breaks(text)
    return _collapse_and_strip(text)


@dataclass(frozen=True)
class CleanedPage:
    """Cleaned text of a single page.

    Attributes:
        page_number: 1-based page number, copied unchanged from extraction.
        text: Cleaned text (may be empty).
    """

    page_number: int
    text: str

    @property
    def has_text(self) -> bool:
        """``False`` if the page is empty after cleaning."""
        return bool(self.text)


def clean_pages(pages: Sequence[PageText]) -> list[CleanedPage]:
    """Clean every page, keeping count, order and page numbers.

    Pages that become empty are kept with empty text. The input is not
    modified (``PageText`` is frozen; a new list is returned). Logs one
    summary line; page text is never logged.

    Raises:
        TypeError: if a page's text is not a ``str``.
    """
    cleaned = [
        CleanedPage(page_number=page.page_number, text=clean_page_text(page.text))
        for page in pages
    ]
    logger.info(
        "Cleaned %d page(s): %d -> %d characters, %d page(s) empty after cleaning",
        len(cleaned),
        sum(len(page.text) for page in pages),
        sum(len(page.text) for page in cleaned),
        sum(1 for page in cleaned if not page.has_text),
    )
    return cleaned


# --- Document-level: repeated headers and footers -----------------------------

#: Number of non-empty lines at the top and at the bottom of a page that are
#: candidates for running headers / footers.
EDGE_LINES = 2

#: A key must appear on at least this many pages to count as repeated.
MIN_REPEAT_PAGES = 3

#: ... and on at least this fraction of the pages that have text.
MIN_REPEAT_RATIO = 0.5

_DIGIT_RUN = re.compile(r"\d+")
_WHITESPACE_RUN = re.compile(r"\s+")

#: Anchored pattern for a page-number-like key (digit runs already replaced
#: by "#"). The key must contain at least one "#" and consist only of "#",
#: the tokens "page", "pg", "p.", "slide", "of", the separators "/", "-",
#: en dash U+2013, em dash U+2014, "|", "." and spaces. Matches e.g. "#",
#: "page #", "page # of #", "# / #", "#/#", "- # -", "slide #", "p. #",
#: "pg. #"; rejects "step #", "example #", "chapter #", "unit #".
PAGE_NUMBER_KEY_PATTERN = re.compile(
    r"""
    ^
    (?=.*\#)                          # at least one number
    (?:\#|page|pg|p\.|slide|of|[/|.\-\u2013\u2014\ ])+
    $
    """,
    re.VERBOSE,
)


class KeyKind(StrEnum):
    """How an edge line is compared across pages."""

    #: Lowercase, whitespace collapsed, digits kept ("Unit 3" -> "unit 3").
    EXACT = "exact"
    #: Digit runs replaced by "#"; only for page-number-like lines
    #: ("Page 3 of 36" -> "page # of #").
    PAGE_NUMBER = "page_number"


@dataclass(frozen=True)
class RepeatedKey:
    """A normalised edge line found repeated across pages.

    Attributes:
        key: Comparison key of the given ``kind``.
        kind: ``EXACT`` for constant lines, ``PAGE_NUMBER`` for page-number
            lines whose numbers vary.
        page_count: Number of text pages whose edge region contains the key.
    """

    key: str
    kind: KeyKind
    page_count: int


@dataclass(frozen=True)
class HeaderFooterResult:
    """Result of :func:`remove_repeated_headers_footers`.

    Attributes:
        pages: Cleaned pages, same count, order and page numbers as the input.
        repeated_top: Keys detected as running headers, most frequent first.
        repeated_bottom: Keys detected as running footers, most frequent first.
        lines_removed: Total number of lines removed across all pages.
    """

    pages: tuple[CleanedPage, ...]
    repeated_top: tuple[RepeatedKey, ...]
    repeated_bottom: tuple[RepeatedKey, ...]
    lines_removed: int


_LineKey = tuple[KeyKind, str]


def _exact_key(line: str) -> str:
    """Return the exact key: lowercase, whitespace collapsed, digits kept."""
    return _WHITESPACE_RUN.sub(" ", line.lower()).strip()


def _line_key(line: str) -> _LineKey:
    """Return the single comparison key of a line and its kind.

    Page-number-like lines get a ``PAGE_NUMBER`` key ("Page 3 of 36" ->
    "page # of #"); every other line gets an ``EXACT`` key, so lines whose
    numbers vary ("Step 1", "Step 2") never share a key.
    """
    exact = _exact_key(line)
    numbered = _DIGIT_RUN.sub("#", exact)
    if PAGE_NUMBER_KEY_PATTERN.match(numbered):
        return KeyKind.PAGE_NUMBER, numbered
    return KeyKind.EXACT, exact


def _edge_indices(lines: Sequence[str]) -> tuple[list[int], list[int]]:
    """Return indices of the first and last ``EDGE_LINES`` non-empty lines.

    On a page with few lines the two lists may overlap.
    """
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    return non_empty[:EDGE_LINES], non_empty[-EDGE_LINES:]


def _count_edge_keys(
    pages: Sequence[CleanedPage],
) -> tuple[dict[_LineKey, int], dict[_LineKey, int]]:
    """Count, per region, on how many pages each key appears (once per page)."""
    top: dict[_LineKey, int] = {}
    bottom: dict[_LineKey, int] = {}
    for page in pages:
        lines = page.text.split("\n")
        top_idx, bottom_idx = _edge_indices(lines)
        for counts, indices in ((top, top_idx), (bottom, bottom_idx)):
            for key in {_line_key(lines[index]) for index in indices}:
                counts[key] = counts.get(key, 0) + 1
    return top, bottom


def _repeated_keys(
    counts: dict[_LineKey, int], text_pages: int
) -> tuple[RepeatedKey, ...]:
    """Keep keys meeting both the absolute and the ratio threshold."""
    threshold = max(MIN_REPEAT_PAGES, MIN_REPEAT_RATIO * text_pages)
    repeated = [
        RepeatedKey(key=key, kind=kind, page_count=count)
        for (kind, key), count in counts.items()
        if count >= threshold
    ]
    return tuple(
        sorted(repeated, key=lambda item: (-item.page_count, item.kind, item.key))
    )


def _key_set(repeated: tuple[RepeatedKey, ...]) -> frozenset[_LineKey]:
    """Return the (kind, key) pairs of repeated keys for fast lookup."""
    return frozenset((item.kind, item.key) for item in repeated)


def _strip_edge_lines(
    page: CleanedPage, top_keys: frozenset[_LineKey], bottom_keys: frozenset[_LineKey]
) -> tuple[CleanedPage, int]:
    """Remove repeated keys from the page's matching edge region only."""
    lines = page.text.split("\n")
    top_idx, bottom_idx = _edge_indices(lines)
    drop = {i for i in top_idx if _line_key(lines[i]) in top_keys}
    drop |= {i for i in bottom_idx if _line_key(lines[i]) in bottom_keys}
    if not drop:
        return page, 0
    kept = "\n".join(line for index, line in enumerate(lines) if index not in drop)
    return CleanedPage(page.page_number, _collapse_and_strip(kept)), len(drop)


def remove_repeated_headers_footers(
    pages: Sequence[CleanedPage],
) -> HeaderFooterResult:
    """Remove running headers, footers and page-number lines across pages.

    Algorithm:

    1. The edge region of a page is its first ``EDGE_LINES`` (top) and last
       ``EDGE_LINES`` (bottom) non-empty lines; on short pages a line can be
       in both.
    2. Each edge line gets exactly one comparison key (see :class:`KeyKind`):
       - ``PAGE_NUMBER`` if, after lowercasing, collapsing whitespace and
         replacing every digit run by "#", the result matches
         :data:`PAGE_NUMBER_KEY_PATTERN` ("Page 3 of 36" and "Page 17 of 36"
         -> "page # of #"; "5" -> "#"; "- 3 -" -> "- # -");
       - ``EXACT`` otherwise: lowercase, whitespace collapsed, digits kept
         ("Unit 3" -> "unit 3"). Lines whose numbers vary but are not
         page-number-like ("Step 1", "Step 2") therefore never share a key.
    3. For TOP and BOTTOM separately, count on how many text pages each
       (kind, key) pair appears (at most once per page).
    4. A key is repeated if its count is ``>= MIN_REPEAT_PAGES`` and
       ``>= MIN_REPEAT_RATIO`` x the number of pages that have text. Empty
       pages do not count.
    5. Lines with a repeated key are removed only from the matching edge
       region; the same text in the middle of a page is kept.
    6. If fewer than ``MIN_REPEAT_PAGES`` pages have text, nothing changes.
    7. Changed pages get the same final newline collapse and strip as
       :func:`clean_page_text`.

    Pure: no I/O, the input is not modified, and page count, order and page
    numbers are preserved. Pages that become empty are kept.

    Not idempotent; designed to run once per document. After the first pass
    the next lines move into the edge regions (e.g. a chapter title under a
    removed header) and the ratio is recomputed over fewer text pages, so a
    second pass may remove more.

    Known trade-offs:

    - Only page-number-like lines are compared with digits normalised, so
      running headers that vary by number but are not page numbers
      ("Chapter 1" / "Chapter 2", "Lecture 3" / "Lecture 4") are kept.
    - A section number alone on an edge line ("3.2") matches the
      page-number pattern ("#.#") and is removed if it is repeated in that
      form on enough pages.
    - Content that really starts or ends most pages with the same line
      (e.g. a fixed heading on every slide) is removed as a header/footer.
    """
    text_pages = sum(1 for page in pages if page.has_text)
    if text_pages < MIN_REPEAT_PAGES:
        result = HeaderFooterResult(tuple(pages), (), (), 0)
    else:
        top_counts, bottom_counts = _count_edge_keys(pages)
        repeated_top = _repeated_keys(top_counts, text_pages)
        repeated_bottom = _repeated_keys(bottom_counts, text_pages)
        top_keys = _key_set(repeated_top)
        bottom_keys = _key_set(repeated_bottom)
        stripped = [_strip_edge_lines(page, top_keys, bottom_keys) for page in pages]
        result = HeaderFooterResult(
            pages=tuple(page for page, _ in stripped),
            repeated_top=repeated_top,
            repeated_bottom=repeated_bottom,
            lines_removed=sum(count for _, count in stripped),
        )
    logger.info(
        "Header/footer removal on %d page(s) (%d with text): %d line(s) removed, "
        "%d repeated top key(s), %d repeated bottom key(s)",
        len(result.pages),
        text_pages,
        result.lines_removed,
        len(result.repeated_top),
        len(result.repeated_bottom),
    )
    return result
