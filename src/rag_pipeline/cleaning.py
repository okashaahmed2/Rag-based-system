"""Page-level text cleaning for extracted PDF text.

The public entry points are :func:`clean_page_text`, a pure, idempotent
function that cleans the text of one page, and :func:`clean_pages`, which
cleans every page returned by :func:`rag_pipeline.extraction.extract_pdf`
while keeping page count, order and page numbers unchanged.

Cleaning only normalises representation (line endings, invisible
characters, ligatures, whitespace, hyphenated line breaks). It never changes
case, removes punctuation or drops non-ASCII text, so formulas, Roman Urdu,
Urdu script and emoji survive unchanged. Header/footer removal and chunking
happen in later steps.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

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
    "  "
    "           "
    "  　"
)

#: Invisible characters that are deleted outright. ZWNJ (U+200C) and ZWJ
#: (U+200D) are deliberately kept: ZWNJ is meaningful in Urdu/Persian script
#: and ZWJ joins emoji sequences.
INVISIBLE_CHARS: frozenset[str] = frozenset(
    "­"  # soft hyphen
    "​"  # zero-width space
    "﻿"  # byte order mark / zero-width no-break space
)

#: Characters that start a new line besides "\n" (after "\r\n" is handled):
#: CR, form feed, vertical tab, NEL (U+0085) and LINE SEPARATOR (U+2028).
LINE_BREAK_CHARS: frozenset[str] = frozenset("\r\f\v\u0085 ")

#: Characters that start a new paragraph: PARAGRAPH SEPARATOR (U+2029).
PARAGRAPH_BREAK_CHARS: frozenset[str] = frozenset(" ")

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
    text = _collapse_blank_lines(text)
    return text.strip()


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
