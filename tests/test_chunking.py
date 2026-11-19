"""Tests for rag_pipeline.chunking."""

from __future__ import annotations

import copy
import logging

import pytest

from rag_pipeline.chunking import TextChunk, chunk_pages
from rag_pipeline.cleaning import CleanedPage

# --- Helpers ------------------------------------------------------------------


def _pages(*texts: str) -> list[CleanedPage]:
    return [CleanedPage(page_number=i + 1, text=t) for i, t in enumerate(texts)]


def _sentence(i: int, words: int = 8) -> str:
    return " ".join(f"w{i}x{j}" for j in range(words)) + "."


def _paragraph(start: int, sentences: int) -> str:
    return " ".join(_sentence(start + k) for k in range(sentences))


def _assert_invariants(
    pages: list[CleanedPage], chunks: list[TextChunk], size: int
) -> None:
    """Check every structural invariant of a chunking result."""
    texts = {page.page_number: page.text for page in pages}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    page_order = [page.page_number for page in pages]
    assert [page_order.index(c.page_number) for c in chunks] == sorted(
        page_order.index(c.page_number) for c in chunks
    )
    for chunk in chunks:
        assert 0 < len(chunk.text) <= size
        page_text = texts[chunk.page_number]
        assert 0 <= chunk.char_start < chunk.char_end <= len(page_text)
        assert chunk.text == page_text[chunk.char_start : chunk.char_end]
        assert chunk.text == chunk.text.strip()
    for page in pages:
        covered = [False] * len(page.text)
        page_chunks = [c for c in chunks if c.page_number == page.page_number]
        for chunk in page_chunks:
            for pos in range(chunk.char_start, chunk.char_end):
                covered[pos] = True
        missing = [
            pos for pos, ch in enumerate(page.text) if not ch.isspace() and not covered[pos]
        ]
        assert missing == []
        starts = [c.char_start for c in page_chunks]
        ends = [c.char_end for c in page_chunks]
        assert starts == sorted(starts) and len(set(starts)) == len(starts)
        assert ends == sorted(ends) and len(set(ends)) == len(ends)


def _overlaps(chunks: list[TextChunk]) -> list[int]:
    """``prev.char_end - next.char_start`` for consecutive same-page chunks.

    Positive means overlap; zero or negative means no overlap (a negative
    value is the whitespace gap between the chunks).
    """
    return [
        a.char_end - b.char_start
        for a, b in zip(chunks, chunks[1:])
        if a.page_number == b.page_number
    ]


# --- Varied inputs for the invariants -----------------------------------------

LONG_URL = "https://example.com/" + "a1b2c3" * 60
VARIED_INPUTS: dict[str, list[str]] = {
    "single_short": ["Hello world."],
    "paragraphs": ["\n\n".join(_paragraph(i * 10, 6) for i in range(8))],
    "lines": ["\n".join(_sentence(i, 12) for i in range(60))],
    "one_long_sentence": [" ".join(f"word{i}" for i in range(500))],
    "long_url": [f"See {LONG_URL} for details.\n\nNext paragraph here."],
    "mixed_pages": [
        _paragraph(0, 30),
        "",
        "Short page.",
        "Title\n\n" + "\n".join(_sentence(i, 20) for i in range(30)),
    ],
    "unicode": [
        "Yeh ek misaal hai. " * 40,
        "یہ ایک مثال ہے۔ " * 60,
        "Emoji test \U0001f600\U0001f44d\U0001f3fd party! " * 50,
    ],
    "abbreviations": ["See e.g. Fig. 3 and Dr. Smith et al. for more. " * 40],
    "whitespace_edges": ["  \n  padded text inside   \n\n  more  \n"],
}


@pytest.mark.parametrize("name", sorted(VARIED_INPUTS))
@pytest.mark.parametrize(("size", "overlap"), [(800, 120), (100, 0), (150, 60), (37, 36)])
def test_invariants_on_varied_inputs(name: str, size: int, overlap: int) -> None:
    pages = _pages(*VARIED_INPUTS[name])
    before = copy.deepcopy(pages)

    chunks = chunk_pages(pages, size, overlap)

    _assert_invariants(pages, chunks, size)
    assert all(o <= overlap for o in _overlaps(chunks))
    assert pages == before
    assert chunk_pages(pages, size, overlap) == chunks


def test_returns_text_chunk_objects_that_are_frozen() -> None:
    chunk = chunk_pages(_pages("Some text."), 800, 120)[0]

    assert isinstance(chunk, TextChunk)
    with pytest.raises(AttributeError):
        chunk.text = "changed"  # type: ignore[misc]


# --- Behaviour ----------------------------------------------------------------


def test_short_page_is_exactly_one_chunk() -> None:
    text = "Line one.\nLine two.\n\nSecond paragraph."

    chunks = chunk_pages(_pages(text), 800, 120)

    assert chunks == [TextChunk(0, 1, text, 0, len(text))]


def test_page_of_exactly_chunk_size_is_one_chunk() -> None:
    text = "x" * 50 + " " + "y" * 49

    chunks = chunk_pages(_pages(text), 100, 10)

    assert [c.text for c in chunks] == [text]


def test_split_prefers_paragraph_breaks_over_lines() -> None:
    para_a = "\n".join(_sentence(i, 6) for i in range(8))
    para_b = "\n".join(_sentence(i + 20, 6) for i in range(8))
    assert len(para_a) < 400 < len(para_a) + len(para_b)

    chunks = chunk_pages(_pages(para_a + "\n\n" + para_b), 400, 0)

    assert [c.text for c in chunks] == [para_a, para_b]


def test_split_prefers_line_breaks_over_sentences() -> None:
    line_a = " ".join(_sentence(i, 5) for i in range(6))
    line_b = " ".join(_sentence(i + 20, 5) for i in range(6))
    assert len(line_a) < 300 < len(line_a) + len(line_b)

    chunks = chunk_pages(_pages(line_a + "\n" + line_b), 300, 0)

    assert [c.text for c in chunks] == [line_a, line_b]


def test_split_prefers_sentences_over_words() -> None:
    sentences = [_sentence(i, 10) for i in range(6)]
    text = " ".join(sentences)
    size = len(" ".join(sentences[:2])) + 20

    chunks = chunk_pages(_pages(text), size, 0)

    assert all(c.text.endswith(".") for c in chunks)
    assert " ".join(c.text for c in chunks) == text


def test_long_sentence_is_split_at_word_boundaries() -> None:
    words = [f"token{i}" for i in range(200)]
    text = " ".join(words)

    chunks = chunk_pages(_pages(text), 100, 0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert all(part in words for part in chunk.text.split(" "))
    assert " ".join(c.text for c in chunks) == text


def test_long_word_is_hard_cut() -> None:
    chunks = chunk_pages(_pages(LONG_URL), 100, 0)

    assert [len(c.text) for c in chunks[:-1]] == [100] * (len(chunks) - 1)
    assert "".join(c.text for c in chunks) == LONG_URL


def test_long_word_between_words_keeps_neighbours_whole() -> None:
    text = f"before {LONG_URL} after"

    chunks = chunk_pages(_pages(text), 100, 0)

    _assert_invariants(_pages(text), chunks, 100)
    assert chunks[0].text.startswith("before")
    assert chunks[-1].text.endswith("after")


def test_consecutive_chunks_overlap_by_at_most_chunk_overlap() -> None:
    text = " ".join(_sentence(i, 6) for i in range(80))

    chunks = chunk_pages(_pages(text), 300, 80)
    overlaps = _overlaps(chunks)

    assert overlaps and all(0 < o <= 80 for o in overlaps)
    for prev, nxt in zip(chunks, chunks[1:]):
        shared = text[nxt.char_start : prev.char_end]
        assert prev.text.endswith(shared) and nxt.text.startswith(shared)


def test_overlap_is_whole_units_not_mid_word() -> None:
    text = " ".join(f"word{i}" for i in range(300))

    chunks = chunk_pages(_pages(text), 120, 30)

    for chunk in chunks:
        assert chunk.char_start == 0 or text[chunk.char_start - 1] == " "
        assert chunk.char_end == len(text) or text[chunk.char_end] == " "


def test_zero_overlap_means_no_overlap() -> None:
    text = " ".join(_sentence(i, 6) for i in range(80))

    chunks = chunk_pages(_pages(text), 300, 0)

    assert len(chunks) > 1
    assert all(o <= 0 for o in _overlaps(chunks))
    assert " ".join(c.text for c in chunks) == text


def test_overlap_larger_than_any_unit_still_progresses() -> None:
    text = " ".join(f"w{i}" for i in range(500))

    chunks = chunk_pages(_pages(text), 10, 9)

    _assert_invariants(_pages(text), chunks, 10)
    assert len(chunks) < 1000


def _prose_sentence(i: int) -> str:
    """A varied sentence of roughly 60-110 characters."""
    words = ["data", "model", "vector", "margin", "kernel", "cluster", "point"]
    count = 9 + (i * 7) % 8
    return (
        f"Sentence {i} "
        + " ".join(words[(i + j) % len(words)] for j in range(count))
        + ("." if i % 3 else "?")
    )


def _prose_page() -> str:
    """Paragraphs of 300-700 characters, several sentences each."""
    paragraphs: list[str] = []
    index = 0
    for target in (320, 560, 410, 585, 380, 500, 450, 340):
        sentences: list[str] = []
        while len(" ".join(sentences)) < target:
            sentences.append(_prose_sentence(index))
            index += 1
        paragraphs.append(" ".join(sentences))
    assert all(300 <= len(p) <= 700 for p in paragraphs)
    return "\n\n".join(paragraphs)


def test_prose_paragraphs_always_overlap() -> None:
    text = _prose_page()
    pages = _pages(text)

    chunks = chunk_pages(pages, 800, 120)
    overlaps = _overlaps(chunks)

    _assert_invariants(pages, chunks, 800)
    assert len(overlaps) >= 3
    assert all(0 < o <= 120 for o in overlaps)


def test_overlap_starts_at_sentence_start_when_available() -> None:
    sentences = [_prose_sentence(i) for i in range(40)]
    text = " ".join(sentences)

    chunks = chunk_pages(_pages(text), 400, 150)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert 0 < prev.char_end - nxt.char_start <= 150
        assert text[nxt.char_start - 2] in ".?!" and text[nxt.char_start - 1] == " "


def test_overlap_starts_after_line_break_as_sentence_start() -> None:
    lines = [f"item {i} without end punctuation here" for i in range(40)]
    text = "\n".join(lines)

    chunks = chunk_pages(_pages(text), 200, 60)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert 0 < prev.char_end - nxt.char_start <= 60
        assert text[nxt.char_start - 1] == "\n"


def test_overlap_starts_at_word_start_without_sentence_start() -> None:
    text = " ".join(f"word{i}" for i in range(400))

    chunks = chunk_pages(_pages(text), 200, 50)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        overlap = prev.char_end - nxt.char_start
        assert 0 < overlap <= 50
        assert text[nxt.char_start - 1] == " "
        # earliest word start in the window: no word start before it inside
        window_start = prev.char_end - 50
        earlier = text[window_start : nxt.char_start - 1]
        assert " " not in earlier


def test_sentence_start_preferred_over_earlier_word_start() -> None:
    first = " ".join(f"f{i:02d}" for i in range(37)) + "."  # 148 chars
    second = "Next " + " ".join(f"g{i:03d}" for i in range(60))
    text = f"{first} {second}"
    sentence_pos = len(first) + 1

    chunks = chunk_pages(_pages(text), 200, 60)

    prev, nxt = chunks[0], chunks[1]
    window_start = prev.char_end - 60
    assert window_start < sentence_pos
    assert " " in text[window_start:sentence_pos - 1]  # earlier word start exists
    assert nxt.char_start == sentence_pos


def test_long_paragraph_gives_overlapping_word_aligned_chunks() -> None:
    text = " ".join(f"term{i}" for i in range(1000))
    assert len(text) > 5 * 800

    chunks = chunk_pages(_pages(text), 800, 120)

    _assert_invariants(_pages(text), chunks, 800)
    assert len(chunks) > 5
    assert all(0 < o <= 120 for o in _overlaps(chunks))
    for chunk in chunks:
        assert chunk.char_start == 0 or text[chunk.char_start - 1] == " "


@pytest.mark.parametrize(("size", "overlap"), [(100, 99), (100, 50), (800, 120)])
def test_text_without_spaces_terminates_and_keeps_invariants(
    size: int, overlap: int
) -> None:
    text = "x" * 5000

    chunks = chunk_pages(_pages(text), size, overlap)

    _assert_invariants(_pages(text), chunks, size)
    assert all(o <= 0 for o in _overlaps(chunks))
    assert "".join(c.text for c in chunks) == text


def test_progress_when_overlap_would_not_reach_next_unit() -> None:
    # The next unit is so long that starting early in the window would not
    # leave room for it; a later start or no overlap must be chosen.
    text = "a b c d e f g h " + "y" * 95

    chunks = chunk_pages(_pages(text), 100, 99)

    _assert_invariants(_pages(text), chunks, 100)


def test_overlap_never_crosses_pages() -> None:
    page_a = " ".join(_sentence(i, 6) for i in range(20))
    page_b = " ".join(_sentence(i + 100, 6) for i in range(20))

    chunks = chunk_pages(_pages(page_a, page_b), 200, 100)

    first_b = next(c for c in chunks if c.page_number == 2)
    assert first_b.char_start == 0
    assert first_b.text.startswith(_sentence(100, 6))
    for chunk in chunks:
        assert ("w100x" in chunk.text) <= (chunk.page_number == 2)


def test_chunks_never_cross_pages() -> None:
    chunks = chunk_pages(_pages("Page one.", "Page two."), 800, 120)

    assert [(c.page_number, c.text) for c in chunks] == [(1, "Page one."), (2, "Page two.")]


def test_empty_pages_produce_no_chunks_and_index_stays_sequential() -> None:
    pages = [
        CleanedPage(1, ""),
        CleanedPage(2, "First."),
        CleanedPage(3, ""),
        CleanedPage(4, "   \n\n "),
        CleanedPage(5, "Second."),
    ]

    chunks = chunk_pages(pages, 800, 120)

    assert [(c.chunk_index, c.page_number) for c in chunks] == [(0, 2), (1, 5)]


def test_no_pages_gives_no_chunks() -> None:
    assert chunk_pages([], 800, 120) == []


def test_page_numbers_are_copied_not_renumbered() -> None:
    pages = [CleanedPage(7, "seven"), CleanedPage(3, "three")]

    chunks = chunk_pages(pages, 800, 120)

    assert [(c.chunk_index, c.page_number) for c in chunks] == [(0, 7), (1, 3)]


def test_abbreviations_do_not_lose_or_duplicate_text() -> None:
    text = "As shown in Fig. 3 and e.g. Table 2, the result holds. " * 20
    text = text.strip()

    chunks = chunk_pages(_pages(text), 120, 0)

    _assert_invariants(_pages(text), chunks, 120)
    assert " ".join(c.text for c in chunks) == text


@pytest.mark.parametrize(
    "text",
    [
        "Yeh ek misaal hai ke chunking kaise kaam karti hai. " * 30,
        "یہ ایک مثال ہے۔ " * 60,
        "Party time \U0001f389\U0001f44d\U0001f3fd \U0001f468‍\U0001f469‍\U0001f467 ok! "
        * 40,
    ],
    ids=["roman_urdu", "urdu", "emoji"],
)
def test_multilingual_text_chunks_without_loss(text: str) -> None:
    text = text.strip()

    chunks = chunk_pages(_pages(text), 100, 0)

    _assert_invariants(_pages(text), chunks, 100)
    assert " ".join(c.text for c in chunks) == text


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_sizes_raise_value_error(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages(_pages("text"), size, overlap)


@pytest.mark.parametrize(("size", "overlap"), [(800.0, 120), (800, "120"), (True, 0)])
def test_non_int_sizes_raise_type_error(size: object, overlap: object) -> None:
    with pytest.raises(TypeError):
        chunk_pages(_pages("text"), size, overlap)  # type: ignore[arg-type]


def test_invalid_sizes_raise_even_with_no_pages() -> None:
    with pytest.raises(ValueError):
        chunk_pages([], 100, 100)


def test_logs_one_summary_line_without_text(caplog: pytest.LogCaptureFixture) -> None:
    secret = "confidential-marker-xyz"

    with caplog.at_level(logging.DEBUG, logger="rag_pipeline.chunking"):
        chunk_pages(_pages(f"{secret} " * 50, ""), 100, 20)

    records = [r for r in caplog.records if r.name == "rag_pipeline.chunking"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "2 page(s)" in records[0].getMessage()
    assert secret not in records[0].getMessage()
