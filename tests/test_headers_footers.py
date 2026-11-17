"""Tests for document-level header/footer removal in rag_pipeline.cleaning."""

from __future__ import annotations

import logging

import pytest

from rag_pipeline.cleaning import (
    PAGE_NUMBER_KEY_PATTERN,
    CleanedPage,
    HeaderFooterResult,
    KeyKind,
    RepeatedKey,
    remove_repeated_headers_footers,
)

EXACT = KeyKind.EXACT
PAGE = KeyKind.PAGE_NUMBER

WORDS = [
    "apple", "banana", "cherry", "damson", "elder", "fig", "grape", "hazel",
    "iris", "juniper", "kiwi", "lemon", "mango", "nectar", "olive", "peach",
]


def _body(i: int) -> str:
    """Three distinct content lines; keys differ by words, not only digits."""
    a, b, c = WORDS[i % 16], WORDS[(i + 5) % 16], WORDS[(i + 11) % 16]
    return f"{a} is explained here\n{b} is compared with {a}\nwhy {c} matters"


def _pages(texts: list[str], start: int = 1) -> list[CleanedPage]:
    return [CleanedPage(page_number=start + i, text=t) for i, t in enumerate(texts)]


def _texts(result: HeaderFooterResult) -> list[str]:
    return [page.text for page in result.pages]


# --- Removal ------------------------------------------------------------------


def test_identical_header_on_all_pages_is_removed() -> None:
    pages = _pages([f"ML Notes\n{_body(i)}" for i in range(5)])
    result = remove_repeated_headers_footers(pages)
    assert _texts(result) == [_body(i) for i in range(5)]
    assert result.repeated_top == (RepeatedKey("ml notes", EXACT, 5),)
    assert result.repeated_bottom == ()
    assert result.lines_removed == 5


def test_page_n_of_m_footer_is_removed() -> None:
    pages = _pages([f"{_body(n)}\nPage {n} of 12" for n in range(1, 13)])
    result = remove_repeated_headers_footers(pages)
    assert _texts(result) == [_body(n) for n in range(1, 13)]
    assert result.repeated_bottom == (RepeatedKey("page # of #", PAGE, 12),)


def test_bare_page_numbers_at_bottom_are_removed() -> None:
    pages = _pages([f"{_body(n)}\n{n + 1}" for n in range(4)])
    result = remove_repeated_headers_footers(pages)
    assert _texts(result) == [_body(n) for n in range(4)]
    assert result.repeated_bottom == (RepeatedKey("#", PAGE, 4),)


def test_header_and_footer_removed_and_blank_lines_tidied() -> None:
    pages = _pages([f"Course X\n\n{_body(n)}\n\n\n{n}" for n in range(1, 5)])
    result = remove_repeated_headers_footers(pages)
    assert _texts(result) == [_body(n) for n in range(1, 5)]
    assert result.lines_removed == 8


def test_header_on_second_edge_line_is_removed() -> None:
    pages = _pages([f"Course X\nUnit 3\n{_body(n)}" for n in range(4)])
    result = remove_repeated_headers_footers(pages)
    assert _texts(result) == [_body(n) for n in range(4)]
    assert {k.key for k in result.repeated_top} == {"course x", "unit 3"}


# --- Lines that must be kept --------------------------------------------------


def test_line_repeated_at_top_of_only_two_of_ten_pages_is_kept() -> None:
    texts = [_body(i) for i in range(10)]
    texts[0] = "Summary\n" + texts[0]
    texts[5] = "Summary\n" + texts[5]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == texts
    assert result.lines_removed == 0


def test_key_below_ratio_is_kept() -> None:
    # 3 of 10 pages meets MIN_REPEAT_PAGES but not the 50 % ratio.
    texts = [_body(i) for i in range(10)]
    for i in (0, 3, 6):
        texts[i] = "Recap\n" + texts[i]
    result = remove_repeated_headers_footers(_pages(texts))
    assert result.repeated_top == ()
    assert _texts(result) == texts


def test_header_text_in_middle_of_page_is_kept() -> None:
    texts = [f"ML Notes\n{_body(i)}" for i in range(4)]
    texts.append("ML Notes\nalpha intro\nbeta intro\nML Notes\ngamma end\ndelta end")
    result = remove_repeated_headers_footers(_pages(texts))
    assert result.pages[-1].text == "alpha intro\nbeta intro\nML Notes\ngamma end\ndelta end"


def test_header_key_at_bottom_is_not_removed_by_top_rule() -> None:
    texts = [f"ML Notes\n{_body(i)}" for i in range(4)]
    texts.append(f"{_body(9)}\nML Notes")
    result = remove_repeated_headers_footers(_pages(texts))
    assert result.pages[-1].text == f"{_body(9)}\nML Notes"


def test_roman_urdu_content_kept_unless_repeated_edge_line() -> None:
    texts = [
        "SVM Notes\nSVM kya hai?\nYe ek classifier hai.\nmargin zaroori hai",
        "SVM Notes\nKernel trick kya hai?\nData ko upar le jao.\nlinear nahi hai",
        "SVM Notes\nSVM kya hai?\nDobara samjho.\nexample dekho",
        "SVM Notes\nC parameter kya hai?\nPenalty hai.\nsoft margin",
    ]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [t.split("\n", 1)[1] for t in texts]
    # "SVM kya hai?" is a top edge line on 2 of 4 pages: below MIN_REPEAT_PAGES.
    assert result.repeated_top == (RepeatedKey("svm notes", EXACT, 4),)
    assert result.pages[0].text.startswith("SVM kya hai?")
    assert result.pages[2].text.startswith("SVM kya hai?")


def test_repeated_roman_urdu_edge_line_is_removed() -> None:
    texts = [f"Yaad rakho: exam ahem hai\n{_body(i)}" for i in range(4)]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [_body(i) for i in range(4)]


# --- Thresholds and empty pages -----------------------------------------------


def test_fewer_than_three_text_pages_changes_nothing() -> None:
    pages = _pages([f"Header\n{_body(0)}", f"Header\n{_body(1)}", "", ""])
    result = remove_repeated_headers_footers(pages)
    assert list(result.pages) == pages
    assert result.lines_removed == 0
    assert result.repeated_top == () and result.repeated_bottom == ()


def test_empty_pages_do_not_count_towards_ratio_and_are_kept() -> None:
    # 3 header pages of 3 text pages (100 %); counting the 4 empty pages
    # would give 3/7 < 50 % and nothing would be removed.
    texts = ["", f"Header\n{_body(1)}", "", f"Header\n{_body(2)}", "",
             f"Header\n{_body(3)}", ""]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == ["", _body(1), "", _body(2), "", _body(3), ""]
    assert result.repeated_top == (RepeatedKey("header", EXACT, 3),)


def test_page_with_only_header_becomes_empty_but_is_kept() -> None:
    texts = [f"Header\n{_body(0)}", f"Header\n{_body(1)}", "Header", f"Header\n{_body(3)}"]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [_body(0), _body(1), "", _body(3)]
    assert not result.pages[2].has_text
    assert result.pages[2].page_number == 3


def test_short_pages_varying_numbered_lines_are_kept() -> None:
    # On short pages every line is an edge line; lines that differ only in
    # digits but are not page-number-like keep their exact key and survive.
    texts = [f"Header\nContent of page {n}." for n in range(1, 5)]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [f"Content of page {n}." for n in range(1, 5)]
    assert result.repeated_top == (RepeatedKey("header", EXACT, 4),)


def test_single_line_page_counts_once_per_region() -> None:
    result = remove_repeated_headers_footers(_pages(["Header"] * 3))
    assert result.repeated_top == (RepeatedKey("header", EXACT, 3),)
    assert result.repeated_bottom == (RepeatedKey("header", EXACT, 3),)
    assert result.lines_removed == 3
    assert _texts(result) == ["", "", ""]


# --- Invariants and result object ---------------------------------------------


def test_preserves_count_order_and_page_numbers() -> None:
    pages = [
        CleanedPage(page_number=9, text=f"H\n{_body(9)}\n1"),
        CleanedPage(page_number=2, text=""),
        CleanedPage(page_number=4, text=f"H\n{_body(4)}\n2"),
        CleanedPage(page_number=5, text=f"H\n{_body(5)}\n3"),
    ]
    result = remove_repeated_headers_footers(pages)
    assert [page.page_number for page in result.pages] == [9, 2, 4, 5]
    assert _texts(result) == [_body(9), "", _body(4), _body(5)]


def test_input_is_not_mutated() -> None:
    pages = _pages([f"Header\n{_body(i)}" for i in range(4)])
    snapshot = list(pages)
    result = remove_repeated_headers_footers(pages)
    assert pages == snapshot
    assert result.pages is not pages


def test_unchanged_pages_are_returned_equal() -> None:
    pages = _pages([_body(i) for i in range(5)])
    result = remove_repeated_headers_footers(pages)
    assert list(result.pages) == pages
    assert result.lines_removed == 0


def test_result_reports_keys_counts_and_line_total() -> None:
    texts = [f"Doc Title\n{_body(n)}\nPage {n} of 6" for n in range(1, 6)]
    texts.append(f"Other start\n{_body(6)}\nPage 6 of 6")
    result = remove_repeated_headers_footers(_pages(texts))
    assert result.repeated_top == (RepeatedKey("doc title", EXACT, 5),)
    assert result.repeated_bottom == (RepeatedKey("page # of #", PAGE, 6),)
    assert result.lines_removed == 11
    assert result.pages[5].text == f"Other start\n{_body(6)}"


def test_result_is_frozen() -> None:
    result = remove_repeated_headers_footers([])
    with pytest.raises(AttributeError):
        result.lines_removed = 1  # type: ignore[misc]


def test_empty_input() -> None:
    assert remove_repeated_headers_footers([]) == HeaderFooterResult((), (), (), 0)


def test_logs_one_summary_line_without_text_or_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pages = _pages([f"Secret Header\n{_body(i)}" for i in range(4)])
    with caplog.at_level(logging.DEBUG, logger="rag_pipeline.cleaning"):
        remove_repeated_headers_footers(pages)
    records = [r for r in caplog.records if r.name == "rag_pipeline.cleaning"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "4 line(s) removed" in message
    assert "secret" not in message.lower()
    assert "explained" not in message


def test_skip_path_also_logs_one_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="rag_pipeline.cleaning"):
        remove_repeated_headers_footers(_pages(["Header\nbody"]))
    assert len([r for r in caplog.records if r.name == "rag_pipeline.cleaning"]) == 1


# --- Two key kinds: exact vs page-number-like ---------------------------------


def test_slide_deck_header_removed_step_titles_kept() -> None:
    texts = [
        f"Lecture 5\nStep {n}\n{WORDS[n]} is defined here\nwhy {WORDS[n + 6]} matters"
        for n in range(1, 7)
    ]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [t.split("\n", 1)[1] for t in texts]
    assert result.repeated_top == (RepeatedKey("lecture 5", EXACT, 6),)
    assert result.lines_removed == 6


@pytest.mark.parametrize("title", ["Example", "Chapter"])
def test_varying_numbered_header_is_kept(title: str) -> None:
    # "Chapter N" running headers are kept: documented trade-off.
    texts = [f"{title} {n}\n{_body(n)}" for n in range(1, 7)]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == texts
    assert result.lines_removed == 0
    assert result.repeated_top == ()


@pytest.mark.parametrize(
    ("template", "key"),
    [
        ("- {n} -", "- # -"),
        ("{n}/12", "#/#"),
        ("{n} / 12", "# / #"),
        ("Slide {n}", "slide #"),
        ("p. {n}", "p. #"),
        ("Pg {n}", "pg #"),
        ("Page {n}", "page #"),
        ("{n} | Page", "# | page"),
        ("\u2013 {n} \u2013", "\u2013 # \u2013"),
        ("\u2014 {n} \u2014", "\u2014 # \u2014"),
    ],
)
def test_page_number_footers_are_removed(template: str, key: str) -> None:
    texts = [f"{_body(n)}\n{template.format(n=n)}" for n in range(1, 13)]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [_body(n) for n in range(1, 13)]
    assert result.repeated_bottom == (RepeatedKey(key, PAGE, 12),)


def test_constant_numbered_header_removed_by_exact_key() -> None:
    texts = [f"Unit 3\n{_body(n)}" for n in range(6)]
    result = remove_repeated_headers_footers(_pages(texts))
    assert _texts(result) == [_body(n) for n in range(6)]
    assert result.repeated_top == (RepeatedKey("unit 3", EXACT, 6),)


def test_constant_page_number_like_line_reported_once_as_page_number() -> None:
    texts = [f"{_body(n)}\nPage 1" for n in range(4)]
    result = remove_repeated_headers_footers(_pages(texts))
    assert result.repeated_bottom == (RepeatedKey("page #", PAGE, 4),)
    assert _texts(result) == [_body(n) for n in range(4)]


@pytest.mark.parametrize(
    "key",
    ["#", "page #", "page # of #", "# / #", "#/#", "- # -", "slide #", "p. #",
     "pg. #", "# of #", "| # |", "#.#"],
)
def test_page_number_pattern_accepts(key: str) -> None:
    assert PAGE_NUMBER_KEY_PATTERN.match(key)


@pytest.mark.parametrize(
    "key",
    ["step #", "example #", "chapter #", "unit #", "lecture #", "page", "- -",
     "", "content of page #.", "# kya hai", "#x", "slides #"],
)
def test_page_number_pattern_rejects(key: str) -> None:
    assert not PAGE_NUMBER_KEY_PATTERN.match(key)
