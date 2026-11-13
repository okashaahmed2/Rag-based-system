"""Tests for rag_pipeline.cleaning."""

from __future__ import annotations

import logging
import sys
import unicodedata

import pytest

from rag_pipeline.cleaning import (
    SPACE_SEPARATORS,
    CleanedPage,
    clean_page_text,
    clean_pages,
)
from rag_pipeline.extraction import PageText

# --- Character rules ----------------------------------------------------------


def test_ligature_fi_is_expanded() -> None:
    assert clean_page_text("ﬁle") == "file"


@pytest.mark.parametrize(
    ("ligature", "expected"),
    [
        ("ﬀ", "ff"),
        ("ﬁ", "fi"),
        ("ﬂ", "fl"),
        ("ﬃ", "ffi"),
        ("ﬄ", "ffl"),
        ("ﬅ", "st"),
        ("ﬆ", "st"),
    ],
)
def test_all_ligatures_are_expanded(ligature: str, expected: str) -> None:
    assert clean_page_text(f"a{ligature}b") == f"a{expected}b"


def test_superscript_is_preserved_nfc_not_nfkc() -> None:
    assert clean_page_text("x² + y²") == "x² + y²"


def test_decomposed_accent_is_composed_by_nfc() -> None:
    assert clean_page_text("café") == "café"


def test_nbsp_becomes_space() -> None:
    assert clean_page_text("10 kg") == "10 kg"


@pytest.mark.parametrize("space", sorted(SPACE_SEPARATORS))
def test_every_space_separator_becomes_space(space: str) -> None:
    assert clean_page_text(f"a{space}b") == "a b"


def test_space_separator_table_matches_unicodedata() -> None:
    zs = {
        chr(code)
        for code in range(sys.maxunicode + 1)
        if unicodedata.category(chr(code)) == "Zs"
    }
    assert SPACE_SEPARATORS == zs - {" "}


@pytest.mark.parametrize("char", ["​", "﻿", "­"])
def test_invisible_characters_are_removed(char: str) -> None:
    assert clean_page_text(f"data{char}base") == "database"


def test_emoji_zwj_sequence_survives() -> None:
    family = "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466"
    assert clean_page_text(f"Family: {family}") == f"Family: {family}"


@pytest.mark.parametrize(
    "text",
    [
        "می‌کنم",  # Persian "mikonam" with ZWNJ
        "بچوں ‌کا",  # Urdu text with ZWNJ
    ],
)
def test_zwnj_in_urdu_persian_survives(text: str) -> None:
    assert clean_page_text(text) == text


def test_nul_and_c0_controls_are_removed() -> None:
    assert clean_page_text("a\x00b\x01c\x1bd\x1fe") == "abcde"


@pytest.mark.parametrize("char", ["\x7f", "\x80", "\x90", "\x9f"])
def test_del_and_c1_controls_are_removed(char: str) -> None:
    assert clean_page_text(f"ab{char}cd") == "abcd"


def test_tab_is_kept_as_whitespace_not_removed() -> None:
    assert clean_page_text("a\tb") == "a b"


def test_removed_invisible_char_before_combining_mark_is_idempotent() -> None:
    once = clean_page_text("e​́")
    assert once == "é"
    assert clean_page_text(once) == once


# --- Line endings -------------------------------------------------------------


def test_crlf_becomes_lf() -> None:
    assert clean_page_text("a\r\nb") == "a\nb"


def test_lone_cr_becomes_lf() -> None:
    assert clean_page_text("a\rb") == "a\nb"


def test_form_feed_and_vertical_tab_become_lf() -> None:
    assert clean_page_text("a\fb\vc") == "a\nb\nc"


@pytest.mark.parametrize("char", [" ", "\u0085"])
def test_line_separator_and_nel_become_lf(char: str) -> None:
    assert clean_page_text(f"a{char}b") == "a\nb"


def test_paragraph_separator_becomes_paragraph_break() -> None:
    assert clean_page_text("a b") == "a\n\nb"


def test_paragraph_separator_does_not_exceed_two_newlines() -> None:
    assert clean_page_text("a\n \nb") == "a\n\nb"


def test_crlf_blank_line_stays_one_paragraph_break() -> None:
    assert clean_page_text("a\r\n\r\nb") == "a\n\nb"


# --- Hyphenated line breaks ---------------------------------------------------


def test_hyphenated_break_is_joined() -> None:
    assert clean_page_text("informa-\ntion") == "information"


def test_hyphenated_break_with_spaces_after_newline_is_joined() -> None:
    assert clean_page_text("informa-\n   tion") == "information"


def test_hyphenated_break_with_spaces_before_newline_is_joined() -> None:
    assert clean_page_text("informa-  \ntion") == "information"


def test_consecutive_hyphenated_breaks_are_all_joined() -> None:
    assert clean_page_text("ab-\ncd-\nef") == "abcdef"


def test_capitalised_word_with_lowercase_split_is_joined() -> None:
    # Only the letter directly before "-" matters, so "Data-\nscience" joins.
    assert clean_page_text("Data-\nscience") == "Datascience"


def test_hyphenated_break_unicode_lowercase_is_joined() -> None:
    assert clean_page_text("groß-\närtig") == "großärtig"


@pytest.mark.parametrize(
    "text",
    [
        "COVID-\n19",
        "Data-\nScience",
        "data-\nScience",
        "list:\n- item one\n- item two",
        "word -\nnext",
        "informa-\n\ntion",
        "x-\n-y",
    ],
)
def test_hyphenated_break_not_joined(text: str) -> None:
    assert clean_page_text(text) == text


def test_hyphen_inside_line_is_kept() -> None:
    assert clean_page_text("state-of-the-art") == "state-of-the-art"


# --- Whitespace and newlines --------------------------------------------------


def test_spaces_and_tabs_collapse() -> None:
    assert clean_page_text("a  \t  b\t\tc") == "a b c"


def test_trailing_and_leading_spaces_on_lines_are_removed() -> None:
    assert clean_page_text("  first  \n\t second \t") == "first\nsecond"


@pytest.mark.parametrize("count", [3, 4, 7])
def test_many_newlines_become_two(count: int) -> None:
    assert clean_page_text("a" + "\n" * count + "b") == "a\n\nb"


def test_blank_lines_with_spaces_collapse_to_two_newlines() -> None:
    assert clean_page_text("a\n  \n\t\n \nb") == "a\n\nb"


def test_single_and_double_newlines_preserved() -> None:
    text = "line one\nline two\n\nparagraph two\n- bullet"
    assert clean_page_text(text) == text


@pytest.mark.parametrize("text", ["", " ", "\n\n\t ", " ​\r\n\f", "\x00"])
def test_empty_or_whitespace_only_returns_empty(text: str) -> None:
    assert clean_page_text(text) == ""


def test_page_is_stripped() -> None:
    assert clean_page_text("\n\n  hello world  \n\n") == "hello world"


# --- Content that must survive ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "SVM kya hai?",
        "ڈیٹا سائنس",  # Urdu script
        "Great result \U0001f680\U0001f44d",
        "Family: \U0001f468‍\U0001f469",
        "E = mc², H₂O, α ≤ β!",
        "UPPER lower MiXeD, punctuation: ;.?!()[]{}",
    ],
)
def test_meaningful_content_is_not_altered(text: str) -> None:
    assert clean_page_text(text) == text


# --- Properties ---------------------------------------------------------------

MIXED_SAMPLES = [
    "",
    "plain text",
    "﻿  Intro duction\r\n\r\n\r\n\r\nThe informa-\n  tion is​ here.\f",
    "COVID-\n19 and Data-\nScience\n\n\n\n- item\n-  item two  ",
    "x² + ﬁle ­\tcafé\x00\n\n\nSVM kya hai? \U0001f680",
    "ab-\ncd-\nef\r\rgh\v\vij",
    "e​́ a-\nété ڈیٹا",
    " \n \n \n ",
    "a-\n​b",
    "a -\n\nb\n \n \nc",
    "line next\u0085third para  \n\nend",
    "del\x7f c1\x80\x90\x9f     ",
    "\U0001f468‍\U0001f469 می‌کنم a-\n‌b",
    "a- b x-\u0085y",
]


@pytest.mark.parametrize("text", MIXED_SAMPLES)
def test_clean_page_text_is_idempotent(text: str) -> None:
    once = clean_page_text(text)
    assert clean_page_text(once) == once


@pytest.mark.parametrize("value", [None, 123, b"bytes", ["a"]])
def test_non_str_raises_type_error(value: object) -> None:
    with pytest.raises(TypeError):
        clean_page_text(value)  # type: ignore[arg-type]


# --- clean_pages --------------------------------------------------------------


def _pages() -> tuple[PageText, ...]:
    return (
        PageText(page_number=1, text="  First   page\r\n", has_text=True),
        PageText(page_number=2, text="​ \n\n", has_text=True),
        PageText(page_number=3, text="", has_text=False),
        PageText(page_number=4, text="informa-\ntion", has_text=True),
    )


def test_clean_pages_preserves_count_order_and_page_numbers() -> None:
    pages = _pages()
    cleaned = clean_pages(pages)
    assert [page.page_number for page in cleaned] == [1, 2, 3, 4]
    assert [page.text for page in cleaned] == ["First page", "", "", "information"]


def test_clean_pages_keeps_page_that_becomes_empty() -> None:
    cleaned = clean_pages(_pages())
    assert cleaned[1] == CleanedPage(page_number=2, text="")
    assert not cleaned[1].has_text
    assert cleaned[0].has_text


def test_clean_pages_does_not_mutate_input() -> None:
    pages_list = list(_pages())
    snapshot = list(pages_list)
    result = clean_pages(pages_list)
    assert pages_list == snapshot
    assert result is not pages_list


def test_clean_pages_non_consecutive_page_numbers_unchanged() -> None:
    pages = [
        PageText(page_number=7, text="b", has_text=True),
        PageText(page_number=2, text="a", has_text=True),
    ]
    assert [page.page_number for page in clean_pages(pages)] == [7, 2]


def test_clean_pages_empty_input() -> None:
    assert clean_pages([]) == []


def test_cleaned_page_is_frozen() -> None:
    page = CleanedPage(page_number=1, text="x")
    with pytest.raises(AttributeError):
        page.text = "y"  # type: ignore[misc]


def test_clean_pages_logs_one_summary_without_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG, logger="rag_pipeline.cleaning"):
        clean_pages(_pages())
    records = [r for r in caplog.records if r.name == "rag_pipeline.cleaning"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "4 page(s)" in message
    assert "First" not in message
    assert "information" not in message
