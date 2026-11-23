"""Tests for rag_pipeline.evaluation (pure: golden set loader and metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_pipeline.config import PROJECT_ROOT
from rag_pipeline.evaluation import (
    ExpectedSource,
    GoldenSetError,
    first_relevant_rank,
    hit_at_k,
    load_golden_set,
    mrr_at_k,
)

DOC_A = "a" * 64
DOC_B = "b" * 64


def answerable(qid: str = "q01", **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": qid,
        "question": "What is a decision tree?",
        "expected": [{"doc_id": DOC_A, "filename": "Lecture07.pdf", "pages": [4, 5]}],
        "tags": ["definition"],
        "notes": "slide 4",
    }
    entry.update(overrides)
    return entry


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_entries(path: Path, entries: list[dict[str, Any]]) -> Path:
    return write_lines(path, [json.dumps(e, ensure_ascii=False) for e in entries])


# --- load_golden_set: valid input ---------------------------------------------------


def test_load_valid_file_with_blank_lines(tmp_path: Path) -> None:
    negative = {"id": "q02", "question": "Who won the 1950 World Cup?", "expected": []}
    path = write_lines(
        tmp_path / "g.jsonl",
        ["", json.dumps(answerable()), "   ", json.dumps(negative), ""],
    )

    questions = load_golden_set(path)

    assert [q.id for q in questions] == ["q01", "q02"]
    first, second = questions
    assert first.expected == (ExpectedSource(DOC_A, "Lecture07.pdf", (4, 5)),)
    assert first.tags == ("definition",)
    assert first.notes == "slide 4"
    assert first.line_number == 2
    assert first.is_answerable
    assert second.expected == () and not second.is_answerable
    assert second.tags == () and second.notes is None
    assert second.line_number == 4


def test_roman_urdu_and_emoji_round_trip(tmp_path: Path) -> None:
    question = "Decision tree kya hai aur ye kaise kaam karta hai? \U0001F333 é"
    notes = "Roman Urdu: \"ye sawal lecture 7 se hai\" — \U0001F600"
    path = write_entries(tmp_path / "g.jsonl", [answerable(question=question, notes=notes)])

    loaded = load_golden_set(path)[0]

    assert loaded.question == question
    assert loaded.notes == notes


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(answerable()).encode("utf-8") + b"\n")

    assert load_golden_set(path)[0].id == "q01"


def test_example_file_in_repo_is_valid() -> None:
    questions = load_golden_set(PROJECT_ROOT / "eval" / "golden_set.example.jsonl")

    assert [q.is_answerable for q in questions] == [True, False]
    assert all("FAKE" in q.question for q in questions)


# --- load_golden_set: errors with line numbers ---------------------------------------


def _bad(entry: dict[str, Any]) -> str:
    return json.dumps(entry)


def _expected(**overrides: Any) -> dict[str, Any]:
    source: dict[str, Any] = {"doc_id": DOC_A, "filename": "x.pdf", "pages": [1]}
    source.update(overrides)
    return answerable(expected=[source])


_ERROR_CASES: dict[str, tuple[str, str]] = {
    "invalid_json": ('{"id": "q", ', "invalid JSON"),
    "not_object": ("[1, 2]", "must be a JSON object"),
    "unknown_key": (_bad(answerable(extra=1)), "unknown key"),
    "missing_question": (_bad({"id": "q", "expected": []}), "missing key.*question"),
    "missing_expected": (_bad({"id": "q", "question": "x"}), "missing key.*expected"),
    "empty_question": (_bad(answerable(question="   ")), "'question' must be a non-empty"),
    "question_not_str": (_bad(answerable(question=5)), "'question' must be a non-empty"),
    "empty_id": (_bad(answerable(qid=" ")), "'id' must be a non-empty"),
    "id_not_str": (_bad(answerable(qid=7)), "'id' must be a non-empty"),
    "expected_not_list": (_bad(answerable(expected={})), "'expected' must be a list"),
    "expected_item_not_object": (_bad(answerable(expected=["x"])), r"expected\[1\] must be an object"),
    "expected_unknown_key": (_bad(_expected(chunk_id="x")), "unknown key.*chunk_id"),
    "expected_missing_key": (
        _bad(answerable(expected=[{"doc_id": DOC_A, "pages": [1]}])),
        "lacks key.*filename",
    ),
    "doc_id_short": (_bad(_expected(doc_id="abc")), "64 lowercase hex"),
    "doc_id_upper": (_bad(_expected(doc_id="A" * 64)), "64 lowercase hex"),
    "doc_id_not_hex": (_bad(_expected(doc_id="g" * 64)), "64 lowercase hex"),
    "empty_filename": (_bad(_expected(filename="")), "filename must be a non-empty"),
    "pages_empty": (_bad(_expected(pages=[])), "pages must be a non-empty list"),
    "pages_not_list": (_bad(_expected(pages=4)), "pages must be a non-empty list"),
    "page_zero": (_bad(_expected(pages=[0])), "integers >= 1"),
    "page_float": (_bad(_expected(pages=[1.0])), "integers >= 1"),
    "page_bool": (_bad(_expected(pages=[True])), "integers >= 1"),
    "page_str": (_bad(_expected(pages=["4"])), "integers >= 1"),
    "pages_duplicate": (_bad(_expected(pages=[4, 4])), "duplicates"),
    "same_doc_twice": (
        _bad(
            answerable(
                expected=[
                    {"doc_id": DOC_A, "filename": "x.pdf", "pages": [1]},
                    {"doc_id": DOC_A, "filename": "y.pdf", "pages": [2]},
                ]
            )
        ),
        "same doc_id twice",
    ),
    "tags_not_list": (_bad(answerable(tags="definition")), "'tags' must be a list of strings"),
    "tag_not_str": (_bad(answerable(tags=["ok", 3])), "'tags' must be a list of strings"),
    "notes_not_str": (_bad(answerable(notes=["x"])), "'notes' must be a string"),
}


@pytest.mark.parametrize("case", sorted(_ERROR_CASES))
def test_each_validation_error_names_its_line(tmp_path: Path, case: str) -> None:
    bad_line, pattern = _ERROR_CASES[case]
    good = json.dumps(answerable("q-good"))
    path = write_lines(tmp_path / "g.jsonl", [good, "", bad_line])

    with pytest.raises(GoldenSetError, match=pattern) as info:
        load_golden_set(path)

    assert info.value.line_number == 3
    assert str(info.value).startswith("line 3: ")


def test_duplicate_id_names_both_lines(tmp_path: Path) -> None:
    path = write_entries(tmp_path / "g.jsonl", [answerable("q1"), answerable("q2"), answerable("q1")])

    with pytest.raises(GoldenSetError, match=r"line 3: duplicate id 'q1' \(first on line 1\)"):
        load_golden_set(path)


def test_file_without_questions_raises(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "g.jsonl", ["", "  "])

    with pytest.raises(GoldenSetError, match="holds no question") as info:
        load_golden_set(path)
    assert info.value.line_number is None


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="cannot read"):
        load_golden_set(tmp_path / "missing.jsonl")


def test_invalid_utf8_raises(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_bytes(b'{"id": "q\xff"}\n')

    with pytest.raises(GoldenSetError, match="cannot read"):
        load_golden_set(path)


# --- first_relevant_rank -------------------------------------------------------------


EXPECTED = (ExpectedSource(DOC_A, "Lecture07.pdf", (4,)),)


def test_rank_match_at_one() -> None:
    assert first_relevant_rank([(DOC_A, 4), (DOC_B, 1)], EXPECTED) == 1


def test_rank_match_at_three() -> None:
    assert first_relevant_rank([(DOC_B, 4), (DOC_A, 3), (DOC_A, 4)], EXPECTED) == 3


def test_rank_no_match_is_none() -> None:
    assert first_relevant_rank([(DOC_B, 4), (DOC_A, 5)], EXPECTED) is None


def test_rank_empty_results_is_none() -> None:
    assert first_relevant_rank([], EXPECTED) is None


def test_rank_negative_question_is_none() -> None:
    assert first_relevant_rank([(DOC_A, 4)], ()) is None


def test_rank_any_acceptable_page_counts() -> None:
    expected = (ExpectedSource(DOC_A, "x.pdf", (4, 5)),)

    assert first_relevant_rank([(DOC_B, 1), (DOC_A, 5), (DOC_A, 4)], expected) == 2


def test_rank_any_expected_document_counts() -> None:
    expected = (ExpectedSource(DOC_A, "a.pdf", (9,)), ExpectedSource(DOC_B, "b.pdf", (2,)))

    assert first_relevant_rank([(DOC_A, 1), (DOC_B, 2)], expected) == 2


def test_rank_same_page_twice_gives_first_position() -> None:
    assert first_relevant_rank([(DOC_B, 1), (DOC_A, 4), (DOC_A, 4)], EXPECTED) == 2


def test_rank_matches_doc_id_despite_different_filename() -> None:
    # The labelled filename is "Lecture07.pdf"; results carry only doc_id + page,
    # and a renamed copy ("Lecture07 (1).pdf") has the same doc_id.
    assert first_relevant_rank([(DOC_A, 4)], EXPECTED) == 1


def test_rank_same_page_of_other_document_is_no_match() -> None:
    assert first_relevant_rank([(DOC_B, 4)], EXPECTED) is None


# --- hit_at_k / mrr_at_k ---------------------------------------------------------------


WORKED = [1, 3, None]


def test_worked_example_hit_at_1() -> None:
    assert hit_at_k(WORKED, 1) == pytest.approx(1 / 3)


def test_worked_example_hit_at_5() -> None:
    assert hit_at_k(WORKED, 5) == pytest.approx(2 / 3)


def test_worked_example_mrr_at_5() -> None:
    assert mrr_at_k(WORKED, 5) == pytest.approx((1 + 1 / 3 + 0) / 3)


def test_rank_above_k_is_a_miss() -> None:
    assert hit_at_k([3, 6], 5) == pytest.approx(1 / 2)
    assert mrr_at_k([3, 6], 5) == pytest.approx((1 / 3 + 0) / 2)
    assert hit_at_k([2], 1) == 0.0
    assert mrr_at_k([2], 1) == 0.0


def test_rank_equal_to_k_is_a_hit() -> None:
    assert hit_at_k([5], 5) == 1.0
    assert mrr_at_k([5], 5) == pytest.approx(0.2)


@pytest.mark.parametrize("metric", [hit_at_k, mrr_at_k])
def test_empty_ranks_raise(metric: Any) -> None:
    with pytest.raises(ValueError, match="at least one answerable question"):
        metric([], 5)


@pytest.mark.parametrize("metric", [hit_at_k, mrr_at_k])
@pytest.mark.parametrize("k", [0, -1, True, 2.0])
def test_invalid_k_raises(metric: Any, k: Any) -> None:
    with pytest.raises(ValueError, match="k must be"):
        metric([1], k)


@pytest.mark.parametrize("metric", [hit_at_k, mrr_at_k])
@pytest.mark.parametrize("rank", [0, -2, 1.0, True])
def test_invalid_rank_raises(metric: Any, rank: Any) -> None:
    with pytest.raises(ValueError, match="ranks must be"):
        metric([1, rank], 5)
