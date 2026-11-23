"""Tests for scripts/compare_eval.py (synthetic result JSONs built with evaluate())."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rag_pipeline.config import PROJECT_ROOT
from rag_pipeline.evaluation import ExpectedSource, GoldenQuestion, RetrievedHit, evaluate


def _load_script() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "compare_eval.py"
    spec = importlib.util.spec_from_file_location("compare_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_eval", module)
    spec.loader.exec_module(module)
    return module


compare_eval = _load_script()

DOC_A = "a" * 64
DOC_B = "b" * 64
TAGS = {"q1": ("definition",), "q2": ("fact",), "q3": ("fact", "roman_urdu"), "n1": ("negative",)}


def question(qid: str) -> GoldenQuestion:
    expected = () if qid.startswith("n") else (ExpectedSource(DOC_A, "a.pdf", (1,)),)
    return GoldenQuestion(qid, f"text of {qid}", expected, TAGS[qid], None, 1)


def hits(rank: int | None, top_sim: float) -> list[RetrievedHit]:
    return [
        RetrievedHit(DOC_A if i == rank else DOC_B, 1 if i == rank else 90 + i,
                     round(top_sim - 0.01 * i, 6), "a.pdf" if i == rank else "b.pdf")
        for i in range(1, 11)
    ]


def write_result(
    path: Path, config: str, ranks: dict[str, int | None], ids: tuple[str, ...] = ("q1", "q2", "q3", "n1"),
    dirty: bool = False,
) -> Path:
    questions = [question(qid) for qid in ids]
    table = {f"text of {qid}": hits(ranks.get(qid), 0.8 if qid != "n1" else 0.3) for qid in ids}
    result = evaluate(questions, lambda text, top_k: table[text][:top_k])
    payload = {
        "run": {"date": "2026-11-23", "git": {"commit": "f" * 40, "dirty": dirty}},
        "manifest": {"config": config},
        "ingest": [],
        "results": result.to_dict(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return write_result(tmp_path / "base.json", "minilm_600_120", {"q1": 1, "q2": 3, "q3": 2})


def run_main(capsys: pytest.CaptureFixture[str], *paths: Path) -> tuple[int, str, str]:
    code = compare_eval.main([str(p) for p in paths])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_metrics_side_by_side_with_deltas(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "minilm_400_80", {"q1": 1, "q2": 1, "q3": 2})

    code, out, _ = run_main(capsys, base, other)

    assert code == 0
    lines = out.splitlines()
    overall = lines.index("OVERALL (n 4, answerable 3)")
    hit1 = lines[overall + 1]
    assert hit1.split()[:3] == ["hit@1", "0.333", "0.667"] and "(+0.333)" in hit1
    mrr = lines[overall + 5]
    # base MRR (1 + 1/3 + 1/2)/3 = 0.611; other (1 + 1 + 1/2)/3 = 0.833
    assert mrr.split()[:3] == ["MRR@10", "0.611", "0.833"] and "(+0.222)" in mrr
    assert "negative (n 1, answerable 0)" in lines
    neg = lines.index("negative (n 1, answerable 0)")
    assert lines[neg + 1].split() == ["hit@1", "n/a", "n/a"]
    assert all(line.isascii() for line in lines)


def test_rank_table_better_worse_same(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 2, "q2": 1, "q3": 2})

    _, out, _ = run_main(capsys, base, other)

    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("q")}
    assert rows["q1"].split()[1:4] == ["1", "2", "worse"]
    assert rows["q2"].split()[1:4] == ["3", "1", "better"]
    assert rows["q3"].split()[1:4] == ["2", "2", "same"]
    assert "n1" not in rows  # negatives have no rank row


def test_question_leaving_top5_is_flagged(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 7, "q3": None})

    _, out, _ = run_main(capsys, base, other)

    assert "7 worse  OUT-TOP5" in out
    assert "- worse  OUT-TOP5" in out
    assert "  q2: rank 7 in cfg (in top 5 in base)" in out
    assert "  q3: rank - in cfg (in top 5 in base)" in out


def test_no_flags_when_all_in_top5(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 5, "q3": 4})

    _, out, _ = run_main(capsys, base, other)

    flags = out.split("Answerable questions outside the top 5:")[1].splitlines()
    assert flags[1] == "  none"


def test_similarity_summary_per_file(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 1, "q3": 1})

    _, out, _ = run_main(capsys, base, other)

    sim = [line for line in out.splitlines() if "answerable top-1" in line]
    assert len(sim) == 2
    assert sim[0].startswith("minilm_600_120: answerable top-1 min 0.790 / median 0.790")
    assert "negatives: n1 0.290" in sim[0]


def test_three_files_compare_against_base(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write_result(tmp_path / "a.json", "cfg_a", {"q1": 1, "q2": 1, "q3": 1})
    b = write_result(tmp_path / "b.json", "cfg_b", {"q1": 2, "q2": 3, "q3": 2})

    code, out, _ = run_main(capsys, base, a, b)

    assert code == 0
    assert "OTHER cfg_a: a.json" in out and "OTHER cfg_b: b.json" in out
    assert "cfg_a vs minilm_600_120: PASS (replace base)" in out
    assert "cfg_b vs minilm_600_120: FAIL (keep base)" in out


# --- refusal ---------------------------------------------------------------------------------


def test_different_question_ids_are_refused(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 1}, ids=("q1", "q2", "n1"))

    code, out, err = run_main(capsys, base, other)

    assert code == 2 and out == ""
    assert "Different golden sets" in err and "only in base: ['q3']" in err


def test_different_question_order_is_refused(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 3, "q3": 2},
                         ids=("q2", "q1", "q3", "n1"))

    code, _, err = run_main(capsys, base, other)

    assert code == 2 and "or a different order" in err


def test_different_answerable_flags_are_refused(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 3, "q3": 2})
    payload = json.loads(other.read_text(encoding="utf-8"))
    payload["results"]["questions"][3]["answerable"] = True
    other.write_text(json.dumps(payload), encoding="utf-8")

    code, _, err = run_main(capsys, base, other)

    assert code == 2 and "answerable flags differ" in err


def test_different_ks_are_refused(tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str]) -> None:
    other = write_result(tmp_path / "o.json", "cfg", {"q1": 1, "q2": 3, "q3": 2})
    payload = json.loads(other.read_text(encoding="utf-8"))
    payload["results"]["ks"] = [1, 5]
    other.write_text(json.dumps(payload), encoding="utf-8")

    code, _, err = run_main(capsys, base, other)

    assert code == 2 and "Different ks" in err


@pytest.mark.parametrize("content", ["{not json", json.dumps({"no": "results"})])
def test_unreadable_or_foreign_file_is_refused(
    tmp_path: Path, base: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(content, encoding="utf-8")

    code, _, err = run_main(capsys, base, bad)

    assert code == 2 and "ERROR:" in err


# --- decision rule ------------------------------------------------------------------------------


def _rule(tmp_path: Path, base: Path, ranks: dict[str, Any]) -> tuple[bool, list[str]]:
    other = write_result(tmp_path / "o.json", "cfg", ranks)
    return compare_eval.rule_check(compare_eval.load_run(base), compare_eval.load_run(other))


def test_rule_passes_with_enough_gain_and_all_in_top5(tmp_path: Path, base: Path) -> None:
    passed, lines = _rule(tmp_path, base, {"q1": 1, "q2": 1, "q3": 2})  # MRR +0.222

    assert passed
    assert all(line.startswith("  [x]") for line in lines[1:])


def test_rule_fails_when_gain_is_below_threshold(tmp_path: Path, base: Path) -> None:
    # base (1 + 1/3 + 1/2)/3 = 0.611; other (1 + 1/2 + 1/3)/3 = 0.611 -> gain 0.000
    passed, lines = _rule(tmp_path, base, {"q1": 1, "q2": 2, "q3": 3})

    assert not passed
    assert "  [ ] MRR gain >= 0.050 (got +0.000)" in lines


def test_rule_fails_when_a_question_leaves_top5(tmp_path: Path, base: Path) -> None:
    # Big MRR gain on two questions, but q3 drops to rank 6.
    passed, lines = _rule(tmp_path, base, {"q1": 1, "q2": 1, "q3": 6})

    assert not passed
    assert "  [ ] hit@5 = 1.000 (got 0.667)" in lines
    assert "  [ ] no answerable question outside top 5 (outside: ['q3'])" in lines
