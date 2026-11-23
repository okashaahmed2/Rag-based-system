"""Compare evaluation result files side by side (ASCII output).

Usage::

    .venv\\Scripts\\python.exe scripts\\compare_eval.py BASE.json OTHER.json [MORE.json ...]

Shows, for every file against BASE: overall and per-tag hit@k and MRR with
deltas, a per-question rank table (better / worse / same), a flag for every
answerable question outside the top 5, the similarity summaries, and the
pre-registered decision rule check. Files built from different golden sets
(different question ids) or with different k values are refused (exit 2).

Input files are written by ``scripts/run_eval.py`` and contain no question
or chunk text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The "top 5" of the decision rule and the outside-top-5 flag.
TOP_K_GUARD = 5
#: Pre-registered rule: replace the base only if hit@5 stays 1.000, MRR improves by
#: at least this much, and no answerable question leaves the top 5.
MIN_MRR_GAIN = 0.05
REQUIRED_HIT_AT_GUARD = 1.0
FLOAT_TOLERANCE = 1e-9


class CompareError(Exception):
    """The result files cannot be compared."""


@dataclass(frozen=True)
class EvalRun:
    """One loaded result file."""

    path: Path
    label: str
    payload: dict[str, Any]

    @property
    def results(self) -> dict[str, Any]:
        return self.payload["results"]

    @property
    def mrr_key(self) -> str:
        return f"mrr_at_{self.results['mrr_k']}"

    def question_ids(self) -> list[str]:
        return [q["id"] for q in self.results["questions"]]

    def ranks(self) -> dict[str, int | None]:
        return {q["id"]: q["first_relevant_rank"] for q in self.results["questions"]}

    def answerable(self) -> dict[str, bool]:
        return {q["id"]: q["answerable"] for q in self.results["questions"]}


def load_run(path: Path) -> EvalRun:
    """Read one result file; the label is its manifest config (else the file stem)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompareError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict) or "results" not in payload:
        raise CompareError(f"{path} is not a run_eval result file (no 'results')")
    label = str((payload.get("manifest") or {}).get("config") or path.stem)
    return EvalRun(path=path, label=label, payload=payload)


def check_comparable(runs: Sequence[EvalRun]) -> None:
    """Same question ids (same order), same answerable flags, same ks and MRR depth."""
    base = runs[0]
    for other in runs[1:]:
        if other.question_ids() != base.question_ids():
            only_base = sorted(set(base.question_ids()) - set(other.question_ids()))
            only_other = sorted(set(other.question_ids()) - set(base.question_ids()))
            raise CompareError(
                f"Different golden sets: {base.path.name} vs {other.path.name} "
                f"(only in base: {only_base or '-'}; only in other: {only_other or '-'}; "
                "or a different order)"
            )
        if other.answerable() != base.answerable():
            raise CompareError(
                f"Different golden sets: answerable flags differ in {other.path.name}"
            )
        for key in ("ks", "mrr_k"):
            if other.results[key] != base.results[key]:
                raise CompareError(
                    f"Different {key}: {base.path.name} {base.results[key]} vs "
                    f"{other.path.name} {other.results[key]}"
                )


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _delta(value: float | None, base: float | None) -> str:
    if value is None or base is None:
        return ""
    return f"({value - base:+.3f})"


def _rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _rank_key(rank: int | None) -> float:
    """Missing rank (not in the retrieved depth) sorts after every real rank."""
    return float("inf") if rank is None else float(rank)


def _outside_guard(rank: int | None) -> bool:
    return rank is None or rank > TOP_K_GUARD


#: Narrowest column: fits a value with its delta, e.g. "0.615 (-0.077)".
MIN_COLUMN_WIDTH = 16


def common_model(runs: Sequence[EvalRun]) -> str | None:
    """The model slug all labels share, if every label is ``<slug>_<size>_<overlap>``."""
    parts = [run.label.rsplit("_", 2) for run in runs]
    if len(runs) < 2 or any(len(p) != 3 for p in parts):
        return None
    slugs = {p[0] for p in parts}
    return slugs.pop() if len(slugs) == 1 else None


def column_labels(runs: Sequence[EvalRun]) -> list[str]:
    """Unambiguous column labels: the chunk part ("600_120") when all runs share a
    model, else the full config names. Never truncated."""
    if common_model(runs) is not None:
        return ["_".join(run.label.rsplit("_", 2)[1:]) for run in runs]
    return [run.label for run in runs]


def _column_width(labels: Sequence[str]) -> int:
    return max(MIN_COLUMN_WIDTH, *(len(label) for label in labels))


def metric_lines(runs: Sequence[EvalRun]) -> list[str]:
    """Overall + per-tag table, one block per group, deltas vs the base."""
    base = runs[0]
    ks = [str(k) for k in base.results["ks"]]
    metrics = [(f"hit@{k}", ("hit_at", k)) for k in ks] + [
        (f"MRR@{base.results['mrr_k']}", (base.mrr_key, None))
    ]
    labels = column_labels(runs)
    width = _column_width(labels)
    header = f"{'group / metric':<22}" + "".join(f"{label:>{width + 2}}" for label in labels)
    lines = [header, "-" * len(header)]
    groups = ["overall", *base.results["per_tag"].keys()]
    for group in groups:
        summaries = [
            run.results["overall"] if group == "overall" else run.results["per_tag"].get(group)
            for run in runs
        ]
        first = summaries[0] or {}
        lines.append(
            f"{group.upper() if group == 'overall' else group} "
            f"(n {first.get('questions', 0)}, answerable {first.get('answerable', 0)})"
        )
        for name, (key, k) in metrics:
            values: list[float | None] = []
            for summary in summaries:
                if not summary:
                    values.append(None)
                elif k is None:
                    values.append(summary.get(key))
                else:
                    hit_at = summary.get("hit_at")
                    values.append(None if hit_at is None else hit_at.get(k))
            cells = [f"{_num(values[0]):>{width + 2}}"]
            for value in values[1:]:
                cell = f"{_num(value)} {_delta(value, values[0])}".strip()
                cells.append(f"{cell:>{width + 2}}")
            lines.append(f"  {name:<20}" + "".join(cells))
    return lines


def rank_lines(runs: Sequence[EvalRun]) -> list[str]:
    """Per-question ranks vs the base, with better / worse / same and top-5 flags."""
    base = runs[0]
    base_ranks = base.ranks()
    answerable = base.answerable()
    labels = column_labels(runs)
    # A cell holds e.g. "10 better  OUT-TOP5" (19 characters).
    width = max(22, *(len(label) for label in labels[1:]))
    header = f"{'id':<6}{labels[0]:>{max(6, len(labels[0]) + 2)}}" + "".join(
        f"{label:>{width + 4}}" for label in labels[1:]
    )
    base_width = max(6, len(labels[0]) + 2)
    lines = [header, "-" * len(header)]
    flags: list[str] = []
    for qid in base.question_ids():
        if not answerable[qid]:
            continue
        row = f"{qid:<6}{_rank(base_ranks[qid]):>{base_width}}"
        for run in runs[1:]:
            rank = run.ranks()[qid]
            if _rank_key(rank) < _rank_key(base_ranks[qid]):
                change = "better"
            elif _rank_key(rank) > _rank_key(base_ranks[qid]):
                change = "worse"
            else:
                change = "same"
            mark = "  OUT-TOP5" if _outside_guard(rank) else ""
            row += f"{_rank(rank) + ' ' + change + mark:>{width + 4}}"
            if _outside_guard(rank):
                left = "" if _outside_guard(base_ranks[qid]) else " (in top 5 in base)"
                flags.append(
                    f"  {qid}: rank {_rank(rank)} in {run.label}{left}"
                )
        lines.append(row)
    lines.append("")
    lines.append(f"Answerable questions outside the top {TOP_K_GUARD}:")
    lines += flags or ["  none"]
    return lines


def similarity_lines(runs: Sequence[EvalRun]) -> list[str]:
    lines = []
    for run in runs:
        sim = run.results["similarity"]
        top1 = sim["answerable_top1"]
        negatives = ", ".join(f"{qid} {_num(v)}" for qid, v in sim["negative_top1"].items())
        lines.append(
            f"{run.label}: answerable top-1 min {_num(top1['min'])} / median "
            f"{_num(top1['median'])} / max {_num(top1['max'])} "
            f"({top1['without_results']} without results); negatives: {negatives or '-'}"
        )
    return lines


def rule_check(base: EvalRun, other: EvalRun) -> tuple[bool, list[str]]:
    """The pre-registered decision rule for replacing ``base`` by ``other``."""
    hit_guard = (other.results["overall"]["hit_at"] or {}).get(str(TOP_K_GUARD))
    base_mrr = base.results["overall"][base.mrr_key]
    other_mrr = other.results["overall"][other.mrr_key]
    left = [
        qid
        for qid, rank in other.ranks().items()
        if other.answerable()[qid] and _outside_guard(rank)
    ]
    gain = None if base_mrr is None or other_mrr is None else other_mrr - base_mrr
    checks = [
        (
            hit_guard is not None and abs(hit_guard - REQUIRED_HIT_AT_GUARD) <= FLOAT_TOLERANCE,
            f"hit@{TOP_K_GUARD} = {REQUIRED_HIT_AT_GUARD:.3f} (got {_num(hit_guard)})",
        ),
        (
            gain is not None and gain >= MIN_MRR_GAIN - FLOAT_TOLERANCE,
            f"MRR gain >= {MIN_MRR_GAIN:.3f} (got {'n/a' if gain is None else f'{gain:+.3f}'})",
        ),
        (not left, f"no answerable question outside top {TOP_K_GUARD} (outside: {left or 'none'})"),
    ]
    passed = all(ok for ok, _ in checks)
    lines = [f"{other.label} vs {base.label}: {'PASS (replace base)' if passed else 'FAIL (keep base)'}"]
    lines += [f"  [{'x' if ok else ' '}] {text}" for ok, text in checks]
    return passed, lines


def render(runs: Sequence[EvalRun]) -> list[str]:
    """The full comparison report (ASCII)."""
    lines = ["Files:"]
    for index, (run, label) in enumerate(zip(runs, column_labels(runs))):
        git = (run.payload.get("run") or {}).get("git") or {}
        lines.append(
            f"  {'BASE ' if index == 0 else 'OTHER'} [{label}] {run.label}: {run.path.name} "
            f"(commit {str(git.get('commit'))[:10]}, dirty {git.get('dirty')})"
        )
    model = common_model(runs)
    if model is not None:
        lines.append(f"  (all runs use model {model}; columns show chunk_size_overlap)")
    lines += ["", "Metrics (delta vs BASE):", *metric_lines(runs)]
    lines += ["", "Ranks of answerable questions (vs BASE):", *rank_lines(runs)]
    lines += ["", "Similarity:", *similarity_lines(runs)]
    lines += ["", "Pre-registered decision rule:"]
    for other in runs[1:]:
        lines += rule_check(runs[0], other)[1]
    return [line.encode("ascii", "replace").decode("ascii") for line in lines]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare run_eval result files.")
    parser.add_argument("base", type=Path, help="baseline result JSON")
    parser.add_argument("others", type=Path, nargs="+", help="result JSONs to compare")
    args = parser.parse_args(argv)
    try:
        runs = [load_run(path) for path in [args.base, *args.others]]
        check_comparable(runs)
    except CompareError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2
    sys.stdout.write("\n".join(render(runs)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
