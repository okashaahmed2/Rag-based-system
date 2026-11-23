"""Diagnosis record (Phase 7 Step 3b): does reasoning leak into message.content?

MANUAL ONLY: not run by pytest and not part of the pipeline. Needs a running
Ollama with both models pulled and the existing eval store
``data/eval/stores/all-minilm-l6-v2_600_120``. It is kept as the record of
why ``granite4.2:3b`` was rejected as the default model.

For golden questions q02, q13 and q14 it builds the exact RAG messages once
(eval store read path, top_k 5, same settings overrides as
``scripts/rag_smoke.py``) and sends them as RAW ``/api/chat`` requests with
httpx (not ``llm.chat``) in four variants, so the raw JSON can be inspected:

    V1 granite4.2:3b  "think": false
    V2 granite4.2:3b  no "think" key
    V3 llama3.2:3b    no "think" key
    V4 llama3.2:3b    "think": false

Each request: stream false, temperature 0, num_ctx 4096, num_predict 2048,
timeout 300 s. Per run it reports HTTP status, prompt/completion tokens and
timings, whether ``message.thinking`` is non-empty, whether the content holds
``<think>`` / ``</think>``, the character counts before and after the last
``</think>``, the text after it, and ``check_answer`` on the raw content and
on the stripped text.

What it measured (2026-11-23, Ollama 0.40.17, CPU only):

* granite4.2:3b + ``think: false``: no ``thinking`` field, but the reasoning
  was written into ``content`` and ended with a bare ``</think>`` (q02 848 and
  q13 732 completion tokens, 55-105 s wall). Its chat template defaults
  ``enable_thinking`` to True and ends the prompt with ``<think>``.
* granite4.2:3b without ``think``: reasoning moved to ``message.thinking``,
  but was still generated (q02 hit the 2048-token limit with empty content).
* llama3.2:3b (with or without ``think: false``): clean answers of 3-38
  tokens, about 2 s generation, correct status on all three questions.

Nothing is written; output goes to stdout only (answer text included, which
is acceptable in this manual diagnosis script only).

Usage::

    .venv\\Scripts\\python.exe scripts\\llm_think_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import httpx

from rag_pipeline.answering import build_rag_messages, check_answer
from rag_pipeline.config import Settings
from rag_pipeline.evaluation import load_golden_set
from rag_pipeline.retrieval import build_retrieval_context, retrieve

EVAL_STORE = Path("data/eval/stores/all-minilm-l6-v2_600_120")
GOLDEN_SET = Path("eval/golden_set.jsonl")
QUESTION_IDS = ("q02", "q13", "q14")
TOP_K = 5
CHAT_URL = "http://127.0.0.1:11434/api/chat"
TAGS_URL = "http://127.0.0.1:11434/api/tags"
TIMEOUT_SECONDS = 300.0
OPTIONS = {"temperature": 0, "num_ctx": 4096, "num_predict": 2048}
#: (name, model, think value or None for "no think key")
VARIANTS: tuple[tuple[str, str, bool | None], ...] = (
    ("V1", "granite4.2:3b", False),
    ("V2", "granite4.2:3b", None),
    ("V3", "llama3.2:3b", None),
    ("V4", "llama3.2:3b", False),
)
CLOSE_TAG = "</think>"


def out(line: str = "") -> None:
    print(line, flush=True)


def installed_models(client: httpx.Client) -> set[str]:
    response = client.get(TAGS_URL)
    response.raise_for_status()
    return {m.get("name", "") for m in response.json().get("models", [])}


def run_variant(
    client: httpx.Client, model: str, think: bool | None, messages: list[dict[str, str]]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": OPTIONS,
    }
    if think is not None:
        body["think"] = think
    started = time.perf_counter()
    try:
        response = client.post(CHAT_URL, json=body)
    except httpx.TimeoutException as exc:
        return {"http": f"TIMEOUT {type(exc).__name__}", "wall": time.perf_counter() - started}
    wall = time.perf_counter() - started
    result: dict[str, Any] = {"http": response.status_code, "wall": wall}
    try:
        data = response.json()
    except ValueError:
        result["error"] = response.text[:300]
        return result
    if response.status_code != 200:
        result["error"] = str(data.get("error", data))[:300]
        return result
    message = data.get("message") or {}
    result.update(
        content=message.get("content") or "",
        thinking=message.get("thinking") or "",
        done_reason=data.get("done_reason"),
        prompt_tokens=data.get("prompt_eval_count", 0),
        prompt_eval_s=data.get("prompt_eval_duration", 0) / 1e9,
        completion_tokens=data.get("eval_count", 0),
        eval_s=data.get("eval_duration", 0) / 1e9,
    )
    return result


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    base = Settings()
    store = base.project_root / EVAL_STORE
    settings = Settings(chroma_path=store / "chroma", status_db_path=store / "status.db")
    questions = {q.id: q for q in load_golden_set(base.project_root / GOLDEN_SET)}
    context = build_retrieval_context(settings)
    prepared = {}
    for qid in QUESTION_IDS:
        chunks = retrieve(questions[qid].question, context, TOP_K)
        prepared[qid] = (build_rag_messages(questions[qid].question, chunks), len(chunks))
        out(f"{qid}: {len(chunks)} chunks, user message {len(prepared[qid][0][1]['content'])} chars")

    rows = []
    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        installed = installed_models(client)
        out(f"installed models: {sorted(installed)}")
        for qid in QUESTION_IDS:
            messages, n_sources = prepared[qid]
            for name, model, think in VARIANTS:
                label = f"{qid} {name} {model} think={'absent' if think is None else think}"
                if model not in installed:
                    out(f"\n=== {label}: SKIPPED (model not installed)")
                    rows.append((qid, name, {"http": "skipped"}, None, None, None))
                    continue
                r = run_variant(client, model, think, messages)
                out(f"\n=== {label}")
                if "content" not in r:
                    out(f"http={r['http']} wall={r['wall']:.2f}s error={r.get('error', '-')}")
                    rows.append((qid, name, r, None, None, None))
                    continue
                content: str = r["content"]
                has_open = "<think>" in content
                has_close = CLOSE_TAG in content
                if has_close:
                    cut = content.rfind(CLOSE_TAG)
                    before, after = content[:cut], content[cut + len(CLOSE_TAG):]
                else:
                    before, after = "", content
                raw_status = check_answer(content, n_sources).status.value
                after_status = check_answer(after, n_sources).status.value
                out(
                    f"http={r['http']} done_reason={r['done_reason']} "
                    f"prompt_tokens={r['prompt_tokens']} prompt_eval={r['prompt_eval_s']:.2f}s "
                    f"completion_tokens={r['completion_tokens']} eval={r['eval_s']:.2f}s "
                    f"wall={r['wall']:.2f}s"
                )
                out(
                    f"thinking_field_nonempty={bool(r['thinking'])} (len {len(r['thinking'])}) "
                    f"content_has_<think>={has_open} content_has_</think>={has_close} "
                    f"chars_before_last_</think>={len(before)} chars_after={len(after)}"
                )
                out(f"check_answer raw={raw_status} after_last_</think>={after_status}")
                out(f"text after last </think> (or whole content):\n{after.strip()}")
                rows.append((qid, name, r, (has_open, has_close, len(before), len(after)),
                             raw_status, after_status))

    out("\n=== TABLE")
    out(
        f"{'q':<4}{'var':<4}{'http':>8}{'p_tok':>7}{'p_s':>7}{'c_tok':>7}{'eval_s':>8}"
        f"{'wall_s':>8}  {'thinkF':<7}{'<think>':<8}{'</think>':<9}{'before':>7}{'after':>7}"
        f"  {'raw':<14}{'after':<14}"
    )
    for qid, name, r, tags, raw_status, after_status in rows:
        if tags is None:
            out(f"{qid:<4}{name:<4}{str(r['http']):>8}  {r.get('error', '')}")
            continue
        has_open, has_close, n_before, n_after = tags
        out(
            f"{qid:<4}{name:<4}{r['http']:>8}{r['prompt_tokens']:>7}{r['prompt_eval_s']:>7.2f}"
            f"{r['completion_tokens']:>7}{r['eval_s']:>8.2f}{r['wall']:>8.2f}  "
            f"{str(bool(r['thinking'])):<7}{str(has_open):<8}{str(has_close):<9}"
            f"{n_before:>7}{n_after:>7}  {raw_status:<14}{after_status:<14}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
