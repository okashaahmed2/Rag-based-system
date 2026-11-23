"""Manual smoke check against the REAL local Ollama server (not run by pytest).

Runs :func:`rag_pipeline.llm.check_ollama`, then one chat with a fixed system
and user message, and logs the model, timings, token counts, tokens/s,
``done_reason``, ``truncated`` and the answer text. Logging the answer is
acceptable here only because this is a manual script with a fixed,
non-confidential prompt; application code never logs answers.

Settings come from ``.env`` / ``RAG_*`` variables (``RAG_OLLAMA_BASE_URL``,
``RAG_LLM_MODEL``, ...). Ollama must already be running; this script never
starts it or pulls a model.

Usage::

    .venv\\Scripts\\python.exe scripts\\llm_smoke.py
"""

from __future__ import annotations

import logging
import sys
import time

from rag_pipeline.config import get_settings
from rag_pipeline.llm import LLMError, chat, check_ollama

logger = logging.getLogger("llm_smoke")

SYSTEM_MESSAGE = "Answer in one sentence."
USER_MESSAGE = "What is a decision tree?"


def _rate(tokens: int, seconds: float) -> float:
    """Tokens per second, 0.0 if no time was measured."""
    return tokens / seconds if seconds > 0 else 0.0


def main() -> int:
    """Run the smoke check; return 0 on success, 1 on any LLM error."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    logging.getLogger("rag_pipeline.llm").setLevel(logging.INFO)
    settings = get_settings()
    logger.info(
        "Settings: base_url=%s model=%s temperature=%s num_ctx=%d timeout=%.1f s",
        settings.ollama_base_url,
        settings.llm_model,
        settings.llm_temperature,
        settings.llm_num_ctx,
        settings.llm_timeout_seconds,
    )

    try:
        info = check_ollama(settings)
        logger.info(
            "check_ollama: version=%s model_present=%s installed=%s",
            info["version"],
            info["model_present"],
            info["installed_models"],
        )
        started = time.perf_counter()
        result = chat(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": USER_MESSAGE},
            ],
            settings,
        )
        wall = time.perf_counter() - started
    except LLMError as exc:
        logger.error("FAIL: %s: %s", type(exc).__name__, exc)
        return 1

    logger.info(
        "model=%s done_reason=%s truncated=%s",
        result.model,
        result.done_reason,
        result.truncated,
    )
    logger.info(
        "load %.2f s | prompt %d tok in %.3f s (%.1f tok/s) | "
        "generation %d tok in %.2f s (%.1f tok/s) | total %.2f s (wall %.2f s)",
        result.load_seconds,
        result.prompt_tokens,
        result.prompt_eval_seconds,
        _rate(result.prompt_tokens, result.prompt_eval_seconds),
        result.completion_tokens,
        result.eval_seconds,
        _rate(result.completion_tokens, result.eval_seconds),
        result.total_seconds,
        wall,
    )
    logger.info("answer: %s", result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
