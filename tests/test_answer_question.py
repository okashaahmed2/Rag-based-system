"""Tests for rag_pipeline.answering.answer_question and RAG_QUERY_TOP_K.

Only fake ``retrieve_fn`` / ``chat_fn`` are used: no embedding model, no
vector store, no Ollama.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import pytest
from pydantic import ValidationError

from rag_pipeline.answering import (
    NOT_FOUND_MESSAGE,
    AnswerStatus,
    Citation,
    RagAnswer,
    answer_question,
    build_rag_messages,
)
from rag_pipeline.config import Settings
from rag_pipeline.llm import LLMResponse, LLMTimeoutError, LLMUnavailableError
from rag_pipeline.retrieval import RetrievalContext, RetrievalError
from rag_pipeline.vector_store import RetrievedChunk

QUESTION = "QUESTION-SECRET what is a decision tree?"
CONTEXT = cast(RetrievalContext, object())


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_chunk(n: int) -> RetrievedChunk:
    doc_id = f"{n:x}" * 64
    return RetrievedChunk(
        chunk_id=f"{doc_id[:64]}:{n}",
        doc_id=doc_id[:64],
        filename=f"doc{n}.pdf",
        page_number=10 + n,
        chunk_index=n,
        text=f"CHUNK-SECRET-{n} decision trees split data.",
        distance=0.1 * n,
        similarity=1 - 0.1 * n,
    )


CHUNKS = [make_chunk(n) for n in (1, 2, 3)]


def make_response(
    text: str,
    *,
    prompt_tokens: int = 900,
    completion_tokens: int = 40,
    truncated: bool = False,
    thinking_leaked: bool = False,
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="granite4.2:3b",
        done_reason="length" if truncated else "stop",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        load_seconds=0.5,
        prompt_eval_seconds=7.5,
        eval_seconds=2.0,
        total_seconds=10.0,
        truncated=truncated,
        thinking_leaked=thinking_leaked,
    )


class FakeRetrieve:
    def __init__(self, chunks: Sequence[RetrievedChunk] = CHUNKS, exc: Exception | None = None):
        self.chunks = list(chunks)
        self.exc = exc
        self.calls: list[tuple[str, RetrievalContext, int]] = []

    def __call__(self, question: str, context: RetrievalContext, top_k: int) -> list[RetrievedChunk]:
        self.calls.append((question, context, top_k))
        if self.exc is not None:
            raise self.exc
        return self.chunks


class FakeChat:
    def __init__(self, response: LLMResponse | None = None, exc: Exception | None = None):
        self.response = response if response is not None else make_response("Trees [S1].")
        self.exc = exc
        self.calls: list[tuple[list[dict[str, str]], Settings]] = []

    def __call__(self, messages: list[dict[str, str]], settings: Settings) -> LLMResponse:
        self.calls.append((messages, settings))
        if self.exc is not None:
            raise self.exc
        return self.response


def run(
    retrieve_fn: FakeRetrieve | None = None,
    chat_fn: FakeChat | None = None,
    settings: Settings | None = None,
    **kwargs: object,
) -> RagAnswer:
    return answer_question(
        QUESTION,
        CONTEXT,
        settings if settings is not None else make_settings(),
        retrieve_fn=retrieve_fn if retrieve_fn is not None else FakeRetrieve(),
        chat_fn=chat_fn if chat_fn is not None else FakeChat(),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- settings


def test_query_top_k_default() -> None:
    assert make_settings().query_top_k == 5


@pytest.mark.parametrize("value", ["1", "20"])
def test_query_top_k_bounds_accepted(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_QUERY_TOP_K", value)

    assert make_settings().query_top_k == int(value)


@pytest.mark.parametrize("bad", ["0", "-1", "21", "100"])
def test_invalid_query_top_k_raises(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_QUERY_TOP_K", bad)

    with pytest.raises(ValidationError, match="RAG_QUERY_TOP_K must be between 1 and 20"):
        make_settings()


def test_llm_max_answer_tokens_default() -> None:
    assert make_settings().llm_max_answer_tokens == 256


@pytest.mark.parametrize("value", ["16", "2048"])
def test_llm_max_answer_tokens_bounds_accepted(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_ANSWER_TOKENS", value)

    assert make_settings().llm_max_answer_tokens == int(value)


@pytest.mark.parametrize("bad", ["15", "0", "-1", "2049"])
def test_invalid_llm_max_answer_tokens_raises(bad: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_ANSWER_TOKENS", bad)

    with pytest.raises(
        ValidationError, match="RAG_LLM_MAX_ANSWER_TOKENS must be between 16 and 2048"
    ):
        make_settings()


def test_non_integer_llm_max_answer_tokens_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_ANSWER_TOKENS", "lots")

    with pytest.raises(ValidationError):
        make_settings()


def test_llm_model_default_is_llama3_2_3b() -> None:
    assert make_settings().llm_model == "llama3.2:3b"


def test_non_integer_query_top_k_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_QUERY_TOP_K", "five")

    with pytest.raises(ValidationError):
        make_settings()


# --------------------------------------------------------------------------- top_k / calls


def test_top_k_default_comes_from_settings() -> None:
    retrieve_fn = FakeRetrieve()

    result = run(retrieve_fn, settings=make_settings(query_top_k=7))

    assert retrieve_fn.calls == [(QUESTION, CONTEXT, 7)]
    assert result.top_k == 7


def test_explicit_top_k_overrides_settings() -> None:
    retrieve_fn = FakeRetrieve()

    result = run(retrieve_fn, settings=make_settings(query_top_k=7), top_k=3)

    assert retrieve_fn.calls == [(QUESTION, CONTEXT, 3)]
    assert result.top_k == 3


def test_messages_passed_to_chat_equal_build_rag_messages() -> None:
    chat_fn = FakeChat()
    settings = make_settings()

    run(chat_fn=chat_fn, settings=settings)

    assert len(chat_fn.calls) == 1
    messages, passed_settings = chat_fn.calls[0]
    assert messages == build_rag_messages(QUESTION, CHUNKS)
    assert passed_settings is settings


# --------------------------------------------------------------------------- statuses


def test_no_chunks_returns_not_found_without_calling_chat() -> None:
    chat_fn = FakeChat()

    result = run(FakeRetrieve(chunks=[]), chat_fn)

    assert chat_fn.calls == []
    assert result == RagAnswer(
        status=AnswerStatus.NOT_FOUND,
        answer_text=NOT_FOUND_MESSAGE,
        citations=(),
        sources=(),
        invalid_citations=(),
        top_k=5,
        llm_called=False,
        prompt_tokens=0,
        completion_tokens=0,
        load_seconds=0.0,
        prompt_eval_seconds=0.0,
        eval_seconds=0.0,
        total_seconds=0.0,
        truncated=False,
        thinking_leaked=False,
    )


def test_answered_keeps_markers_and_maps_citations() -> None:
    text = "Trees split data [S2]. They use features [S2][S3]."

    result = run(chat_fn=FakeChat(make_response(f"  {text}\n")))

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer_text == text
    assert result.llm_called is True
    assert [c.source_number for c in result.citations] == [2, 3]
    assert result.invalid_citations == ()


def test_citation_s2_maps_to_second_chunk() -> None:
    result = run(chat_fn=FakeChat(make_response("Trees [S2].")))

    second = CHUNKS[1]
    assert result.citations == (
        Citation(
            source_number=2,
            doc_id=second.doc_id,
            filename=second.filename,
            page_number=second.page_number,
            chunk_index=second.chunk_index,
            similarity=second.similarity,
        ),
    )
    assert (result.citations[0].filename, result.citations[0].page_number) == ("doc2.pdf", 12)


def test_sources_list_every_chunk_in_order() -> None:
    result = run(chat_fn=FakeChat(make_response("Trees [S1].")))

    assert [(s.source_number, s.filename, s.page_number) for s in result.sources] == [
        (1, "doc1.pdf", 11),
        (2, "doc2.pdf", 12),
        (3, "doc3.pdf", 13),
    ]


def test_citations_are_in_source_number_order() -> None:
    result = run(chat_fn=FakeChat(make_response("A [S3]. B [S1].")))

    assert [c.source_number for c in result.citations] == [1, 3]


@pytest.mark.parametrize("text", ["NOT_FOUND", "**NOT_FOUND**.", "NOT_FOUND - not in sources"])
def test_model_not_found_uses_message(text: str) -> None:
    result = run(chat_fn=FakeChat(make_response(text)))

    assert result.status is AnswerStatus.NOT_FOUND
    assert result.answer_text == NOT_FOUND_MESSAGE
    assert result.llm_called is True
    assert result.citations == ()
    assert len(result.sources) == 3


def test_no_citations_keeps_model_text() -> None:
    result = run(chat_fn=FakeChat(make_response("Trees split data, see [12].")))

    assert result.status is AnswerStatus.NO_CITATIONS
    assert result.answer_text == "Trees split data, see [12]."
    assert result.citations == ()


def test_bad_citations_keep_text_and_valid_citations() -> None:
    result = run(chat_fn=FakeChat(make_response("Trees [S1][S4][S9].")))

    assert result.status is AnswerStatus.BAD_CITATIONS
    assert result.answer_text == "Trees [S1][S4][S9]."
    assert [c.source_number for c in result.citations] == [1]
    assert result.invalid_citations == (4, 9)


def test_empty_answer() -> None:
    result = run(chat_fn=FakeChat(make_response("  \n ")))

    assert result.status is AnswerStatus.EMPTY
    assert result.answer_text == ""
    assert result.llm_called is True


def test_llm_counts_and_timings_copied() -> None:
    result = run(chat_fn=FakeChat(make_response("Trees [S1].", truncated=True)))

    assert (result.prompt_tokens, result.completion_tokens) == (900, 40)
    assert (result.load_seconds, result.prompt_eval_seconds, result.eval_seconds) == (0.5, 7.5, 2.0)
    assert result.total_seconds == 10.0
    assert result.truncated is True
    assert result.thinking_leaked is False


@pytest.mark.parametrize("leaked", [True, False])
def test_thinking_leaked_copied_from_llm_response(leaked: bool) -> None:
    result = run(chat_fn=FakeChat(make_response("Trees [S1].", thinking_leaked=leaked)))

    assert result.thinking_leaked is leaked


def test_not_found_on_last_line_after_explanation() -> None:
    text = "The sources do not discuss PCA [S2].\n\nNOT_FOUND"

    result = run(chat_fn=FakeChat(make_response(text)))

    assert result.status is AnswerStatus.NOT_FOUND
    assert result.answer_text == NOT_FOUND_MESSAGE
    assert result.citations == ()


# --------------------------------------------------------------------------- warnings / logs


def test_context_full_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.answering")
    response = make_response("Trees [S1].", prompt_tokens=4000, completion_tokens=96)

    run(chat_fn=FakeChat(response), settings=make_settings(llm_num_ctx=4096))

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "context may be full" in warnings[0].getMessage()


def test_no_warning_below_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.answering")
    response = make_response("Trees [S1].", prompt_tokens=4000, completion_tokens=95)

    run(chat_fn=FakeChat(response), settings=make_settings(llm_num_ctx=4096))

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    "answer", ["ANSWER-SECRET [S1][S2].", "ANSWER-SECRET no cites", "ANSWER-SECRET [S9]", "NOT_FOUND"]
)
def test_no_question_chunk_or_answer_text_in_logs(
    answer: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    response = make_response(answer, prompt_tokens=5000)

    run(chat_fn=FakeChat(response))

    assert caplog.records
    assert "QUESTION-SECRET" not in caplog.text
    assert "CHUNK-SECRET" not in caplog.text
    assert "ANSWER-SECRET" not in caplog.text


def test_one_info_line_per_answer(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="rag_pipeline.answering")

    run(chat_fn=FakeChat(make_response("Trees [S2].")))

    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert "status=answered" in infos[0]
    assert "3 sources" in infos[0]
    assert "cited=[2]" in infos[0]


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "exc", [LLMUnavailableError("down"), LLMTimeoutError("slow"), RuntimeError("other")]
)
def test_chat_errors_propagate(exc: Exception) -> None:
    with pytest.raises(type(exc)) as excinfo:
        run(chat_fn=FakeChat(exc=exc))

    assert excinfo.value is exc


def test_retrieval_errors_propagate_and_chat_not_called() -> None:
    exc = RetrievalError("store broken")
    chat_fn = FakeChat()

    with pytest.raises(RetrievalError) as excinfo:
        run(FakeRetrieve(exc=exc), chat_fn)

    assert excinfo.value is exc
    assert chat_fn.calls == []


def test_empty_question_with_chunks_raises_value_error() -> None:
    chat_fn = FakeChat()

    with pytest.raises(ValueError, match="question must be a non-empty string"):
        answer_question("  ", CONTEXT, make_settings(), retrieve_fn=FakeRetrieve(), chat_fn=chat_fn)

    assert chat_fn.calls == []
