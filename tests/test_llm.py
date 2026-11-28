"""Tests for rag_pipeline.llm and the LLM settings (httpx.MockTransport only, no network)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from rag_pipeline.config import Settings
from rag_pipeline.llm import (
    LLMError,
    LLMModelNotFoundError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    build_chat_payload,
    chat,
    check_ollama,
    validate_messages,
)

PROMPT_SECRET = "PROMPT-SECRET-what-is-a-decision-tree"
SYSTEM_SECRET = "SYSTEM-SECRET-answer-in-one-sentence"
ANSWER_SECRET = "ANSWER-SECRET-a-tree-of-decisions"
THINKING_SECRET = "THINKING-SECRET-let-me-reason"

MESSAGES = [
    {"role": "system", "content": SYSTEM_SECRET},
    {"role": "user", "content": PROMPT_SECRET},
]

Handler = Callable[[httpx.Request], httpx.Response]


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


class Recorder:
    """MockTransport handler that records requests and replays one handler."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def chat_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "granite4.2:3b",
        "created_at": "2026-11-23T10:00:00Z",
        "message": {"role": "assistant", "content": ANSWER_SECRET},
        "done": True,
        "done_reason": "stop",
        "total_duration": 7_410_000_000,
        "load_duration": 2_590_000_000,
        "prompt_eval_count": 748,
        "prompt_eval_duration": 6_327_000_000,
        "eval_count": 18,
        "eval_duration": 1_070_000_000,
    }
    body.update(overrides)
    return body


def json_handler(body: Any, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, json=body)


def raising_handler(exc_type: type[httpx.TransportError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc_type("simulated", request=request)

    return handler


# --------------------------------------------------------------------------- settings


def test_llm_settings_defaults() -> None:
    settings = make_settings()

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.llm_model == "llama3.2:3b"
    assert settings.llm_temperature == 0.0
    assert settings.llm_num_ctx == 4096
    assert settings.llm_timeout_seconds == 120.0


def test_explicit_localhost_base_url_is_kept_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434")

    assert make_settings().ollama_base_url == "http://localhost:11434"


def test_llm_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OLLAMA_BASE_URL", " http://127.0.0.1:9999/ ")
    monkeypatch.setenv("RAG_LLM_MODEL", "  llama3.2:3b  ")
    monkeypatch.setenv("RAG_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("RAG_LLM_NUM_CTX", "512")
    monkeypatch.setenv("RAG_LLM_TIMEOUT_SECONDS", "30.5")

    settings = make_settings()

    assert settings.ollama_base_url == "http://127.0.0.1:9999"
    assert settings.llm_model == "llama3.2:3b"
    assert settings.llm_temperature == 0.7
    assert settings.llm_num_ctx == 512
    assert settings.llm_timeout_seconds == 30.5


@pytest.mark.parametrize("url", ["http://localhost:11434//", "HTTPS://host:1/"])
def test_base_url_trailing_slashes_removed(url: str) -> None:
    assert not make_settings(ollama_base_url=url).ollama_base_url.endswith("/")


@pytest.mark.parametrize("bad_url", ["", "   ", "localhost:11434", "ftp://host", "http://", "/"])
def test_invalid_base_url_raises(bad_url: str) -> None:
    with pytest.raises(ValidationError, match="RAG_OLLAMA_BASE_URL must be an http"):
        make_settings(ollama_base_url=bad_url)


@pytest.mark.parametrize("bad_model", ["", "   "])
def test_empty_llm_model_raises(bad_model: str) -> None:
    with pytest.raises(ValidationError, match="RAG_LLM_MODEL must not be empty"):
        make_settings(llm_model=bad_model)


@pytest.mark.parametrize("value", [0.0, 1.0, 2.0])
def test_temperature_bounds_accepted(value: float) -> None:
    assert make_settings(llm_temperature=value).llm_temperature == value


@pytest.mark.parametrize("bad_value", [-0.01, 2.01, float("nan")])
def test_invalid_temperature_raises(bad_value: float) -> None:
    with pytest.raises(ValidationError, match="RAG_LLM_TEMPERATURE must be between 0 and 2"):
        make_settings(llm_temperature=bad_value)


@pytest.mark.parametrize("bad_value", [511, 0, -1])
def test_invalid_num_ctx_raises(bad_value: int) -> None:
    with pytest.raises(ValidationError, match="RAG_LLM_NUM_CTX must be >= 512"):
        make_settings(llm_num_ctx=bad_value)


@pytest.mark.parametrize("bad_value", [0, -1, float("nan")])
def test_invalid_timeout_raises(bad_value: float) -> None:
    with pytest.raises(ValidationError, match="RAG_LLM_TIMEOUT_SECONDS must be > 0"):
        make_settings(llm_timeout_seconds=bad_value)


@pytest.mark.parametrize(
    ("name", "value"),
    [("RAG_LLM_NUM_CTX", "4k"), ("RAG_LLM_TEMPERATURE", "warm"), ("RAG_LLM_TIMEOUT_SECONDS", "")],
)
def test_non_numeric_llm_settings_raise(
    name: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        make_settings()


# --------------------------------------------------------------------------- chat request


def test_chat_request_body_is_exactly_as_expected() -> None:
    recorder = Recorder(json_handler(chat_body()))
    settings = make_settings(llm_temperature=0.3, llm_num_ctx=2048, llm_max_answer_tokens=128)

    chat(MESSAGES, settings, client=recorder.client())

    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:11434/api/chat"
    assert json.loads(request.content) == {
        "model": "llama3.2:3b",
        "messages": MESSAGES,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 128},
    }


def test_chat_uses_configured_base_url_and_model() -> None:
    recorder = Recorder(json_handler(chat_body(model="other:1b")))
    settings = make_settings(ollama_base_url="http://10.0.0.5:8080/", llm_model="other:1b")

    chat(MESSAGES, settings, client=recorder.client())

    request = recorder.requests[0]
    assert str(request.url) == "http://10.0.0.5:8080/api/chat"
    assert json.loads(request.content)["model"] == "other:1b"


def test_chat_copies_messages_without_mutating_input() -> None:
    recorder = Recorder(json_handler(chat_body()))
    messages = [{"role": "user", "content": PROMPT_SECRET}]

    chat(messages, make_settings(), client=recorder.client())

    assert messages == [{"role": "user", "content": PROMPT_SECRET}]


# --------------------------------------------------------------------------- chat parsing


def test_chat_parses_success_and_converts_ns_to_seconds() -> None:
    recorder = Recorder(json_handler(chat_body()))

    result = chat(MESSAGES, make_settings(), client=recorder.client())

    assert result == LLMResponse(
        text=ANSWER_SECRET,
        model="granite4.2:3b",
        done_reason="stop",
        prompt_tokens=748,
        completion_tokens=18,
        load_seconds=pytest.approx(2.59),
        prompt_eval_seconds=pytest.approx(6.327),
        eval_seconds=pytest.approx(1.07),
        total_seconds=pytest.approx(7.41),
        truncated=False,
        thinking_leaked=False,
    )


def test_default_num_predict_is_max_answer_tokens() -> None:
    recorder = Recorder(json_handler(chat_body()))

    chat(MESSAGES, make_settings(), client=recorder.client())

    assert json.loads(recorder.requests[0].content)["options"]["num_predict"] == 256


# --------------------------------------------------------------------------- format / num_predict

#: The request body every chat sent before ``format`` / ``num_predict`` existed.
LEGACY_BODY = {
    "model": "llama3.2:3b",
    "messages": MESSAGES,
    "stream": False,
    "think": False,
    "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 128},
}
LEGACY_SETTINGS = {"llm_temperature": 0.3, "llm_num_ctx": 2048, "llm_max_answer_tokens": 128}
SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
    "required": ["items"],
    "$defs": {"Item": {"type": "object", "properties": {"term": {"type": "string"}}}},
}


def test_payload_without_new_args_equals_legacy_payload() -> None:
    payload = build_chat_payload(list(MESSAGES), make_settings(**LEGACY_SETTINGS))

    assert payload == LEGACY_BODY
    assert list(payload) == list(LEGACY_BODY)  # same key order
    assert list(payload["options"]) == ["temperature", "num_ctx", "num_predict"]
    assert "format" not in payload


def test_payload_with_explicit_none_equals_legacy_payload() -> None:
    payload = build_chat_payload(
        list(MESSAGES), make_settings(**LEGACY_SETTINGS), format=None, num_predict=None
    )

    assert payload == LEGACY_BODY


def test_request_bytes_without_new_args_are_byte_identical() -> None:
    recorder = Recorder(json_handler(chat_body()))

    chat(MESSAGES, make_settings(**LEGACY_SETTINGS), client=recorder.client())

    expected = httpx.Request("POST", "http://127.0.0.1:11434/api/chat", json=LEGACY_BODY)
    assert recorder.requests[0].content == expected.content


def test_format_json_string_is_sent_unchanged() -> None:
    recorder = Recorder(json_handler(chat_body()))

    chat(MESSAGES, make_settings(**LEGACY_SETTINGS), client=recorder.client(), format="json")

    body = json.loads(recorder.requests[0].content)
    assert body == {**LEGACY_BODY, "format": "json"}


def test_format_schema_dict_is_sent_unchanged() -> None:
    recorder = Recorder(json_handler(chat_body()))
    schema = json.loads(json.dumps(SCHEMA))

    chat(MESSAGES, make_settings(**LEGACY_SETTINGS), client=recorder.client(), format=schema)

    body = json.loads(recorder.requests[0].content)
    assert body["format"] == SCHEMA
    assert {k: v for k, v in body.items() if k != "format"} == LEGACY_BODY
    assert schema == SCHEMA  # input not mutated


def test_num_predict_override_replaces_only_num_predict() -> None:
    recorder = Recorder(json_handler(chat_body()))

    chat(MESSAGES, make_settings(**LEGACY_SETTINGS), client=recorder.client(), num_predict=768)

    body = json.loads(recorder.requests[0].content)
    assert body["options"] == {"temperature": 0.3, "num_ctx": 2048, "num_predict": 768}
    assert "format" not in body


def test_format_and_num_predict_together() -> None:
    payload = build_chat_payload(
        list(MESSAGES), make_settings(**LEGACY_SETTINGS), format=SCHEMA, num_predict=1
    )

    assert payload["format"] is SCHEMA
    assert payload["options"]["num_predict"] == 1


@pytest.mark.parametrize(
    "bad_format", ["", "JSON", "yaml", " json", {}, [], ["json"], 1, True, b"json"]
)
def test_invalid_format_raises_before_any_request(bad_format: Any) -> None:
    recorder = Recorder(json_handler(chat_body()))

    with pytest.raises(ValueError, match="format must be"):
        chat(MESSAGES, make_settings(), client=recorder.client(), format=bad_format)

    assert recorder.requests == []


@pytest.mark.parametrize("bad_value", [0, -1, True, False, 1.5, "768", 768.0])
def test_invalid_num_predict_raises_before_any_request(bad_value: Any) -> None:
    recorder = Recorder(json_handler(chat_body()))

    with pytest.raises(ValueError, match="num_predict must be an int >= 1"):
        chat(MESSAGES, make_settings(), client=recorder.client(), num_predict=bad_value)

    assert recorder.requests == []


# --------------------------------------------------------------------------- leaked thinking


def content_body(content: str) -> dict[str, Any]:
    return chat_body(message={"role": "assistant", "content": content})


@pytest.mark.parametrize(
    ("content", "expected_text", "leaked"),
    [
        (f"{THINKING_SECRET} reasoning\n</think>\n{ANSWER_SECRET} [S1]  ", f"{ANSWER_SECRET} [S1]", True),
        (
            f"<think>{THINKING_SECRET}</think>draft</think>\n  {ANSWER_SECRET}\n",
            ANSWER_SECRET,
            True,
        ),
        (f"<think>\n{THINKING_SECRET} never closed", "", True),
        (f"{THINKING_SECRET}</think>", "", True),
        (f"  {ANSWER_SECRET} [S1]\n", f"  {ANSWER_SECRET} [S1]\n", False),
        ("a < b and <thinking> is not a tag", "a < b and <thinking> is not a tag", False),
    ],
)
def test_leaked_thinking_is_stripped(content: str, expected_text: str, leaked: bool) -> None:
    result = chat(MESSAGES, make_settings(), client=Recorder(json_handler(content_body(content))).client())

    assert result.text == expected_text
    assert result.thinking_leaked is leaked


def test_leaked_thinking_warning_has_counts_only(caplog: pytest.LogCaptureFixture) -> None:
    content = f"{THINKING_SECRET}\n</think>\n{ANSWER_SECRET}"
    caplog.set_level(logging.DEBUG)

    chat(MESSAGES, make_settings(), client=Recorder(json_handler(content_body(content))).client())

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"({len(content)} chars)" in warnings[0]
    assert f"kept the {len(ANSWER_SECRET)} chars" in warnings[0]
    assert THINKING_SECRET not in caplog.text
    assert ANSWER_SECRET not in caplog.text


def test_no_leak_warning_without_tags(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.llm")

    result = chat(MESSAGES, make_settings(), client=Recorder(json_handler(chat_body())).client())

    assert result.thinking_leaked is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_thinking_field_and_leak_are_independent(caplog: pytest.LogCaptureFixture) -> None:
    body = chat_body(
        message={"role": "assistant", "content": f"x</think>{ANSWER_SECRET}", "thinking": "t"}
    )
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.llm")

    result = chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())

    assert result.text == ANSWER_SECRET
    assert result.thinking_leaked is True
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


def test_llm_response_is_frozen() -> None:
    result = chat(MESSAGES, make_settings(), client=Recorder(json_handler(chat_body())).client())

    with pytest.raises(AttributeError):
        result.text = "changed"  # type: ignore[misc]


def test_missing_counts_and_durations_default_to_zero() -> None:
    body = {"model": "granite4.2:3b", "message": {"role": "assistant", "content": "x"}}

    result = chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())

    assert (result.prompt_tokens, result.completion_tokens) == (0, 0)
    assert result.total_seconds == 0.0
    assert result.done_reason is None
    assert result.truncated is False


def test_thinking_field_is_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    body = chat_body(
        message={"role": "assistant", "content": ANSWER_SECRET, "thinking": THINKING_SECRET}
    )
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.llm")

    result = chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())

    assert result.text == ANSWER_SECRET
    assert THINKING_SECRET not in result.text
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"({len(THINKING_SECRET)} chars)" in warnings[0].getMessage()
    assert THINKING_SECRET not in caplog.text


def test_empty_thinking_field_gives_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    body = chat_body(message={"role": "assistant", "content": "x", "thinking": ""})
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.llm")

    chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_done_reason_length_sets_truncated_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="rag_pipeline.llm")

    result = chat(
        MESSAGES,
        make_settings(),
        client=Recorder(json_handler(chat_body(done_reason="length"))).client(),
    )

    assert result.truncated is True
    assert result.done_reason == "length"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "cut" in warnings[0].getMessage()


def test_no_prompt_or_answer_text_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    body = chat_body(
        done_reason="length",
        message={"role": "assistant", "content": ANSWER_SECRET, "thinking": THINKING_SECRET},
    )
    caplog.set_level(logging.DEBUG)

    chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())

    assert caplog.records, "expected log records"
    for secret in (PROMPT_SECRET, SYSTEM_SECRET, ANSWER_SECRET, THINKING_SECRET):
        assert secret not in caplog.text


def test_no_prompt_text_in_logs_on_errors(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    handlers: list[Handler] = [
        raising_handler(httpx.ConnectError),
        raising_handler(httpx.ReadTimeout),
        json_handler({"error": "boom"}, status=500),
        json_handler({"error": "model not found"}, status=404),
        lambda request: httpx.Response(200, content=b"not json"),
    ]
    for handler in handlers:
        with pytest.raises(LLMError):
            chat(MESSAGES, make_settings(), client=Recorder(handler).client())

    for secret in (PROMPT_SECRET, SYSTEM_SECRET):
        assert secret not in caplog.text


# --------------------------------------------------------------------------- chat errors


@pytest.mark.parametrize("exc_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError])
def test_connection_errors_raise_unavailable(exc_type: type[httpx.TransportError]) -> None:
    recorder = Recorder(raising_handler(exc_type))

    with pytest.raises(LLMUnavailableError, match=r"Ollama is not running at http://127\.0\.0\.1:11434"):
        chat(MESSAGES, make_settings(), client=recorder.client())


@pytest.mark.parametrize("exc_type", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout])
def test_timeouts_raise_llm_timeout_error(exc_type: type[httpx.TransportError]) -> None:
    recorder = Recorder(raising_handler(exc_type))

    with pytest.raises(LLMTimeoutError, match="within 120.0 s"):
        chat(MESSAGES, make_settings(), client=recorder.client())


def test_404_raises_model_not_found_with_pull_hint() -> None:
    recorder = Recorder(json_handler({"error": "model 'llama3.2:3b' not found"}, status=404))

    with pytest.raises(LLMModelNotFoundError, match=r"run: ollama pull llama3\.2:3b"):
        chat(MESSAGES, make_settings(), client=recorder.client())


def test_500_raises_response_error_with_detail() -> None:
    recorder = Recorder(json_handler({"error": "out of memory"}, status=500))

    with pytest.raises(LLMResponseError, match="HTTP 500: out of memory"):
        chat(MESSAGES, make_settings(), client=recorder.client())


def test_error_detail_is_shortened() -> None:
    recorder = Recorder(lambda request: httpx.Response(500, text="x" * 5000))

    with pytest.raises(LLMResponseError) as excinfo:
        chat(MESSAGES, make_settings(), client=recorder.client())

    assert len(str(excinfo.value)) < 300


def test_invalid_json_raises_response_error() -> None:
    recorder = Recorder(lambda request: httpx.Response(200, content=b"{not json"))

    with pytest.raises(LLMResponseError, match="invalid JSON"):
        chat(MESSAGES, make_settings(), client=recorder.client())


def test_non_object_json_raises_response_error() -> None:
    with pytest.raises(LLMResponseError, match="not an object"):
        chat(MESSAGES, make_settings(), client=Recorder(json_handler([1, 2])).client())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"message": None}, "no 'message' object"),
        ({"message": "text"}, "no 'message' object"),
        ({"message": {"role": "assistant"}}, "message.content"),
        ({"message": {"role": "assistant", "content": 5}}, "message.content"),
        ({"model": None}, "no 'model' name"),
        ({"done_reason": 3}, "'done_reason' is not a string"),
        ({"eval_count": "18"}, "'eval_count' is not a count"),
        ({"total_duration": -1}, "'total_duration' is not a count"),
        ({"prompt_eval_count": True}, "'prompt_eval_count' is not a count"),
    ],
)
def test_bad_fields_raise_response_error(overrides: dict[str, Any], message: str) -> None:
    body = chat_body(**overrides)

    with pytest.raises(LLMResponseError, match=message):
        chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())


@pytest.mark.parametrize("field", ["message", "model"])
def test_absent_required_field_raises_response_error(field: str) -> None:
    body = chat_body()
    del body[field]

    with pytest.raises(LLMResponseError):
        chat(MESSAGES, make_settings(), client=Recorder(json_handler(body)).client())


def test_all_errors_share_llm_error_base() -> None:
    for exc_type in (LLMUnavailableError, LLMModelNotFoundError, LLMTimeoutError, LLMResponseError):
        assert issubclass(exc_type, LLMError)


# --------------------------------------------------------------------------- message validation


@pytest.mark.parametrize(
    ("messages", "message"),
    [
        ([], "messages must not be empty"),
        ([{"role": "tool", "content": "x"}], "invalid role 'tool'"),
        ([{"role": "USER", "content": "x"}], "invalid role 'USER'"),
        ([{"content": "x"}], "invalid role None"),
        ([{"role": "user", "content": ""}], "non-empty string content"),
        ([{"role": "user", "content": " \n\t"}], "non-empty string content"),
        ([{"role": "user", "content": None}], "non-empty string content"),
        ([{"role": "user"}], "non-empty string content"),
        ([{"role": "user", "content": "x", "images": []}], "unknown keys"),
        (["just a string"], "must be a mapping"),
        (
            [{"role": "system", "content": "ok"}, {"role": "user", "content": ""}],
            "message 1",
        ),
    ],
)
def test_invalid_messages_raise_before_any_request(messages: Any, message: str) -> None:
    recorder = Recorder(json_handler(chat_body()))

    with pytest.raises(ValueError, match=message):
        chat(messages, make_settings(), client=recorder.client())

    assert recorder.requests == []


def test_validate_messages_accepts_all_roles() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]

    assert validate_messages(messages) == messages


# --------------------------------------------------------------------------- check_ollama


def tags_router(models: list[dict[str, Any]], version: str = "0.40.17") -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": version})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": models})
        return httpx.Response(404, json={"error": "unexpected path"})

    return handler


def test_check_ollama_model_present() -> None:
    recorder = Recorder(tags_router([{"name": "nomic:latest"}, {"name": "llama3.2:3b"}]))

    info = check_ollama(make_settings(), client=recorder.client())

    assert info == {
        "base_url": "http://127.0.0.1:11434",
        "version": "0.40.17",
        "model": "llama3.2:3b",
        "model_present": True,
        "installed_models": ["nomic:latest", "llama3.2:3b"],
    }
    assert [(r.method, str(r.url)) for r in recorder.requests] == [
        ("GET", "http://127.0.0.1:11434/api/version"),
        ("GET", "http://127.0.0.1:11434/api/tags"),
    ]


def test_check_ollama_untagged_model_matches_latest() -> None:
    recorder = Recorder(tags_router([{"name": "llama3.2:latest"}]))

    info = check_ollama(make_settings(llm_model="llama3.2"), client=recorder.client())

    assert info["model_present"] is True


def test_check_ollama_tag_must_match_exactly() -> None:
    recorder = Recorder(tags_router([{"name": "llama3.2:latest"}]))

    with pytest.raises(LLMModelNotFoundError):
        check_ollama(make_settings(), client=recorder.client())


def test_check_ollama_missing_model() -> None:
    recorder = Recorder(tags_router([{"name": "granite4.2:3b"}]))

    with pytest.raises(LLMModelNotFoundError, match=r"run: ollama pull llama3\.2:3b"):
        check_ollama(make_settings(), client=recorder.client())


def test_check_ollama_no_models_installed() -> None:
    with pytest.raises(LLMModelNotFoundError):
        check_ollama(make_settings(), client=Recorder(tags_router([])).client())


def test_check_ollama_unreachable() -> None:
    recorder = Recorder(raising_handler(httpx.ConnectError))

    with pytest.raises(LLMUnavailableError, match=r"Ollama is not running at http://127\.0\.0\.1:11434"):
        check_ollama(make_settings(), client=recorder.client())


def test_check_ollama_timeout() -> None:
    with pytest.raises(LLMTimeoutError):
        check_ollama(make_settings(), client=Recorder(raising_handler(httpx.ReadTimeout)).client())


@pytest.mark.parametrize(
    "handler",
    [
        json_handler({"error": "boom"}, status=500),
        json_handler({"models": "nope"}),
        lambda request: httpx.Response(200, content=b"<html>"),
    ],
)
def test_check_ollama_bad_responses_raise_response_error(handler: Handler) -> None:
    with pytest.raises(LLMResponseError):
        check_ollama(make_settings(), client=Recorder(handler).client())


def test_check_ollama_missing_version_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})

    with pytest.raises(LLMResponseError, match="no 'version'"):
        check_ollama(make_settings(), client=Recorder(handler).client())


# --------------------------------------------------------------------------- own client


def test_chat_without_injected_client_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[httpx.Client] = []
    real_client = httpx.Client

    def fake_client(*args: Any, **kwargs: Any) -> httpx.Client:
        client = real_client(*args, transport=httpx.MockTransport(json_handler(chat_body())), **kwargs)
        created.append(client)
        return client

    monkeypatch.setattr("rag_pipeline.llm.httpx.Client", fake_client)

    result = chat(MESSAGES, make_settings(llm_timeout_seconds=42.0))

    assert result.text == ANSWER_SECRET
    assert len(created) == 1
    assert created[0].timeout == httpx.Timeout(42.0)
    assert created[0].is_closed


def test_injected_client_is_not_closed() -> None:
    client = Recorder(json_handler(chat_body())).client()

    chat(MESSAGES, make_settings(), client=client)

    assert not client.is_closed
