"""Tests for rag_pipeline.structured_output and the extraction token setting (no network)."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rag_pipeline.config import Settings
from rag_pipeline.llm import LLMResponse, LLMResponseError, LLMTimeoutError, LLMUnavailableError
from rag_pipeline.structured_output import (
    CONSTRAINT_KEYWORDS,
    InvalidReason,
    StructuredResult,
    StructuredStatus,
    call_structured,
    grammar_schema,
)

OUTPUT_SECRET = "OUTPUT-SECRET-term-value"
PROMPT_SECRET = "PROMPT-SECRET-source-text"
MESSAGES = [
    {"role": "system", "content": "SYSTEM-SECRET"},
    {"role": "user", "content": PROMPT_SECRET},
]


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- models


def constrained_models() -> tuple[type[BaseModel], type[BaseModel]]:
    """A nested model using every constraint keyword (and fields named like keywords)."""

    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        term: str = Field(min_length=1, max_length=80, description="The term.")
        source: str = Field(pattern=r"^S[1-9][0-9]*$")
        score: int = Field(ge=0, le=10, multiple_of=2)
        ratio: float = Field(gt=0, lt=1)
        email: str = Field(json_schema_extra={"format": "email"})
        kind: Literal["concept", "method"]
        note: str | None = Field(default=None, max_length=5)
        format: str = Field(pattern=r"^x$")
        pattern: str = Field(max_length=3)

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Inner] = Field(min_length=1, max_length=10)
        tags: list[str] = Field(default_factory=list, max_length=3)

    return Inner, Outer


def unconstrained_models() -> tuple[type[BaseModel], type[BaseModel]]:
    """The same models (same names) without any constraint keyword."""

    class Inner(BaseModel):
        model_config = ConfigDict(extra="forbid")
        term: str = Field(description="The term.")
        source: str
        score: int
        ratio: float
        email: str
        kind: Literal["concept", "method"]
        note: str | None = None
        format: str
        pattern: str

    class Outer(BaseModel):
        model_config = ConfigDict(extra="forbid")
        items: list[Inner]
        tags: list[str] = Field(default_factory=list)

    return Inner, Outer


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")
    term: str = Field(min_length=1, max_length=20)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[Item] = Field(max_length=2)


def schema_nodes(node: Any) -> list[dict[str, Any]]:
    """Every schema object (not property-name maps) in a JSON schema."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for key, value in node.items():
            if key in ("properties", "$defs"):
                for sub in value.values():
                    found.extend(schema_nodes(sub))
            elif key in ("items", "additionalProperties") and isinstance(value, dict):
                found.extend(schema_nodes(value))
            elif key in ("anyOf", "allOf", "oneOf"):
                for sub in value:
                    found.extend(schema_nodes(sub))
    return found


# --------------------------------------------------------------------------- grammar_schema


def test_original_schema_really_uses_every_constraint_keyword() -> None:
    _, outer = constrained_models()
    used = {key for node in schema_nodes(outer.model_json_schema()) for key in node}

    assert CONSTRAINT_KEYWORDS <= used
    assert "$defs" in outer.model_json_schema()


def test_grammar_schema_has_no_constraint_keyword_at_any_depth() -> None:
    _, outer = constrained_models()

    schema = grammar_schema(outer)

    for node in schema_nodes(schema):
        assert not CONSTRAINT_KEYWORDS & set(node), node


def test_grammar_schema_otherwise_equals_original_schema() -> None:
    _, constrained = constrained_models()
    _, unconstrained = unconstrained_models()

    assert grammar_schema(constrained) == unconstrained.model_json_schema()


def test_grammar_schema_keeps_structure_keywords_and_field_names() -> None:
    _, outer = constrained_models()

    schema = grammar_schema(outer)
    inner = schema["$defs"]["Inner"]

    assert schema["properties"]["items"]["items"] == {"$ref": "#/$defs/Inner"}
    assert schema["additionalProperties"] is False
    assert inner["additionalProperties"] is False
    assert "format" in inner["properties"] and "pattern" in inner["properties"]
    assert "format" in inner["required"] and "pattern" in inner["required"]
    assert inner["properties"]["kind"]["enum"] == ["concept", "method"]
    assert inner["properties"]["term"]["description"] == "The term."
    assert inner["properties"]["term"]["title"] == "Term"
    assert inner["properties"]["format"] == {"title": "Format", "type": "string"}


def test_grammar_schema_leaves_the_original_schema_unchanged() -> None:
    _, outer = constrained_models()
    original = outer.model_json_schema()
    snapshot = copy.deepcopy(original)

    stripped = grammar_schema(outer)
    stripped["$defs"]["Inner"]["properties"]["term"]["title"] = "changed"

    assert original == snapshot
    assert outer.model_json_schema() == snapshot


def test_grammar_schema_returns_a_new_dict_each_call() -> None:
    first = grammar_schema(Output)
    second = grammar_schema(Output)

    assert first == second
    assert first is not second
    assert first["$defs"] is not second["$defs"]


def test_grammar_schema_is_json_serialisable() -> None:
    assert json.loads(json.dumps(grammar_schema(Output))) == grammar_schema(Output)


# --------------------------------------------------------------------------- setting


def test_llm_max_extraction_tokens_default() -> None:
    assert make_settings().llm_max_extraction_tokens == 768


def test_llm_max_extraction_tokens_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_EXTRACTION_TOKENS", "1024")

    assert make_settings().llm_max_extraction_tokens == 1024


@pytest.mark.parametrize("value", ["16", "2048"])
def test_llm_max_extraction_tokens_bounds_accepted(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_EXTRACTION_TOKENS", value)

    assert make_settings().llm_max_extraction_tokens == int(value)


@pytest.mark.parametrize("bad", ["15", "2049", "0", "-1"])
def test_invalid_llm_max_extraction_tokens_raises(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_EXTRACTION_TOKENS", bad)

    with pytest.raises(
        ValidationError, match="RAG_LLM_MAX_EXTRACTION_TOKENS must be between 16 and 2048"
    ):
        make_settings()


def test_non_integer_llm_max_extraction_tokens_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_EXTRACTION_TOKENS", "lots")

    with pytest.raises(ValidationError):
        make_settings()


def test_extraction_tokens_do_not_change_answer_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LLM_MAX_EXTRACTION_TOKENS", "100")

    assert make_settings().llm_max_answer_tokens == 256


# --------------------------------------------------------------------------- call_structured


def make_response(
    text: str, *, truncated: bool = False, thinking_leaked: bool = False
) -> LLMResponse:
    return LLMResponse(
        text=text,
        model="llama3.2:3b",
        done_reason="length" if truncated else "stop",
        prompt_tokens=500,
        completion_tokens=70,
        load_seconds=0.1,
        prompt_eval_seconds=0.2,
        eval_seconds=0.3,
        total_seconds=0.7,
        truncated=truncated,
        thinking_leaked=thinking_leaked,
    )


class FakeChat:
    def __init__(self, response: LLMResponse | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, str]],
        settings: Settings,
        *,
        format: Any,
        num_predict: int | None,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": messages, "settings": settings, "format": format, "num_predict": num_predict}
        )
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


VALID = json.dumps({"items": [{"term": OUTPUT_SECRET[:20]}]})


def run(response: LLMResponse, settings: Settings | None = None) -> StructuredResult[Output]:
    return call_structured(MESSAGES, settings or make_settings(), Output, FakeChat(response))


def test_ok_path_returns_validated_value_and_counts() -> None:
    result = run(make_response(VALID))

    assert result.status is StructuredStatus.OK
    assert result.reason is None
    assert result.error_count == 0
    assert isinstance(result.value, Output)
    assert result.value.items[0].term == OUTPUT_SECRET[:20]
    assert (result.model, result.done_reason) == ("llama3.2:3b", "stop")
    assert (result.prompt_tokens, result.completion_tokens) == (500, 70)
    assert (result.load_seconds, result.prompt_eval_seconds) == (0.1, 0.2)
    assert (result.eval_seconds, result.total_seconds) == (0.3, 0.7)


def test_chat_fn_receives_grammar_schema_and_extraction_tokens() -> None:
    chat_fn = FakeChat(make_response(VALID))
    settings = make_settings(llm_max_extraction_tokens=333)

    call_structured(MESSAGES, settings, Output, chat_fn)

    (call,) = chat_fn.calls
    assert call["format"] == grammar_schema(Output)
    assert "maxLength" not in json.dumps(call["format"])
    assert call["num_predict"] == 333
    assert call["settings"] is settings
    assert call["messages"] == MESSAGES


def test_truncated_valid_json_is_still_truncated() -> None:
    result = run(make_response(VALID, truncated=True))

    assert result.status is StructuredStatus.INVALID_OUTPUT
    assert result.reason is InvalidReason.TRUNCATED
    assert result.value is None
    assert result.error_count == 0


def test_truncated_broken_json_is_truncated() -> None:
    result = run(make_response('{"items": [{"term": "Decision', truncated=True))

    assert result.reason is InvalidReason.TRUNCATED


def test_truncated_wins_over_thinking_leak() -> None:
    result = run(make_response(VALID, truncated=True, thinking_leaked=True))

    assert result.reason is InvalidReason.TRUNCATED


def test_thinking_leak_is_invalid_even_with_valid_json() -> None:
    result = run(make_response(VALID, thinking_leaked=True))

    assert result.status is StructuredStatus.INVALID_OUTPUT
    assert result.reason is InvalidReason.THINKING_LEAKED
    assert result.value is None


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_content_is_empty(text: str) -> None:
    result = run(make_response(text))

    assert result.reason is InvalidReason.EMPTY
    assert result.value is None


@pytest.mark.parametrize(
    ("text", "errors"),
    [
        ("not json at all", 1),
        ('```json\n{"items": []}\n```', 1),
        ('{"items": [{"term": "x"}]} trailing', 1),
        ('{"items": [{"term": "x", "extra": 1}]}', 1),
        ('{"items": [], "extra": 1}', 1),
        ('{"items": [{"term": ""}]}', 1),
        ('{"items": [{"term": "' + "x" * 21 + '"}]}', 1),
        ('{"items": [{"term": "a"}, {"term": "b"}, {"term": "c"}]}', 1),
        ("{}", 1),
        ('{"items": [{}, {"term": 5}]}', 2),
        ("[]", 1),
    ],
)
def test_schema_violations_are_schema_validation(text: str, errors: int) -> None:
    result = run(make_response(text))

    assert result.status is StructuredStatus.INVALID_OUTPUT
    assert result.reason is InvalidReason.SCHEMA_VALIDATION
    assert result.error_count == errors
    assert result.value is None


@pytest.mark.parametrize(
    "exc",
    [LLMUnavailableError("down"), LLMTimeoutError("slow"), LLMResponseError("bad")],
)
def test_llm_errors_propagate_unchanged(exc: Exception) -> None:
    with pytest.raises(type(exc)) as info:
        call_structured(MESSAGES, make_settings(), Output, FakeChat(exc=exc))

    assert info.value is exc


def test_result_is_frozen() -> None:
    result = run(make_response(VALID))

    with pytest.raises(AttributeError):
        result.status = StructuredStatus.INVALID_OUTPUT  # type: ignore[misc]


@pytest.mark.parametrize(
    "text",
    [
        json.dumps({"items": [{"term": OUTPUT_SECRET[:20]}]}),
        json.dumps({"items": [{"term": OUTPUT_SECRET}]}),
        json.dumps({"items": [], OUTPUT_SECRET: 1}),
        OUTPUT_SECRET,
    ],
)
def test_no_prompt_or_output_text_in_logs(text: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    run(make_response(text))

    assert caplog.records
    assert OUTPUT_SECRET[:10] not in caplog.text
    assert PROMPT_SECRET not in caplog.text
    assert "SYSTEM-SECRET" not in caplog.text
