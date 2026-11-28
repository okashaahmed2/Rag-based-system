"""Structured LLM output: a Pydantic model as Ollama grammar, validated fail-closed.

:func:`call_structured` sends one chat with ``format`` set to
:func:`grammar_schema` of a Pydantic model and validates the reply with the
SAME model. The result is either ``OK`` with the validated value or
``INVALID_OUTPUT`` with one :class:`InvalidReason`; there is no retry and no
repair of broken output.

Why constraint keywords are removed from the grammar (Phase 10 Step 0 probe,
Ollama 0.40.19, llama3.2:3b): with ``maxLength`` in the schema the grammar
silently cut a term ("stochastic gradient descent" -> "stochastic gradient")
instead of failing. Lengths, patterns, item counts and ranges are therefore
checked only by Pydantic after generation, where a violation is visible
(``SCHEMA_VALIDATION``) rather than hidden in plausible but wrong data. The
grammar keeps only the structure (types, properties, required, items,
``$defs``/``$ref``, ``additionalProperties``, ``enum``).

``done_reason == "length"`` (truncated) is rejected even if the cut text
happens to be valid JSON: a cut list may simply be missing items.

Privacy: prompt and output text are never logged, only status, reason and
counts (Pydantic error TYPES, never their locations or inputs, which can hold
model output).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from rag_pipeline.config import Settings
from rag_pipeline.llm import ChatFormat, LLMResponse

logger = logging.getLogger(__name__)

#: JSON Schema keywords removed by :func:`grammar_schema` (checked by Pydantic instead).
CONSTRAINT_KEYWORDS: frozenset[str] = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
    }
)
#: Keywords whose value maps NAMES to sub-schemas (the names are data, never keywords).
_SCHEMA_MAPS: frozenset[str] = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)
#: Keywords whose value is one sub-schema (or a bool such as ``additionalProperties``).
_SCHEMA_VALUES: frozenset[str] = frozenset(
    {
        "items",
        "additionalProperties",
        "not",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
#: Keywords whose value is a list of sub-schemas.
_SCHEMA_LISTS: frozenset[str] = frozenset({"anyOf", "allOf", "oneOf", "prefixItems"})

ModelT = TypeVar("ModelT", bound=BaseModel)


def _strip_schema(node: Any) -> Any:
    """Copy of one schema node without constraint keywords (sub-schemas recursed)."""
    if isinstance(node, list):
        return [_strip_schema(item) for item in node]
    if not isinstance(node, Mapping):
        return node
    stripped: dict[str, Any] = {}
    for key, value in node.items():
        if key in CONSTRAINT_KEYWORDS:
            continue
        if key in _SCHEMA_MAPS and isinstance(value, Mapping):
            stripped[key] = {name: _strip_schema(sub) for name, sub in value.items()}
        elif key in _SCHEMA_VALUES or key in _SCHEMA_LISTS:
            stripped[key] = _strip_schema(value)
        else:
            stripped[key] = _copy_value(value)
    return stripped


def _copy_value(value: Any) -> Any:
    """Deep copy of a non-schema value (``enum``, ``default``, ``title``, ...)."""
    if isinstance(value, Mapping):
        return {key: _copy_value(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def grammar_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """``model_cls.model_json_schema()`` without constraint keywords, as a new dict.

    Removes :data:`CONSTRAINT_KEYWORDS` from every schema node at any depth
    (inside ``properties``, ``$defs``, ``items``, ``anyOf`` ...). Property and
    definition NAMES are never removed, so a field called ``format`` or
    ``pattern`` is kept. Everything else (``type``, ``properties``,
    ``required``, ``items``, ``$defs``, ``$ref``, ``additionalProperties``,
    ``title``, ``description``, ``enum``, ...) is kept unchanged.
    """
    return cast(dict[str, Any], _strip_schema(model_cls.model_json_schema()))


class StructuredStatus(enum.Enum):
    """Outcome of :func:`call_structured`."""

    OK = "ok"
    INVALID_OUTPUT = "invalid_output"


class InvalidReason(enum.Enum):
    """Why an output was rejected (first matching check wins)."""

    TRUNCATED = "truncated"
    THINKING_LEAKED = "thinking_leaked"
    EMPTY = "empty"
    SCHEMA_VALIDATION = "schema_validation"


@dataclass(frozen=True)
class StructuredResult(Generic[ModelT]):
    """Result of :func:`call_structured`.

    Attributes:
        status: ``OK`` or ``INVALID_OUTPUT``.
        value: The validated model instance (``OK`` only), else ``None``.
        reason: Why the output was rejected (``INVALID_OUTPUT`` only), else ``None``.
        error_count: Number of Pydantic errors (``SCHEMA_VALIDATION`` only), else 0.
        model: Model name reported by Ollama.
        done_reason: Why generation stopped, as reported by Ollama.
        prompt_tokens, completion_tokens: Token counts from Ollama.
        load_seconds, prompt_eval_seconds, eval_seconds, total_seconds: Ollama timings.
    """

    status: StructuredStatus
    value: ModelT | None
    reason: InvalidReason | None
    error_count: int
    model: str
    done_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    load_seconds: float
    prompt_eval_seconds: float
    eval_seconds: float
    total_seconds: float


class StructuredChatFn(Protocol):
    """``chat(messages, settings, *, format, num_predict)``, e.g. a partial of ``llm.chat``."""

    def __call__(
        self,
        messages: list[dict[str, str]],
        settings: Settings,
        *,
        format: ChatFormat | None,
        num_predict: int | None,
    ) -> LLMResponse: ...


def _result(
    response: LLMResponse,
    value: ModelT | None,
    reason: InvalidReason | None,
    error_count: int = 0,
) -> StructuredResult[ModelT]:
    return StructuredResult(
        status=StructuredStatus.OK if reason is None else StructuredStatus.INVALID_OUTPUT,
        value=value,
        reason=reason,
        error_count=error_count,
        model=response.model,
        done_reason=response.done_reason,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        load_seconds=response.load_seconds,
        prompt_eval_seconds=response.prompt_eval_seconds,
        eval_seconds=response.eval_seconds,
        total_seconds=response.total_seconds,
    )


def _check(response: LLMResponse, model_cls: type[ModelT]) -> StructuredResult[ModelT]:
    """Apply the fail-closed checks in order: truncated, thinking, empty, schema."""
    if response.truncated:
        return _result(response, None, InvalidReason.TRUNCATED)
    if response.thinking_leaked:
        return _result(response, None, InvalidReason.THINKING_LEAKED)
    if not response.text.strip():
        return _result(response, None, InvalidReason.EMPTY)
    try:
        value = model_cls.model_validate_json(response.text)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        logger.warning(
            "Structured output failed validation as %s: %d error(s), types %s",
            model_cls.__name__,
            len(errors),
            sorted({str(error["type"]) for error in errors}),
        )
        return _result(response, None, InvalidReason.SCHEMA_VALIDATION, len(errors))
    return _result(response, value, None)


def call_structured(
    messages: Sequence[Mapping[str, str]],
    settings: Settings,
    model_cls: type[ModelT],
    chat_fn: StructuredChatFn,
) -> StructuredResult[ModelT]:
    """Ask the LLM for JSON matching ``model_cls`` and validate it (fail closed, no retry).

    ``chat_fn`` gets ``format=grammar_schema(model_cls)`` and
    ``num_predict=settings.llm_max_extraction_tokens``. The reply is
    ``INVALID_OUTPUT`` when, checked in this order, it was truncated
    (``done_reason == "length"``, even if the text is valid JSON), reasoning
    leaked into it, it is blank, or ``model_cls.model_validate_json`` fails
    (``error_count`` = number of Pydantic errors). Otherwise ``OK`` with the
    validated value.

    Raises:
        LLMError: from ``chat_fn``, propagated unchanged.
        ValueError: invalid messages (from ``chat_fn``), propagated unchanged.
    """
    response = chat_fn(
        [dict(message) for message in messages],
        settings,
        format=grammar_schema(model_cls),
        num_predict=settings.llm_max_extraction_tokens,
    )
    result = _check(response, model_cls)
    logger.info(
        "Structured output %s: status=%s, reason=%s, errors=%d, done_reason=%s, "
        "prompt_tokens=%d, completion_tokens=%d, total %.2f s",
        model_cls.__name__,
        result.status.value,
        result.reason.value if result.reason else None,
        result.error_count,
        result.done_reason,
        result.prompt_tokens,
        result.completion_tokens,
        result.total_seconds,
    )
    return result
