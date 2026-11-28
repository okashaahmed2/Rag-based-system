"""Minimal client for a local Ollama server (non-streaming chat only).

:func:`chat` sends one ``POST /api/chat`` request and returns an
:class:`LLMResponse`; :func:`check_ollama` reports the server version and
whether the configured model is installed.

Every chat request sets ``"stream": false`` and ``"think": false``. Without
``think: false``, thinking-capable models (e.g. ``granite4.2:3b``) write their
reasoning into the answer text. If a response still carries a ``thinking``
field it is dropped and only its length is logged.

Privacy: prompt and answer text are never logged, only counts and timings.

Both functions accept an optional ``httpx.Client`` (tests inject one built
on ``httpx.MockTransport``); otherwise a short-lived client with the
configured timeout is used. Request URLs are always absolute, so an injected
client needs no ``base_url``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from rag_pipeline.config import Settings

logger = logging.getLogger(__name__)

#: Message roles accepted by :func:`chat`.
VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})
_MESSAGE_KEYS: frozenset[str] = frozenset({"role", "content"})
#: Longest Ollama error text copied into an exception message.
_MAX_ERROR_DETAIL = 200
_NS_PER_SECOND = 1_000_000_000
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class LLMError(Exception):
    """Base class for every error raised by this module."""


class LLMUnavailableError(LLMError):
    """The Ollama server cannot be reached."""


class LLMModelNotFoundError(LLMError):
    """The configured model is not installed in Ollama."""


class LLMTimeoutError(LLMError):
    """Ollama did not answer within the configured timeout."""


class LLMResponseError(LLMError):
    """Ollama returned a bad HTTP status, invalid JSON or missing fields."""


@dataclass(frozen=True)
class LLMResponse:
    """One non-streamed chat answer with Ollama's counts and timings.

    Attributes:
        text: Answer text (any ``thinking`` content is never included).
        model: Model name reported by Ollama.
        done_reason: Why generation stopped (``"stop"``, ``"length"``, ...),
            ``None`` if Ollama did not say.
        prompt_tokens: Prompt tokens evaluated (``prompt_eval_count``).
        completion_tokens: Tokens generated (``eval_count``).
        load_seconds: Model load time.
        prompt_eval_seconds: Prompt evaluation time.
        eval_seconds: Generation time.
        total_seconds: Total time measured by Ollama.
        truncated: True if the answer was cut (``done_reason == "length"``).
        thinking_leaked: True if reasoning tags were found in the content and
            removed (see :func:`strip_leaked_thinking`).
    """

    text: str
    model: str
    done_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    load_seconds: float
    prompt_eval_seconds: float
    eval_seconds: float
    total_seconds: float
    truncated: bool
    thinking_leaked: bool


def strip_leaked_thinking(content: str) -> tuple[str, bool]:
    """Remove reasoning a model wrote into the answer; return ``(text, leaked)``.

    Some templates (e.g. ``granite4.2:3b``) open ``<think>`` in the prompt even
    with ``think: false``, so the content holds the reasoning and a bare
    ``</think>`` before the real answer. With ``</think>`` present only the
    text after the LAST one is kept (stripped); with ``<think>`` but no
    ``</think>`` everything is reasoning and the text is ``""``; without
    either tag the content is returned unchanged.
    """
    if _THINK_CLOSE in content:
        return content.rsplit(_THINK_CLOSE, 1)[1].strip(), True
    if _THINK_OPEN in content:
        return "", True
    return content, False


def validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Check chat messages and return plain ``{"role", "content"}`` copies.

    Raises:
        ValueError: ``messages`` is empty, a message has keys other than
            ``role``/``content``, an unknown role, or content that is not a
            string or is empty / whitespace only.
    """
    if not messages:
        raise ValueError("messages must not be empty")
    checked: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"message {index} must be a mapping, got {type(message).__name__}")
        extra = set(message) - _MESSAGE_KEYS
        if extra:
            raise ValueError(f"message {index} has unknown keys: {sorted(extra)}")
        role = message.get("role")
        if role not in VALID_ROLES:
            raise ValueError(
                f"message {index} has invalid role {role!r}; "
                f"expected one of {sorted(VALID_ROLES)}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} ({role}) must have non-empty string content")
        checked.append({"role": role, "content": content})
    return checked


#: Output format accepted by :func:`chat`: ``"json"`` or a JSON schema (dict).
ChatFormat = str | dict[str, Any]


def validate_format(format: object) -> None:
    """Check an Ollama ``format`` value: the string ``"json"`` or a non-empty dict.

    Raises:
        ValueError: any other value.
    """
    if isinstance(format, str):
        if format != "json":
            raise ValueError(f'format must be "json" or a JSON schema dict, got {format!r}')
        return
    if not isinstance(format, dict) or not format:
        raise ValueError(
            f'format must be "json" or a non-empty JSON schema dict, got {type(format).__name__}'
        )


def validate_num_predict(num_predict: object) -> int:
    """Check a ``num_predict`` override: an int >= 1 (bool rejected).

    Raises:
        ValueError: any other value.
    """
    if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict < 1:
        raise ValueError(f"num_predict must be an int >= 1, got {num_predict!r}")
    return num_predict


def build_chat_payload(
    messages: list[dict[str, str]],
    settings: Settings,
    *,
    format: ChatFormat | None = None,
    num_predict: int | None = None,
) -> dict[str, Any]:
    """JSON body of a ``POST /api/chat`` request (always non-streaming, no thinking).

    Without ``format`` and ``num_predict`` the body is exactly the same as
    before these options existed. ``format`` (``"json"`` or a JSON schema
    dict) is added as the top-level ``format`` key only when given;
    ``num_predict`` replaces ``options.num_predict``
    (``settings.llm_max_answer_tokens``) only when given.

    Raises:
        ValueError: invalid ``format`` or ``num_predict``.
    """
    if format is not None:
        validate_format(format)
    predict = (
        settings.llm_max_answer_tokens if num_predict is None else validate_num_predict(num_predict)
    )
    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": predict,
        },
    }
    if format is not None:
        payload["format"] = format
    return payload


@contextmanager
def _client_scope(settings: Settings, client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Yield the injected client, or a short-lived one closed afterwards."""
    if client is not None:
        yield client
        return
    with httpx.Client(timeout=settings.llm_timeout_seconds) as own_client:
        yield own_client


def _send(
    client: httpx.Client,
    method: str,
    path: str,
    settings: Settings,
    payload: dict[str, Any] | None = None,
) -> httpx.Response:
    """Send one request, mapping transport failures to :class:`LLMError` subclasses."""
    url = f"{settings.ollama_base_url}{path}"
    try:
        return client.request(method, url, json=payload)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.error(
            "Ollama unreachable at %s (%s %s): %s", settings.ollama_base_url, method, path, exc
        )
        raise LLMUnavailableError(
            f"Ollama is not running at {settings.ollama_base_url} ({exc})"
        ) from exc
    except httpx.TimeoutException as exc:
        logger.error(
            "Ollama request %s %s timed out after %.1f s",
            method,
            path,
            settings.llm_timeout_seconds,
        )
        raise LLMTimeoutError(
            f"Ollama did not answer {method} {path} within "
            f"{settings.llm_timeout_seconds} s (RAG_LLM_TIMEOUT_SECONDS)"
        ) from exc
    except httpx.TransportError as exc:
        logger.error("Ollama transport error on %s %s: %s", method, path, exc)
        raise LLMUnavailableError(
            f"Ollama is not running at {settings.ollama_base_url} ({exc})"
        ) from exc


def _error_detail(response: httpx.Response) -> str:
    """Ollama's ``error`` text (or the raw body), shortened for messages."""
    try:
        data = response.json()
    except ValueError:
        detail = response.text
    else:
        detail = data.get("error", response.text) if isinstance(data, dict) else response.text
    return str(detail)[:_MAX_ERROR_DETAIL]


def _json_object(response: httpx.Response, path: str) -> dict[str, Any]:
    """Parse a successful response body as a JSON object."""
    if response.status_code >= 400:
        detail = _error_detail(response)
        logger.error("Ollama %s returned HTTP %d: %s", path, response.status_code, detail)
        raise LLMResponseError(f"Ollama {path} returned HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
    except ValueError as exc:
        logger.error("Ollama %s returned invalid JSON (%d bytes)", path, len(response.content))
        raise LLMResponseError(f"Ollama {path} returned invalid JSON") from exc
    if not isinstance(data, dict):
        logger.error("Ollama %s returned JSON %s, expected an object", path, type(data).__name__)
        raise LLMResponseError(f"Ollama {path} returned JSON that is not an object")
    return data


def _count(data: Mapping[str, Any], key: str) -> int:
    """Non-negative integer field, 0 if absent (Ollama omits some counts)."""
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LLMResponseError(f"Ollama response field {key!r} is not a count: {value!r}")
    return value


def _seconds(data: Mapping[str, Any], key: str) -> float:
    """Nanosecond duration field converted to seconds, 0.0 if absent."""
    return _count(data, key) / _NS_PER_SECOND


def _model_not_found(settings: Settings) -> LLMModelNotFoundError:
    return LLMModelNotFoundError(
        f"Model {settings.llm_model!r} is not installed in Ollama at "
        f"{settings.ollama_base_url}; run: ollama pull {settings.llm_model}"
    )


def parse_chat_response(data: Mapping[str, Any]) -> LLMResponse:
    """Build an :class:`LLMResponse` from a ``/api/chat`` JSON object.

    A non-empty ``thinking`` field is dropped (WARNING with its length only);
    ``done_reason == "length"`` sets ``truncated`` (WARNING).

    Raises:
        LLMResponseError: ``message``, ``message.content`` or ``model`` is
            missing or has the wrong type, or a count/duration is invalid.
    """
    message = data.get("message")
    if not isinstance(message, Mapping):
        raise LLMResponseError("Ollama /api/chat response has no 'message' object")
    text = message.get("content")
    if not isinstance(text, str):
        raise LLMResponseError("Ollama /api/chat response has no string 'message.content'")
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise LLMResponseError("Ollama /api/chat response has no 'model' name")
    done_reason = data.get("done_reason")
    if done_reason is not None and not isinstance(done_reason, str):
        raise LLMResponseError(
            f"Ollama response field 'done_reason' is not a string: {done_reason!r}"
        )

    thinking = message.get("thinking") or data.get("thinking")
    if thinking:
        logger.warning(
            "Ollama returned a 'thinking' field (%d chars) despite think=false; "
            "it was dropped from the answer",
            len(str(thinking)),
        )

    answer, leaked = strip_leaked_thinking(text)
    if leaked:
        logger.warning(
            "Reasoning leaked into the answer content (%d chars); kept the %d chars "
            "after the last </think> (empty if the reasoning was never closed)",
            len(text),
            len(answer),
        )

    result = LLMResponse(
        text=answer,
        model=model,
        done_reason=done_reason,
        prompt_tokens=_count(data, "prompt_eval_count"),
        completion_tokens=_count(data, "eval_count"),
        load_seconds=_seconds(data, "load_duration"),
        prompt_eval_seconds=_seconds(data, "prompt_eval_duration"),
        eval_seconds=_seconds(data, "eval_duration"),
        total_seconds=_seconds(data, "total_duration"),
        truncated=done_reason == "length",
        thinking_leaked=leaked,
    )
    if result.truncated:
        logger.warning(
            "LLM answer was cut at the length limit (done_reason=length, "
            "prompt_tokens=%d, completion_tokens=%d)",
            result.prompt_tokens,
            result.completion_tokens,
        )
    return result


def chat(
    messages: Sequence[Mapping[str, Any]],
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    format: ChatFormat | None = None,
    num_predict: int | None = None,
) -> LLMResponse:
    """Send a non-streaming chat request to Ollama and return the answer.

    Messages are validated before any request is sent. The request always
    uses ``stream: false`` and ``think: false``, with ``temperature`` and
    ``num_ctx`` from ``settings``. Optional ``format`` (``"json"`` or a JSON
    schema dict) and ``num_predict`` are passed to
    :func:`build_chat_payload`; without them the request is unchanged.

    Raises:
        ValueError: invalid ``messages``, ``format`` or ``num_predict`` (no
            request is sent).
        LLMUnavailableError: the server cannot be reached.
        LLMModelNotFoundError: Ollama answers 404 (model not installed).
        LLMTimeoutError: no answer within ``settings.llm_timeout_seconds``.
        LLMResponseError: other HTTP error status, invalid JSON or missing fields.
    """
    payload = build_chat_payload(
        validate_messages(messages), settings, format=format, num_predict=num_predict
    )
    with _client_scope(settings, client) as http:
        response = _send(http, "POST", "/api/chat", settings, payload)
    if response.status_code == 404:
        logger.error(
            "Ollama model %s not found at %s: %s",
            settings.llm_model,
            settings.ollama_base_url,
            _error_detail(response),
        )
        raise _model_not_found(settings)
    result = parse_chat_response(_json_object(response, "/api/chat"))
    logger.info(
        "LLM chat %s: %d message(s), prompt_tokens=%d, completion_tokens=%d, "
        "done_reason=%s, load %.2f s, prompt %.2f s, eval %.2f s, total %.2f s",
        result.model,
        len(payload["messages"]),
        result.prompt_tokens,
        result.completion_tokens,
        result.done_reason,
        result.load_seconds,
        result.prompt_eval_seconds,
        result.eval_seconds,
        result.total_seconds,
    )
    return result


def _installed_model_names(data: Mapping[str, Any]) -> list[str]:
    """Model names from a ``/api/tags`` JSON object."""
    models = data.get("models")
    if not isinstance(models, list):
        raise LLMResponseError("Ollama /api/tags response has no 'models' list")
    names: list[str] = []
    for entry in models:
        if isinstance(entry, Mapping):
            name = entry.get("name") or entry.get("model")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _is_installed(model: str, installed: Sequence[str]) -> bool:
    """True if ``model`` is installed; a name without a tag means ``:latest``."""
    candidates = {model} if ":" in model else {model, f"{model}:latest"}
    return any(name in candidates for name in installed)


def check_ollama(settings: Settings, *, client: httpx.Client | None = None) -> dict[str, Any]:
    """Check that Ollama is reachable and the configured model is installed.

    Returns:
        ``{"base_url", "version", "model", "model_present", "installed_models"}``;
        ``model_present`` is always True on return (a missing model raises).

    Raises:
        LLMUnavailableError: the server cannot be reached.
        LLMModelNotFoundError: ``settings.llm_model`` is not installed.
        LLMTimeoutError: no answer within the timeout.
        LLMResponseError: bad status, invalid JSON or missing fields.
    """
    with _client_scope(settings, client) as http:
        version_data = _json_object(_send(http, "GET", "/api/version", settings), "/api/version")
        tags_data = _json_object(_send(http, "GET", "/api/tags", settings), "/api/tags")
    version = version_data.get("version")
    if not isinstance(version, str) or not version:
        raise LLMResponseError("Ollama /api/version response has no 'version'")
    installed = _installed_model_names(tags_data)
    if not _is_installed(settings.llm_model, installed):
        logger.error(
            "Ollama %s at %s: model %s not installed (%d model(s) installed)",
            version,
            settings.ollama_base_url,
            settings.llm_model,
            len(installed),
        )
        raise _model_not_found(settings)
    logger.info(
        "Ollama %s at %s: model %s installed",
        version,
        settings.ollama_base_url,
        settings.llm_model,
    )
    return {
        "base_url": settings.ollama_base_url,
        "version": version,
        "model": settings.llm_model,
        "model_present": True,
        "installed_models": installed,
    }
