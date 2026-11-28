"""Pure JSON checks of the n8n workflow files (no n8n needed).

``n8n/workflows/rag_ingest_scheduled.json`` is checked strictly. Exports in
``n8n/reference/`` are only reviewed: findings are reported as warnings and
never fail the suite.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from rag_pipeline.config import PROJECT_ROOT

WORKFLOW_PATH = PROJECT_ROOT / "n8n" / "workflows" / "rag_ingest_scheduled.json"
REFERENCE_DIR = PROJECT_ROOT / "n8n" / "reference"
API_BASE = "http://127.0.0.1:8000"

EXPECTED_NODES = {
    "Schedule every 5 min",
    "Manual run",
    "Start scan",
    "Already running?",
    "Skip: scan already running",
    "Wait 3s",
    "Check status",
    "Still running?",
    "Job succeeded?",
    "Done",
    "Scan not succeeded",
}

#: (source, output index, target); output 0 = true branch of an "if" node.
EXPECTED_CONNECTIONS = {
    ("Schedule every 5 min", 0, "Start scan"),
    ("Manual run", 0, "Start scan"),
    ("Start scan", 0, "Already running?"),
    ("Already running?", 0, "Skip: scan already running"),
    ("Already running?", 1, "Wait 3s"),
    ("Wait 3s", 0, "Check status"),
    ("Check status", 0, "Still running?"),
    ("Still running?", 0, "Wait 3s"),
    ("Still running?", 1, "Job succeeded?"),
    ("Job succeeded?", 0, "Done"),
    ("Job succeeded?", 1, "Scan not succeeded"),
}


# --- helpers (also used for the reference review) -------------------------------


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    assert isinstance(data, dict)
    return data


def walk(value: Any) -> Iterator[tuple[str | None, Any]]:
    """Yield every ``(key, value)`` pair of nested dicts/lists (key None in lists)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from walk(item)


def all_strings(value: Any) -> list[str]:
    strings = [v for _, v in walk(value) if isinstance(v, str)]
    strings += [k for k, _ in walk(value) if isinstance(k, str)]
    return strings


def connections(workflow: dict[str, Any]) -> set[tuple[str, int, str]]:
    found: set[tuple[str, int, str]] = set()
    for source, kinds in workflow["connections"].items():
        for index, targets in enumerate(kinds.get("main", [])):
            for target in targets or []:
                assert target["type"] == "main"
                found.add((source, index, target["node"]))
    return found


def http_urls(workflow: dict[str, Any]) -> list[str]:
    return [
        node["parameters"]["url"]
        for node in workflow["nodes"]
        if node["type"] == "n8n-nodes-base.httpRequest"
    ]


def safety_findings(workflow: dict[str, Any]) -> list[str]:
    """Return URL / credentials / localhost / pinData problems (empty = clean)."""
    findings: list[str] = []
    for url in http_urls(workflow):
        if not url.lstrip("=").startswith(API_BASE):
            findings.append(f"URL does not start with {API_BASE}: {url}")
    if any(key == "credentials" for key, _ in walk(workflow)):
        findings.append('contains a "credentials" key')
    if workflow.get("pinData"):
        findings.append("contains non-empty pinData")
    for text in all_strings(workflow):
        if "localhost" in text.lower():
            findings.append(f'string contains "localhost": {text}')
    return findings


def node(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [n for n in workflow["nodes"] if n["name"] == name]
    assert len(matches) == 1, name
    return matches[0]


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return load(WORKFLOW_PATH)


# --- strict checks of the workflow --------------------------------------------------


def test_workflow_parses_and_has_basic_settings(workflow: dict[str, Any]) -> None:
    assert workflow["name"] == "RAG ingest (scheduled)"
    assert workflow["active"] is False
    assert workflow["settings"] == {"executionOrder": "v1", "timezone": "Asia/Karachi"}


def test_exact_node_names(workflow: dict[str, Any]) -> None:
    names = [n["name"] for n in workflow["nodes"]]
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_NODES


def test_node_ids_are_unique_uuid4(workflow: dict[str, Any]) -> None:
    import uuid

    ids = [n["id"] for n in workflow["nodes"]]
    assert len(ids) == len(set(ids))
    for value in ids:
        assert uuid.UUID(value).version == 4


def test_exact_connections(workflow: dict[str, Any]) -> None:
    assert connections(workflow) == EXPECTED_CONNECTIONS


def test_loop_back_and_both_triggers(workflow: dict[str, Any]) -> None:
    found = connections(workflow)
    assert ("Still running?", 0, "Wait 3s") in found
    assert ("Schedule every 5 min", 0, "Start scan") in found
    assert ("Manual run", 0, "Start scan") in found


def test_no_credentials_pindata_or_localhost(workflow: dict[str, Any]) -> None:
    assert "pinData" not in workflow
    assert safety_findings(workflow) == []


def test_every_url_targets_the_local_api(workflow: dict[str, Any]) -> None:
    urls = http_urls(workflow)
    assert len(urls) == 2
    for url in urls:
        assert url.lstrip("=").startswith(API_BASE)


@pytest.mark.parametrize("name", ["Start scan", "Check status"])
def test_http_nodes_retry_and_timeout(workflow: dict[str, Any], name: str) -> None:
    http = node(workflow, name)
    assert http["type"] == "n8n-nodes-base.httpRequest"
    assert http["typeVersion"] == 4.4
    assert http["retryOnFail"] is True
    assert http["maxTries"] == 3
    assert http["waitBetweenTries"] == 5000
    assert http["parameters"]["options"]["timeout"] == 10000
    assert http["parameters"]["options"]["response"]["response"]["responseFormat"] == "json"
    assert "onError" not in http  # default: stop the workflow


def test_start_scan_is_post_without_body(workflow: dict[str, Any]) -> None:
    params = node(workflow, "Start scan")["parameters"]
    assert params["method"] == "POST"
    assert params["url"] == f"{API_BASE}/ingest/scan"
    assert "sendBody" not in params


def test_check_status_uses_job_id_from_start_scan(workflow: dict[str, Any]) -> None:
    params = node(workflow, "Check status")["parameters"]
    assert "method" not in params  # GET is the default
    assert params["url"] == (
        "=http://127.0.0.1:8000/status/{{ $('Start scan').first().json.job_id }}"
    )


def test_already_running_condition(workflow: dict[str, Any]) -> None:
    conditions = node(workflow, "Already running?")["parameters"]["conditions"]
    [condition] = conditions["conditions"]
    assert condition["leftValue"] == "={{ $json.already_running }}"
    assert condition["operator"] == {"type": "boolean", "operation": "true", "singleValue": True}


def test_still_running_guard(workflow: dict[str, Any]) -> None:
    conditions = node(workflow, "Still running?")["parameters"]["conditions"]
    assert conditions["combinator"] == "and"
    running, guard = conditions["conditions"]
    assert running["leftValue"] == "={{ ['queued','running'].includes($json.status) }}"
    assert running["operator"]["operation"] == "true"
    assert guard["leftValue"] == "={{ $runIndex }}"
    assert guard["rightValue"] == 40
    assert guard["operator"] == {"type": "number", "operation": "lt"}


def test_job_succeeded_condition_and_error_message(workflow: dict[str, Any]) -> None:
    [condition] = node(workflow, "Job succeeded?")["parameters"]["conditions"]["conditions"]
    assert condition["leftValue"] == "={{ $json.status }}"
    assert condition["rightValue"] == "succeeded"
    assert condition["operator"] == {"type": "string", "operation": "equals"}
    stop = node(workflow, "Scan not succeeded")
    assert stop["type"] == "n8n-nodes-base.stopAndError"
    assert stop["parameters"] == {
        "errorType": "errorMessage",
        "errorMessage": "=Scan job {{ $json.job_id }} ended with status '{{ $json.status }}'",
    }


def test_wait_is_three_seconds(workflow: dict[str, Any]) -> None:
    wait = node(workflow, "Wait 3s")
    assert wait["typeVersion"] == 1.1
    assert wait["parameters"] == {"amount": 3, "unit": "seconds"}


def test_schedule_every_five_minutes(workflow: dict[str, Any]) -> None:
    schedule = node(workflow, "Schedule every 5 min")
    assert schedule["type"] == "n8n-nodes-base.scheduleTrigger"
    assert schedule["typeVersion"] == 1.3
    assert schedule["parameters"]["rule"]["interval"] == [
        {"field": "minutes", "minutesInterval": 5}
    ]


# --- report-only review of the reference exports -------------------------------------


def test_reference_exports_reviewed_report_only() -> None:
    """Warn about findings in ``n8n/reference/*.json``; never fail on them."""
    for path in sorted(REFERENCE_DIR.glob("*.json")):
        try:
            findings = safety_findings(load(path))
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            findings = [f"could not be reviewed: {type(exc).__name__}: {exc}"]
        for finding in findings:
            warnings.warn(f"{path.name}: {finding}", UserWarning, stacklevel=1)
