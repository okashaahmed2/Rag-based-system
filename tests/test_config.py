"""Tests for rag_pipeline.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_pipeline import config
from rag_pipeline.config import PROJECT_ROOT, Settings, get_settings


def test_project_root_is_derived_from_package_location() -> None:
    assert PROJECT_ROOT == Path(config.__file__).resolve().parents[2]
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_defaults_load_without_env_file() -> None:
    settings = Settings(_env_file=None)

    assert settings.project_root == PROJECT_ROOT
    assert settings.inbox_dir == PROJECT_ROOT / "data" / "inbox"
    assert settings.processed_dir == PROJECT_ROOT / "data" / "processed"
    assert settings.failed_dir == PROJECT_ROOT / "data" / "failed"
    assert settings.log_dir == PROJECT_ROOT / "logs"
    assert settings.log_level == "INFO"


def test_project_root_does_not_depend_on_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.project_root == PROJECT_ROOT
    assert settings.inbox_dir == PROJECT_ROOT / "data" / "inbox"


def test_env_var_overrides_default_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_LOG_LEVEL", "debug")

    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"


def test_env_var_overrides_default_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_INBOX_DIR", str(tmp_path / "my_inbox"))

    settings = Settings(_env_file=None)

    assert settings.inbox_dir == tmp_path / "my_inbox"


def test_env_file_is_read(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_LOG_LEVEL=warning\nUNRELATED_VAR=x\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.log_level == "WARNING"


def test_env_var_takes_precedence_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RAG_LOG_LEVEL=WARNING\n", encoding="utf-8")
    monkeypatch.setenv("RAG_LOG_LEVEL", "ERROR")

    settings = Settings(_env_file=env_file)

    assert settings.log_level == "ERROR"


def test_unknown_rag_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_NOT_A_SETTING", "whatever")

    settings = Settings(_env_file=None)

    assert not hasattr(settings, "not_a_setting")


@pytest.mark.parametrize("bad_level", ["verbose", "", "TRACE"])
def test_invalid_log_level_raises_validation_error(
    bad_level: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_LOG_LEVEL", bad_level)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_relative_paths_resolve_under_project_root(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        project_root=tmp_path,
        inbox_dir=Path("custom/inbox"),
        log_dir=Path("my_logs"),
    )

    assert settings.inbox_dir == tmp_path / "custom" / "inbox"
    assert settings.log_dir == tmp_path / "my_logs"
    assert settings.processed_dir == tmp_path / "data" / "processed"


def test_absolute_path_is_kept_as_given(tmp_path: Path) -> None:
    absolute = tmp_path / "elsewhere" / "failed"

    settings = Settings(_env_file=None, project_root=tmp_path / "root", failed_dir=absolute)

    assert settings.failed_dir == absolute


def test_settings_do_not_create_directories(tmp_path: Path) -> None:
    Settings(_env_file=None, project_root=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_chunk_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120


def test_chunk_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "0")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 0


def test_chunk_size_minimum_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "99")

    settings = Settings(_env_file=None)

    assert (settings.chunk_size, settings.chunk_overlap) == (100, 99)


@pytest.mark.parametrize(
    ("size", "overlap", "message"),
    [
        ("99", "0", "RAG_CHUNK_SIZE must be >= 100"),
        ("0", "0", "RAG_CHUNK_SIZE must be >= 100"),
        ("-800", "0", "RAG_CHUNK_SIZE must be >= 100"),
        ("800", "-1", "RAG_CHUNK_OVERLAP must be >= 0"),
        ("800", "800", "RAG_CHUNK_OVERLAP must be >= 0"),
        ("800", "900", "RAG_CHUNK_OVERLAP must be >= 0"),
    ],
)
def test_invalid_chunk_settings_raise_clear_error(
    size: str, overlap: str, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", size)
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", overlap)

    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


@pytest.mark.parametrize("bad_value", ["eight hundred", "8.5", ""])
def test_non_integer_chunk_size_raises(
    bad_value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", bad_value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached() -> None:
    first = get_settings()

    assert get_settings() is first

    get_settings.cache_clear()
    assert get_settings() is not first
