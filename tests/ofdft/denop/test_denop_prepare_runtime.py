from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mldft.ofdft.denop import prepare_runtime as prepare_runtime_module


class FakeLogger:
    """Small logger stub that records ``remove/add/info`` calls for assertions."""

    def __init__(self):
        self.calls = []

    def remove(self):
        self.calls.append(("remove",))

    def add(self, *args, **kwargs):
        self.calls.append(("add", args, kwargs))
        return len(self.calls)

    def info(self, message):
        self.calls.append(("info", message))


def test_prepare_logger_uses_denop_dir_for_continue_runs(monkeypatch, tmp_path: Path):
    """Check continue mode forces logging to ``<denop_dir>/ofdft.log``."""
    fake_logger = FakeLogger()
    monkeypatch.setattr(prepare_runtime_module, "logger", fake_logger)

    log_file = prepare_runtime_module.prepare_logger(
        OmegaConf.create(
            {
                "continue_denop": True,
                "denop_dir": str(tmp_path),
                "log_file": str(tmp_path / "ignored.log"),
            }
        )
    )

    assert log_file == tmp_path / "ofdft.log"
    assert fake_logger.calls[0] == ("remove",)
    assert fake_logger.calls[1][0] == "add"
    assert callable(fake_logger.calls[1][1][0])
    assert "format" in fake_logger.calls[1][2]
    assert fake_logger.calls[1][2]["colorize"]
    assert fake_logger.calls[1][2]["enqueue"]
    assert fake_logger.calls[2][0] == "add"
    assert fake_logger.calls[2][1][0] == tmp_path / "ofdft.log"
    assert fake_logger.calls[2][2]["level"] == "TRACE"
    assert fake_logger.calls[2][2]["rotation"] == "10 MB"
    assert fake_logger.calls[2][2]["enqueue"]
    assert fake_logger.calls[2][2]["backtrace"]
    assert fake_logger.calls[2][2]["diagnose"]
    assert fake_logger.calls[-1] == ("info", f'Logging to "{tmp_path / "ofdft.log"}"')


def test_prepare_logger_raises_when_log_file_is_missing(monkeypatch):
    """Ensure non-continue mode fails fast when no ``log_file`` is provided."""
    fake_logger = FakeLogger()
    monkeypatch.setattr(prepare_runtime_module, "logger", fake_logger)

    with pytest.raises(
        ValueError,
        match="The 'log_file' configuration must be specified for logging to function properly.",
    ):
        prepare_runtime_module.prepare_logger(
            OmegaConf.create({"continue_denop": False, "log_file": None})
        )


def test_prepare_logger_uses_explicit_log_file_when_not_continue(monkeypatch, tmp_path: Path):
    """Ensure non-continue runs respect the configured ``log_file`` path."""
    fake_logger = FakeLogger()
    monkeypatch.setattr(prepare_runtime_module, "logger", fake_logger)

    log_file = prepare_runtime_module.prepare_logger(
        OmegaConf.create({"continue_denop": False, "log_file": str(tmp_path / "custom.log")})
    )

    assert log_file == tmp_path / "custom.log"
    add_calls = [call for call in fake_logger.calls if call[0] == "add"]
    assert len(add_calls) == 2
    assert add_calls[1][1][0] == tmp_path / "custom.log"


def test_prepare_logger_raises_when_continue_dir_does_not_exist(monkeypatch, tmp_path: Path):
    """Ensure continue mode rejects a missing ``denop_dir``."""
    fake_logger = FakeLogger()
    monkeypatch.setattr(prepare_runtime_module, "logger", fake_logger)
    missing_dir = tmp_path / "missing_run_dir"

    with pytest.raises(
        ValueError,
        match="does not exist or is not a directory",
    ):
        prepare_runtime_module.prepare_logger(
            OmegaConf.create(
                {
                    "continue_denop": True,
                    "denop_dir": str(missing_dir),
                    "log_file": str(tmp_path / "ignored.log"),
                }
            )
        )
