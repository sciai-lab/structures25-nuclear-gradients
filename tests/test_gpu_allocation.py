"""Tests for automatic GPU allocation (:mod:`mldft.utils.gpu_allocation`).

GPUs and the current host are mocked, so these run on a GPU-less CI machine.
"""

import contextlib
import json
import subprocess
from datetime import datetime

import pytest

import mldft.utils.gpu_allocation.allocation as ga
import mldft.utils.gpu_allocation.discovery as discovery
import mldft.utils.gpu_allocation.remote_scripts.proxy_tmux_session as proxy
import mldft.utils.gpu_allocation.remote_scripts.run_tmux_job as run_job
import mldft.utils.gpu_allocation.remote_tmux as rt
from mldft.utils.gpu_allocation.allocation import (
    _fallback_hosts,
    _host_label,
    _num_gpus_from_env,
    ensure_gpu_allocation,
)
from mldft.utils.gpu_allocation.discovery import (
    parse_free_gpu_ids,
    select_allocation,
)
from mldft.utils.gpu_allocation.remote_tmux import (
    _build_remote_job_command,
    _build_tmux_create_command,
    _build_tmux_proxy_command,
    _create_remote_tmux_session,
    _proxy_remote_tmux_session,
    _remote_tmux_banner_lines,
    _session_name_from_timestamp,
)

NVIDIA_SMI_SAMPLE = "\n".join(
    [
        "0, GPU-0, 12000, 12000",  # fully used -> busy
        "1, GPU-1, 100, 12000",  # ~0.8% used -> free
        "2, GPU-2, 0, 12000",  # idle -> free
        "3, GPU-3, 11000, 12000",  # ~92% used -> busy
    ]
)


def _probe_from(mapping):
    return lambda host: list(mapping.get(host, []))


def _patch_ensure(
    monkeypatch,
    gpus_by_host,
    *,
    local_host="compgpu12",
    env=None,
    fallback_hosts=ga.FALLBACK_HOSTS,
):
    """Stub the collaborators of ``ensure_gpu_allocation`` for a test.

    Returns ``(env, events)`` where ``env`` is the dict standing in for
    ``os.environ`` and ``events`` records the confirm/remote-launch calls in order.
    """
    env = {} if env is None else env
    events: list = []
    monkeypatch.setattr(ga.os, "environ", env)
    monkeypatch.setattr(ga.socket, "gethostname", lambda: local_host)
    if fallback_hosts is not None:
        monkeypatch.setattr(ga, "_fallback_hosts", lambda cwd: tuple(fallback_hosts))
    monkeypatch.setattr(ga, "probe_host", lambda host, lh: list(gpus_by_host.get(host, [])))
    monkeypatch.setattr(
        ga, "_confirm_allocation", lambda host, ids: events.append(("confirm", host, ids))
    )
    monkeypatch.setattr(
        ga, "_run_in_remote_tmux", lambda host, ids, cwd: events.append(("remote_tmux", host, ids))
    )
    return env, events


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_free_gpu_ids_threshold():
    assert parse_free_gpu_ids(NVIDIA_SMI_SAMPLE, "") == [1, 2]


def test_parse_free_gpu_ids_handles_blank_lines_and_padding():
    assert parse_free_gpu_ids("\n  0, GPU-0, 0, 8000 \n\n", "") == [0]


def test_parse_free_gpu_ids_skips_zero_total():
    gpu_output = "0, GPU-0, 0, 0\n1, GPU-1, 0, 8000\n"
    assert parse_free_gpu_ids(gpu_output, "") == [1]


def test_parse_free_gpu_ids_skips_low_memory_gpu_with_process():
    # GPU 0 is barely using memory but already has a foreign compute process,
    # so it must not be offered even though it passes the memory threshold.
    gpu_output = "\n".join(
        [
            "0, GPU-occupied, 100, 80000",
            "1, GPU-free, 100, 80000",
        ]
    )
    process_output = "GPU-occupied\n"
    assert parse_free_gpu_ids(gpu_output, process_output) == [1]


def test_probe_host_is_conservative_when_process_query_fails(monkeypatch):
    calls = []

    def fake_query_host(host, local_host, query):
        calls.append(query)
        if query == discovery.GPU_QUERY:
            return "0, GPU-free, 0, 80000\n"
        return None

    monkeypatch.setattr(discovery, "_query_host", fake_query_host)

    assert discovery.probe_host("compgpu12", "compgpu12") == []
    assert calls == [discovery.GPU_QUERY, discovery.PROCESS_QUERY]


# --------------------------------------------------------------------------- #
# host selection
# --------------------------------------------------------------------------- #
def test_select_prefers_first_host():
    probe = _probe_from({"compgpu12": [2, 3], "compgpu8": [0, 1]})
    assert select_allocation(["compgpu12", "compgpu8"], 1, probe) == (("compgpu12", [2]), 2)


def test_select_falls_back_to_remote():
    probe = _probe_from({"compgpu12": [], "compgpu8": [4, 5]})
    assert select_allocation(["compgpu12", "compgpu8"], 2, probe) == (("compgpu8", [4, 5]), 2)


def test_select_returns_none_when_insufficient():
    probe = _probe_from({"compgpu12": [1], "compgpu8": [2]})
    assert select_allocation(["compgpu12", "compgpu8"], 2, probe) == (None, 1)


def test_select_never_mixes_hosts():
    # Each host has a single free GPU; requesting two must NOT combine them.
    probe = _probe_from({"compgpu12": [0], "compgpu8": [1]})
    assert select_allocation(["compgpu12", "compgpu8"], 2, probe) == (None, 1)


# --------------------------------------------------------------------------- #
# num_gpus from environment
# --------------------------------------------------------------------------- #
def test_fallback_hosts_uses_default_when_config_missing(tmp_path):
    assert _fallback_hosts(tmp_path) == ga.FALLBACK_HOSTS


def test_fallback_hosts_reads_repo_gpu_config(tmp_path):
    nested = tmp_path / "subdir"
    nested.mkdir()
    (tmp_path / ".gpus").write_text("custom1\n\n# comment\ncustom2\n", encoding="utf-8")

    assert _fallback_hosts(nested) == ("custom1", "custom2")


def test_host_label_adds_known_gpu_type():
    assert _host_label("compgpu12") == "compgpu12 (H200)"


def test_host_label_leaves_unknown_custom_host_plain():
    assert _host_label("custom2") == "custom2"


def test_num_gpus_from_env_allows_many():
    assert _num_gpus_from_env({"MLDFT_NUM_GPUS": "5"}) == 5


def test_num_gpus_from_env_defaults_to_one():
    assert _num_gpus_from_env({}) == 1


def test_num_gpus_from_env_errors_invalid():
    with pytest.raises(SystemExit) as exc_info:
        _num_gpus_from_env({"MLDFT_NUM_GPUS": "1,2"})

    assert str(exc_info.value) == "MLDFT_NUM_GPUS must be a positive integer, got '1,2'."


def test_session_name_from_timestamp_uses_command_name_prefix(monkeypatch):
    now = datetime(2026, 6, 2, 10, 14, 15)
    monkeypatch.setattr("sys.argv", ["mldft_train", "foo=bar"])

    assert _session_name_from_timestamp(now) == "mldft_train_20260602_101415"


def test_session_name_from_timestamp_sanitizes_command_name(monkeypatch):
    now = datetime(2026, 6, 2, 10, 14, 15)
    monkeypatch.setattr("sys.argv", ["/tmp/mldft denop.py"])

    assert _session_name_from_timestamp(now) == "mldft_denop_py_20260602_101415"


# --------------------------------------------------------------------------- #
# ensure_gpu_allocation
# --------------------------------------------------------------------------- #
def test_ensure_respects_existing_cuda_visible_devices(monkeypatch):
    env, events = _patch_ensure(
        monkeypatch, {"compgpu12": [2, 3]}, env={"CUDA_VISIBLE_DEVICES": "5"}
    )
    probed: list[str] = []
    monkeypatch.setattr(ga, "probe_host", lambda host, lh: probed.append(host) or [])

    ensure_gpu_allocation()

    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert probed == []  # never probes when user already chose
    assert events == []


def test_ensure_respects_empty_cuda_visible_devices(monkeypatch):
    env, events = _patch_ensure(
        monkeypatch, {"compgpu12": [2, 3]}, env={"CUDA_VISIBLE_DEVICES": ""}
    )
    probed: list[str] = []
    monkeypatch.setattr(ga, "probe_host", lambda host, lh: probed.append(host) or [])

    ensure_gpu_allocation()

    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert probed == []
    assert events == []


def test_ensure_sets_local_gpus(monkeypatch):
    env, events = _patch_ensure(monkeypatch, {"compgpu12": [2, 3]}, env={"MLDFT_NUM_GPUS": "2"})

    ensure_gpu_allocation()

    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert events == [("confirm", "compgpu12", [2, 3])]


def test_ensure_uses_configured_fallback_hosts(monkeypatch, tmp_path):
    (tmp_path / ".gpus").write_text("custom1\ncustom2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    env, events = _patch_ensure(
        monkeypatch,
        {"compgpu12": [], "custom1": [], "custom2": [4]},
        fallback_hosts=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        ensure_gpu_allocation()

    assert exc_info.value.code == 1
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert events == [("confirm", "custom2", [4]), ("remote_tmux", "custom2", [4])]


def test_ensure_reports_custom_gpu_host_without_unknown_type(monkeypatch, tmp_path, capsys):
    (tmp_path / ".gpus").write_text("custom1\ncustom2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _patch_ensure(
        monkeypatch,
        {"compgpu12": [], "custom1": [], "custom2": [4]},
        fallback_hosts=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        ensure_gpu_allocation()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Running on custom2 with GPU 4 via remote tmux" in err
    assert "custom2 (" not in err


def test_ensure_reports_known_gpu_host_type(monkeypatch, capsys):
    _patch_ensure(monkeypatch, {"compgpu12": [2]})

    ensure_gpu_allocation()

    err = capsys.readouterr().err
    assert "Using GPU 2 on compgpu12 (H200)" in err


def test_ensure_runs_remote_tmux_without_touching_local_env(monkeypatch):
    env, events = _patch_ensure(monkeypatch, {"compgpu12": [], "compgpu8": [1]})

    with pytest.raises(SystemExit) as exc_info:
        ensure_gpu_allocation()

    assert exc_info.value.code == 1
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert events == [("confirm", "compgpu8", [1]), ("remote_tmux", "compgpu8", [1])]


def test_ensure_exits_if_remote_tmux_unexpectedly_returns(monkeypatch):
    env, events = _patch_ensure(monkeypatch, {"compgpu12": [], "compgpu8": [1]})

    with pytest.raises(SystemExit) as exc_info:
        ensure_gpu_allocation()

    assert exc_info.value.code == 1
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert events == [("confirm", "compgpu8", [1]), ("remote_tmux", "compgpu8", [1])]


def test_ensure_exits_when_no_gpu_available(monkeypatch):
    _patch_ensure(monkeypatch, {})
    with pytest.raises(SystemExit):
        ensure_gpu_allocation()


def test_ensure_reports_when_no_gpu_anywhere(monkeypatch, capsys):
    _patch_ensure(monkeypatch, {})
    with pytest.raises(SystemExit):
        ensure_gpu_allocation()
    err = capsys.readouterr().err
    assert "[mldft-gpu] Finding free GPU..." in err
    assert "[mldft-gpu] No free GPU could be found on any of the servers:" in err


def test_ensure_reports_partial_availability(monkeypatch, capsys):
    # Request 2 GPUs but only 1 is free on any single host.
    _patch_ensure(monkeypatch, {"compgpu12": [1]}, env={"MLDFT_NUM_GPUS": "2"})
    with pytest.raises(SystemExit):
        ensure_gpu_allocation()
    err = capsys.readouterr().err
    assert "[mldft-gpu] Finding free GPUs..." in err
    assert "Only 1 GPU free on any single host, but 2 GPUs requested." in err


def test_ensure_skips_recheck_after_quick_confirmation(monkeypatch):
    env, events = _patch_ensure(monkeypatch, {}, fallback_hosts=())
    probed_hosts: list[str] = []
    monkeypatch.setattr(ga, "probe_host", lambda host, lh: probed_hosts.append(host) or [2])
    clock = iter([0.0, ga.RECHECK_AFTER_SECONDS])
    monkeypatch.setattr(ga.time, "monotonic", lambda: next(clock))

    ensure_gpu_allocation()

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert probed_hosts == ["compgpu12"]  # no second probe when confirmed in time


def test_ensure_rechecks_after_slow_confirmation_and_keeps_free_gpus(monkeypatch):
    env, events = _patch_ensure(monkeypatch, {}, fallback_hosts=())
    probed_hosts: list[str] = []
    monkeypatch.setattr(ga, "probe_host", lambda host, lh: probed_hosts.append(host) or [2])
    clock = iter([0.0, 31.0])
    monkeypatch.setattr(ga.time, "monotonic", lambda: next(clock))

    ensure_gpu_allocation()

    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert probed_hosts == ["compgpu12", "compgpu12"]  # selection probe + re-check
    assert events == [("confirm", "compgpu12", [2])]


def test_ensure_reallocates_when_gpus_taken_during_confirmation(monkeypatch, capsys):
    env, events = _patch_ensure(monkeypatch, {}, fallback_hosts=())
    # Round 1 finds GPU 2, the slow confirmation triggers a re-check that shows
    # it taken; round 2 finds GPU 3, which is confirmed in time.
    probes = iter([[2], [], [3]])
    monkeypatch.setattr(ga, "probe_host", lambda host, lh: next(probes))
    clock = iter([0.0, 31.0, 40.0, 41.0])
    monkeypatch.setattr(ga.time, "monotonic", lambda: next(clock))

    ensure_gpu_allocation()

    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert events == [("confirm", "compgpu12", [2]), ("confirm", "compgpu12", [3])]
    err = capsys.readouterr().err
    assert "GPU 2 on compgpu12 (H200) no longer free, searching again..." in err


def test_build_remote_job_command_uses_local_interactive_shell(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mldft_train", "foo=bar"])
    monkeypatch.setenv("SHELL", "/bin/zsh")

    command = _build_remote_job_command([1], "mldft-train-1234")

    assert command.startswith("sh -c ")
    assert "sleep 0.2" in command
    assert "exec /bin/zsh -ic " in command
    assert '"shell": "/bin/zsh"' in command  # also pinned for the inner job shell
    assert "python3 -c" in command
    assert "CUDA_VISIBLE_DEVICES" in command
    assert "mldft_train" in command
    assert "foo=bar" in command
    assert "Command exited with status" in command
    assert "mldft-train-1234" in command


def test_current_command_pins_absolute_launcher_path(monkeypatch):
    # The remote host must run the exact same console script as the launching
    # host, not whatever its own PATH resolves the bare command name to.
    monkeypatch.setattr("sys.argv", ["/some/venv/bin/mldft_train", "foo=bar"])

    assert rt._current_command() == "/some/venv/bin/mldft_train foo=bar"


def test_current_command_resolves_path_launcher(monkeypatch, tmp_path):
    launcher = tmp_path / "bin" / "mldft_train"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    monkeypatch.setenv("PATH", str(launcher.parent))
    monkeypatch.setattr("sys.argv", ["mldft_train", "foo=bar"])

    assert rt._current_command() == f"{launcher} foo=bar"


def test_current_command_resolves_relative_launcher_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["bin/mldft_train"])

    assert rt._current_command() == str(tmp_path / "bin" / "mldft_train")


def test_build_remote_job_command_falls_back_to_remote_shell(monkeypatch):
    monkeypatch.setattr("sys.argv", ["mldft_train"])
    monkeypatch.delenv("SHELL", raising=False)

    command = _build_remote_job_command([1], "mldft-train-1234")

    assert 'exec "${SHELL:-/bin/bash}" -ic ' in command
    assert '"shell": ""' in command  # the inner job shell falls back remotely too


def test_remote_tmux_banner_lines_include_only_tmux_specific_context():
    lines = _remote_tmux_banner_lines("mldft-train-1234")

    assert lines == (
        "[mldft-gpu] tmux session: mldft-train-1234",
        "[mldft-gpu] Streaming through launcher proxy; remote tmux remains persistent.",
        "",
    )


def test_build_tmux_create_command_uses_python_and_tmux_start_directory(tmp_path):
    command = _build_tmux_create_command(
        "mldft-train-1234",
        tmp_path,
        "exec /bin/zsh -ic 'echo ok'",
    )

    assert command.startswith("python3 -c ")
    assert rt.TMUX_MISSING_SENTINEL in command
    assert ".gpulogs/mldft-train-1234.log" in command
    assert ".gpulogs/mldft-train-1234.status" in command
    assert "shutil.which" in command
    assert "tmux" in command
    assert "new-session" in command
    assert '"-c", cwd' in command
    assert str(tmp_path) in command
    assert "set-option" in command
    assert "history-limit" in command
    assert "50000" in command
    assert "pipe-pane" in command
    assert "cat >> " in command


def test_build_tmux_proxy_command_streams_and_forwards_input():
    command = _build_tmux_proxy_command("mldft-train-1234")

    assert command.startswith("python3 -c ")
    assert "tmux attach-session" not in command
    assert "Proxy stdin/stdout between the launcher and a persistent tmux session" in command
    assert "send-keys" in command
    assert "C-c" in command
    assert "kill-session" in command
    assert "ProxyDisconnected" in command
    assert "stdout.buffer.write" in command
    assert ".gpulogs/mldft-train-1234.log" in command
    assert ".gpulogs/mldft-train-1234.status" in command
    assert "unlink" in command


def test_create_remote_tmux_session_uses_ssh(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr("sys.argv", ["mldft_train", "foo=bar"])
    monkeypatch.setenv("SHELL", "/bin/zsh")

    def fake_run(command, *, check, capture_output, text):
        captured.append(
            {
                "command": command,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(rt.subprocess, "run", fake_run)

    _create_remote_tmux_session("compgpu8", "mldft-train-1234", [1], tmp_path)

    call = captured[0]
    command = call["command"]
    assert call["check"] is False
    assert call["capture_output"] is True
    assert call["text"] is True
    assert command[0] == "ssh"
    assert command[-2] == "compgpu8"
    assert command[-1].startswith("python3 -c ")
    assert "new-session" in command[-1]
    assert "mldft-train-1234" in command[-1]
    assert '"-c", cwd' in command[-1]
    assert str(tmp_path) in command[-1]
    assert ".gpulogs/mldft-train-1234.log" in command[-1]
    assert ".gpulogs/mldft-train-1234.status" in command[-1]
    assert "set-option" in command[-1]
    assert "history-limit" in command[-1]
    assert "50000" in command[-1]
    assert "pipe-pane" in command[-1]
    assert "cat >> " in command[-1]
    assert '\\"cuda\\": \\"1\\"' in command[-1]
    assert "CUDA_VISIBLE_DEVICES" in command[-1]
    assert "[mldft-gpu] Running on compgpu8 with GPU 1 via remote tmux" not in command[-1]
    assert "CUDA_VISIBLE_DEVICES=1 mldft_train foo=bar" not in command[-1]
    assert (
        "[mldft-gpu] Streaming through launcher proxy; remote tmux remains persistent."
        in command[-1]
    )
    assert "Command exited with status" in command[-1]


def test_create_remote_tmux_session_shows_ssh_setup_hint(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["mldft_train"])
    monkeypatch.setattr(
        rt.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            255,
            stdout="",
            stderr="ssh: Could not resolve hostname compgpu8: Name or service not known\n",
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        _create_remote_tmux_session("compgpu8", "mldft-train-1234", [1], tmp_path)

    message = str(exc_info.value)
    assert "Could not resolve hostname compgpu8" in message
    assert "ssh compgpu8" in message
    assert "ssh-keygen -t ed25519 -f ~/.ssh/cluster" in message
    assert "Host compgpu*" in message
    assert "IdentityFile ~/.ssh/cluster" in message
    assert "ssh-keyscan -t ed25519 compgpu8 >> ~/.ssh/known_hosts" in message


def test_create_remote_tmux_session_errors_when_tmux_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["mldft_train"])
    monkeypatch.setattr(
        rt.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 127, stdout="", stderr=f"{rt.TMUX_MISSING_SENTINEL}\n"
        ),
    )

    with pytest.raises(SystemExit, match="does not provide tmux"):
        _create_remote_tmux_session("compgpu8", "mldft-train-1234", [1], tmp_path)


def test_proxy_remote_tmux_session_exits_with_proxy_status(monkeypatch):
    monkeypatch.setattr(rt, "_run_remote_tmux_proxy", lambda host, session_name: 12)

    with pytest.raises(SystemExit) as exc_info:
        _proxy_remote_tmux_session("compgpu8", "mldft-train-1234")

    assert exc_info.value.code == 12


def test_run_in_remote_tmux_creates_session_then_proxies(monkeypatch, tmp_path, capsys):
    events = []

    monkeypatch.setattr(rt, "_session_name_from_timestamp", lambda: "mldft-train-1234")
    monkeypatch.setattr(
        rt,
        "_create_remote_tmux_session",
        lambda host, session_name, gpu_ids, cwd: events.append(
            ("create", host, session_name, gpu_ids, cwd)
        ),
    )

    def fake_proxy(host, session_name):
        events.append(("proxy", host, session_name))
        raise SystemExit(0)

    monkeypatch.setattr(rt, "_proxy_remote_tmux_session", fake_proxy)

    with pytest.raises(SystemExit) as exc_info:
        rt.run_in_remote_tmux("compgpu8", [1], tmp_path)

    assert exc_info.value.code == 0
    assert events == [
        ("create", "compgpu8", "mldft-train-1234", [1], tmp_path),
        ("proxy", "compgpu8", "mldft-train-1234"),
    ]
    err = capsys.readouterr().err
    assert "Created tmux session mldft-train-1234 on compgpu8" in err


def test_run_remote_tmux_proxy_streams_without_tty(monkeypatch):
    captured_command: list[str] = []
    signal_calls = []
    context_events = []

    class FakeProcess:
        def wait(self):
            return 12

    def fake_popen(command):
        captured_command.extend(command)
        return FakeProcess()

    def fake_signal(signum, handler):
        signal_calls.append((signum, handler))
        return "old-handler"

    @contextlib.contextmanager
    def fake_forward_mode():
        context_events.append("enter")
        yield
        context_events.append("exit")

    monkeypatch.setattr(rt.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(rt.signal, "signal", fake_signal)
    monkeypatch.setattr(rt, "_forward_stdin_control_bytes", fake_forward_mode)

    status = rt._run_remote_tmux_proxy("compgpu8", "mldft-train-1234")

    assert status == 12
    assert captured_command[0] == "ssh"
    assert "-t" not in captured_command
    assert captured_command[-2] == "compgpu8"
    assert captured_command[-1].startswith("python3 -c ")
    assert "Proxy stdin/stdout between the launcher" in captured_command[-1]
    assert "tmux attach-session" not in captured_command[-1]
    assert context_events == ["enter", "exit"]
    assert signal_calls[0] == (rt.signal.SIGINT, rt.signal.SIG_IGN)
    assert signal_calls[1] == (rt.signal.SIGINT, "old-handler")


def test_terminal_attrs_for_proxy_input_uses_cbreak_without_local_signals():
    old_attrs = [0, 0, 0, rt.termios.ICANON | rt.termios.ECHO | rt.termios.ISIG, 0, 0, [0] * 32]

    proxy_attrs = rt._terminal_attrs_for_proxy_input(old_attrs)

    # The helper must not mutate the saved attrs; those are restored afterwards.
    assert proxy_attrs is not old_attrs
    assert proxy_attrs[6] is not old_attrs[6]

    # Cbreak mode gives byte-by-byte input and disables local echo while streaming.
    assert proxy_attrs[3] & rt.termios.ICANON == 0
    assert proxy_attrs[3] & rt.termios.ECHO == 0
    assert proxy_attrs[6][rt.termios.VMIN] == 1
    assert proxy_attrs[6][rt.termios.VTIME] == 0

    # Ctrl-C must be forwarded to remote tmux, not converted into local SIGINT.
    assert proxy_attrs[3] & rt.termios.ISIG == 0

    # The original attrs should still describe the normal terminal mode.
    assert old_attrs[3] & rt.termios.ISIG
    assert old_attrs[6][rt.termios.VMIN] == 0


def test_forward_stdin_control_bytes_restores_terminal_attrs(monkeypatch):
    old_attrs = [0, 0, 0, rt.termios.ICANON | rt.termios.ECHO | rt.termios.ISIG, 0, 0, [0] * 32]
    proxy_attrs = [0, 0, 0, 0, 0, 0, [0] * 32]
    applied_attrs = []

    class FakeStdin:
        def isatty(self):
            return True

        def fileno(self):
            return 99

    def fake_tcgetattr(fd):
        assert fd == 99
        return [*old_attrs[:6], list(old_attrs[6])]

    def fake_tcsetattr(fd, when, attrs):
        assert fd == 99
        assert when == rt.termios.TCSADRAIN
        applied_attrs.append([*attrs[:6], list(attrs[6])])

    monkeypatch.setattr(rt.sys, "stdin", FakeStdin())
    monkeypatch.setattr(rt.termios, "tcgetattr", fake_tcgetattr)
    monkeypatch.setattr(rt.termios, "tcsetattr", fake_tcsetattr)
    monkeypatch.setattr(rt, "_terminal_attrs_for_proxy_input", lambda attrs: proxy_attrs)

    with rt._forward_stdin_control_bytes():
        pass

    assert applied_attrs == [proxy_attrs, old_attrs]


def test_proxy_returns_failure_when_session_disappears_without_status(monkeypatch, tmp_path):
    log_path = tmp_path / "job.log"
    status_path = tmp_path / "job.status"
    log_path.write_text("partial output\n", encoding="utf-8")

    monkeypatch.setattr(proxy, "LOG_SETTLE_POLLS", 0)
    monkeypatch.setattr(proxy, "_stdin_fd", lambda: None)
    monkeypatch.setattr(proxy, "_tmux_session_exists", lambda session_name: False)

    assert proxy._proxy_until_done("mldft-train-1234", log_path, status_path) == 1


def test_proxy_maps_stdin_bytes_to_tmux_keys(monkeypatch):
    commands = []
    monkeypatch.setattr(proxy, "_run_tmux", lambda command: commands.append(command))

    proxy._forward_stdin_bytes("mldft-train-1234", b"hello\n\x03tail\x04")

    assert commands == [
        ["tmux", "send-keys", "-t", "mldft-train-1234", "-l", "hello"],
        ["tmux", "send-keys", "-t", "mldft-train-1234", "Enter"],
        ["tmux", "send-keys", "-t", "mldft-train-1234", "C-c"],
        ["tmux", "send-keys", "-t", "mldft-train-1234", "-l", "tail"],
        ["tmux", "send-keys", "-t", "mldft-train-1234", "C-d"],
    ]


def test_proxy_cleans_up_only_after_status_and_final_output(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "job.log"
    status_path = tmp_path / "job.status"
    log_path.write_text("final output\n", encoding="utf-8")
    status_path.write_text("7\n", encoding="utf-8")
    killed = []

    monkeypatch.setattr(proxy, "LOG_SETTLE_POLLS", 0)
    monkeypatch.setattr(proxy, "_stdin_fd", lambda: None)
    monkeypatch.setattr(proxy, "_tmux_session_exists", lambda session_name: True)
    monkeypatch.setattr(
        proxy, "_kill_tmux_session", lambda session_name: killed.append(session_name)
    )

    status = proxy._proxy_until_done("mldft-train-1234", log_path, status_path)

    assert status == 7
    captured = capsys.readouterr()
    assert captured.out == "final output\n"
    assert "Final output delivered; closed remote tmux session." in captured.err
    assert killed == ["mldft-train-1234"]
    assert not log_path.exists()
    assert not status_path.exists()


def test_proxy_disconnect_does_not_cleanup(monkeypatch, tmp_path):
    log_path = tmp_path / "job.log"
    status_path = tmp_path / "job.status"
    log_path.write_text("partial\n", encoding="utf-8")
    status_path.write_text("7\n", encoding="utf-8")

    def fail_copy(log_path, offset):
        raise proxy.ProxyDisconnected

    def fail_cleanup(*args):
        raise AssertionError("disconnect must preserve remote tmux state")

    monkeypatch.setattr(proxy, "_stdin_fd", lambda: None)
    monkeypatch.setattr(proxy, "_copy_new_log_bytes", fail_copy)
    monkeypatch.setattr(proxy, "_cleanup_delivered_session", fail_cleanup)

    with pytest.raises(proxy.ProxyDisconnected):
        proxy._proxy_until_done("mldft-train-1234", log_path, status_path)

    assert log_path.exists()
    assert status_path.exists()


def test_run_tmux_job_writes_status_and_keeps_session_alive(monkeypatch, tmp_path, capsys):
    # Spec: after the command exits, the job wrapper records the status and keeps
    # the tmux session alive (by exec'ing an interactive shell) so that only a
    # connected proxy may later clean it up.
    status_path = tmp_path / "job.status"
    config = {
        "command": "mldft_train",
        "cuda": "1",
        "shell": "/bin/zsh",
        "status_path": str(status_path),
        "banner_lines": ["banner line"],
    }
    run_job_calls = []
    execl_calls = []
    monkeypatch.setattr("sys.argv", ["run_tmux_job.py", json.dumps(config)])
    monkeypatch.setattr(
        run_job, "_run_job", lambda command, cuda, shell: run_job_calls.append(shell) or 7
    )
    monkeypatch.setattr(run_job.os, "execl", lambda *args: execl_calls.append(args))
    # The pinned launching-host shell must win over the remote $SHELL, which is
    # unreliable over non-interactive SSH.
    monkeypatch.setenv("SHELL", "/bin/bash")

    run_job.main()

    assert status_path.read_text(encoding="utf-8") == "7\n"
    out = capsys.readouterr().out
    assert "banner line" in out
    assert "Command exited with status 7." in out
    assert run_job_calls == ["/bin/zsh"]
    assert execl_calls == [("/bin/zsh", "/bin/zsh", "-i")]
