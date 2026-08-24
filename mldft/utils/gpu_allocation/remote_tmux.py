"""Remote tmux session handling for GPU launcher commands."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import termios
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from typing import NoReturn

from mldft.utils.gpu_allocation.ssh import SSH_OPTIONS, ssh_setup_hint

TMUX_MISSING_SENTINEL = "__MLDFT_TMUX_MISSING__"
TMUX_HISTORY_LIMIT = 50_000
REMOTE_SCRIPT_PACKAGE = "mldft.utils.gpu_allocation.remote_scripts"
SESSION_NAME_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _current_command_argv() -> list[str]:
    """Return the current console command with its original arguments.

    The executable is pinned to its resolved absolute path so the remote host
    runs the exact same console script -- and, via its shebang, the same
    virtualenv python -- as the launching host, instead of whatever the remote
    shell's ``PATH`` resolves the bare command name to. The identical mount
    setup on all our servers makes local paths valid on every host.
    """
    executable = sys.argv[0]
    if not Path(executable).parent.parts:
        executable = shutil.which(executable) or executable
    return [str(Path(executable).absolute()), *sys.argv[1:]]


def _current_command() -> str:
    """Return the current console command as a shell-escaped string."""
    return shlex.join(_current_command_argv())


def _session_name_prefix() -> str:
    """Return a tmux-safe prefix based on the current console command."""
    command_name = Path(sys.argv[0]).name
    safe_name = SESSION_NAME_UNSAFE_CHARS.sub("_", command_name).strip("_-")
    return safe_name or "mldft"


def _session_name_from_timestamp(now: datetime | None = None) -> str:
    """Return a tmux-safe command-prefixed timestamped session name."""
    when = now or datetime.now()
    return f"{_session_name_prefix()}_{when:%Y%m%d_%H%M%S}"


def _remote_session_path(session_name: str, suffix: str) -> str:
    """Return a remote per-session path under the user's home directory."""
    return f"~/.gpulogs/{session_name}.{suffix}"


def _remote_script_source(script_name: str) -> str:
    """Return the source for a bundled remote-side helper script."""
    return files(REMOTE_SCRIPT_PACKAGE).joinpath(script_name).read_text(encoding="utf-8")


def _remote_python_command(script_name: str, config: Mapping[str, object]) -> str:
    """Return a remote shell command that runs a bundled script with JSON config."""
    source = _remote_script_source(script_name)
    return shlex.join(["python3", "-c", source, json.dumps(config)])


def _interactive_shell_command(command: str) -> str:
    """Return ``command`` wrapped in the user's interactive shell.

    The remote ``$SHELL`` is unreliable over non-interactive SSH (it is often
    ``/bin/bash`` even for zsh users), so the shell rc files that set up the
    environment, e.g. the ``uv`` project virtualenv, would not be sourced.
    We therefore pin the launching machine's ``$SHELL``, assuming the shared
    cluster home provides the same shell at the same path on the remote host.
    """
    local_shell = os.environ.get("SHELL")
    shell = shlex.quote(local_shell) if local_shell else '"${SHELL:-/bin/bash}"'
    # The brief delay lets the session-creation script attach ``pipe-pane`` for
    # log capture before the command starts producing output.
    inner = f"sleep 0.2; exec {shell} -ic {shlex.quote(command)}"
    return shlex.join(["sh", "-c", inner])


def _remote_tmux_banner_lines(session_name: str) -> tuple[str, ...]:
    """Return tmux-specific lines shown at the top of the remote pane."""
    return (
        f"[mldft-gpu] tmux session: {session_name}",
        "[mldft-gpu] Streaming through launcher proxy; remote tmux remains persistent.",
        "",
    )


def _build_remote_job_command(
    gpu_ids: Sequence[int], session_name: str, banner_lines: Sequence[str] = ()
) -> str:
    """Return the tmux job command to execute on the remote host."""
    cuda = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    command = _remote_python_command(
        "run_tmux_job.py",
        {
            "command": _current_command(),
            "cuda": cuda,
            # Pin the launching host's shell so the job sources the same rc
            # files; see the docstring of _interactive_shell_command.
            "shell": os.environ.get("SHELL", ""),
            "status_path": _remote_session_path(session_name, "status"),
            "banner_lines": list(banner_lines),
        },
    )
    return _interactive_shell_command(command)


def _build_tmux_create_command(session_name: str, cwd: Path, remote_job_command: str) -> str:
    """Return the remote shell command that creates the tmux session."""
    return _remote_python_command(
        "create_tmux_session.py",
        {
            "session_name": session_name,
            "cwd": str(cwd),
            "remote_job_command": remote_job_command,
            "log_path": _remote_session_path(session_name, "log"),
            "status_path": _remote_session_path(session_name, "status"),
            "tmux_missing_sentinel": TMUX_MISSING_SENTINEL,
            "history_limit": TMUX_HISTORY_LIMIT,
        },
    )


def _build_tmux_proxy_command(session_name: str) -> str:
    """Return the remote shell command that proxies tmux output and input."""
    return _remote_python_command(
        "proxy_tmux_session.py",
        {
            "session_name": session_name,
            "log_path": _remote_session_path(session_name, "log"),
            "status_path": _remote_session_path(session_name, "status"),
        },
    )


def _create_remote_tmux_session(
    host: str, session_name: str, gpu_ids: Sequence[int], cwd: Path
) -> None:
    """Create a detached tmux session on ``host`` that owns the remote job."""
    remote_job_command = _build_remote_job_command(
        gpu_ids, session_name, _remote_tmux_banner_lines(session_name)
    )
    remote_command = _build_tmux_create_command(session_name, cwd, remote_job_command)
    try:
        result = subprocess.run(
            ["ssh", *SSH_OPTIONS, host, remote_command],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SystemExit(
            f"Failed to create remote tmux session on {host}: {exc}\n\n" f"{ssh_setup_hint(host)}"
        ) from exc

    if result.returncode == 0:
        return

    stderr = result.stderr.strip()
    stdout = result.stdout.strip()
    if TMUX_MISSING_SENTINEL in stderr:
        raise SystemExit(
            f"Remote host {host} does not provide tmux. "
            "Remote GPU fallback requires tmux for persistence."
        )

    detail = stderr or stdout or f"ssh exited with status {result.returncode}."
    raise SystemExit(
        f"Failed to create tmux session {session_name} on {host}: {detail}\n\n"
        f"{ssh_setup_hint(host)}"
    )


def _terminal_attrs_for_proxy_input(old_attrs: list) -> list:
    """Return terminal attrs that make local control keys readable as bytes.

    The launcher proxy needs every keystroke -- including Ctrl-C -- to arrive on
    stdin as a raw byte, so it can be forwarded over SSH and become a ``tmux
    send-keys`` on the remote host. In the terminal's normal (cooked) mode this
    does not happen: input is line-buffered, echoed locally, and Ctrl-C/Ctrl-Z
    are swallowed to raise local signals instead.

    ``termios.tcgetattr`` returns ``[iflag, oflag, cflag, lflag, ispeed, ospeed,
    cc]``. On a copy we clear three input-processing flags in ``lflag`` and make
    reads return on the first byte. Output post-processing (``oflag``) is left
    untouched, so streamed newlines are still rendered correctly.
    """
    attrs = old_attrs[:]
    attrs[6] = list(old_attrs[6])
    # ICANON off: deliver input byte-by-byte instead of line-by-line.
    # ECHO off: the remote pane already echoes, so do not echo locally.
    # ISIG off: pass Ctrl-C/Ctrl-Z through as bytes instead of local signals.
    attrs[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG)
    attrs[6][termios.VMIN] = 1
    attrs[6][termios.VTIME] = 0
    return attrs


@contextlib.contextmanager
def _forward_stdin_control_bytes() -> Iterator[None]:
    """Temporarily let Ctrl-C and friends pass through stdin as bytes."""
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, _terminal_attrs_for_proxy_input(old_attrs))
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def _proxy_remote_tmux_session(host: str, session_name: str) -> NoReturn:
    """Proxy local input/output to the persistent remote tmux session."""
    status = _run_remote_tmux_proxy(host, session_name)
    raise SystemExit(status)


def _run_remote_tmux_proxy(host: str, session_name: str) -> int:
    """Stream tmux output and forward stdin without attaching a tmux client."""
    process = subprocess.Popen(
        ["ssh", *SSH_OPTIONS, host, _build_tmux_proxy_command(session_name)]
    )
    old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with _forward_stdin_control_bytes():
            return process.wait()
    finally:
        signal.signal(signal.SIGINT, old_sigint)


def run_in_remote_tmux(host: str, gpu_ids: Sequence[int], cwd: Path) -> NoReturn:
    """Run the current command inside a detached tmux session on ``host``."""
    session_name = _session_name_from_timestamp()
    _create_remote_tmux_session(host, session_name, gpu_ids, cwd)
    print(f"[mldft-gpu] Created tmux session {session_name} on {host}", file=sys.stderr)
    _proxy_remote_tmux_session(host, session_name)
