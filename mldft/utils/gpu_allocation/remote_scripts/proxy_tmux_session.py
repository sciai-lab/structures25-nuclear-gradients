"""Proxy stdin/stdout between the launcher and a persistent tmux session."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path

POLL_SECONDS = 0.1
LOG_SETTLE_POLLS = 3
KILL_SESSION_SETTLE_SECONDS = 0.1

CONTROL_KEYS = {
    0x03: "C-c",
    0x04: "C-d",
    0x09: "Tab",
    0x7F: "BSpace",
}


class ProxyDisconnected(Exception):
    """Raised when the launcher connection disappears while proxying output."""


def _tmux_session_exists(session_name: str) -> bool:
    """Return whether the tmux session still exists."""
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _run_tmux(command: list[str]) -> None:
    """Run a tmux command best-effort and suppress helper noise."""
    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _kill_tmux_session(session_name: str) -> None:
    """Best-effort kill for a session whose final output was delivered."""
    _run_tmux(["tmux", "kill-session", "-t", session_name])
    time.sleep(KILL_SESSION_SETTLE_SECONDS)


def _copy_new_log_bytes(log_path: Path, offset: int) -> int:
    """Copy newly appended log bytes to stdout and return the new offset."""
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(offset)
            data = log_file.read()
    except OSError:
        return offset

    if data:
        try:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        except BrokenPipeError as exc:
            raise ProxyDisconnected from exc
    return offset + len(data)


def _copy_settled_log_bytes(log_path: Path, offset: int) -> int:
    """Copy final log bytes after the job status marker appears."""
    for _ in range(LOG_SETTLE_POLLS):
        time.sleep(POLL_SECONDS)
        offset = _copy_new_log_bytes(log_path, offset)
    return offset


def _read_job_status(status_path: Path, fallback: int) -> int:
    """Return the recorded job status, or ``fallback`` if it cannot be read."""
    try:
        return int(status_path.read_text(encoding="utf-8", errors="replace").strip() or fallback)
    except (OSError, ValueError):
        return fallback


def _tmux_key_commands(data: bytes) -> Iterable[tuple[str, str]]:
    """Yield tmux input operations for bytes read from launcher stdin."""
    literal = bytearray()

    def flush() -> Iterable[tuple[str, str]]:
        if literal:
            yield ("literal", bytes(literal).decode("utf-8", errors="replace"))
            literal.clear()

    index = 0
    while index < len(data):
        byte = data[index]
        if byte in (0x0D, 0x0A):
            yield from flush()
            if byte == 0x0D and index + 1 < len(data) and data[index + 1] == 0x0A:
                index += 1
            yield ("key", "Enter")
        elif byte in CONTROL_KEYS:
            yield from flush()
            yield ("key", CONTROL_KEYS[byte])
        else:
            literal.append(byte)
        index += 1

    yield from flush()


def _forward_stdin_bytes(session_name: str, data: bytes) -> None:
    """Forward launcher stdin bytes into the tmux pane."""
    for kind, value in _tmux_key_commands(data):
        if kind == "literal":
            _run_tmux(["tmux", "send-keys", "-t", session_name, "-l", value])
        else:
            _run_tmux(["tmux", "send-keys", "-t", session_name, value])


def _stdin_fd() -> int | None:
    """Return stdin's file descriptor, or ``None`` when it cannot be used."""
    try:
        return sys.stdin.fileno()
    except OSError:
        return None


def _read_available_stdin(stdin_fd: int) -> bytes | None:
    """Return currently available stdin bytes, or ``None`` on EOF."""
    readable, _, _ = select.select([stdin_fd], [], [], 0)
    if not readable:
        return b""
    data = os.read(stdin_fd, 4096)
    return data or None


def _cleanup_delivered_session(session_name: str, log_path: Path, status_path: Path) -> None:
    """Remove remote state only after final output reached the launcher."""
    if _tmux_session_exists(session_name):
        _kill_tmux_session(session_name)
    for path in (log_path, status_path):
        path.unlink(missing_ok=True)


def _proxy_until_done(session_name: str, log_path: Path, status_path: Path) -> int:
    """Proxy output/input until the job finishes or the launcher disconnects."""
    offset = 0
    stdin_fd = _stdin_fd()

    while True:
        offset = _copy_new_log_bytes(log_path, offset)

        if stdin_fd is not None:
            data = _read_available_stdin(stdin_fd)
            if data is None:
                stdin_fd = None  # stdin reached EOF; stop polling it
            elif data:
                _forward_stdin_bytes(session_name, data)

        if status_path.is_file():
            offset = _copy_settled_log_bytes(log_path, offset)
            status = _read_job_status(status_path, 0)
            _cleanup_delivered_session(session_name, log_path, status_path)
            print(
                "[mldft-gpu] Final output delivered; closed remote tmux session.",
                file=sys.stderr,
                flush=True,
            )
            return status

        if not _tmux_session_exists(session_name):
            _copy_settled_log_bytes(log_path, offset)
            return 1

        time.sleep(POLL_SECONDS)


def main() -> None:
    """Main entry point for proxying a remote tmux session."""
    config = json.loads(sys.argv[1])
    session_name = config["session_name"]
    log_path = Path(config["log_path"]).expanduser()
    status_path = Path(config["status_path"]).expanduser()

    try:
        status = _proxy_until_done(session_name, log_path, status_path)
    except ProxyDisconnected:
        sys.exit(0)
    sys.exit(status)


if __name__ == "__main__":
    main()
