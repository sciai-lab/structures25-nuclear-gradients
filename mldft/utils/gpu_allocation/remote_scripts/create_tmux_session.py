"""Create and configure the remote tmux session for a GPU job."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _run_required(command: Sequence[str]) -> None:
    """Run ``command``; exit with its output and status when it fails."""
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode == 0:
        return

    detail = result.stderr or result.stdout
    if detail:
        print(detail, end="", file=sys.stderr)
    sys.exit(result.returncode)


def _run_optional(command: Sequence[str]) -> None:
    """Run a cosmetic ``command``, ignoring failures on older tmux versions."""
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    """Main entry point for creating and configuring the remote tmux session."""
    config = json.loads(sys.argv[1])
    session_name = config["session_name"]
    cwd = config["cwd"]
    remote_job_command = config["remote_job_command"]
    log_path = Path(config["log_path"]).expanduser()
    status_path = Path(config["status_path"]).expanduser()

    if shutil.which("tmux") is None:
        print(config["tmux_missing_sentinel"], file=sys.stderr)
        sys.exit(127)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    status_path.unlink(missing_ok=True)

    _run_required(["tmux", "new-session", "-d", "-s", session_name, "-c", cwd, remote_job_command])
    _run_optional(["tmux", "set-option", "-t", session_name, "mouse", "on"])
    _run_optional(
        ["tmux", "set-option", "-t", session_name, "history-limit", str(config["history_limit"])]
    )
    # pipe-pane captures the remote job's output for the launcher proxy and later recovery.
    _run_required(
        ["tmux", "pipe-pane", "-o", "-t", session_name, "cat >> " + shlex.quote(str(log_path))]
    )


if __name__ == "__main__":
    main()
