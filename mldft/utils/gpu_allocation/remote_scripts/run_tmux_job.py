"""Run the selected GPU command inside a remote tmux pane."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_job(command: str, cuda: str, shell: str) -> int:
    """Run the requested user command with ``CUDA_VISIBLE_DEVICES`` set."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda
    try:
        return subprocess.run([shell, "-ic", command], env=env, check=False).returncode
    except OSError as exc:
        print(f"[mldft-gpu] Failed to start command: {exc}", file=sys.stderr, flush=True)
        return 127


def main() -> None:
    """Main entry point for running the GPU job inside tmux."""
    config = json.loads(sys.argv[1])
    command = config["command"]
    banner_lines = config["banner_lines"]
    cuda = config["cuda"]
    # The launching host's shell; the remote $SHELL over non-interactive SSH is
    # unreliable (often /bin/bash even for zsh users) and would source the
    # wrong rc files.
    shell = config["shell"] or os.environ.get("SHELL") or "/bin/bash"
    status_path = Path(config["status_path"]).expanduser()

    status_path.parent.mkdir(parents=True, exist_ok=True)
    for line in banner_lines:
        print(line, flush=True)

    status = _run_job(command, cuda, shell)
    print(f"\n[mldft-gpu] Command exited with status {status}.", flush=True)
    status_path.write_text(f"{status}\n", encoding="utf-8")

    # Keep the tmux session alive after the job ends; only a connected proxy
    # that has delivered the final output is allowed to kill it.
    os.execl(shell, shell, "-i")


if __name__ == "__main__":
    main()
