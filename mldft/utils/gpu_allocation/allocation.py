"""Automatic GPU allocation for the console-script entry points.

When one of the ``[project.scripts]`` commands runs (``mldft_train``,
``mldft_denop``, ...), we transparently pick free GPUs and export
``CUDA_VISIBLE_DEVICES`` *before* the ML stack is imported. If the local machine
has no sufficiently free GPUs, we create a persistent tmux session on a remote
host and proxy the current terminal to it (never mixing GPUs from different
hosts).

Behavior is controlled without CLI flags, so that hydra keeps full control over
``argv``:

- If ``CUDA_VISIBLE_DEVICES`` is already set, we do nothing (manual override wins).
- ``MLDFT_NUM_GPUS``: number of GPUs to request (default ``1``). A GPU counts
  as free when it has no active compute process and ``memory.used /
  memory.total < 0.1``.
- ``.gpus``: optional fallback-host list, one host per line, searched from the
  current directory upward. If absent, the hard-coded fallback hosts are used.
- If confirming the selection takes longer than 30 seconds, the chosen GPUs are
  probed again and the search restarts when they were taken in the meantime.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterable, MutableMapping, Sequence

from mldft.utils.gpu_allocation.discovery import (
    probe_host,
    select_allocation,
)
from mldft.utils.gpu_allocation.remote_tmux import (
    _current_command,
)
from mldft.utils.gpu_allocation.remote_tmux import (
    run_in_remote_tmux as _run_in_remote_tmux,
)


def _gpus_word(count: int) -> str:
    """Return ``"GPU"`` or ``"GPUs"`` depending on ``count``.

    You can override the default fallback hosts by creating a ``.gpus`` file in the
    project root. See the docstring of this file for details.
    """
    return "GPU" if count == 1 else "GPUs"


FALLBACK_HOSTS: tuple[str, ...] = (
    # NVIDIA A100
    "compgpu7",
    "compgpu8",
    "compgpu9",
    "compgpu10",
    "compgpu13",
    "compgpu14",
)

FALLBACK_HOSTS_CONFIG = ".gpus"

# If the user takes longer than this to confirm the selection, the chosen GPUs
# are probed again before use, since they may have been taken in the meantime.
RECHECK_AFTER_SECONDS = 30

GPU_TYPES_BY_HOST: dict[str, str] = {
    "compgpu1": "Titan Xp",
    "compgpu2": "Titan Xp",
    "compgpu3": "Titan Xp",
    "compgpu4": "GeForce RTX 2080",
    "compgpu5": "GeForce RTX 2080",
    "compgpu6": "GeForce RTX 2080",
    "compgpu7": "A100",
    "compgpu8": "A100",
    "compgpu9": "A100",
    "compgpu10": "A100",
    "compgpu11": "H100",
    "compgpu12": "H200",
    "compgpu13": "A100",
    "compgpu14": "A100",
    "compgpu15": "RTX Pro 6000",
    "hcigpu05": "Quadro RTX 6000",
    "hcigpu06": "Quadro RTX 6000",
    "hcigpu07": "Quadro RTX 6000",
    "hcigpu08": "Quadro RTX 8000",
    "hgsgpu01": "Quadro P5000",
}


def _host_label(host: str) -> str:
    """Return ``host`` with the GPU type appended when it is known."""
    gpu_type = GPU_TYPES_BY_HOST.get(host)
    if gpu_type is None:
        return host
    return f"{host} ({gpu_type})"


def _fallback_hosts_config_path(start: Path) -> Path | None:
    """Return the nearest fallback-host config at or above ``start``."""
    for directory in (start, *start.parents):
        path = directory / FALLBACK_HOSTS_CONFIG
        if path.is_file():
            return path
    return None


def _fallback_hosts(start: Path) -> tuple[str, ...]:
    """Return configured fallback hosts, or defaults when no config exists."""
    config_path = _fallback_hosts_config_path(start)
    if config_path is None:
        return FALLBACK_HOSTS

    hosts: list[str] = []
    for line in config_path.read_text(encoding="utf-8").splitlines():
        host = line.strip()
        if host and not host.startswith("#"):
            hosts.append(host)
    return tuple(hosts)


def _num_gpus_from_env(environ: MutableMapping[str, str]) -> int:
    """Return the requested GPU count from ``MLDFT_NUM_GPUS`` (>= 1)."""
    raw_num_gpus = environ.get("MLDFT_NUM_GPUS")
    if raw_num_gpus is None or raw_num_gpus == "":
        return 1

    message = f"MLDFT_NUM_GPUS must be a positive integer, got {raw_num_gpus!r}."
    try:
        requested = int(raw_num_gpus)
    except ValueError as exc:
        raise SystemExit(message) from exc

    if requested < 1:
        raise SystemExit(message)
    return requested


def _dedup(items: Iterable[str]) -> tuple[str, ...]:
    """Return items in first-seen order without duplicates."""
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def _report(message: str) -> None:
    """Print a short allocator status line."""
    print(f"[mldft-gpu] {message}", file=sys.stderr, flush=True)


def _prompt_or_abort(message: str) -> None:
    """Print ``message`` and wait for Enter, aborting on Ctrl-C or EOF."""
    print(f"[mldft-gpu] {message}", file=sys.stderr, end="", flush=True)
    try:
        confirmed = sys.stdin.readline()
    except KeyboardInterrupt as exc:
        print(file=sys.stderr)
        raise SystemExit("GPU allocation confirmation aborted.") from exc
    if confirmed == "":
        raise SystemExit("GPU allocation confirmation aborted.")


def _confirm_allocation(host: str, gpu_ids: Sequence[int]) -> None:
    """Prompt for confirmation before using the selected GPUs."""
    cuda = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    word = _gpus_word(len(gpu_ids))
    _prompt_or_abort(
        f"Press Enter to use {word} {cuda} on {_host_label(host)} (Ctrl-C to abort): "
    )


def ensure_gpu_allocation() -> None:
    """Export ``CUDA_VISIBLE_DEVICES`` or proxy to a remote tmux run if unset."""
    environ = os.environ
    if "CUDA_VISIBLE_DEVICES" in environ:
        return

    num_gpus = _num_gpus_from_env(environ)
    local_host = socket.gethostname().split(".")[0]

    hosts = _dedup([local_host, *_fallback_hosts(Path.cwd())])
    gpus_word = _gpus_word(num_gpus)
    _report(f"Finding free {gpus_word}...")
    while True:
        probed_at = time.monotonic()
        selection, best_free = select_allocation(
            hosts, num_gpus, lambda host: probe_host(host, local_host)
        )
        if selection is None:
            if best_free >= 1:
                _report(
                    f"Only {best_free} {_gpus_word(best_free)} free on any single host, "
                    f"but {num_gpus} {gpus_word} requested."
                )
            _report(
                f"No free {gpus_word} could be found on any of the servers: {', '.join(hosts)}"
            )
            raise SystemExit(1)

        host, gpu_ids = selection
        _confirm_allocation(host, gpu_ids)
        if time.monotonic() - probed_at <= RECHECK_AFTER_SECONDS:
            break
        if set(gpu_ids) <= set(probe_host(host, local_host)):
            break
        taken = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        _report(
            f"{_gpus_word(len(gpu_ids))} {taken} on {_host_label(host)} "
            "no longer free, searching again..."
        )

    cuda = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    word = _gpus_word(len(gpu_ids))
    command = _current_command()

    on_local = host == local_host
    if on_local:
        _report(f"Using {word} {cuda} on {_host_label(host)}")
    else:
        _report(f"Running on {_host_label(host)} with {word} {cuda} via remote tmux")
    print(f"CUDA_VISIBLE_DEVICES={cuda} {command}\n", file=sys.stderr, flush=True)

    if on_local:
        environ["CUDA_VISIBLE_DEVICES"] = cuda
    else:
        _run_in_remote_tmux(host, gpu_ids, Path.cwd())
        raise SystemExit(1)
