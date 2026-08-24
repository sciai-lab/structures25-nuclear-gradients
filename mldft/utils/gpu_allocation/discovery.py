"""GPU discovery and allocation helpers."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Sequence

from mldft.utils.gpu_allocation.ssh import SSH_OPTIONS

# A GPU is considered free if its used memory is below this fraction of its total
# memory, and it has no active compute process.
FREE_MEM_FRACTION = 0.1

# nvidia-smi is queried directly instead of parsing gpustat: nvidia-smi ships
# with every driver, needs no virtualenv, and the CSV output is trivial to
# parse robustly. We need two queries:
# 1. Enumerate all GPUs with their memory usage and UUIDs
# 2. List the UUIDs of all GPUs with active compute processes to exclude them
GPU_QUERY: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=index,uuid,memory.used,memory.total",
    "--format=csv,noheader,nounits",
)
PROCESS_QUERY: tuple[str, ...] = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid",
    "--format=csv,noheader,nounits",
)


def parse_free_gpu_ids(gpu_output: str, process_output: str) -> list[int]:
    """Parse the two nvidia-smi query outputs into free GPU indices.

    A GPU counts as free when it has no active compute process and its
    used-memory fraction is below ``FREE_MEM_FRACTION``.
    """
    busy_uuids = {line.strip() for line in process_output.splitlines() if line.strip()}

    free_gpu_ids: list[int] = []
    for line in gpu_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        index_str, uuid, used_str, total_str = (part.strip() for part in line.split(","))
        total = float(total_str)

        if total <= 0 or uuid in busy_uuids:
            continue

        if float(used_str) / total < FREE_MEM_FRACTION:
            free_gpu_ids.append(int(index_str))

    return free_gpu_ids


def _query_host(host: str, local_host: str, query: Sequence[str]) -> str | None:
    """Run an nvidia-smi query on ``host`` and return its stdout, or ``None``."""
    if host == local_host:
        command: list[str] = list(query)
    else:
        command = ["ssh", *SSH_OPTIONS, host, shlex.join(query)]

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None

    return result.stdout if result.returncode == 0 else None


def probe_host(host: str, local_host: str) -> list[int]:
    """Return the free GPU indices on ``host`` (locally or via SSH)."""
    gpu_output = _query_host(host, local_host, GPU_QUERY)
    if gpu_output is None:
        return []

    process_output = _query_host(host, local_host, PROCESS_QUERY)
    if process_output is None:
        return []
    try:
        return parse_free_gpu_ids(gpu_output, process_output)
    except (ValueError, IndexError):
        return []


def select_allocation(
    hosts: Sequence[str], num_gpus: int, probe: Callable[[str], list[int]]
) -> tuple[tuple[str, list[int]] | None, int]:
    """Pick the first host with at least ``num_gpus`` free GPUs.

    Returns the ``(host, gpu_ids)`` selection (or ``None`` if no single host has
    enough) together with the largest number of free GPUs seen on any one host,
    so the caller can report how close it got. GPUs are only ever taken from a
    single host, so we never end up using one GPU on one node and another GPU on
    a different node.
    """
    best_free = 0
    for host in hosts:
        free = probe(host)
        best_free = max(best_free, len(free))
        if len(free) >= num_gpus:
            return (host, list(free[:num_gpus])), best_free
    return None, best_free
