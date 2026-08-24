import multiprocessing
import os

import psutil
import threadpoolctl
import torch
from loguru import logger
from pyscf import lib


def set_num_threads(num_threads: int):
    """Set the number of threads to use in pyscf and numpy and torch."""
    # Locally pyscf lib works, on the cluster os.environ["OMP_NUM_THREADS"] = str(num_threads_per_process) is needed
    lib.num_threads(num_threads)
    # Other numpy backends might need other environment variables to be set, but this has worked for our purposes
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    # Limit MKL/OpenBLAS thread pools at runtime via their own APIs (env vars are read-only after import).
    threadpoolctl.threadpool_limits(limits=num_threads)


def resolve_processes_and_threads(
    num_processes: int | None = None,
    num_threads_per_process: int | None = None,
) -> tuple[int, int]:
    """Resolve process and thread counts without applying global side effects."""
    num_threads_per_process = num_threads_per_process if num_threads_per_process is not None else 1
    num_processes = (
        num_processes
        if num_processes is not None
        else max((multiprocessing.cpu_count() - 1) // num_threads_per_process, 1)
    )
    return num_processes, num_threads_per_process


def configure_processes_and_threads(
    num_processes: int | None = None,
    num_threads_per_process: int | None = None,
) -> tuple[int, int]:
    """Configure the number of processes and threads per process and set them in pyscf and numpy.

    If the number of threads is not specified, it is
    defaulted to 1. If the number of processes is not specified, the maximum available number of processes given the
    number of cpu cores and threads per process is used.

    Args:
        num_processes: The number of processes to use.
        num_threads_per_process: The number of threads per process to use.
    """
    num_processes, num_threads_per_process = resolve_processes_and_threads(
        num_processes, num_threads_per_process
    )
    set_num_threads(num_threads_per_process)
    return num_processes, num_threads_per_process


def get_memory_usage_in_mb():
    """Get the memory usage of the current process in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    mega_byte = 1048576
    return memory_info.rss / mega_byte  # in bytes


def resolve_max_memory_per_process(max_memory_per_process: int | float | None = 4000):
    """Resolve the maximum memory to use per process in MB without side effects."""
    return max_memory_per_process if max_memory_per_process is not None else 4000


def configure_max_memory_per_process(max_memory_per_process: int | float | None = 4000):
    """Configure the maximum memory to use per process in MB.

    Args:
        max_memory_per_process: The maximum memory to use per process in MB.
    """
    max_memory_per_process = resolve_max_memory_per_process(max_memory_per_process)
    logger.info(f"Setting max memory per process to {max_memory_per_process} MB.")
    os.environ["PYSCF_MAX_MEMORY"] = str(max_memory_per_process)
    return max_memory_per_process
