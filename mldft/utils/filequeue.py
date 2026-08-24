# python
"""Minimal distributed file-based work queue."""

import fcntl
import os
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import IO, Iterable

from loguru import logger


class TaskLease:
    """Context manager for a single leased task.

    Starts a heartbeat while the task is in use and marks completion on exit.
    """

    def __init__(self, queue: "FileQueue", task: Path):
        """Initialize the lease for the given task in the given queue.

        Args:
            queue: The FileQueue instance managing the task.
            task: The Path to the task file (in the FileQueue) that has been leased.
        """
        self.queue = queue
        self.task = task
        self._stop_evt = threading.Event()
        self._hb_thread: threading.Thread | None = None

    def _start_heartbeat(self) -> None:
        if self._hb_thread is not None:
            return
        self._hb_thread = threading.Thread(
            target=self.queue._heartbeat_loop,
            args=(self.task, self._stop_evt),
            daemon=True,
        )
        self._hb_thread.start()

    def __enter__(self) -> "TaskLease":
        self._start_heartbeat()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop_evt.set()
        if self._hb_thread:
            self._hb_thread.join(timeout=1.0)
        # On KeyboardInterrupt, requeue for later processing
        if exc_type is KeyboardInterrupt:
            self.queue._requeue_task(self.task)
        else:
            # Success if no exception escaped the `with` block.
            self.queue._mark_done(self.task, success=exc_type is None)


class FileQueue:
    def __init__(
        self,
        root: str,
        queue_dir: str = "filequeue",
        stale_after: float = 600,
        hb_interval: float | None = 120,
        recover_interval: float = 3600,
    ):
        """Initialize the file queue in the given root directory.

        The directory will contain subdirectories: todo, running, done, failed. When a job is running,
        a heartbeat file (.hb) will be created alongside it in the running/ directory. If it becomes stale,
        it will be moved back to todo/ for another worker to pick up.

        Args:
            root: Root directory for the queue.
            stale_after: Time in seconds to consider a running task stale and recover it.
            hb_interval: Time in seconds between heartbeats. Defaults to stale_after/5.
            recover_interval: Time in seconds between checks for stale tasks. Will only check once the next task is leased.
        """
        self.root = Path(root).resolve() / queue_dir
        self.todo = self.root / "todo"
        self.running = self.root / "running"
        self.done = self.root / "done"
        self.failed = self.root / "failed"
        self.lockfile = self.root / "queue.lock"
        self.stale_after = stale_after
        self.hb_interval = hb_interval or max(1.0, stale_after / 5)
        self.recover_interval = recover_interval
        self.last_recover = None

        for d in [self.todo, self.running, self.done, self.failed]:
            d.mkdir(parents=True, exist_ok=True)
        self.lockfile.touch(exist_ok=True)

    def _acquire_lock(self):
        lock_fd = open(self.lockfile, "r+")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return lock_fd

    @staticmethod
    def _release_lock(lock_fd: IO) -> None:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

    def _acquire_task(self) -> Path | None:
        st = time.perf_counter()
        while True:
            retry = False
            lock = self._acquire_lock()
            try:
                task_path: Path | None = None
                # Pick the first file, faster than reading all and sorting.
                with os.scandir(self.todo) as entries:
                    for entry in entries:
                        if entry.is_file():
                            task_path = Path(entry.path)
                            break
                if task_path is None:
                    return None
                dest = self.running / task_path.name
                try:
                    os.replace(str(task_path), str(dest))
                    # Create an initial heartbeat at lease time so tasks leased in a batch but not yet
                    # entered can still be recovered if the worker dies before starting them.
                    self._heartbeat_once(dest)
                    duration = (time.perf_counter() - st) * 1000
                    logger.trace(f"Acquired task {task_path.name} in {duration:.3f} ms")
                    return dest
                except FileNotFoundError:
                    logger.debug(
                        f"Task {task_path.name} disappeared before it could be leased; retrying..."
                    )
                    retry = True
            finally:
                self._release_lock(lock)
            if retry:
                time.sleep(0.05)
                continue

    def _acquire_tasks(self, max_items: int) -> list[Path]:
        if max_items <= 0:
            return []
        st = time.perf_counter()
        leased: list[Path] = []
        lock = self._acquire_lock()
        try:
            with os.scandir(self.todo) as entries:
                for entry in entries:
                    if len(leased) >= max_items:
                        break
                    if not entry.is_file():
                        continue
                    src = Path(entry.path)
                    dest = self.running / src.name
                    try:
                        os.replace(str(src), str(dest))
                        # Seed heartbeat on lease to make not-yet-entered batched tasks recoverable.
                        self._heartbeat_once(dest)
                        leased.append(dest)
                    except FileNotFoundError:
                        continue
        finally:
            self._release_lock(lock)
        if leased:
            duration = (time.perf_counter() - st) * 1000
            logger.trace(f"Acquired {len(leased)} tasks in one batch in {duration:.3f} ms")
        return leased

    def _mark_done(self, path: Path, success: bool):
        dest = self.done if success else self.failed
        if not path.exists():
            state = "done" if success else "failed"
            logger.warning(f"Task {path.name} missing from running/ while marking as {state}.")
            return
        try:
            os.replace(str(path), dest / path.name)
        except FileNotFoundError:
            logger.warning(
                f"Task {path.name} vanished during finalization; assuming already handled."
            )
        hb = path.with_suffix(path.suffix + ".hb")
        hb.unlink(missing_ok=True)

    def _requeue_task(self, path: Path):
        """Move a running task back to todo/ for re-processing (internal use)."""
        if not path.exists():
            logger.warning(f"Cannot requeue task {path.name} as it no longer exists")
            return
        dest = self.todo / path.name
        try:
            os.replace(str(path), dest)
        except FileNotFoundError:
            logger.warning(f"Failed to requeue task {path.name} because it disappeared; skipping.")
            return
        hb = path.with_suffix(path.suffix + ".hb")
        hb.unlink(missing_ok=True)
        logger.info(f"Requeued task {path.name} back to todo/")

    def requeue(self, path: Path):
        """Move a running task back to todo/ for re-processing (public API)."""
        src = self.running / path.name
        if not src.exists():
            logger.warning(f"Cannot requeue task {path.name} as it is not in running/")
            return
        dest = self.todo / path.name
        try:
            os.replace(str(src), dest)
        except FileNotFoundError:
            logger.warning(f"Failed to requeue task {path.name} because it disappeared; skipping.")
            return
        hb = src.with_suffix(src.suffix + ".hb")
        hb.unlink(missing_ok=True)
        logger.info(f"Requeued task {path.name} back to todo/")

    def _heartbeat_once(self, task: Path):
        hb = task.with_suffix(task.suffix + ".hb")
        hb.touch()

    def _heartbeat_loop(self, task: Path, stop_evt):
        while not stop_evt.is_set():
            self._heartbeat_once(task)
            stop_evt.wait(self.hb_interval)

    def _atomic_touch(self, path: Path):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
        os.close(fd)

    def recover_stale(self):
        """Move stale leases back to `todo/` based on heartbeat age."""
        st = time.perf_counter()
        now = time.time()
        for hb in self.running.glob("*.hb"):
            try:
                age = now - hb.stat().st_mtime
            except FileNotFoundError:
                logger.warning(f"[recovery] hb {hb.name} vanished after globbing.")
                continue
            if age > self.stale_after:
                task = self.running / hb.stem
                if task.exists():
                    logger.info(f"[recovery] Moving stale task {task.name} back to todo/")
                    try:
                        os.replace(str(task), self.todo / task.name)
                    except FileNotFoundError:
                        logger.warning(
                            f"[recovery] Task {task.name} vanished before it could be moved back."
                        )
                        continue
                hb.unlink(missing_ok=True)
        duration = (time.perf_counter() - st) * 1000
        logger.trace(f"Recovered stale tasks in {duration:.3f} ms")

    def seed_once(self, ids: Iterable[int], suffix: str = ".task"):
        """Idempotently add ONLY missing tasks for the given ids.

        Safe under many concurrent writers. Skips ids present in any state.
        """
        st = time.perf_counter()
        lock = self._acquire_lock()
        try:
            # Build a set of existing task filenames for O(1) membership.
            existing = set()
            for d in (self.todo, self.running, self.done, self.failed):
                existing.update(p.name for p in d.iterdir())

            created = 0
            for idx in ids:
                name = f"{idx:012}{suffix}"
                if name in existing:
                    continue
                try:
                    self._atomic_touch(self.todo / name)
                    created += 1
                except FileExistsError:
                    # Raced with another seeder for same id.
                    pass
            duration = (time.perf_counter() - st) * 1000
            logger.info(f"Seeded {created} new tasks into todo/ in {duration:.3f} ms")
        finally:
            self._release_lock(lock)

    def lease(self) -> TaskLease | None:
        """Acquire one task and return a `TaskLease`, or None if empty."""
        now = time.monotonic()
        if self.last_recover is None or now - self.last_recover > self.recover_interval:
            self.recover_stale()
            self.last_recover = now
        task = self._acquire_task()
        if task is None:
            return None
        lease = TaskLease(self, task)
        lease._start_heartbeat()
        return lease

    def lease_many(self, max_items: int) -> list[TaskLease]:
        """Acquire up to `max_items` tasks and return their `TaskLease`s."""
        now = time.monotonic()
        if self.last_recover is None or now - self.last_recover > self.recover_interval:
            self.recover_stale()
            self.last_recover = now
        tasks = self._acquire_tasks(max_items)
        leases = [TaskLease(self, task) for task in tasks]
        for lease in leases:
            lease._start_heartbeat()
        return leases
