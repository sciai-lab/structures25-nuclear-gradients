"""Stage training data from one DFT_DATA root to another."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

from tqdm import tqdm


@dataclass
class StageStats:
    copied: int = 0
    skipped: int = 0
    missing_optional: int = 0


def resolve_stage_path(path_value: str, field_name: str) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(path_value))
    if "$" in expanded:
        raise ValueError(
            f"Could not resolve '{field_name}={path_value}'. "
            "Make sure all referenced environment variables are set on the compute node."
        )
    return Path(expanded).resolve()


def get_local_rank() -> int:
    for env_var in ("SLURM_LOCALID", "LOCAL_RANK"):
        rank_value = os.environ.get(env_var)
        if rank_value is None:
            continue
        try:
            return int(rank_value)
        except ValueError:
            continue
    return 0


def wait_for_stage_marker(path: Path, timeout_s: int, poll_s: float = 1.0) -> None:
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        if path.exists():
            return
        sleep(poll_s)
    raise TimeoutError(f"Timed out waiting for staged data marker at {path}.")


def infer_label_subdir(transform_name: str) -> str:
    if transform_name in {"", "none", "labels"}:
        return "labels"
    return f"labels_{transform_name}"


def _resolve_under_root(path: str | Path, root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path {p} is not under root {root}") from exc
    return p


def _copy_file(
    src: Path,
    dst: Path,
    stats: StageStats,
    dry_run: bool,
    log_skipped: bool = True,
    skip_log_limit: int = 20,
) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    if src.is_dir():
        if dry_run:
            stats.copied += 1
            return
        if dst.exists() and not dst.is_dir():
            raise IsADirectoryError(
                f"Cannot copy directory {src} to existing non-directory destination {dst}"
            )
        shutil.copytree(src, dst, dirs_exist_ok=True)
        stats.copied += 1
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        stats.skipped += 1
        if log_skipped and stats.skipped <= skip_log_limit:
            print(f"[skip] already exists: {dst}")
        return
    if dry_run:
        stats.copied += 1
        return
    tmp = dst.parent / f".{dst.name}.tmp.{os.getpid()}"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    stats.copied += 1


def _add_task(src: Path, dst: Path, tasks: dict[Path, Path]) -> None:
    if dst not in tasks:
        tasks[dst] = src


def _add_manifest_tasks(
    manifest_src: Path,
    src_root: Path,
    dst_root: Path,
    tasks: dict[Path, Path],
) -> None:
    if not manifest_src.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_src}")
    manifest_rel = manifest_src.relative_to(src_root)
    _add_task(manifest_src, dst_root / manifest_rel, tasks)

    with manifest_src.open("r") as f:
        manifest = json.load(f)
    for shard in manifest.get("shards", []):
        shard_ref = Path(str(shard["output_file"]))
        if shard_ref.is_absolute():
            shard_src = _resolve_under_root(shard_ref, src_root)
        else:
            shard_src = (src_root / shard_ref).resolve()
        shard_dst = dst_root / shard_src.relative_to(src_root)
        _add_task(shard_src, shard_dst, tasks)


def stage_training_data(
    src_dft_data: str | Path,
    dst_dft_data: str | Path,
    dataset: str,
    transform: str,
    target_key: str,
    *,
    label_subdir: str | None = None,
    split_file: str | Path | None = None,
    include_denop_h5: bool = False,
    extra_paths: list[str | Path] | None = None,
    dry_run: bool = False,
    skip_log_limit: int = 20,
) -> StageStats:
    src_root = Path(src_dft_data).resolve()
    dst_root = Path(dst_dft_data).resolve()
    src_dataset_dir = src_root / dataset
    dst_dataset_dir = dst_root / dataset
    if not src_dataset_dir.exists():
        raise FileNotFoundError(f"Missing source dataset dir: {src_dataset_dir}")

    stats = StageStats()
    tasks: dict[Path, Path] = {}

    split_src = _resolve_under_root(split_file or (Path(dataset) / "split.pkl"), src_root)
    _add_task(split_src, dst_root / split_src.relative_to(src_root), tasks)

    split_yaml_src = split_src.with_suffix(".yaml")
    if split_yaml_src.exists():
        _add_task(split_yaml_src, dst_root / split_yaml_src.relative_to(src_root), tasks)
    else:
        stats.missing_optional += 1
        print(f"[optional-missing] {split_yaml_src}")

    dataset_info_src = src_dataset_dir / "dataset_info.yaml"
    if dataset_info_src.exists():
        _add_task(dataset_info_src, dst_dataset_dir / "dataset_info.yaml", tasks)
    else:
        stats.missing_optional += 1
        print(f"[optional-missing] {dataset_info_src}")

    for extra in extra_paths or []:
        extra_src = _resolve_under_root(extra, src_root)
        _add_task(extra_src, dst_root / extra_src.relative_to(src_root), tasks)

    resolved_label_subdir = label_subdir or infer_label_subdir(transform)
    train_manifest_src = (
        src_dataset_dir / f"{resolved_label_subdir}_h5_{target_key}" / "manifest.json"
    )
    _add_manifest_tasks(train_manifest_src, src_root, dst_root, tasks)

    if include_denop_h5:
        denop_manifest_src = src_dataset_dir / f"labels_denop_h5_{target_key}" / "manifest.json"
        _add_manifest_tasks(denop_manifest_src, src_root, dst_root, tasks)

    planned = [(src, dst) for dst, src in tasks.items()]
    print(f"Planned copy operations: {len(planned)} file(s)")
    for src, dst in tqdm(planned, desc="Staging", unit="file", smoothing=0.0):
        _copy_file(
            src,
            dst,
            stats=stats,
            dry_run=dry_run,
            skip_log_limit=skip_log_limit,
        )

    print(
        "Done:",
        f"copied={stats.copied}",
        f"skipped={stats.skipped}",
        f"missing_optional={stats.missing_optional}",
        f"label_subdir={resolved_label_subdir}",
        f"denop_h5={include_denop_h5}",
    )
    if stats.skipped > skip_log_limit:
        print(f"[skip] ... and {stats.skipped - skip_log_limit} more existing file(s)")
    return stats
