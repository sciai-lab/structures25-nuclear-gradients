"""Utilities for creating reproducibility snapshots."""

import io
import os
import subprocess
import tarfile
from pathlib import Path

from mldft.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _add_git_info(tar: tarfile.TarFile, project_root: Path) -> None:
    """Add basic Git info (commit, dirty flag, and status) to a tar archive."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
        ).strip()

        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True
            ).strip()
        )

        status = subprocess.check_output(["git", "status", "-vv"], cwd=project_root, text=True)

        info = f"Commit: {commit}\nDirty: {'Yes' if dirty else 'No'}\n\n{status}"

        data = info.encode()
        tarinfo = tarfile.TarInfo("git_info.txt")
        tarinfo.size = len(data)
        tar.addfile(tarinfo, io.BytesIO(data))

    except subprocess.CalledProcessError:
        log.warning("Could not retrieve git information")


def create_code_backup(output_dir: Path) -> None:
    """Create compressed archive of source code and git info.

    Args:
        output_dir: Directory where the code_backup.tar.gz will be created
    """
    project_root = Path(os.environ["PROJECT_ROOT"])
    archive_path = output_dir / "code_backup.tar.gz"

    log.info(f"Creating code backup at {archive_path}")

    with tarfile.open(archive_path, "w:gz") as tar:
        # Add source code folders
        folder_path = project_root / "mldft"
        if folder_path.exists():
            log.info("Adding mldft to archive")
            tar.add(folder_path, arcname="mldft")
        else:
            log.warning("Folder mldft not found")

        # Add git information
        _add_git_info(tar, project_root)
