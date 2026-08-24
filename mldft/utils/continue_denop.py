"""Utility functions to continue density optimization from a given one."""

import re
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict


def get_existing_sample_paths(denop_dir: str | Path) -> list:
    """Extract the sample trajectory paths that are already present in the given denop_dir.

    Args:
        denop_dir: Path to the directory of the density optimization that should be continued.

    Returns:
        existing_paths: List of existing sample_trajectory paths of the format '<dataset>_<mol_id>.pt'.
    """
    denop_dir = Path(str(denop_dir))

    existing_sample_dir = denop_dir / "sample_trajectories"

    existing_paths = [file_path for file_path in existing_sample_dir.glob("sample*.pt")]

    return existing_paths


def get_missing_paths_and_scf_iters(
    all_paths: list[str | Path],
    all_scf_iterations: list[str | Path],
    existing_paths: list[str | Path],
) -> tuple[list, list]:
    """Extracts missing paths and number of SCF iterations given already existing paths."""

    missing_paths = []
    missing_scf_iterations = []
    existing_mol_ids = {extract_id_existing_sample(path.name) for path in existing_paths}
    for i, query_path in enumerate(all_paths):
        query_mol_id = extract_id_val_sample(query_path.name)
        if query_mol_id not in existing_mol_ids:
            missing_paths.append(query_path)
            missing_scf_iterations.append(all_scf_iterations[i])

    return missing_paths, missing_scf_iterations


def extract_id_existing_sample(filename: str):
    """Extracts the numeric ID from existing_sample_names (sample_XXX_0030225.pt)."""
    match = re.match(
        r"sample_([^_]+)_(\d+)\.pt", filename
    )  # regex pattern of saved sample trajectories
    if match:
        return int(match.group(2))  # Take the number sequence
    else:
        return None


def extract_id_val_sample(filename: str):
    """Extracts the numeric ID from val_samples (0030225.zarr.zip)."""
    match = re.match(r"(\d+)\.zarr\.zip", filename)  # regex pattern of data
    if match:
        return int(match.group(1))
    else:
        return None


def merge_denop_configs(cfg: DictConfig) -> DictConfig:
    """Merge the given config with the one located at the denop_dir to continue density
    optimization in a consistent manner."""

    assert cfg.get("denop_dir") is not None, "continue_denop is True but denop_dir is None"
    denop_dir = Path(cfg.get("denop_dir"))
    existing_cfg_path = denop_dir / "hparams_resolved.yaml"

    if not existing_cfg_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {existing_cfg_path}")

    # these keys should always be overridden
    continue_keys = {
        "n_molecules",
        "continue_denop",
        "denop_dir",
        "plot_summary",
        "device",
        "num_devices",
        "num_threads",
        "torch_threads_per_worker",
        "batch_size",
        "batch_timeout",
        "plot_every_n",
    }
    # for backward compatibility, we initialize the merged cfg with the existing one as OmegaConf.merge would
    # fail on unmatching keys
    merged_cfg = OmegaConf.load(existing_cfg_path)

    existing_keys = set(OmegaConf.to_container(merged_cfg, resolve=True).keys())
    new_keys = set(OmegaConf.to_container(cfg, resolve=True).keys())

    # Point out keys that will remain from the existing cfg but differ from the new cfg
    for overlap_key in (existing_keys & new_keys) - continue_keys:
        existing_value = merged_cfg.get(overlap_key)
        new_value = cfg.get(overlap_key)
        if existing_value != new_value:
            logger.warning(
                f"Using existing config value for differing key {overlap_key}: \n"
                f"  existing: {existing_value}\n"
                f"  current:  {new_value}"
            )

    # for future compatibility we want to keep all keys that are in the new but not in the existing cfg
    to_be_added_keys = new_keys - existing_keys
    continue_keys.update(to_be_added_keys)

    with open_dict(merged_cfg):
        for key in continue_keys:
            if cfg.get(key) is not None:
                merged_cfg[key] = cfg.get(key)
        # add git logging of continued run
        if "git" in cfg:
            merged_cfg["git_continued"] = cfg.get("git")

    return merged_cfg
