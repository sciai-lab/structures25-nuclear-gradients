import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def get_path_from_environment_variable(name: str) -> Path:
    """Return the path to the directory specified by the environment variable `name`.

    Args:
        name: The name of the environment variable.

    Returns:
        The path to the directory specified by the environment variable.

    Raises:
        KeyError: If the environment variable is not set.
        AssertionError: If the path does not exist.
    """
    try:
        result = os.environ[name]
    except KeyError:
        raise KeyError(f"{name} environment variable not set.")
    if not os.path.exists(result):
        logger.warning(
            f'Path "{result}", read from environment variable "{name}", does not exist.'
        )
    return Path(result)


def get_mldft_data_path() -> Path:
    """Return the path to the mldft data directory."""
    return get_path_from_environment_variable("DFT_DATA")


def get_mldft_model_path() -> Path:
    """Return the path to the mldft model directory."""
    return get_path_from_environment_variable("DFT_MODELS")


def get_dataset_from_absolute_path(path: Path) -> str:
    """Return the dataset name from an absolute path."""
    return path.relative_to(get_mldft_data_path()).parts[0]


def parse_run_path(run_path: Path | str) -> Path:
    """Parse the run path, making it absolute if necessary."""
    run_path = Path(str(run_path))
    if not run_path.is_absolute():
        run_dir = get_mldft_model_path() / "train" / "runs"
        run_list = sorted(list(run_dir.glob("*")))
        matching_runs = [run for run in run_list if run.name.startswith(run_path.name)]
        if len(matching_runs) == 1:
            run_path = matching_runs[0]
        elif len(matching_runs) == 0:
            raise ValueError(f"No runs matched the name {run_path.name} in {run_dir}.")
        else:
            matching_runs_str = "\n  ".join([run.name for run in matching_runs])
            raise ValueError(
                f"Unique run could not be inferred as {len(matching_runs)} runs matched the name {run_path.name} in {run_dir}:\n"
                f"  {matching_runs_str}."
            )
    return run_path


CheckpointChoice = Literal["last", "best_energy", "best_density", "best_val_loss"] | str


def run_to_checkpoint_path(
    run_path: Path | str,
    ckpt_choice: CheckpointChoice = "last",
) -> Path:
    """Get the path to the checkpoint of a run.

    If ckpt_choice is `None`, it is assumed that there exists only one checkpoint with name of the form 'epoch_*.ckpt'
    in `run_path/checkpoints`.

    Args:
        run_path: Path to the run directory
        ckpt_choice: Type of checkpoint to load. Options: "last", "best_energy", "best_density",
            "best_val_loss", or a specific checkpoint filename (e.g., "last_epoch_042.ckpt")
    """
    checkpoint_dir = Path(run_path) / "checkpoints"
    assert checkpoint_dir.is_dir(), f"No checkpoint directory found at {checkpoint_dir}"

    # mapping from ckpt_choice to checkpoint file name
    checkpoint_patterns = {
        "last": "last*.ckpt",  # don't use _epoch for backwards compatibility
        "best_energy": "gs_abs_energy_error_epoch_*.ckpt",
        "best_density": "gs_l2_norm_epoch_*.ckpt",
        "best_val_loss": "val_loss_total_epoch_*.ckpt",
    }

    if ckpt_choice is None:
        checkpoint_list = list(checkpoint_dir.glob("epoch*.ckpt"))
        assert len(checkpoint_list) == 1, (
            f"Found {len(checkpoint_list)} checkpoints starting with epoch in {checkpoint_dir},"
            f"but need one to infer the best checkpoint."
        )
        logger.info(
            f"'ckpt_choice' is None. Will use only available ckpt_path: {checkpoint_list[0]}"
        )
        return checkpoint_list[0]

    if ckpt_choice not in checkpoint_patterns:
        # assume it's a specific checkpoint filename
        specific_checkpoint = checkpoint_dir / ckpt_choice
        if specific_checkpoint.exists():
            logger.info(f"Using specific checkpoint '{ckpt_choice}'.")
            return specific_checkpoint
        else:
            raise ValueError(
                f"Invalid ckpt_choice '{ckpt_choice}'. Valid options: {list(checkpoint_patterns.keys())} "
                f"or a specific checkpoint filename. File not found: {specific_checkpoint}"
            )

    pattern = checkpoint_patterns[ckpt_choice]
    checkpoint_list = list(checkpoint_dir.glob(pattern))  # find matches

    if len(checkpoint_list) == 0:
        raise FileNotFoundError(
            f"No checkpoints found matching pattern '{pattern}' in {checkpoint_dir}"
        )
    elif len(checkpoint_list) == 1:
        checkpoint_path = checkpoint_list[0]
    else:
        if ckpt_choice == "last":
            # Pick the latest epoch across all checkpoints.
            # If epochs are equal, prefer the cache checkpoint variant.
            checkpoint_path = max(
                checkpoint_list,
                key=lambda p: (
                    int(
                        p.stem.split("last_with_cache_epoch_")[-1]
                        if "last_with_cache_epoch_" in p.stem
                        else p.stem.split("last_epoch_")[-1]
                    ),
                    int("last_with_cache_epoch_" in p.stem),
                ),
            )
        else:
            raise ValueError(
                f"Too many checkpoints found in {checkpoint_dir}. Checkpoints found: {checkpoint_list} for pattern {pattern}"
            )
    logger.info(f"Using checkpoint '{checkpoint_path.name}' for ckpt_choice='{ckpt_choice}'.")
    return checkpoint_path
