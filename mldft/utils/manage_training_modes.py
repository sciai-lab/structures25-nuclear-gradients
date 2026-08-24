from enum import StrEnum
from pathlib import Path
from typing import Tuple

import torch
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from mldft.utils import RankedLogger
from mldft.utils.environ import CheckpointChoice, parse_run_path, run_to_checkpoint_path


class TrainingModes(StrEnum):
    """Training mode options for continuing or starting training runs."""

    RESUME = "resume"
    OVERRIDE = "override_settings"
    LOAD_WEIGHTS = "load_weights"
    FINETUNE = "finetune"


log = RankedLogger(__name__, rank_zero_only=True)
_CONTROL_OVERRIDES = {"training_mode", "run_path", "ckpt_choice"}


def _override_key(override: str) -> str:
    """Return override key/group part, stripping +/~ prefixes and value."""
    key = override.strip().lstrip("+~").split("=", 1)[0].strip()
    if "@" in key:
        group, package = key.split("@", 1)
        return package or group
    return key


def _compose_from_run_overrides(run_path: Path, task_overrides: list[str]) -> DictConfig:
    """Compose a config from old run overrides plus current task overrides."""
    candidates = [
        run_path / ".hydra" / "overrides.yaml",
        run_path / ".hydra_resumed" / "overrides.yaml",
    ]
    base_overrides = None
    for override_file in candidates:
        if override_file.is_file():
            loaded = OmegaConf.load(override_file)
            base_overrides = [] if loaded is None else list(loaded)
            break
    if base_overrides is None:
        raise FileNotFoundError(
            f"Could not find overrides.yaml in {run_path}/.hydra or {run_path}/.hydra_resumed."
        )

    # Control keys should come only from the current CLI invocation, not from the old run.
    base_overrides = [
        override
        for override in base_overrides
        if _override_key(override) not in _CONTROL_OVERRIDES
    ]
    return compose(config_name="train.yaml", overrides=[*base_overrides, *task_overrides])


def get_weights_ckpt(
    cfg: DictConfig, run_path: Path, ckpt_choice: CheckpointChoice
) -> Path | None:
    """Get checkpoint path for loading weights based on ckpt_choice config."""
    weight_ckpt_path = None
    if ckpt_choice is not None:
        # This should only take a new checkpoint if ckpt_choice is provided
        weight_ckpt_path = run_to_checkpoint_path(run_path, ckpt_choice)
    return weight_ckpt_path


def set_training_mode(
    training_mode: str, cfg: DictConfig
) -> Tuple[DictConfig, str | Path | None, Path | None]:
    """Handle different training modes.

    Available modes are:

    *  ``None``: No training mode provided. Start a new run using provided config. Mode used by default
    *  ``resume``: for resuming training in the same folder.
       This does not allow for overriding any settings and always starts from the last checkpoint.
       The checkpoint started from is not deleted, due to internal PyTorchLightning mechanism.
    *  ``override_settings``: starts a new training run using the config from the run path, but allows for overriding settings.
       Useful for starting the same runs with slightly different settings (lr, momentum, etc.),
       not for fine-tuning. Doesn't allow for loading model weights and will throw an error if attempting to provide checkpoint choice
    *  ``load_weights``: loads the weights from run_path. Defaults to last checkpoint.
    *  ``finetune``: loads both the weights and original settings from run_path. Allows for overriding settings.
       You can select which checkpoint to load with override for ckpt_choice. See run_to_checkpoint_path function in utils/environ.py for details

    Returns modified config file, checkpoint path and weight checkpoint path.
    """
    ckpt_path = None
    weight_ckpt_path = None
    if training_mode is not None:
        run_path = cfg.get("run_path")
        assert (
            run_path is not None
        ), "Training mode was set, but run_path was not provided. Did you forget to provide run_path?"
        run_path = parse_run_path(run_path)
        config_path = run_path / "hparams.yaml"
        original_settings_cfg = OmegaConf.load(config_path)
        # According to Hydra documentation this only contains command-line task overrides.
        # Hydra-specific overrides (hydra.run.dir etc.) are not included.
        overrides = list(HydraConfig.get().overrides.task)

        if training_mode == TrainingModes.RESUME:
            log.info("Resuming training from checkpoint!")
            log.warning("Original last_epoch checkpoint will not be deleted.")
            assert cfg.get("ckpt_choice") in [None, "last"], (
                "Only ckpt_choice 'last' is allowed in resume mode as this contains the state of "
                "the 'best_' checkpoints. "
            )
            accepted_overrides = [
                "training_mode",
                "run_path",
                "ckpt_choice",
                "density_optimizer",
                "denop_settings",
                "hydra",
            ]
            unsupported_overrides = []
            for override in overrides:
                if not any(
                    _override_key(override).startswith(accepted) for accepted in accepted_overrides
                ):
                    unsupported_overrides.append(override)
            if len(unsupported_overrides) > 0:
                raise ValueError(
                    f"Unsupported command line overrides in resume mode: {unsupported_overrides}."
                )
            try:
                import git

                current_file_dir = Path(__file__).resolve().parent
                repo = git.Repo(current_file_dir, search_parent_directories=True)
                sha = repo.head.object.hexsha
                if repo.is_dirty(untracked_files=True):
                    log.warning("Repo is dirty, resuming training with uncommitted changes!")
                org_cfg_resolved = OmegaConf.load(run_path / "hparams_resolved.yaml")
                old_sha = org_cfg_resolved.git.sha
                if old_sha is None:
                    log.warning("No git sha found in original config, cannot compare!")
                elif old_sha != sha:
                    log.warning(
                        f"Git sha has changed since original run! Original sha: {old_sha}, current sha: {sha}"
                    )
            except ImportError:
                log.warning("GitPython is not installed, cannot check git sha!")
            except Exception as e:
                log.warning(f"Could not access git repository, cannot check git sha! Error: {e}")

            # load last checkpoint
            ckpt_path = run_to_checkpoint_path(run_path, "last")
            cfg = OmegaConf.merge(original_settings_cfg, OmegaConf.from_dotlist(overrides))
        elif training_mode == TrainingModes.OVERRIDE:
            log.info("Using original settings from checkpoint with overrides!")
            cfg = _compose_from_run_overrides(run_path, overrides)
            ckpt_choice = cfg.get("ckpt_choice", None)
            assert ckpt_choice is None, (
                f"Mode set to override_settings, but ckpt_choice is {ckpt_choice}. "
                f"Only None is allowed since this mode starts a new training."
            )
        elif training_mode == TrainingModes.LOAD_WEIGHTS:
            # Only load the weights from provided checkpoint
            ckpt_choice = cfg.get("ckpt_choice", None)
            assert (
                ckpt_choice is not None
            ), "Mode set to load_weights, but no checkpoint choice provided"
            weight_ckpt_path = get_weights_ckpt(cfg, run_path, ckpt_choice)
        elif training_mode == TrainingModes.FINETUNE:
            log.info("Using original settings from checkpoint with overrides!")
            cfg = _compose_from_run_overrides(run_path, overrides)
            ckpt_choice = cfg.get("ckpt_choice", None)
            assert (
                ckpt_choice is not None
            ), "Mode set to finetune, but no checkpoint choice provided"
            weight_ckpt_path = get_weights_ckpt(cfg, run_path, ckpt_choice)
        else:
            raise ValueError(
                f"Provided training mode is not recognized. Provided training mode: {training_mode}. Available modes: {[mode.value for mode in TrainingModes]}"
            )
    return cfg, ckpt_path, weight_ckpt_path


def set_model_weights(
    training_mode: str,
    model: LightningModule,
    ckpt_path: Path | None,
    weight_ckpt_path: Path | None,
):
    """Load model weights from checkpoint for finetune or load_weights modes."""
    # Only if we choose to override settings
    if training_mode in (TrainingModes.FINETUNE, TrainingModes.LOAD_WEIGHTS) and (
        weight_ckpt_path is not None
    ):
        # this only loads weights
        assert (
            ckpt_path is None
        ), "Cannot load both model and weights from checkpoint! Model weights would be overwritten."
        state = torch.load(weight_ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        # Backwards compatibility for old checkpoints
        for key in list(state.keys()):  # avoid modifying the keys while iterating over dict
            if "gbf_module" in key:
                state[key.replace("gbf_module", "radial_embedding_module")] = state.pop(key)
        log.info("Overwriting model weights!")
        model.load_state_dict(state)
    else:
        log.info("Not loading any weights")
