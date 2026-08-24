from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from mldft.utils.continue_denop import merge_denop_configs
from mldft.utils.environ import parse_run_path, run_to_checkpoint_path


def prepare_chkpt_path(run_path, cfg: DictConfig) -> tuple[Path | str, Path | str | None]:
    """Resolve checkpoint paths for the primary model and optional initial-guess model.

    Args:
        run_path (Path | str): Parsed training run directory.
        cfg (DictConfig): Denop runtime config containing checkpoint selection fields.

    Returns:
        tuple[Path | str, Path | str | None]: Tuple of ``(ckpt_path, guess_ckpt_path)``
        where ``guess_ckpt_path`` is ``None`` when no separate guess checkpoint is
        configured.
    """
    # Side effects:
    # - None. This helper only resolves and returns checkpoint paths.

    ckpt_path = run_to_checkpoint_path(run_path, cfg.ckpt_choice)
    if cfg.get("guess_path") is not None:
        guess_ckpt_path = run_to_checkpoint_path(cfg.guess_path, cfg.guess_ckpt_choice)
    else:
        guess_ckpt_path = None

    return ckpt_path, guess_ckpt_path


def get_split_filepath(
    cfg: DictConfig,
    model_config: DictConfig | ListConfig,
    model_config_resolved: DictConfig | ListConfig,
) -> str | Path:
    """Resolve the split-file path used to select molecules.

    Args:
        cfg (DictConfig): Runtime config that may directly provide ``split_file_path``.
        model_config (DictConfig | ListConfig): Unresolved model config loaded from the
            training run.
        model_config_resolved (DictConfig | ListConfig): Resolved model config loaded
            from the training run.

    Returns:
        str | Path: Path to the split pickle file, either from runtime config or model
        config.
    """
    # Side effects:
    # - None. This helper only computes and returns a split-file path.

    split_file_path = cfg.get("split_file_path")

    if split_file_path is None:
        if hasattr(model_config.data.datamodule.split_file, "output_file"):
            split_file_path = model_config_resolved.data.datamodule.split_file.output_file
        else:
            split_file_path = model_config.data.datamodule.split_file
    return split_file_path


def prepare_config(
    cfg: DictConfig,
) -> tuple[DictConfig, DictConfig | ListConfig, str | Path, str | Path | None]:
    """Load and enrich denop configuration from a training run directory.

    Args:
        cfg (DictConfig): User/runtime denop config.

    Returns:
        tuple[DictConfig, DictConfig | ListConfig, str | Path, str | Path | None]:
        Tuple ``(cfg, model_config, ckpt_path, guess_ckpt_path)`` with resolved run
        paths and checkpoint references injected into ``cfg``.
    """
    # Side effects:
    # - Mutates ``cfg`` in place by injecting resolved run, split, and checkpoint paths.
    # - Loads model config files from disk and may merge continue-mode config overrides.

    run_path = parse_run_path(cfg.run_path)
    cfg.run_path = str(run_path)

    model_config_path = run_path / "hparams.yaml"
    model_config_path_resolved = run_path / "hparams_resolved.yaml"
    model_config = OmegaConf.load(model_config_path)
    model_config_resolved = OmegaConf.load(model_config_path_resolved)

    split_file_path = get_split_filepath(
        cfg, model_config=model_config, model_config_resolved=model_config_resolved
    )
    cfg.split_file_path = str(split_file_path)

    ckpt_path, guess_ckpt_path = prepare_chkpt_path(run_path, cfg)

    with open_dict(cfg):
        cfg.ckpt_path = str(ckpt_path)
        cfg.guess_ckpt_path = str(guess_ckpt_path)

    if cfg.get("continue_denop"):
        cfg = merge_denop_configs(cfg)

    return cfg, model_config, ckpt_path, guess_ckpt_path
