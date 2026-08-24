from pathlib import Path

from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm


def prepare_logger(cfg: DictConfig) -> Path:
    """Configure loguru sinks for console and optional file logging.

    Args:
        cfg (DictConfig): Runtime config that may define ``log_file`` and
            ``continue_denop``.

    Returns:
        Path | None: Path to the active log file, or ``None`` when file logging is
        disabled.
    """
    # Side effects:
    # - Reconfigures global loguru handlers.
    # - May create/append to a log file on disk when file logging is enabled.

    log_file = cfg.get(
        "log_file"
    )  # this returns None if "log_file" is not defined in cfg, which is caught below

    if cfg.continue_denop:
        denop_dir = Path(str(cfg.denop_dir))

        if denop_dir is None:
            raise ValueError(
                "The 'denop_dir' configuration must be specified when 'continue_denop' is True."
            )

        if not denop_dir.is_dir():
            raise ValueError(
                f"The specified 'denop_dir' ({denop_dir}) does not exist or is not a directory."
            )

        log_file = denop_dir / "ofdft.log"
    elif log_file is not None:
        log_file = Path(log_file)
    else:
        raise ValueError(
            "The 'log_file' configuration must be specified for logging to function properly."
        )

    logger.remove()
    logger_format = (
        "<green>{time:HH:mm:ss}</green>|<level>{level: <8}</level>|<level>{message}</level>"
    )
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=logger_format,
        colorize=True,
        enqueue=True,
    )
    if log_file is not None:
        log_file = Path(log_file)
        logger.add(
            log_file,
            level="TRACE",
            rotation="10 MB",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )
        logger.info(f'Logging to "{log_file}"')

    return log_file
