"""OmegaConf resolvers for the mldft package.

The separators of the override in the run names are ``__`` and ``-`` per default (``key1-value1__key2-value2``).
You can change them by setting the environment variables ``MLDFT_RUN_NAME_KEY_VALUE_SEP`` and
``MLDFT_RUN_NAME_OVERRIDE_SEP``, e.g. by adding the following to your shell configuration file:

.. code-block:: bash

    export MLDFT_RUN_NAME_KEY_VALUE_SEP="="
    export MLDFT_RUN_NAME_OVERRIDE_SEP=", "
"""

import os
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

from mldft.ofdft.functional_factory import requires_grid
from mldft.utils.counter_file import get_and_increment_counter
from mldft.utils.environ import get_mldft_model_path, parse_run_path
from mldft.utils.manage_training_modes import TrainingModes

KEY_VALUE_SEP = os.environ.get("MLDFT_RUN_NAME_KEY_VALUE_SEP", "-")
OVERRIDE_SEP = os.environ.get("MLDFT_RUN_NAME_OVERRIDE_SEP", "__")
SLASH_REPLACEMENT = os.environ.get("MLDFT_RUN_NAME_SLASH_REPLACEMENT", "_")


def split_overrides_string(overrides_string: str) -> dict:
    """Splits an override string of the form "key1.subkey1=value1,key2=value2,key3/subkey3=value3"
    by commas, which are not in brackets.

    Note: This is still not perfect, e.g. "'" will lead to problems.

    Args:
        overrides_string (str): The override string to split.

    Returns:
        dict: The split override string as a dictionary with keys (before '=') and values (after '=').
    """
    # parse the string and split by ',' which are not inside brackets or quotation marks
    delimiter_map = {"[": "]", "(": ")", "{": "}", '"': '"', "'": "'"}
    all_delimiters = set(delimiter_map.keys()) | set(delimiter_map.values())
    stack = []  # stack of expected closing delimiters
    split = []  # list of splits
    start = 0  # start of the current override
    for i, c in enumerate(overrides_string):
        if c in all_delimiters:
            if c in delimiter_map.values():
                if stack and stack[-1] == c:
                    stack.pop(-1)
                    continue
            if c not in delimiter_map.values() or not stack:
                stack.append(delimiter_map[c])
        elif c == "," and not stack:
            split.append(overrides_string[start:i])
            start = i + 1
    split = split + [overrides_string[start:]]
    result = {override.split("=")[0]: "=".join(override.split("=")[1:]) for override in split}
    return result


def dir_counter(path: str, length: int = 2, continue_run: bool = False) -> str:
    """Counts the number of runs in a directory. If the continue_run flag is set the dir_counter is
    not triggered. When using SLURM, with the hydra submitit launcher, the counter value is passed
    via MLDFT_RUN_COUNTER env var. Otherwise, the SLURM_JOB_ID is used if available.

    Args:
        path (str): The path to the directory.
        length (int, optional): The length of the counter. Defaults to 2.
        continue_run (bool, optional): Whether to continue a run. Defaults to False.

    Returns:
        str: The counter as a string with leading zeros.
    """
    if continue_run:
        return ""

    # Check if counter was passed via environment variable (e.g. from login node to compute
    # node when using hydra submitit) in this case, don't increase but just use that one
    if "MLDFT_RUN_COUNTER" in os.environ:
        return os.environ["MLDFT_RUN_COUNTER"].zfill(length)
    # If SLURM_JOB_ID is set but not MLDFT_RUN_COUNTER, it is a normal slurm submission
    elif "SLURM_JOB_ID" in os.environ:
        return os.environ["SLURM_JOB_ID"].zfill(length)

    # Get and increment the counter, and set it in environment for child processes
    counter_value = str(get_and_increment_counter(os.path.join(path, ".run_counter"))).zfill(
        length
    )
    # Pin the counter value for this runtime via environment variable
    os.environ["MLDFT_RUN_COUNTER"] = counter_value

    return counter_value


# register a resolver that counts the number of runs in a directory
OmegaConf.register_new_resolver(
    "dir_counter",
    dir_counter,
)


def create_and_return_dir(path):
    """Creates a directory and returns it."""
    os.makedirs(path, exist_ok=False)
    return path


OmegaConf.register_new_resolver("create_and_return_dir", create_and_return_dir)


def slash_to_backslash(path: str) -> str:
    """Converts slashes to backslashes."""
    return path.replace("/", "\\")


OmegaConf.register_new_resolver("slash_to_backslash", slash_to_backslash)


def _get_leaf_key(full_key: str) -> str:
    """Returns the leaf key of a full key: "key1.subkey1/subsubkey1" -> "subsubkey1" """
    return full_key.split(".")[-1].split("/")[-1]


def _normalize_override_value(value: str) -> str:
    """Return a stable string representation for Hydra override values.

    This mainly normalizes numeric inputs so semantically equivalent values such as ``1e-3`` and
    ``0.001`` map to the same run-directory suffix.
    """
    lower_value = value.lower()
    if lower_value in ["true", "false"]:
        return lower_value

    try:
        numeric_value = float(value)
    except ValueError:
        return value

    return format(numeric_value, ".15g")


def leafs_only_override_dirname(overrides_string: str) -> str:
    """Converts an override string of the form
    "key1.subkey1=value1,key2=value2,key3/subkey3=value3" to a more compact form
    "subkey1-value1__key2-value2__subkey3-value3".

    Args:
        overrides_string (str): The override string to compactify, as returned by ``${hydra.job.override_dirname}``.

    Returns:
        str: The compactified override string.
    """
    return OVERRIDE_SEP.join(
        [
            f"{_get_leaf_key(key)}{KEY_VALUE_SEP}{_normalize_override_value(value)}"
            for key, value in split_overrides_string(overrides_string).items()
        ]
    )


OmegaConf.register_new_resolver("leafs_only_override_dirname", leafs_only_override_dirname)


def values_only_override_dirname(overrides_string: str) -> str:
    """Converts an override string of the form
    "key1.subkey1=value1,key2=value2,key3/subkey3=value3" to a more compact form
    "value1__value2__value3".

    Args:
        overrides_string (str): The override string to compactify, as returned by ``${hydra.job.override_dirname}``.

    Returns:
        str: The compactified override string.
    """
    return OVERRIDE_SEP.join(
        _normalize_override_value(value)
        for value in split_overrides_string(overrides_string).values()
    )


OmegaConf.register_new_resolver("values_only_override_dirname", values_only_override_dirname)


def checkpoint_path_to_run_number(ckpt_path: str) -> str:
    """Converts a checkpoint path to a run number. E.g. ".../sciai-
    dft/train/runs/448_maybe2_lower_bound_loss/checkpoints/last.ckpt" gets mapped to "448".

    If the checkpoint path is not in the expected format, an exception is raised.

    Args:
        ckpt_path (str): The checkpoint path to convert.

    Returns:
        str: The run number.

    Raises:
        ValueError: If the checkpoint path is not in the expected format.
    """
    try:
        parts = ckpt_path.split("/")
        run_name = parts[-3]
        run_number = run_name.split("_")[0]
        int(run_number)  # make sure it is an integer, but do not return int to keep leading zeros
        return run_number
    except Exception as e:
        raise ValueError(f"Could not convert checkpoint path to run number: {e}")


def run_path_to_run_name(run_path: str) -> str:
    """Converts a run directory or checkpoint path to a run name."""
    path = Path(str(run_path))
    run_name = path.parent.parent.name if path.parent.name == "checkpoints" else path.name
    return run_name.split("_", 1)[0].replace("/", SLASH_REPLACEMENT)


def smart_override_dirname(overrides_string: str, exclude_keys: list | str = None) -> str:
    """Converts an override string of the form
    "key1.subkey1=str_value1,key2=int_value2,key3/subkey3=float_value3" to a more compact form
    "str_value1__key2-int_value2__subkey3-float_value3", where the subkeys are only shown if the
    value is numeric.

    Args:
        overrides_string (str): The override string to compactify, as returned by ``${hydra.job.override_dirname}``.
        exclude_keys (list | str, optional): A list of keys to exclude from the compactified string. Defaults to None.

    Returns:
        str: The compactified override string.
    """
    # If the length is less than 3, there can't be an override with a = sign
    if len(overrides_string) < 3:
        return ""
    result = []

    if exclude_keys is None:
        exclude_keys = []
    if isinstance(exclude_keys, str):
        exclude_keys = [exclude_keys]

    def is_float_or_bool(value: str) -> bool:
        """Check if the value is a float or a boolean."""
        try:
            float(value)
            return True
        except ValueError:
            return value.lower() in ["true", "false"]

    for full_key, value in split_overrides_string(overrides_string).items():
        # check if key should be excluded
        if full_key in exclude_keys:
            continue
        # don't put the full checkpoint path, it will be too long
        if full_key == "ckpt_path":
            try:
                result.append(f"from_checkpoint_{checkpoint_path_to_run_number(value)}")
            except ValueError:
                logger.warning(
                    f"Could not convert checkpoint path to run number, just adding 'from_checkpoint' to run name. "
                    f"(path: {value})."
                )
                result.append("from_checkpoint")
            continue
        if full_key == "weight_ckpt_path":
            try:
                result.append(f"from_weight_checkpoint_{checkpoint_path_to_run_number(value)}")
            except ValueError:
                logger.warning(
                    f"Could not convert weight checkpoint path to run number, just adding 'from_weight_checkpoint' "
                    f"to run name. (path: {value})."
                )
                result.append("from_weight_checkpoint")
            continue
        # check if value is numeric
        if is_float_or_bool(value):
            result.append(
                f"{_get_leaf_key(full_key)}{KEY_VALUE_SEP}{_normalize_override_value(value)}"
            )
        else:
            result.append(value)

    name = OVERRIDE_SEP.join(result)
    name = name.replace("/", SLASH_REPLACEMENT)
    max_length = os.pathconf(get_mldft_model_path(), "PC_NAME_MAX")
    max_length = max_length - 5  # allows run numbers up to 9999
    return name[:max_length]


OmegaConf.register_new_resolver("smart_override_dirname", smart_override_dirname)


# need to keep path_to_basename for backward compatibility
def path_to_basename(path: str) -> str:
    """Converts a path to its basename."""
    if path is None:
        raise ValueError("Provided path is None, did you forget to provide a path?")
    path = parse_run_path(path)
    return path.name


OmegaConf.register_new_resolver("path_to_basename", path_to_basename)


def resolve_base_dir(run_path: str, denop_dir: str, continue_denop: bool):
    """Returns the model specific directory of the ofdft run, which is either run_path's basename
    or denop_dir's parent directory name."""
    if continue_denop:
        if denop_dir is None:
            raise ValueError(
                "Provided denop_dir is None despite continue_denop being True. Did you forget to provide a denop_dir?"
            )
        return Path(denop_dir).parent.name
    else:
        return path_to_basename(run_path)


OmegaConf.register_new_resolver("resolve_base_dir", resolve_base_dir)


def resolve_run_dir(run_path: str):
    """Function to resolve which directory to run hydra from.

    Needed, because run_path might be just a run number. Returns the parsed run_path, converting it
    from a run number to a full path if necessary. Returns None if run_path is None (this happens
    when training from scratch).
    """
    if run_path is None:
        return None

    return str(parse_run_path(run_path))


OmegaConf.register_new_resolver("resolve_run_dir", resolve_run_dir)


def from_run_number(run_path: str, training_mode: str):
    """Return run_path and training_mode if both provided, empty string otherwise."""
    if training_mode is None or run_path is None:
        return ""

    return f"{training_mode}-from-{run_path_to_run_name(run_path)}"


OmegaConf.register_new_resolver("from_run_number", from_run_number)


def get_run_directory(
    log_dir: str,
    task_name: str,
    name: str,
    override_dirname: str,
    run_path: str = None,
    training_mode: str = None,
):
    """Generate the run directory path based on training mode.

    For resume mode: returns the existing run directory.
    For other modes: creates new directory with format {counter}_{name}_{from-XXX}_{overrides}.
    """
    is_resuming = training_mode == TrainingModes.RESUME

    if is_resuming:
        return str(resolve_run_dir(run_path))

    runs_base = f"{log_dir}/{task_name}/runs"
    counter = dir_counter(runs_base, 3, is_resuming)
    run_suffix = from_run_number(run_path, training_mode)
    override_suffix = smart_override_dirname(override_dirname)

    run_name = f"{counter}_{name}_{run_suffix}_{override_suffix}"
    max_length = os.pathconf(get_mldft_model_path(), "PC_NAME_MAX")
    return runs_base + "/" + run_name[:max_length]


OmegaConf.register_new_resolver("get_run_directory", get_run_directory)


def to_no_basis_transforms_dataset_statistics(
    dataset_statistics_path: str, transformation_name: str
) -> str:
    """Converts a dataset statistics path to no basis transforms by a simple replacement."""
    return dataset_statistics_path.replace(transformation_name, "no_basis_transforms")


OmegaConf.register_new_resolver(
    "to_no_basis_transforms_dataset_statistics", to_no_basis_transforms_dataset_statistics
)

OmegaConf.register_new_resolver("get_len", lambda x: len(x))
OmegaConf.register_new_resolver("requires_grid", requires_grid)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)


# tensorframes resolvers
def parse_reps_import(rep):
    """Imports and calls the parse_reps function from tensorframes.

    The import is done here to make the tests run without the tensorframes package. If the package
    is not installed, using this resolver will raise an ImportError that indicates that the
    tensorframes package is missing.
    """
    from tensorframes.reps.utils import parse_reps

    return parse_reps(rep)


# getting dim of tensor rep:
OmegaConf.register_new_resolver("get_rep_dim", lambda rep: parse_reps_import(rep).dim)
OmegaConf.register_new_resolver("get_tensorrep_dim", lambda rep: parse_reps_import(f"t{rep}").dim)
OmegaConf.register_new_resolver("get_irrep_dim", lambda rep: parse_reps_import(f"i{rep}").dim)

# parse reps:
OmegaConf.register_new_resolver("parse_reps", lambda rep: parse_reps_import(rep))


def mux(condition: bool, true_value, false_value):
    """Returns true_value if condition is True and false_value else."""

    return true_value if condition else false_value


OmegaConf.register_new_resolver("mux", mux)


def check_if_resuming(training_mode: str) -> bool:
    """Check if training_mode is 'resume_training'."""
    if training_mode is None:
        return False
    assert isinstance(training_mode, str), "Training mode must be a string"
    return training_mode == TrainingModes.RESUME


OmegaConf.register_new_resolver("check_if_resuming", check_if_resuming)
