import shutil
from argparse import ArgumentParser
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tqdm import tqdm

from mldft.utils.environ import get_mldft_model_path


def count_tensorboard_steps(tensorboard_file: Path, step_threshold: int) -> int:
    """Count the number of steps in a tensorboard file.

    Args:
        tensorboard_file (Path): Path to the tensorboard file.
        step_threshold (int): Minimum number of steps required to not be moved. If steps >= step_threshold, the function
            returns early to be more efficient.
    """
    steps = 0
    try:
        for record in RawEventFileLoader(str(tensorboard_file)).Load():
            event = Event()
            event.ParseFromString(record)
            steps = max(steps, event.step)
            if steps >= step_threshold:
                break
    except Exception as e:
        print(f"Error reading {tensorboard_file}: {e}")
    return steps


def move_short_runs(main_directory: Path, short_runs_directory: Path, step_threshold: int):
    """Move runs with insufficient steps to a separate directory.

    Args:
        main_directory (Path): Directory containing subdirectories with tensorboard files.
        short_runs_directory (Path): Directory to move runs with insufficient steps.
        step_threshold (int): Minimum number of steps required to not be moved.
    """
    # Create directory for short runs if it doesn't exist
    short_runs_directory.mkdir(parents=True, exist_ok=True)
    subdirs = [subdir for subdir in main_directory.iterdir() if subdir.is_dir()]

    # Iterate through each subfolder in the main directory
    for subdir in tqdm(subdirs, desc="Scanning runs", unit="run"):
        tensorboard_files = list(subdir.glob("events.out.tfevents*"))
        if tensorboard_files:
            for tb_file in tensorboard_files:
                steps = count_tensorboard_steps(tb_file, step_threshold)
                if steps < step_threshold:
                    # Move the whole directory if steps are below threshold
                    dest_path = short_runs_directory / subdir.name
                    try:
                        shutil.move(str(subdir), dest_path)
                        print(f"Moved {subdir} to {dest_path} due to insufficient steps: {steps}")
                    except Exception as e:
                        print(f"Error moving {subdir} to {dest_path}: {e}")
                    break
        else:
            # Move the whole directory if no tensorboard file found
            dest_path = short_runs_directory / subdir.name
            try:
                shutil.move(str(subdir), dest_path)
                print(f"Moved {subdir} to {dest_path} due to missing tensorboard file")
            except Exception as e:
                print(f"Error moving {subdir} to {dest_path}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--step_threshold", type=int, default=2000)
    main_directory = get_mldft_model_path() / "train" / "runs"
    parser.add_argument("--main_directory", type=str, default=main_directory)
    short_runs_directory = main_directory.parent / "short_runs"
    parser.add_argument("--short_runs_directory", type=str, default=short_runs_directory)
    args = parser.parse_args()
    main_directory = Path(args.main_directory)
    short_runs_directory = Path(args.short_runs_directory)
    step_threshold = args.step_threshold
    move_short_runs(main_directory, short_runs_directory, step_threshold)
