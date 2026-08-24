import pickle
from pathlib import Path

import numpy as np
import pytest
from omegaconf import ListConfig

from mldft.ofdft.denop import prepare_paths as prepare_paths_module


@pytest.fixture
def split_dict():
    """Provide a compact train/val/test split dictionary used by path-selection tests."""
    return {
        "train": [("dataset_a", "mol_10.zarr.zip", 3)],
        "val": [("dataset_b", "mol_11.zarr.zip", 4), ("dataset_b", "mol_12.zarr.zip", 5)],
        "test": [("dataset_c", "mol_13.zarr.zip", 6)],
    }


def test_construct_label_path():
    """Ensure label paths are assembled as ``<data>/<dataset>/labels/<file>``."""
    path = prepare_paths_module.construct_label_path(Path("/data"), "qm9", "mol.zarr.zip")

    assert path == Path("/data/qm9/labels/mol.zarr.zip")


def test_get_paths_from_split(split_dict):
    """Verify split extraction returns aligned label paths and SCF iteration counts."""
    paths, scf_iterations = prepare_paths_module.get_paths_from_split(
        split_dict, "val", Path("/data")
    )

    assert paths == [
        Path("/data/dataset_b/labels/mol_11.zarr.zip"),
        Path("/data/dataset_b/labels/mol_12.zarr.zip"),
    ]
    assert scf_iterations == [4, 5]


def test_find_molecule_path_by_id_succeeds(split_dict):
    """Check that molecule-id lookup searches the requested split before other splits."""
    split_dict["train"].append(("dataset_a", "mol_11.zarr.zip", 8))

    path, scf_iterations = prepare_paths_module.find_molecule_path_by_id(
        11, split_dict, Path("/data"), user_split="val"
    )

    assert path == Path("/data/dataset_b/labels/mol_11.zarr.zip")
    assert scf_iterations == 4


def test_find_molecule_path_by_id_fails(split_dict):
    """Confirm that molecule-id lookup returns None when the ID is not found."""
    assert (
        prepare_paths_module.find_molecule_path_by_id(999, split_dict, Path("/data"), "val")
        is None
    )


@pytest.mark.parametrize(
    ("molecule_choice", "seed", "start_idx", "expected"),
    [
        ("always_2", None, 0, np.array([2, 2, 2])),
        ("first_n", None, 1, np.array([1, 2])),
        ("seeded_random", 7, 1, np.random.default_rng(7).choice(6, 3, replace=False)[1:]),
    ],
)
def test_configure_dataset_indices_from_string_variants(
    molecule_choice: str, seed: int | None, start_idx: int, expected: np.ndarray
):
    """Cover deterministic index selection variants for string-based molecule choice."""
    dataset_indices = prepare_paths_module.configure_dataset_indices_from_string(
        dataset_size=6,
        n_molecules=3,
        molecule_choice=molecule_choice,
        seed=seed,
        start_idx=start_idx,
    )

    assert np.array_equal(dataset_indices, expected)


def test_configure_dataset_indices_from_string_random_ignores_start_idx(monkeypatch):
    """Confirm random mode ignores ``start_idx`` and emits a warning."""
    warnings = []
    monkeypatch.setattr(
        prepare_paths_module.logger, "warning", lambda message: warnings.append(message)
    )

    dataset_indices = prepare_paths_module.configure_dataset_indices_from_string(
        dataset_size=6,
        n_molecules=3,
        molecule_choice="random",
        seed=5,
        start_idx=2,
    )

    assert len(dataset_indices) == 3
    assert "Ignoring start_idx" in warnings[0]


@pytest.mark.parametrize(
    ("dataset_size", "n_molecules", "molecule_choice", "seed", "message"),
    [
        (2, 3, "first_n", None, "n_molecules cannot be greater than dataset_size"),
        (3, 2, "seeded_random", None, "seed must be provided"),
        (3, 2, "nope", None, "Unknown molecule choice"),
    ],
)
def test_configure_dataset_indices_from_string_rejects_invalid_inputs(
    dataset_size: int,
    n_molecules: int,
    molecule_choice: str,
    seed: int | None,
    message: str,
):
    """Ensure invalid string-choice inputs raise informative ``ValueError`` messages."""
    with pytest.raises(ValueError, match=message):
        prepare_paths_module.configure_dataset_indices_from_string(
            dataset_size=dataset_size,
            n_molecules=n_molecules,
            molecule_choice=molecule_choice,
            seed=seed,
        )


def test_configure_dataset_indices_from_list_adjusts_length(monkeypatch):
    """Validate list-based selection truncates/clamps length and logs adjustments."""
    warnings = []
    monkeypatch.setattr(
        prepare_paths_module.logger, "warning", lambda message: warnings.append(message)
    )

    shorter = prepare_paths_module.configure_dataset_indices_from_list(
        n_molecules=3, molecule_choice=[7, 8], start_idx=0
    )
    longer = prepare_paths_module.configure_dataset_indices_from_list(
        n_molecules=2, molecule_choice=[7, 8, 9], start_idx=0
    )
    exact = prepare_paths_module.configure_dataset_indices_from_list(
        n_molecules=2, molecule_choice=[4, 5], start_idx=1
    )

    assert np.array_equal(shorter, np.array([7, 8]))
    assert np.array_equal(longer, np.array([7, 8]))
    assert np.array_equal(exact, np.array([5]))
    assert "Setting n_molecules to 2" in warnings[0]
    assert "Using the first 2 molecules" in warnings[1]


def test_configure_dataset_indices_dispatches_and_validates():
    """Check dispatcher handles list configs and rejects invalid counts/types."""
    assert np.array_equal(
        prepare_paths_module.configure_dataset_indices(
            dataset_size=3,
            n_molecules=None,
            molecule_choice=ListConfig([2, 1, 0]),
            seed=None,
        ),
        np.array([2, 1, 0]),
    )

    with pytest.raises(ValueError, match="n_molecules must be non-negative"):
        prepare_paths_module.configure_dataset_indices(
            dataset_size=3, n_molecules=-1, molecule_choice="first_n", seed=None
        )

    with pytest.raises(ValueError, match="Unknown molecule choice"):
        prepare_paths_module.configure_dataset_indices(
            dataset_size=3, n_molecules=1, molecule_choice=1.5, seed=None
        )


def test_prepare_molecule_paths_works(split_dict):
    """Verify explicit molecule IDs are resolved across splits and missing IDs fail."""
    paths, iterations = prepare_paths_module.prepare_molecule_paths(
        data_dir=Path("/data"),
        split_dict=split_dict,
        n_molecules=None,
        molecule_ids=[13, 10],
        molecule_choice="first_n",
        split="val",
        seed=None,
    )

    assert paths == [
        Path("/data/dataset_c/labels/mol_13.zarr.zip"),
        Path("/data/dataset_a/labels/mol_10.zarr.zip"),
    ]
    assert iterations == [6, 3]


def test_prepare_molecule_paths_throws_for_missing_ids(split_dict):
    with pytest.raises(ValueError, match="Molecule 99 not found"):
        prepare_paths_module.prepare_molecule_paths(
            data_dir=Path("/data"),
            split_dict=split_dict,
            n_molecules=None,
            molecule_ids=[99],
            molecule_choice="first_n",
            split="val",
            seed=None,
        )


def test_prepare_paths_filters_missing_when_continue_denop(
    monkeypatch, tmp_path: Path, split_dict
):
    """Ensure continue mode removes already-processed molecules via helper hooks."""
    split_file_path = tmp_path / "split.pkl"
    with open(split_file_path, "wb") as handle:
        pickle.dump(split_dict, handle)

    monkeypatch.setattr(prepare_paths_module, "get_mldft_data_path", lambda: Path("/data"))
    monkeypatch.setattr(
        prepare_paths_module,
        "get_existing_sample_paths",
        lambda denop_dir: [Path("/data/dataset_b/labels/mol_11.zarr.zip")],
    )
    monkeypatch.setattr(
        prepare_paths_module,
        "get_missing_paths_and_scf_iters",
        lambda all_paths, iterations, existing: (all_paths[1:], iterations[1:]),
    )

    loaded_split_dict, paths, iterations = prepare_paths_module.prepare_paths(
        denop_dir=tmp_path,
        n_molecules=2,
        molecule_ids=None,
        molecule_choice="first_n",
        continue_denop=True,
        split_file_path=split_file_path,
        split="val",
        seed=None,
    )

    assert loaded_split_dict == split_dict
    assert paths == [Path("/data/dataset_b/labels/mol_12.zarr.zip")]
    assert iterations == [5]


def test_prepare_paths_keeps_all_paths_without_continue(monkeypatch, tmp_path: Path, split_dict):
    """Check non-continue mode returns the full selected molecule subset unchanged."""
    split_file_path = tmp_path / "split.pkl"
    with open(split_file_path, "wb") as handle:
        pickle.dump(split_dict, handle)

    monkeypatch.setattr(prepare_paths_module, "get_mldft_data_path", lambda: Path("/data"))

    _, paths, iterations = prepare_paths_module.prepare_paths(
        denop_dir=tmp_path,
        n_molecules=1,
        molecule_ids=None,
        molecule_choice="first_n",
        continue_denop=False,
        split_file_path=split_file_path,
        split="val",
        seed=None,
    )

    assert paths == [Path("/data/dataset_b/labels/mol_11.zarr.zip")]
    assert iterations == [4]


def test_prepare_paths_continue_mode_when_no_missing_paths(
    monkeypatch, tmp_path: Path, split_dict
):
    """Ensure continue mode returns empty lists when all selected molecules already exist."""
    split_file_path = tmp_path / "split.pkl"
    with open(split_file_path, "wb") as handle:
        pickle.dump(split_dict, handle)

    info_messages = []
    monkeypatch.setattr(
        prepare_paths_module.logger, "info", lambda message: info_messages.append(message)
    )
    monkeypatch.setattr(prepare_paths_module, "get_mldft_data_path", lambda: Path("/data"))
    monkeypatch.setattr(
        prepare_paths_module,
        "get_existing_sample_paths",
        lambda denop_dir: [Path("/data/dataset_b/labels/mol_11.zarr.zip")],
    )
    monkeypatch.setattr(
        prepare_paths_module,
        "get_missing_paths_and_scf_iters",
        lambda all_paths, iterations, existing: ([], []),
    )

    _, paths, iterations = prepare_paths_module.prepare_paths(
        denop_dir=tmp_path,
        n_molecules=1,
        molecule_ids=None,
        molecule_choice="first_n",
        continue_denop=True,
        split_file_path=split_file_path,
        split="val",
        seed=None,
    )

    assert paths == []
    assert iterations == []
    assert any("All queried samples are already present" in message for message in info_messages)
