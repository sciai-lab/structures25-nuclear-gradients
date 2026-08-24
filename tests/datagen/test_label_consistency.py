"""Consistency check between two label directories.

For every molecule present in both ``REFERENCE_LABELS_DIR`` and ``NEW_LABELS_DIR``
(matched by ``<mol_id>.zarr.zip`` filename), assert that the new labels contain
every array the reference holds with a compatible shape and dtype. This catches a
regenerated / reprocessed label set that drops an array, changes its rank, or
changes its dtype. The tests skip cleanly when the data drive is not mounted.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import zarr

_DFT_DATA = Path(os.environ.get("DFT_DATA", "/path/to/data"))

# Two label directories to compare. The new labels are validated against the
# reference for every molecule (matched by ``<mol_id>.zarr.zip`` filename) that
# exists in both. Point ``NEW_LABELS_DIR`` at the freshly produced / reprocessed
# label directory.
REFERENCE_LABELS_DIR = _DFT_DATA / "OMolPBE" / "labels"
NEW_LABELS_DIR = _DFT_DATA / "OMolTest" / "labels"

# How many shared molecules to compare across the two directories.
N_MOLECULES_TO_COMPARE = 2

ALLOWED_REFERENCE_ONLY_LABELS = frozenset(
    {
        "ks_labels/energies/e_ext",
        "ks_labels/energies/e_hartree",
        "ks_labels/energies/e_kin",
        "ks_labels/energies/e_xc",
        "of_labels/energies/e_kin",
        "of_labels/energies/e_xc",
        "of_labels/spatial/grad_kin",
        "of_labels/spatial/grad_xc",
    }
)


def _collect_array_specs(
    root: zarr.hierarchy.Group,
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Recursively map every array path under ``root`` to its ``(shape, dtype)``."""

    specs: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

    def visit(name: str, obj) -> None:
        # Groups have no ``dtype``; only record leaf arrays.
        if getattr(obj, "dtype", None) is not None:
            specs[name] = (tuple(obj.shape), np.dtype(obj.dtype))

    root.visititems(visit)
    return specs


def assert_labels_consistent(
    reference: zarr.hierarchy.Group,
    candidate: zarr.hierarchy.Group,
    *,
    exact_shape: bool = True,
    ignored_arrays: set[str] | frozenset[str] = frozenset(),
) -> None:
    """Assert ``candidate`` represents every array in ``reference``.

    Args:
        reference: The label group whose contents define the required schema.
        candidate: The new label group being validated.
        exact_shape: If True, every shared array must match shape and dtype
            exactly (the same-molecule case). If False, only the number of
            dimensions and the dtype *kind* must match, which is the invariant
            that holds across different molecules (the per-molecule ``T`` scf-step
            and ``B`` basis axes legitimately differ).
        ignored_arrays: Array paths to exclude from missing, shape, and dtype
            checks.

    Raises:
        AssertionError: If an array is missing, or a shared array's shape/dtype
            does not match.
    """

    ref_specs = {
        name: spec
        for name, spec in _collect_array_specs(reference).items()
        if name not in ignored_arrays
    }
    cand_specs = {
        name: spec
        for name, spec in _collect_array_specs(candidate).items()
        if name not in ignored_arrays
    }

    missing = sorted(set(ref_specs) - set(cand_specs))
    assert not missing, "New labels are missing arrays present in the reference: " + ", ".join(
        missing
    )

    shape_errors: list[str] = []
    dtype_errors: list[str] = []
    for name, (ref_shape, ref_dtype) in ref_specs.items():
        cand_shape, cand_dtype = cand_specs[name]
        if exact_shape:
            if cand_shape != ref_shape:
                shape_errors.append(f"{name}: reference {ref_shape} vs new {cand_shape}")
            if cand_dtype != ref_dtype:
                dtype_errors.append(f"{name}: reference {ref_dtype} vs new {cand_dtype}")
        else:
            if len(cand_shape) != len(ref_shape):
                shape_errors.append(
                    f"{name}: ndim reference {len(ref_shape)} vs new {len(cand_shape)}"
                )
            if cand_dtype.kind != ref_dtype.kind:
                dtype_errors.append(f"{name}: reference {ref_dtype} vs new {cand_dtype}")

    assert not shape_errors, "Shape mismatches between reference and new labels:\n" + "\n".join(
        shape_errors
    )
    assert not dtype_errors, "Dtype mismatches between reference and new labels:\n" + "\n".join(
        dtype_errors
    )


def _matched_label_pairs() -> list[tuple[str, Path, Path]]:
    """First ``N_MOLECULES_TO_COMPARE`` molecules present in both label directories.

    Returns ``(mol_file, reference_path, new_path)`` tuples, matched by filename.
    Returns an empty list if either directory is missing so collection does not
    fail; the test then skips.
    """

    if not REFERENCE_LABELS_DIR.is_dir() or not NEW_LABELS_DIR.is_dir():
        return []
    reference = {p.name: p for p in REFERENCE_LABELS_DIR.glob("*.zarr.zip")}
    new = {p.name: p for p in NEW_LABELS_DIR.glob("*.zarr.zip")}
    common = sorted(set(reference) & set(new))[:N_MOLECULES_TO_COMPARE]
    return [(name, reference[name], new[name]) for name in common]


_MATCHED_PAIRS = _matched_label_pairs()


@pytest.mark.parametrize(
    "mol_file, reference_path, new_path",
    _MATCHED_PAIRS
    or [
        pytest.param(
            None,
            None,
            None,
            marks=pytest.mark.skip(reason="reference/new label directory unavailable"),
        )
    ],
    ids=[name for name, _, _ in _MATCHED_PAIRS] or ["unavailable"],
)
def test_new_labels_match_reference_per_molecule(mol_file, reference_path, new_path):
    """Each shared molecule's new labels must fully match the reference labels."""

    if not REFERENCE_LABELS_DIR.is_dir() or not NEW_LABELS_DIR.is_dir():
        pytest.skip(f"Label directory unavailable: {REFERENCE_LABELS_DIR} or {NEW_LABELS_DIR}")

    reference = zarr.open(str(reference_path), mode="r")
    new_labels = zarr.open(str(new_path), mode="r")
    # PBE has a full SCF trajectory and a different OF basis; compare schema rank
    # and dtype kind.
    assert_labels_consistent(
        reference,
        new_labels,
        exact_shape=False,
        ignored_arrays=ALLOWED_REFERENCE_ONLY_LABELS,
    )
