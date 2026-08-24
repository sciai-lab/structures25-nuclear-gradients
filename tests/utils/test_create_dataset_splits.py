from pathlib import Path

import numpy as np

from mldft.utils.create_dataset_splits import (
    QM9_CHARACTERIZED_SIZE,
    QM9_DIMENET_PERMUTATION_SEED,
    QM9_DIMENET_TRAIN_SIZE,
    QM9_DIMENET_VALIDATION_SIZE,
    QM9_TOTAL_SIZE,
    QM9_UNCHARACTERIZED_URL,
    _get_qm9_split_ids,
    _load_qm9_uncharacterized_ids,
)


def test_get_qm9_split_ids_matches_expected_sizes():
    uncharacterized_ids = np.arange(QM9_CHARACTERIZED_SIZE + 1, QM9_TOTAL_SIZE + 1)

    split_ids = _get_qm9_split_ids(
        uncharacterized_ids,
        permutation_seed=QM9_DIMENET_PERMUTATION_SEED,
        train_size=QM9_DIMENET_TRAIN_SIZE,
        validation_size=QM9_DIMENET_VALIDATION_SIZE,
    )

    assert len(split_ids["train"]) == QM9_DIMENET_TRAIN_SIZE
    assert len(split_ids["validation"]) == QM9_DIMENET_VALIDATION_SIZE
    assert (
        len(split_ids["train"]) + len(split_ids["validation"]) + len(split_ids["test"])
        == QM9_CHARACTERIZED_SIZE
    )
    assert set(split_ids["train"]).isdisjoint(split_ids["validation"])
    assert set(split_ids["train"]).isdisjoint(split_ids["test"])
    assert set(split_ids["validation"]).isdisjoint(split_ids["test"])
    assert set(uncharacterized_ids).isdisjoint(split_ids["train"])
    assert set(uncharacterized_ids).isdisjoint(split_ids["validation"])
    assert set(uncharacterized_ids).isdisjoint(split_ids["test"])


def test_load_qm9_uncharacterized_ids_downloads_from_official_url_when_missing(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("DFT_DATA", str(tmp_path))
    calls = []

    def fake_download_file(url: str, folder: Path, filename: str | None = None) -> Path:
        calls.append((url, folder, filename))
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / (filename or "downloaded.txt")
        target.write_text("header\n1 foo\n3 bar\n", encoding="utf-8")
        return target

    monkeypatch.setattr("mldft.utils.create_dataset_splits.download_file", fake_download_file)

    ids = _load_qm9_uncharacterized_ids()

    assert np.array_equal(ids, np.array([1, 3]))
    assert calls == [
        (
            QM9_UNCHARACTERIZED_URL,
            tmp_path / "QM9_perturbed_fock",
            "uncharacterized.txt",
        )
    ]
