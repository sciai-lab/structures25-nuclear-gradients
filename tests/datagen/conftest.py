import pytest
from pyscf import gto

from mldft.datagen.datasets.dataset import delete_dataset
from mldft.datagen.datasets.md17 import MD17
from mldft.datagen.datasets.misc import MiscXYZ
from mldft.datagen.datasets.qm7x import QM7XEightBatch
from mldft.datagen.datasets.qm9 import QM9

TEST_CHK_FILENAMES = [
    "000001.chk",
    "000002.chk",
    "000003.000001.chk",
    "000003.000002.chk",
    "000004.000002.chk",
]


@pytest.fixture(scope="session")
def md17_dataset(tmp_path_factory):
    """Fixture for MD17 dataset."""
    raw_data_dir = tmp_path_factory.mktemp("dataset") / "test_raw_data_dir"
    kohn_sham_data_dir = tmp_path_factory.mktemp("dataset") / "test_kohn_sham_data_dir"
    filename = "testfilename"
    label_dir = tmp_path_factory.mktemp("dataset") / "test_label_dir"
    md17 = MD17(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_MD17",
    )
    yield md17

    delete_dataset(md17)


@pytest.fixture(scope="session")
def md17_dataset_for_filenametest(tmp_path_factory):
    """Fixture for MD17 dataset."""
    raw_data_dir = tmp_path_factory.mktemp("dataset") / "test_raw_data_dir"
    kohn_sham_data_dir = tmp_path_factory.mktemp("dataset") / "test_kohn_sham_data_dir"
    filename = "testfilename"
    label_dir = tmp_path_factory.mktemp("dataset") / "test_label_dir"
    md17 = MD17(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_MD17",
    )
    for chk_filename in TEST_CHK_FILENAMES:
        (kohn_sham_data_dir / (filename + "_" + chk_filename)).touch()
    yield md17

    delete_dataset(md17)


@pytest.fixture(scope="session")
def qm9_dataset_for_filenametest(tmp_path_factory):
    """Fixture for QM9 dataset."""
    raw_data_dir = tmp_path_factory.mktemp("dataset") / "test_raw_data_dir"
    kohn_sham_data_dir = tmp_path_factory.mktemp("dataset") / "test_kohn_sham_data_dir"
    label_dir = tmp_path_factory.mktemp("dataset") / "test_label_dir"
    filename = "testfilename"
    raw_data_dir.mkdir()
    (raw_data_dir / "dsgdb9nsd_000001.xyz").write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    qm9 = QM9(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_QM9",
    )
    for chk_filename in TEST_CHK_FILENAMES:
        (kohn_sham_data_dir / (filename + "_" + chk_filename)).touch()
    yield qm9

    delete_dataset(qm9)


@pytest.fixture(scope="session")
def qm9_dataset(tmp_path_factory):
    """Fixture for QM9 dataset."""
    raw_data_dir = tmp_path_factory.mktemp("dataset") / "test_raw_data_dir"
    kohn_sham_data_dir = tmp_path_factory.mktemp("dataset") / "test_kohn_sham_data_dir"
    label_dir = tmp_path_factory.mktemp("dataset") / "test_label_dir"
    filename = "testfilename"
    raw_data_dir.mkdir()
    (raw_data_dir / "dsgdb9nsd_000001.xyz").write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    qm9 = QM9(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_QM9",
    )
    yield qm9

    delete_dataset(qm9)


# the QM7XEightBatch is the smallest of the qm7x batches. Though it contains third row
# elements. But for testing purpose i argue we do not care for the elements
@pytest.fixture(scope="session")
def qm7x_dataset_for_filenametest(tmp_path_factory):
    """Fixture for QM7X dataset."""
    raw_data_dir = tmp_path_factory.mktemp("dataset") / "test_raw_data_dir"
    kohn_sham_data_dir = tmp_path_factory.mktemp("dataset") / "test_kohn_sham_data_dir"
    label_dir = tmp_path_factory.mktemp("dataset") / "test_label_dir"
    filename = "testfilename"
    qm7x = QM7XEightBatch(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_qm7x",
        allowed_atomic_numbers=(1, 6, 7, 8, 16, 17),
    )
    for chk_filename in TEST_CHK_FILENAMES:
        (kohn_sham_data_dir / (filename + "_" + chk_filename)).touch()
    yield qm7x

    delete_dataset(qm7x)


@pytest.fixture(scope="session")
def qm7x_dataset(tmp_path_factory):
    """Fixture for QM7X dataset."""
    raw_data_dir = tmp_path_factory.mktemp("dataset") / "test_raw_data_dir"
    kohn_sham_data_dir = tmp_path_factory.mktemp("dataset") / "test_kohn_sham_data_dir"
    label_dir = tmp_path_factory.mktemp("dataset") / "test_label_dir"
    filename = "testfilename"
    raw_data_dir.mkdir()
    (raw_data_dir / "0000001.xyz").write_text("2\n\nH 0 0 0\nH 0 0 0.74\n")
    qm7x = QM7XEightBatch(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_qm7x",
        allowed_atomic_numbers=(1, 6, 7, 8, 16, 17),
    )
    yield qm7x

    delete_dataset(qm7x)


@pytest.fixture(scope="session")
def misc_dataset(tmp_path_factory):
    """Fixture for MiscXYZ dataset."""

    molecule_small = [
        gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"),
        gto.M(atom="H 0 0 0; F 0 0 0.9"),
    ]

    dataset_path = tmp_path_factory.mktemp("dataset")
    raw_data_dir = dataset_path / "test_raw_data_dir"
    kohn_sham_data_dir = dataset_path / "test_kohn_sham_data_dir"
    label_dir = dataset_path / "test_label_dir"

    raw_data_dir.mkdir()

    filename = "testfilename"

    for i, mol in enumerate(molecule_small):
        charges, coords = mol.atom_charges(), mol.atom_coords(unit="angstrom")
        xyz = f"{len(charges)}\n\n"
        for charge, coord in zip(charges, coords):
            xyz += f"{charge} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"

        f = raw_data_dir / f"misc_{i:06}.xyz"
        f.write_text(xyz)

    misc_dataset = MiscXYZ(
        raw_data_dir=raw_data_dir.as_posix(),
        kohn_sham_data_dir=kohn_sham_data_dir.as_posix(),
        label_dir=label_dir.as_posix(),
        filename=filename,
        name="test_misc",
    )

    yield misc_dataset

    delete_dataset(misc_dataset)
