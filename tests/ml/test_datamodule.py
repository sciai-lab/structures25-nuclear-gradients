import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import zarr

from mldft.ml.data.components.dataset import HDF5ShardOFDataset
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.data.datamodule import OFDataModule
from scripts.convert_labels_to_hdf5_shards import (
    ConversionConfig,
    ShardTask,
    convert_shard,
    scan_source_file,
)


@pytest.mark.parametrize("batch_size", [(7), (32)])
def test_datamodule(batch_size, dummy_basis_info, dummy_dataset_path, master_transform_to_torch):
    """Rudimentary test for the current simple DataModule implementation."""
    label_path = Path(dummy_dataset_path)
    data_path = label_path.parent.parent
    split_file = label_path.parent / "train_val_test_split.pkl"
    datamodule = OFDataModule(
        split_file,
        data_path,
        transforms=master_transform_to_torch,
        basis_info=dummy_basis_info,
        batch_size=batch_size,
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    train_loader = datamodule.train_dataloader()
    validation_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    train_length = len(datamodule.train_set)
    validation_length = len(datamodule.val_set)
    test_length = len(datamodule.test_set)
    total_length = train_length + validation_length + test_length
    # Test that each loader is not none, has the correct batch size, and that all positions have 3 dimensions
    for i, loader in enumerate([train_loader, validation_loader, test_loader]):
        assert loader is not None
        assert loader.batch_size == batch_size
        sample = next(iter(loader))
        # If dataset batch size is larger than whole subset, batch size will be smaller.
        if len(loader) > 1:
            assert len(sample) == batch_size
        else:
            assert len(sample) <= batch_size
        assert all(pos.shape[-1] == 3 for pos in sample.pos)


def _create_hdf5_manifest_from_dummy_dataset(
    dummy_dataset_path: Path,
    output_dir: Path,
    ground_state_only: bool = False,
    include_nuclear_gradients: bool = False,
    split_each_source_file: bool = False,
) -> Path:
    source_paths = sorted(dummy_dataset_path.glob("*.zarr"))
    zip_input_dir = output_dir / "_zip_sources"
    zip_input_dir.mkdir(parents=True, exist_ok=True)

    zip_paths: list[Path] = []
    for src in source_paths:
        # Keep the same basename (e.g. *.zarr) so source keys still match split entries.
        dst = zip_input_dir / src.name
        with zipfile.ZipFile(dst, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for file in src.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(src))
        zip_paths.append(dst)

    if include_nuclear_gradients:
        n_nuclear_gradient_files = sum(
            (output_dir / "nuclear_gradients" / path.name).exists() for path in zip_paths
        )
        if n_nuclear_gradient_files == 0:
            raise ValueError(
                "include_nuclear_gradients is enabled but no nuclear gradient files were found "
                f"for label directory '{dummy_dataset_path.name}'."
            )

    infos = [
        scan_source_file(
            path,
            output_dtype="float64",
            include_nuclear_gradients=include_nuclear_gradients,
        )
        for path in zip_paths
    ]
    config = ConversionConfig(
        energy_key="e_kin",
        gradient_key="grad_kin",
        output_dtype="float64",
        zstd_level=0,
        ground_state_only=ground_state_only,
        include_nuclear_gradients=include_nuclear_gradients,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = dummy_dataset_path.parent.name
    label_subdir = dummy_dataset_path.name
    shard_infos = [[info] for info in infos] if split_each_source_file else [infos]
    shards = []
    for shard_id, shard_source_files in enumerate(shard_infos):
        task = ShardTask(
            shard_id=shard_id,
            source_files=shard_source_files,
            output_path=str(output_dir / f"shard_{shard_id:05d}.h5"),
            input_dir=str(zip_input_dir),
            source_key_prefix=f"{dataset_name}/{label_subdir}",
            config=config,
        )
        result = convert_shard(task)
        shard = asdict(result)
        shard["output_file"] = str((output_dir / result.output_file).resolve())
        shards.append(shard)
    manifest = {
        "format": "mldft_h5_sharded_v2_bucketed_row",
        "config": asdict(config),
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f)
    return manifest_path


def _create_dummy_nuclear_gradient_files(
    label_dir: Path,
    output_dir: Path,
    transform_grad_key: str,
    gradient_key: str = "grad_kin",
    outlier_indices: set[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    nuclear_gradient_dir = output_dir / "nuclear_gradients"
    nuclear_gradient_dir.mkdir(parents=True, exist_ok=True)
    outlier_indices = outlier_indices or set()

    expected_values: dict[str, dict[str, np.ndarray]] = {}
    for idx, source_path in enumerate(sorted(label_dir.glob("*.zarr"))):
        root = zarr.open(source_path, mode="r")
        n_atom = int(root["geometry"]["atomic_numbers"].shape[0])
        n_basis = int(root["of_labels"]["spatial"]["coeffs"].shape[1])
        n_nuc_coords = 3 * n_atom
        nuc_grad_label = np.arange(n_nuc_coords, dtype=np.float64) + idx / 10.0
        transform_grad_value = 10.0 if idx in outlier_indices else 0.1
        transform_grad_contracted = np.full(
            (n_nuc_coords, n_basis),
            fill_value=transform_grad_value,
            dtype=np.float64,
        )

        with zarr.ZipStore(nuclear_gradient_dir / source_path.name, mode="w") as store:
            grad_root = zarr.group(store=store)
            grad_root.create_dataset(f"nuc_{gradient_key}", data=nuc_grad_label)
            grad_root.create_dataset(transform_grad_key, data=transform_grad_contracted)

        expected_values[source_path.name] = {
            "nuc_grad_label": nuc_grad_label,
            "transform_grad_contracted": transform_grad_contracted,
        }

    return expected_values


def _copy_labels_to_subdir(source_dir: Path, destination_dir: Path) -> Path:
    shutil.copytree(source_dir, destination_dir)
    return destination_dir


def test_datamodule_hdf5_shards(
    dummy_basis_info, dummy_dataset_path, master_transform_to_torch, tmp_path
):
    label_path = Path(dummy_dataset_path)
    data_path = label_path.parent.parent
    split_file = label_path.parent / "train_val_test_split.pkl"
    manifest_path = _create_hdf5_manifest_from_dummy_dataset(
        dummy_dataset_path=label_path, output_dir=tmp_path / "labels_h5"
    )

    datamodule = OFDataModule(
        split_file,
        data_path,
        transforms=master_transform_to_torch,
        basis_info=dummy_basis_info,
        batch_size=8,
        dataset_kwargs={
            "add_irreps": True,
            "energy_key": "e_kin",
            "gradient_key": "grad_kin",
            "hdf5_manifest_path": str(manifest_path),
        },
    )
    datamodule.setup("fit")
    datamodule.setup("test")
    train_loader = datamodule.train_dataloader()
    validation_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert len(datamodule.train_set) == np.sum(np.arange(1, 9))
    assert len(datamodule.val_set) == 9
    assert len(datamodule.test_set) == 10

    for loader in [train_loader, validation_loader, test_loader]:
        sample = next(iter(loader))
        assert all(pos.shape[-1] == 3 for pos in sample.pos)


def test_datamodule_hdf5_negative_scf_iterations(
    dummy_basis_info, dummy_dataset_path, master_transform_to_torch, tmp_path
):
    label_path = Path(dummy_dataset_path)
    data_path = label_path.parent.parent
    split_file = label_path.parent / "train_val_test_split.pkl"
    manifest_path = _create_hdf5_manifest_from_dummy_dataset(
        dummy_dataset_path=label_path, output_dir=tmp_path / "labels_h5_neg"
    )

    datamodule = OFDataModule(
        split_file,
        data_path,
        transforms=master_transform_to_torch,
        basis_info=dummy_basis_info,
        batch_size=8,
        dataset_kwargs={
            "add_irreps": True,
            "energy_key": "e_kin",
            "gradient_key": "grad_kin",
            "limit_scf_iterations": [-1],
            "keep_initial_guess": False,
            "hdf5_manifest_path": str(manifest_path),
        },
    )
    datamodule.setup("fit")

    assert len(datamodule.train_set) == 8
    expected_last_iterations = list(range(8))
    observed_iterations = [datamodule.train_set[i].scf_iteration for i in range(8)]
    assert observed_iterations == expected_last_iterations


def test_hdf5_ground_state_only_shards(dummy_basis_info, dummy_dataset_path, tmp_path):
    label_path = Path(dummy_dataset_path)
    manifest_path = _create_hdf5_manifest_from_dummy_dataset(
        dummy_dataset_path=label_path,
        output_dir=tmp_path / "labels_denop_h5",
        ground_state_only=True,
    )
    dataset_name = label_path.parent.name
    label_subdir = label_path.name
    source_keys = [f"{dataset_name}/{label_subdir}/dummy_sample_{i}.zarr" for i in range(8)]

    dataset = HDF5ShardOFDataset(
        source_keys=source_keys,
        num_scf_iterations_per_path=[1 + i for i in range(8)],
        basis_info=dummy_basis_info,
        hdf5_manifest_path=manifest_path,
        limit_scf_iterations=-1,
        keep_initial_guess=False,
        additional_keys_at_ground_state={"of_labels/energies/e_kin": Representation.SCALAR},
    )

    assert len(dataset) == 8
    sample = dataset[0]
    assert sample.is_ground_state.item()
    assert "of_labels/energies/ground_state_e_kin" in sample


def test_hdf5_load_nuclear_gradients(dummy_basis_info, dummy_dataset_path, tmp_path):
    transformed_label_dir = _copy_labels_to_subdir(
        Path(dummy_dataset_path),
        tmp_path / "labels_local_frames_global_symmetric_natrep",
    )
    output_dir = tmp_path / "labels_h5_nuclear"
    expected_values = _create_dummy_nuclear_gradient_files(
        label_dir=transformed_label_dir,
        output_dir=output_dir,
        transform_grad_key="local_frames_global_symmetric_natrep_grad_contracted",
    )
    manifest_path = _create_hdf5_manifest_from_dummy_dataset(
        dummy_dataset_path=transformed_label_dir,
        output_dir=output_dir,
        include_nuclear_gradients=True,
    )

    dataset_name = transformed_label_dir.parent.name
    label_subdir = transformed_label_dir.name
    source_key = f"{dataset_name}/{label_subdir}/dummy_sample_2.zarr"
    dataset = HDF5ShardOFDataset(
        source_keys=[source_key],
        num_scf_iterations_per_path=[3],
        basis_info=dummy_basis_info,
        hdf5_manifest_path=manifest_path,
        load_nuclear_gradients=True,
    )

    nonground_sample = dataset[0]
    ground_state_sample = dataset[2]
    expected = expected_values["dummy_sample_2.zarr"]

    assert not nonground_sample.has_nuc_grad_label.item()
    assert np.allclose(
        nonground_sample.nuc_grad_label,
        np.zeros_like(expected["nuc_grad_label"]),
    )
    assert nonground_sample.transform_grad_contracted == expected["nuc_grad_label"].shape[0]

    assert ground_state_sample.is_ground_state.item()
    assert ground_state_sample.has_nuc_grad_label.item()
    assert np.allclose(ground_state_sample.nuc_grad_label, expected["nuc_grad_label"])
    assert np.allclose(
        ground_state_sample.transform_grad_contracted,
        expected["transform_grad_contracted"],
    )


def test_hdf5_ground_state_only_load_nuclear_gradients(
    dummy_basis_info, dummy_dataset_path, tmp_path
):
    transformed_label_dir = _copy_labels_to_subdir(
        Path(dummy_dataset_path),
        tmp_path / "denop_case" / "labels_local_frames_global_symmetric_natrep",
    )
    output_dir = tmp_path / "denop_case" / "labels_denop_h5_nuclear"
    expected_values = _create_dummy_nuclear_gradient_files(
        label_dir=transformed_label_dir,
        output_dir=output_dir,
        transform_grad_key="local_frames_global_symmetric_natrep_grad_contracted",
    )
    manifest_path = _create_hdf5_manifest_from_dummy_dataset(
        dummy_dataset_path=transformed_label_dir,
        output_dir=output_dir,
        ground_state_only=True,
        include_nuclear_gradients=True,
    )

    dataset_name = transformed_label_dir.parent.name
    label_subdir = transformed_label_dir.name
    source_key = f"{dataset_name}/{label_subdir}/dummy_sample_2.zarr"
    dataset = HDF5ShardOFDataset(
        source_keys=[source_key],
        num_scf_iterations_per_path=[3],
        basis_info=dummy_basis_info,
        hdf5_manifest_path=manifest_path,
        limit_scf_iterations=-1,
        keep_initial_guess=False,
        load_nuclear_gradients=True,
    )

    sample = dataset[0]
    expected = expected_values["dummy_sample_2.zarr"]

    assert len(dataset) == 1
    assert sample.is_ground_state.item()
    assert sample.has_nuc_grad_label.item()
    assert np.allclose(sample.nuc_grad_label, expected["nuc_grad_label"])
    assert np.allclose(sample.transform_grad_contracted, expected["transform_grad_contracted"])


def test_hdf5_include_nuclear_gradients_requires_labels(dummy_dataset_path, tmp_path):
    transformed_label_dir = _copy_labels_to_subdir(
        Path(dummy_dataset_path),
        tmp_path / "labels_local_frames_global_symmetric_natrep",
    )

    with pytest.raises(ValueError, match="include_nuclear_gradients is enabled"):
        _create_hdf5_manifest_from_dummy_dataset(
            dummy_dataset_path=transformed_label_dir,
            output_dir=tmp_path / "labels_h5_nuclear_missing",
            include_nuclear_gradients=True,
        )


def test_from_file_skips_nuclear_gradient_outliers(dummy_basis_info, dummy_dataset_path, tmp_path):
    transformed_label_dir = _copy_labels_to_subdir(
        Path(dummy_dataset_path),
        tmp_path / "labels_local_frames_global_symmetric_natrep",
    )
    _create_dummy_nuclear_gradient_files(
        label_dir=transformed_label_dir,
        output_dir=tmp_path,
        transform_grad_key="local_frames_global_symmetric_natrep_grad_contracted",
        outlier_indices={2},
    )

    sample = OFData.from_file(
        path=transformed_label_dir / "dummy_sample_2.zarr",
        scf_iteration=2,
        basis_info=dummy_basis_info,
        load_nuclear_gradients=True,
        nuclear_gradient_outlier_threshold=5.0,
    )

    assert sample.is_ground_state.item()
    assert not sample.has_nuc_grad_label.item()
    assert np.allclose(sample.nuc_grad_label, np.zeros_like(sample.nuc_grad_label))
    assert sample.transform_grad_contracted == sample.nuc_grad_label.shape[0]


def test_hdf5_skips_nuclear_gradient_outliers(dummy_basis_info, dummy_dataset_path, tmp_path):
    transformed_label_dir = _copy_labels_to_subdir(
        Path(dummy_dataset_path),
        tmp_path / "labels_local_frames_global_symmetric_natrep",
    )
    output_dir = tmp_path / "labels_h5_nuclear_outlier"
    _create_dummy_nuclear_gradient_files(
        label_dir=transformed_label_dir,
        output_dir=output_dir,
        transform_grad_key="local_frames_global_symmetric_natrep_grad_contracted",
        outlier_indices={2},
    )
    manifest_path = _create_hdf5_manifest_from_dummy_dataset(
        dummy_dataset_path=transformed_label_dir,
        output_dir=output_dir,
        include_nuclear_gradients=True,
    )

    dataset_name = transformed_label_dir.parent.name
    label_subdir = transformed_label_dir.name
    source_key = f"{dataset_name}/{label_subdir}/dummy_sample_2.zarr"
    dataset = HDF5ShardOFDataset(
        source_keys=[source_key],
        num_scf_iterations_per_path=[3],
        basis_info=dummy_basis_info,
        hdf5_manifest_path=manifest_path,
        load_nuclear_gradients=True,
        nuclear_gradient_outlier_threshold=5.0,
    )

    sample = dataset[2]

    assert sample.is_ground_state.item()
    assert not sample.has_nuc_grad_label.item()
    assert np.allclose(sample.nuc_grad_label, np.zeros_like(sample.nuc_grad_label))
    assert sample.transform_grad_contracted == sample.nuc_grad_label.shape[0]
