"""Convert label zarr files into split-aware HDF5 shards (v2 colocated layout)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import zarr
from tqdm import tqdm

from mldft.utils.environ import get_mldft_data_path

SPLIT_NAMES = ("train", "val", "test")
FORMAT_NAME = "mldft_h5_sharded_v2_bucketed_row"
GROUND_STATE_ENERGY_KEYS = (
    "e_electron",
    "e_ext",
    "e_hartree",
    "e_kin",
    "e_kin_plus_xc",
    "e_kin_plus_xc_minus_nl",
    "e_kin_plus_xc_plus_ecp",
    "e_kin_plus_xc_minus_nl_plus_ecp",
    "e_kin_minus_apbe",
    "e_kinapbe",
    "e_xc",
    "e_tot",
)


@dataclass
class SourceFileInfo:
    path: str
    n_scf_steps: int
    coeff_dim: int
    n_basis: int
    n_atom: int
    estimated_bytes: int
    source_relpath: str = ""
    source_key: str = ""


@dataclass
class ConversionConfig:
    energy_key: str
    gradient_key: str
    output_dtype: str
    zstd_level: int
    ground_state_only: bool = False
    include_nuclear_gradients: bool = False


@dataclass
class ShardTask:
    shard_id: int
    source_files: list[SourceFileInfo]
    output_path: str
    config: ConversionConfig
    input_dir: str = ""
    source_key_prefix: str = ""


@dataclass
class ShardResult:
    shard_id: int
    output_file: str
    n_samples: int
    n_molecules: int
    source_files: list[str]
    n_buckets: int
    bucket_rows_min: int
    bucket_rows_max: int
    bucket_rows_lt64: int


def _to_text(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8")
    return str(x)


def _to_bool(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return bool(x.item())
    return bool(x)


def _compression_kwargs(config: ConversionConfig) -> dict[str, Any]:
    if config.zstd_level == 0:
        return {}
    return hdf5plugin.Zstd(clevel=int(config.zstd_level))


@contextmanager
def _open_zarr_read(path: Path):
    if path.is_dir():
        yield zarr.open(path, mode="r")
        return
    with zarr.ZipStore(path, mode="r") as store:
        yield zarr.open(store, mode="r")


def _chunk_1d(n: int, target: int = 131072) -> tuple[int]:
    """Return a 1D chunk shape capped at `target` elements."""
    return (max(1, min(target, n)),)


def _chunk_2d(n0: int, n1: int, target_rows: int = 4096) -> tuple[int, int]:
    """Return a 2D chunk shape with full-width columns and capped row count."""
    return (max(1, min(target_rows, n0)), n1)


def _estimate_file_bytes(
    n_scf_steps: int,
    coeff_dim: int,
    n_basis: int,
    n_atom: int,
    dtype_size: int,
    has_nuclear_gradients: bool = False,
) -> int:
    sample_bytes = n_scf_steps * ((2 * coeff_dim + 1) * dtype_size + 32)
    molecule_bytes = ((3 * n_atom + n_basis + coeff_dim) * dtype_size) + n_atom + 128
    if has_nuclear_gradients:
        molecule_bytes += ((3 * n_atom) + (3 * n_atom * n_basis)) * dtype_size + 64
    return int(sample_bytes + molecule_bytes)


def scan_source_file(
    path: Path,
    output_dtype: str,
    ground_state_only: bool = False,
    include_nuclear_gradients: bool = False,
) -> SourceFileInfo:
    with _open_zarr_read(path) as root:
        coeff_shape = tuple(root["of_labels"]["spatial"]["coeffs"].shape)
        n_scf_steps = int(coeff_shape[0])
        estimated_n_scf_steps = 1 if ground_state_only else n_scf_steps
        n_basis = int(coeff_shape[1])
        n_channels = 1 if len(coeff_shape) == 2 else int(coeff_shape[2])
        coeff_dim = n_basis * n_channels
        n_atom = int(root["geometry"]["atomic_numbers"].shape[0])
        dtype_size = np.dtype(output_dtype).itemsize
        has_nuclear_gradients = (
            include_nuclear_gradients and _get_nuclear_gradients_path(path).exists()
        )
        estimated_bytes = _estimate_file_bytes(
            n_scf_steps=estimated_n_scf_steps,
            coeff_dim=coeff_dim,
            n_basis=n_basis,
            n_atom=n_atom,
            dtype_size=dtype_size,
            has_nuclear_gradients=has_nuclear_gradients,
        )
        return SourceFileInfo(
            path=str(path),
            n_scf_steps=estimated_n_scf_steps,
            coeff_dim=coeff_dim,
            n_basis=n_basis,
            n_atom=n_atom,
            estimated_bytes=estimated_bytes,
        )


def plan_shards(
    infos: list[SourceFileInfo], target_shard_size_bytes: int
) -> list[list[SourceFileInfo]]:
    """Plan split-local shards by grouping similar coeff_dim and capping shard bytes.

    Files are sorted by (coeff_dim, estimated_bytes desc, path) so each shard tends to contain
    similar tensor widths (better bucket fill) while still respecting the target byte size.
    """
    infos = sorted(infos, key=lambda x: (x.coeff_dim, -x.estimated_bytes, x.path))
    shards: list[list[SourceFileInfo]] = []
    current: list[SourceFileInfo] = []
    current_bytes = 0
    for info in infos:
        if current and current_bytes + info.estimated_bytes > target_shard_size_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(info)
        current_bytes += info.estimated_bytes
    if current:
        shards.append(current)
    return shards


def _read_basis_integrals(spatial_group: zarr.Group, dtype: np.dtype) -> np.ndarray:
    if "dual_basis_integrals" in spatial_group:
        return np.asarray(spatial_group["dual_basis_integrals"][0], dtype=dtype)
    if "basis_integrals" in spatial_group:
        return np.asarray(spatial_group["basis_integrals"][0], dtype=dtype)
    raise KeyError("Neither 'basis_integrals' nor 'dual_basis_integrals' found in source file.")


def _resolve_ecp_label_key(group: zarr.Group, key: str, geometry: zarr.Group, path: Path) -> str:
    if key in group:
        return key
    if "_plus_ecp" in key.lower():
        uses_ecp = "uses_ecp" in geometry and bool(geometry["uses_ecp"][()])
        if not uses_ecp:
            fallback_key = key.replace("_plus_ecp", "")
            if fallback_key in group:
                return fallback_key
    raise KeyError(f"Missing key {key!r} in {path}; not trying an ECP fallback.")


def _check_energy_gradient_key_consistency(
    energy_key: str,
    resolved_energy_key: str,
    gradient_key: str,
    resolved_gradient_key: str,
    path: Path,
) -> None:
    energy_uses_ecp_fallback = resolved_energy_key != energy_key
    gradient_uses_ecp_fallback = resolved_gradient_key != gradient_key
    if energy_uses_ecp_fallback != gradient_uses_ecp_fallback:
        raise ValueError(
            f"Inconsistent ECP key fallback in {path}: energy {energy_key!r} resolved to "
            f"{resolved_energy_key!r}, but gradient {gradient_key!r} resolved to "
            f"{resolved_gradient_key!r}."
        )


def _get_nuclear_gradients_path(source_path: Path) -> Path:
    return source_path.parent.parent / "nuclear_gradients" / source_path.name


def _read_nuclear_gradient_arrays(
    source_path: Path, gradient_key: str, out_dtype: np.dtype, label_dir_name: str | None = None
) -> dict[str, np.ndarray] | None:
    grad_path = _get_nuclear_gradients_path(source_path)
    if not grad_path.exists():
        return None

    if label_dir_name is None:
        label_dir_name = source_path.parent.name

    with _open_zarr_read(grad_path) as grad_root:
        nuc_grad_key = f"nuc_{gradient_key}"
        if nuc_grad_key not in grad_root:
            raise KeyError(
                f"Missing nuclear gradient key '{nuc_grad_key}' in {grad_path} for {source_path}."
            )
        transform_name = label_dir_name[len("labels_") :]
        transform_grad_key = f"{transform_name}_grad_contracted"
        if transform_grad_key not in grad_root:
            raise KeyError(
                f"Missing transform gradient key '{transform_grad_key}' in {grad_path} for "
                f"{source_path}."
            )
        nuc_grad_label = np.asarray(grad_root[nuc_grad_key][:], dtype=out_dtype)
        transform_grad_contracted = np.asarray(grad_root[transform_grad_key][:], dtype=out_dtype)

    return {
        "nuc_grad_label": nuc_grad_label,
        "transform_grad_contracted": transform_grad_contracted,
    }


def _read_source_arrays(
    path: Path,
    out_dtype: np.dtype,
    energy_key: str,
    gradient_key: str,
    ground_state_only: bool,
    include_nuclear_gradients: bool = False,
    label_dir_name: str | None = None,
) -> dict[str, Any]:
    with _open_zarr_read(path) as root:
        geometry = root["geometry"]
        of_labels = root["of_labels"]
        spatial = of_labels["spatial"]
        energies = of_labels["energies"]

        coeffs = np.asarray(spatial["coeffs"][:], dtype=out_dtype)
        resolved_gradient_key = _resolve_ecp_label_key(spatial, gradient_key, geometry, path)
        resolved_energy_key = _resolve_ecp_label_key(energies, energy_key, geometry, path)
        _check_energy_gradient_key_consistency(
            energy_key, resolved_energy_key, gradient_key, resolved_gradient_key, path
        )
        gradients = np.asarray(spatial[resolved_gradient_key][:], dtype=out_dtype)
        energy = np.asarray(energies[resolved_energy_key][:], dtype=out_dtype)

        if coeffs.shape[0] != gradients.shape[0] or coeffs.shape[0] != energy.shape[0]:
            raise ValueError(
                f"Inconsistent n_scf in {path}: coeffs={coeffs.shape}, gradients={gradients.shape}, energy={energy.shape}"
            )
        if coeffs.shape[1:] != gradients.shape[1:]:
            raise ValueError(
                f"Coefficient/gradient shape mismatch in {path}: {coeffs.shape} vs {gradients.shape}"
            )

        n_scf = int(coeffs.shape[0])
        n_basis = int(coeffs.shape[1])
        n_channels = 1 if coeffs.ndim == 2 else int(coeffs.shape[2])
        coeff_dim = n_basis * n_channels

        atom_pos = np.asarray(geometry["atom_pos"][:], dtype=out_dtype)
        atomic_numbers = np.asarray(geometry["atomic_numbers"][:], dtype=np.uint8)
        mol_id = _to_text(geometry["mol_id"][()])

        basis_integrals = _read_basis_integrals(spatial, dtype=out_dtype)
        if basis_integrals.shape != (n_basis,):
            raise ValueError(
                f"basis_integrals shape mismatch in {path}: {basis_integrals.shape} vs {(n_basis,)}"
            )

        if "has_energy_label" in root["ks_labels"]["energies"]:
            has_energy = np.asarray(
                root["ks_labels"]["energies"]["has_energy_label"][:], dtype=np.bool_
            )
        else:
            has_energy = np.ones(n_scf, dtype=np.bool_)

        ground_state_energies = {}
        for key in GROUND_STATE_ENERGY_KEYS:
            try:
                resolved_key = _resolve_ecp_label_key(energies, key, geometry, path)
            except KeyError:
                continue
            ground_state_energies[key] = np.asarray(energies[resolved_key][-1], dtype=out_dtype)
        nuclear_gradient_data = None
        if include_nuclear_gradients:
            nuclear_gradient_data = _read_nuclear_gradient_arrays(
                source_path=path,
                gradient_key=gradient_key,
                out_dtype=out_dtype,
                label_dir_name=label_dir_name,
            )

        if ground_state_only:
            coeffs = coeffs[-1:]
            gradients = gradients[-1:]
            energy = energy[-1:]
            has_energy = has_energy[-1:]
            n_scf = 1

        return {
            "n_scf": n_scf,
            "n_basis": n_basis,
            "n_channels": n_channels,
            "coeff_dim": coeff_dim,
            "coeffs": coeffs,
            "gradients": gradients,
            "energy": energy,
            "has_energy": has_energy,
            "atom_pos": atom_pos,
            "atomic_numbers": atomic_numbers,
            "mol_id": mol_id,
            "basis_integrals": basis_integrals,
            "ground_state_coeff": coeffs[-1],
            "ground_state_energies": ground_state_energies,
            "nuclear_gradient_data": nuclear_gradient_data,
        }


def load_split_sources(
    split_file: Path,
    label_subdir: str,
    data_root: Path,
    split_names: tuple[str, ...],
) -> tuple[dict[str, list[str]], dict[str, SourceFileInfo]]:
    with split_file.open("rb") as f:
        split_dict = pickle.load(f)

    split_source_keys = {name: [] for name in split_names}
    source_infos: dict[str, SourceFileInfo] = {}
    missing_paths: list[str] = []
    total_rows = sum(len(split_dict.get(split_name, [])) for split_name in split_names)

    with tqdm(total=total_rows, desc="Resolve split sources", smoothing=0.0) as pbar:
        for split_name in split_names:
            for dataset_name, label_path, _ in split_dict.get(split_name, []):
                label_path = str(label_path)
                source_key = f"{dataset_name}/{label_subdir}/{label_path}"
                split_source_keys[split_name].append(source_key)
                if source_key in source_infos:
                    pbar.update(1)
                    continue
                path = data_root / str(dataset_name) / label_subdir / label_path
                if not path.exists():
                    missing_paths.append(source_key)
                    pbar.update(1)
                    continue
                source_infos[source_key] = SourceFileInfo(
                    path=str(path),
                    n_scf_steps=0,
                    coeff_dim=0,
                    n_basis=0,
                    n_atom=0,
                    estimated_bytes=0,
                    source_relpath=label_path,
                    source_key=source_key,
                )
                pbar.update(1)

    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        raise FileNotFoundError(
            f"{len(missing_paths)} split entry(ies) were not found for label_subdir '{label_subdir}'. "
            f"First missing: {preview}"
        )
    return split_source_keys, source_infos


def convert_shard(task: ShardTask) -> ShardResult:
    out_dtype = np.dtype(task.config.output_dtype)
    comp_kwargs = _compression_kwargs(task.config)
    use_compression = task.config.zstd_level > 0
    if not task.source_files:
        raise ValueError(f"Shard task {task.shard_id} has no source files.")

    def ds_kwargs(chunks: tuple[int, ...] | None = None) -> dict[str, Any]:
        if chunks is None:
            return dict(comp_kwargs)
        if not use_compression:
            # Uncompressed bucket datasets are read as full rows
            # (`coeff_grad[bucket_row, :, :]`). In contiguous layout HDF5 can still
            # read just that row hyperslab, so forcing chunks would only add overhead.
            return {}
        return {"chunks": chunks, **comp_kwargs}

    sample_mol_index: list[int] = []
    sample_scf_iteration: list[int] = []
    sample_energy: list[float] = []
    sample_has_energy_label: list[bool] = []
    sample_bucket_id: list[int] = []
    sample_bucket_row: list[int] = []

    bucket_dims: list[int] = []
    bucket_dim_to_id: dict[int, int] = {}
    bucket_rows_used: list[int] = []
    bucket_coeff_grad_parts: dict[int, list[np.ndarray]] = {}

    mol_ids: list[str] = []
    mol_n_atom: list[int] = []
    mol_n_basis: list[int] = []
    mol_n_channels: list[int] = []

    atom_ptr = [0]
    atom_pos_parts: list[np.ndarray] = []
    atomic_numbers_parts: list[np.ndarray] = []

    basis_ptr = [0]
    basis_integrals_parts: list[np.ndarray] = []

    gs_coeff_ptr = [0]
    gs_coeff_parts: list[np.ndarray] = []
    ground_state_energy_parts = {key: [] for key in GROUND_STATE_ENERGY_KEYS}
    mol_has_nuc_grad_label: list[bool] = []
    nuc_grad_label_ptr = [0]
    nuc_grad_label_parts: list[np.ndarray] = []
    transform_grad_contracted_ptr = [0]
    transform_grad_contracted_cols: list[int] = []
    transform_grad_contracted_parts: list[np.ndarray] = []

    source_relpaths: list[str] = []
    source_keys: list[str] = []

    for source in task.source_files:
        path = Path(source.path)
        source_relpath = source.source_relpath
        if not source_relpath:
            source_relpath = path.relative_to(Path(task.input_dir)).as_posix()
        source_key = source.source_key
        if not source_key:
            source_key = f"{task.source_key_prefix}/{source_relpath}"
        label_dir_name = Path(source_key).parent.name if source_key else path.parent.name
        source_relpaths.append(source_relpath)
        source_keys.append(source_key)

        data = _read_source_arrays(
            path=path,
            out_dtype=out_dtype,
            energy_key=task.config.energy_key,
            gradient_key=task.config.gradient_key,
            ground_state_only=task.config.ground_state_only,
            include_nuclear_gradients=task.config.include_nuclear_gradients,
            label_dir_name=label_dir_name,
        )

        coeff_dim = int(data["coeff_dim"])
        n_scf = int(data["n_scf"])

        if coeff_dim not in bucket_dim_to_id:
            bucket_id = len(bucket_dims)
            bucket_dims.append(coeff_dim)
            bucket_dim_to_id[coeff_dim] = bucket_id
            bucket_rows_used.append(0)
            bucket_coeff_grad_parts[bucket_id] = []
        else:
            bucket_id = bucket_dim_to_id[coeff_dim]

        coeff_2d = data["coeffs"].reshape(n_scf, coeff_dim)
        grad_2d = data["gradients"].reshape(n_scf, coeff_dim)
        coeff_grad_3d = np.stack((coeff_2d, grad_2d), axis=1)

        row_start = bucket_rows_used[bucket_id]
        bucket_rows_used[bucket_id] += n_scf
        bucket_coeff_grad_parts[bucket_id].append(coeff_grad_3d)

        mol_idx = len(mol_ids)
        mol_ids.append(str(data["mol_id"]))
        mol_n_atom.append(int(data["atomic_numbers"].shape[0]))
        mol_n_basis.append(int(data["n_basis"]))
        mol_n_channels.append(int(data["n_channels"]))

        atom_pos = data["atom_pos"]
        atomic_numbers = data["atomic_numbers"]
        atom_pos_parts.append(atom_pos)
        atomic_numbers_parts.append(atomic_numbers)
        atom_ptr.append(atom_ptr[-1] + int(atomic_numbers.shape[0]))

        basis_integrals = data["basis_integrals"]
        basis_integrals_parts.append(basis_integrals)
        basis_ptr.append(basis_ptr[-1] + int(basis_integrals.shape[0]))

        gs_coeff = data["ground_state_coeff"]
        gs_coeff_parts.append(gs_coeff)
        gs_coeff_ptr.append(gs_coeff_ptr[-1] + int(gs_coeff.shape[0]))
        for key, value in data["ground_state_energies"].items():
            ground_state_energy_parts[key].append(value)

        nuclear_gradient_data = data["nuclear_gradient_data"]
        expected_nuc_grad_dim = 3 * int(data["atomic_numbers"].shape[0])
        if nuclear_gradient_data is None:
            mol_has_nuc_grad_label.append(False)
            nuc_grad_label_ptr.append(nuc_grad_label_ptr[-1])
            transform_grad_contracted_ptr.append(transform_grad_contracted_ptr[-1])
            transform_grad_contracted_cols.append(0)
        else:
            nuc_grad_label = nuclear_gradient_data["nuc_grad_label"]
            transform_grad_contracted = nuclear_gradient_data["transform_grad_contracted"]
            if nuc_grad_label.shape != (expected_nuc_grad_dim,):
                raise ValueError(
                    f"nuc_grad_label shape mismatch in {path}: "
                    f"{nuc_grad_label.shape} vs {(expected_nuc_grad_dim,)}"
                )
            if transform_grad_contracted.ndim != 2:
                raise ValueError(
                    f"transform_grad_contracted must be 2D in {path}, got "
                    f"{transform_grad_contracted.shape}."
                )
            if transform_grad_contracted.shape[0] != expected_nuc_grad_dim:
                raise ValueError(
                    f"transform_grad_contracted row mismatch in {path}: "
                    f"{transform_grad_contracted.shape[0]} vs {expected_nuc_grad_dim}"
                )
            mol_has_nuc_grad_label.append(True)
            nuc_grad_label_parts.append(nuc_grad_label)
            nuc_grad_label_ptr.append(nuc_grad_label_ptr[-1] + int(nuc_grad_label.size))
            transform_grad_contracted_parts.append(transform_grad_contracted.reshape(-1))
            transform_grad_contracted_ptr.append(
                transform_grad_contracted_ptr[-1] + int(transform_grad_contracted.size)
            )
            transform_grad_contracted_cols.append(int(transform_grad_contracted.shape[1]))

        sample_mol_index.extend([mol_idx] * n_scf)
        sample_scf_iteration.extend(range(n_scf))
        sample_energy.extend(data["energy"].tolist())
        sample_has_energy_label.extend(data["has_energy"].tolist())
        sample_bucket_id.extend([bucket_id] * n_scf)
        sample_bucket_row.extend(range(row_start, row_start + n_scf))

    atom_pos_flat = np.concatenate(atom_pos_parts, axis=0)
    atomic_numbers_flat = np.concatenate(atomic_numbers_parts, axis=0)
    basis_integrals_flat = np.concatenate(basis_integrals_parts, axis=0)
    gs_coeff_flat = np.concatenate(gs_coeff_parts, axis=0)

    if len(source_keys) != len(set(source_keys)):
        raise ValueError(f"Duplicate source_keys in shard {task.output_path}.")

    sample_mol_index_arr = np.array(sample_mol_index, dtype=np.int64)
    sample_scf_iteration_arr = np.array(sample_scf_iteration, dtype=np.int32)
    sample_energy_arr = np.array(sample_energy, dtype=out_dtype)
    sample_has_energy_label_arr = np.array(sample_has_energy_label, dtype=np.uint8)
    sample_bucket_id_arr = np.array(sample_bucket_id, dtype=np.int32)
    sample_bucket_row_arr = np.array(sample_bucket_row, dtype=np.int64)
    bucket_dims_arr = np.array(bucket_dims, dtype=np.int32)
    mol_n_atom_arr = np.array(mol_n_atom, dtype=np.int32)
    mol_n_basis_arr = np.array(mol_n_basis, dtype=np.int32)
    mol_n_channels_arr = np.array(mol_n_channels, dtype=np.int8)
    atom_ptr_arr = np.array(atom_ptr, dtype=np.int64)
    basis_ptr_arr = np.array(basis_ptr, dtype=np.int64)
    gs_coeff_ptr_arr = np.array(gs_coeff_ptr, dtype=np.int64)
    mol_has_nuc_grad_label_arr = np.array(mol_has_nuc_grad_label, dtype=np.uint8)
    nuc_grad_label_ptr_arr = np.array(nuc_grad_label_ptr, dtype=np.int64)
    nuc_grad_label_flat = (
        np.concatenate(nuc_grad_label_parts, axis=0)
        if nuc_grad_label_parts
        else np.empty((0,), dtype=out_dtype)
    )
    transform_grad_contracted_ptr_arr = np.array(transform_grad_contracted_ptr, dtype=np.int64)
    transform_grad_contracted_cols_arr = np.array(transform_grad_contracted_cols, dtype=np.int32)
    transform_grad_contracted_flat = (
        np.concatenate(transform_grad_contracted_parts, axis=0)
        if transform_grad_contracted_parts
        else np.empty((0,), dtype=out_dtype)
    )

    output_path = Path(task.output_path)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(tmp_path, "w") as h5f:
        h5f.attrs["format"] = FORMAT_NAME
        h5f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        h5f.attrs["energy_key"] = task.config.energy_key
        h5f.attrs["gradient_key"] = task.config.gradient_key
        h5f.attrs["output_dtype"] = task.config.output_dtype
        h5f.attrs["bucket_chunk_rows"] = 1
        h5f.attrs["sample_bucket_layout"] = "coeff_grad_colocated_v1"
        h5f.attrs["ground_state_only"] = int(task.config.ground_state_only)

        dt_utf8 = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset("source_relpaths", data=source_relpaths, dtype=dt_utf8)
        h5f.create_dataset("source_keys", data=source_keys, dtype=dt_utf8)

        n_samples = len(sample_mol_index)
        g_samples = h5f.create_group("samples")
        g_samples.create_dataset(
            "mol_index",
            data=sample_mol_index_arr,
            **ds_kwargs(_chunk_1d(n_samples)),
        )
        g_samples.create_dataset(
            "scf_iteration",
            data=sample_scf_iteration_arr,
            **ds_kwargs(_chunk_1d(n_samples)),
        )
        g_samples.create_dataset(
            "energy",
            data=sample_energy_arr,
            **ds_kwargs(_chunk_1d(n_samples)),
        )
        g_samples.create_dataset(
            "has_energy_label",
            data=sample_has_energy_label_arr,
            **ds_kwargs(_chunk_1d(n_samples)),
        )
        g_samples.create_dataset(
            "bucket_id",
            data=sample_bucket_id_arr,
            **ds_kwargs(_chunk_1d(n_samples)),
        )
        g_samples.create_dataset(
            "bucket_row",
            data=sample_bucket_row_arr,
            **ds_kwargs(_chunk_1d(n_samples)),
        )

        g_buckets = h5f.create_group("sample_buckets")
        g_buckets.create_dataset(
            "coeff_dim",
            data=bucket_dims_arr,
            **ds_kwargs(_chunk_1d(len(bucket_dims))),
        )
        g_bucket_data = g_buckets.create_group("data")
        for bucket_id, coeff_dim in enumerate(bucket_dims):
            coeff_grad_mat = np.concatenate(bucket_coeff_grad_parts[bucket_id], axis=0)
            g_bucket = g_bucket_data.create_group(str(bucket_id))
            g_bucket.create_dataset(
                "coeff_grad",
                data=coeff_grad_mat,
                **ds_kwargs((1, 2, coeff_dim)),
            )

        n_mol = len(mol_ids)
        g_mol = h5f.create_group("molecules")
        g_mol.create_dataset("mol_id", data=mol_ids, dtype=dt_utf8)
        g_mol.create_dataset("source_key", data=source_keys, dtype=dt_utf8)
        g_mol.create_dataset("source_relpath", data=source_relpaths, dtype=dt_utf8)
        g_mol.create_dataset("n_atom", data=mol_n_atom_arr, **ds_kwargs())
        g_mol.create_dataset("n_basis", data=mol_n_basis_arr, **ds_kwargs())
        g_mol.create_dataset("n_channels", data=mol_n_channels_arr, **ds_kwargs())
        g_mol.create_dataset("atom_ptr", data=atom_ptr_arr, **ds_kwargs())
        g_mol.create_dataset(
            "atom_pos_flat",
            data=atom_pos_flat,
            **ds_kwargs(_chunk_2d(atom_pos_flat.shape[0], 3)),
        )
        g_mol.create_dataset(
            "atomic_numbers_flat",
            data=atomic_numbers_flat,
            **ds_kwargs(_chunk_1d(len(atomic_numbers_flat))),
        )
        g_mol.create_dataset("basis_ptr", data=basis_ptr_arr, **ds_kwargs())
        g_mol.create_dataset(
            "basis_integrals_flat",
            data=basis_integrals_flat,
            **ds_kwargs(_chunk_1d(len(basis_integrals_flat))),
        )
        g_mol.create_dataset("ground_state_coeff_ptr", data=gs_coeff_ptr_arr, **ds_kwargs())
        g_mol.create_dataset(
            "ground_state_coeff_flat",
            data=gs_coeff_flat,
            **ds_kwargs(_chunk_1d(len(gs_coeff_flat))),
        )
        if any(values for values in ground_state_energy_parts.values()):
            g_energy = g_mol.create_group("ground_state_energies")
            for key, values in ground_state_energy_parts.items():
                if values:
                    g_energy.create_dataset(
                        key, data=np.asarray(values, dtype=out_dtype), **ds_kwargs()
                    )
        if np.any(mol_has_nuc_grad_label_arr):
            g_nuc = g_mol.create_group("nuclear_gradients")
            g_nuc.create_dataset("available", data=mol_has_nuc_grad_label_arr, **ds_kwargs())
            g_nuc.create_dataset("nuc_grad_label_ptr", data=nuc_grad_label_ptr_arr, **ds_kwargs())
            g_nuc.create_dataset(
                "nuc_grad_label_flat",
                data=nuc_grad_label_flat,
                **ds_kwargs(_chunk_1d(len(nuc_grad_label_flat))),
            )
            g_nuc.create_dataset(
                "transform_grad_contracted_ptr",
                data=transform_grad_contracted_ptr_arr,
                **ds_kwargs(),
            )
            g_nuc.create_dataset(
                "transform_grad_contracted_cols",
                data=transform_grad_contracted_cols_arr,
                **ds_kwargs(),
            )
            g_nuc.create_dataset(
                "transform_grad_contracted_flat",
                data=transform_grad_contracted_flat,
                **ds_kwargs(_chunk_1d(len(transform_grad_contracted_flat))),
            )

        h5f.attrs["n_samples"] = int(n_samples)
        h5f.attrs["n_molecules"] = int(n_mol)
        h5f.attrs["n_source_files"] = int(len(source_keys))
        h5f.attrs["n_buckets"] = int(len(bucket_dims))

    bucket_rows = np.asarray(bucket_rows_used, dtype=np.int64)
    if len(bucket_rows) == 0:
        bucket_rows_min = 0
        bucket_rows_max = 0
        bucket_rows_lt64 = 0
    else:
        bucket_rows_min = int(bucket_rows.min())
        bucket_rows_max = int(bucket_rows.max())
        bucket_rows_lt64 = int(np.sum(bucket_rows < 64))

    tmp_path.replace(output_path)
    return ShardResult(
        shard_id=task.shard_id,
        output_file=output_path.name,
        n_samples=len(sample_mol_index),
        n_molecules=len(mol_ids),
        source_files=[s.path for s in task.source_files],
        n_buckets=len(bucket_dims),
        bucket_rows_min=bucket_rows_min,
        bucket_rows_max=bucket_rows_max,
        bucket_rows_lt64=bucket_rows_lt64,
    )


def build_manifest(
    output_dir: Path,
    split_file: Path,
    label_subdir: str,
    config: ConversionConfig,
    results: list[ShardResult],
) -> dict[str, Any]:
    data_root = get_mldft_data_path().resolve()
    sample_offsets = [0]
    for result in results:
        sample_offsets.append(sample_offsets[-1] + result.n_samples)

    unique_sources: set[str] = set()
    for result in results:
        unique_sources.update(result.source_files)
    shards = []
    for result in results:
        shard = asdict(result)
        shard["output_file"] = (
            (output_dir / result.output_file).resolve().relative_to(data_root).as_posix()
        )
        shards.append(shard)

    manifest: dict[str, Any] = {
        "format": FORMAT_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "split_file": str(split_file.resolve()),
        "split_dataset_name": split_file.parent.name,
        "label_subdir": label_subdir,
        "n_source_files": len(unique_sources),
        "n_samples_total": int(sum(r.n_samples for r in results)),
        "n_molecules_total": int(sum(r.n_molecules for r in results)),
        "n_shards": len(results),
        "config": asdict(config),
        "paths_relative_to": "DFT_DATA",
        "sample_offsets": sample_offsets,
        "shards": shards,
    }
    return manifest


def _print_bucket_stats(results: list[ShardResult], tag: str) -> None:
    if not results:
        return
    total_samples = int(sum(r.n_samples for r in results))
    total_buckets = int(sum(r.n_buckets for r in results))
    if total_buckets == 0:
        print(f"{tag} bucket stats: buckets=0")
        return
    min_rows = int(min(r.bucket_rows_min for r in results if r.n_buckets > 0))
    max_rows = int(max(r.bucket_rows_max for r in results if r.n_buckets > 0))
    small_buckets = int(sum(r.bucket_rows_lt64 for r in results))
    mean_rows = total_samples / total_buckets
    print(
        f"{tag} bucket stats: buckets={total_buckets}, rows_per_bucket_mean={mean_rows:.1f}, "
        f"min={min_rows}, max={max_rows}, lt64={small_buckets} ({small_buckets / total_buckets:.2%})"
    )


def _auto_target_shard_size_bytes(
    split_name: str, split_bytes: int, ground_state_only: bool
) -> int:
    gb = 1024**3
    if ground_state_only:
        min_gib, max_gib = 0.5, 3.0
    else:
        min_gib, max_gib = 1.5, 8.0
    target_shards = 48 if split_name == "train" else 8
    if split_bytes <= 0:
        return int(min_gib * gb)
    target_gib = split_bytes / (target_shards * gb)
    target_gib = max(min_gib, min(max_gib, target_gib))
    return int(target_gib * gb)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert split.pkl entries into split-aware HDF5 shards."
    )
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--label-subdir", type=str, default="labels")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <split-file.parent>/<label-subdir>_h5_<target-key>[_nuc_grad].",
    )
    parser.add_argument(
        "--target-key",
        type=str,
        required=True,
        help="Uses energy key e_<target-key> and gradient key grad_<target-key>.",
    )
    parser.add_argument(
        "--target-shard-size-gb",
        type=float,
        default=0.0,
        help="Per-split shard size in GiB. Use 0 for auto sizing.",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=0,
        help="Zstd compression level. Use 0 for no compression.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--scan-workers", type=int, default=8)
    parser.add_argument("--ignore-test-split", action="store_true")
    parser.add_argument("--ground-state-only", action="store_true")
    parser.add_argument("--include-nuclear-gradients", action="store_true")
    parser.add_argument(
        "--output-dtype", type=str, default="float64", choices=("float64", "float32")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    has_existing = any(output_dir.iterdir())
    if has_existing and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Use --overwrite.")
    if has_existing and overwrite:
        for old_file in output_dir.glob("*shard_*.h5"):
            old_file.unlink()
        for old_file in output_dir.glob("*shard_*.h5.tmp"):
            old_file.unlink()
        for old_file in output_dir.glob("manifest*.json"):
            old_file.unlink()
        for old_file in output_dir.glob("sample_offsets*.npy"):
            old_file.unlink()


def _require_inside_data_root(path: Path, *, path_name: str, data_root: Path) -> None:
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(
            f"{path_name} must be inside DFT_DATA ({data_root}), got: {resolved_path}"
        ) from exc


def main() -> None:
    args = build_parser().parse_args()

    data_root = get_mldft_data_path().resolve()
    split_file = args.split_file
    label_subdir = args.label_subdir
    split_names = SPLIT_NAMES[:-1] if args.ignore_test_split else SPLIT_NAMES
    target_key = args.target_key
    energy_key = f"e_{target_key}"
    gradient_key = f"grad_{target_key}"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    _require_inside_data_root(split_file, path_name="--split-file", data_root=data_root)
    dataset_name = split_file.parent.name

    output_dir = args.output_dir
    if output_dir is None:
        suffix = "_denop_h5_" if args.ground_state_only else "_h5_"
        output_name = f"{label_subdir}{suffix}{target_key}"
        if args.include_nuclear_gradients:
            output_name += "_nuc_grad"
        output_dir = split_file.parent / output_name
        print(f"--output-dir not provided. Using: {output_dir}")
    _require_inside_data_root(output_dir, path_name="--output-dir", data_root=data_root)
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    if args.target_shard_size_gb < 0:
        raise ValueError(f"target shard size must be >= 0, got {args.target_shard_size_gb}")
    target_shard_size_bytes = (
        int(float(args.target_shard_size_gb) * (1024**3)) if args.target_shard_size_gb > 0 else 0
    )
    if args.zstd_level < 0:
        raise ValueError(f"zstd level must be >= 0, got {args.zstd_level}")

    print(f"Resolving split sources from {split_file} using label_subdir '{label_subdir}' ...")
    split_source_keys, source_infos = load_split_sources(
        split_file=split_file,
        label_subdir=label_subdir,
        data_root=data_root,
        split_names=split_names,
    )
    n_split_entries = int(sum(len(keys) for keys in split_source_keys.values()))
    if n_split_entries == 0:
        raise ValueError(f"No split entries found in {split_file}.")

    source_keys = sorted(source_infos)
    source_paths = [Path(source_infos[key].path) for key in source_keys]
    print(
        f"Selected {n_split_entries} split entries with {len(source_paths)} unique source files "
        f"from label_subdir '{label_subdir}'."
    )
    n_nuclear_gradient_files = sum(
        _get_nuclear_gradients_path(path).exists() for path in source_paths
    )
    if n_nuclear_gradient_files > 0 and args.include_nuclear_gradients:
        print(
            f"Found {n_nuclear_gradient_files} nuclear gradient files; they will be loaded into the HDF5 shards when available."
        )
    elif n_nuclear_gradient_files > 0:
        print(
            f"Found {n_nuclear_gradient_files} nuclear gradient files, but they will be ignored because include_nuclear_gradients is disabled."
        )
    elif args.include_nuclear_gradients:
        raise ValueError(
            "include_nuclear_gradients is enabled but no nuclear gradient files were found "
            f"for label_subdir '{label_subdir}'."
        )
    print(f"Scanning {len(source_paths)} source files ...")
    scan_fn = partial(
        scan_source_file,
        output_dtype=args.output_dtype,
        ground_state_only=bool(args.ground_state_only),
        include_nuclear_gradients=bool(args.include_nuclear_gradients),
    )
    if args.scan_workers > 1:
        with mp.Pool(int(args.scan_workers)) as pool:
            scanned_infos = list(
                tqdm(
                    pool.imap(scan_fn, source_paths),
                    total=len(source_paths),
                    desc="Scan source",
                    smoothing=0.0,
                )
            )
    else:
        scanned_infos = [
            scan_fn(path) for path in tqdm(source_paths, desc="Scan source", smoothing=0.0)
        ]

    for source_key, scanned_info in zip(source_keys, scanned_infos):
        base_info = source_infos[source_key]
        source_infos[source_key] = SourceFileInfo(
            path=scanned_info.path,
            n_scf_steps=scanned_info.n_scf_steps,
            coeff_dim=scanned_info.coeff_dim,
            n_basis=scanned_info.n_basis,
            n_atom=scanned_info.n_atom,
            estimated_bytes=scanned_info.estimated_bytes,
            source_relpath=base_info.source_relpath,
            source_key=source_key,
        )
    split_infos = {
        split_name: [source_infos[source_key] for source_key in split_source_keys[split_name]]
        for split_name in split_names
    }

    print(
        f"Using split-aware sharding from {split_file} for dataset '{dataset_name}'. "
        + ", ".join(
            f"{split_name}={len(split_source_keys[split_name])}" for split_name in split_names
        )
        + "."
    )

    config = ConversionConfig(
        energy_key=energy_key,
        gradient_key=gradient_key,
        output_dtype=args.output_dtype,
        zstd_level=int(args.zstd_level),
        ground_state_only=bool(args.ground_state_only),
        include_nuclear_gradients=bool(args.include_nuclear_gradients),
    )

    tasks: list[ShardTask] = []
    shard_id = 0
    for split_name in split_names:
        split_source_infos = split_infos[split_name]
        split_samples = int(sum(info.n_scf_steps for info in split_source_infos))
        split_bytes = int(sum(info.estimated_bytes for info in split_source_infos))
        split_target_shard_size_bytes = (
            target_shard_size_bytes
            if target_shard_size_bytes > 0
            else _auto_target_shard_size_bytes(
                split_name=split_name,
                split_bytes=split_bytes,
                ground_state_only=bool(args.ground_state_only),
            )
        )
        split_shards = plan_shards(split_source_infos, split_target_shard_size_bytes)
        print(
            f"{split_name}: planned {len(split_shards)} shard(s) for "
            f"{len(split_source_infos)} files, {split_samples} samples, "
            f"~{split_bytes / (1024**3):.2f} GiB estimated, "
            f"target={split_target_shard_size_bytes / (1024**3):.2f} GiB."
        )
        for split_shard_id, shard_infos in enumerate(split_shards):
            tasks.append(
                ShardTask(
                    shard_id=shard_id,
                    source_files=shard_infos,
                    output_path=str(output_dir / f"{split_name}_shard_{split_shard_id:05d}.h5"),
                    config=config,
                )
            )
            shard_id += 1

    if not tasks:
        raise RuntimeError("No shard tasks were created.")

    n_workers = max(1, int(args.num_workers))
    if n_workers > 1 and len(tasks) > 1:
        with mp.Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(convert_shard, tasks),
                    total=len(tasks),
                    desc="Convert shards",
                    smoothing=0.0,
                )
            )
    else:
        results = [
            convert_shard(task) for task in tqdm(tasks, desc="Convert shards", smoothing=0.0)
        ]

    results = sorted(results, key=lambda result: result.shard_id)
    _print_bucket_stats(results, "overall")
    for split_name in split_names:
        split_results = [r for r in results if r.output_file.startswith(f"{split_name}_")]
        _print_bucket_stats(split_results, split_name)
    manifest = build_manifest(
        output_dir=output_dir,
        split_file=split_file,
        label_subdir=label_subdir,
        config=config,
        results=results,
    )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Wrote {len(results)} shard(s) to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Total samples: {manifest['n_samples_total']}, "
        f"total molecules: {manifest['n_molecules_total']}"
    )


if __name__ == "__main__":
    main()
