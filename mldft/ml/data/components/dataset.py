"""Dataset class for machine learning."""

import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import zarr
from omegaconf import ListConfig
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.of_data import (
    OFData,
    Representation,
    is_nuclear_gradient_outlier,
)
from mldft.utils import RankedLogger
from mldft.utils.environ import get_mldft_data_path

log = RankedLogger(__name__, rank_zero_only=True)


class OFDataset(Dataset):
    """Dataset of OF-DFT data.

    Each scf iteration of each molecule is a sample.
    """

    data_class = OFData

    def __init__(
        self,
        paths: List[Path],
        basis_info: BasisInfo,
        transforms: MasterTransformation = None,
        limit_scf_iterations: int | list[int] | None = None,
        keep_initial_guess: bool = True,
        num_scf_iterations_per_path: List[int] | None = None,
        cache_in_memory: bool = False,
        **of_data_kwargs,
    ) -> None:
        """
        Args:
            paths: List of paths to .zarr files.
            basis_info: Basis information for all samples.
            transforms: List of transforms to apply to each sample.
            limit_scf_iterations: Which scf iterations to use. When passing an int s>0 we use all scf iterations larger
                or equal to s. If s<0 we use the s last iterations and when passing a list of ints, we use only the scf
                iterations in the list.
            keep_initial_guess: Whether to keep the initial guess in the dataset when filtering
                with limit_scf_iterations. This is useful if you want to train the energy model on the later scf
                iterations but also want to train the initial guess model on the minao label. Setting this to False
                will not remove the initial guess, limit_scf_iterations=1 and keep_initial_guess=False will remove them.
            num_scf_iterations_per_path: List of number of scf iterations per path. If None, the number of scf
                iterations is read from the paths directly which can take some time.
            of_data_kwargs: Keyword arguments passed to :meth:`OFData.from_file`.
            cache_in_memory: Whether to cache the dataset in memory. This is useful if the dataset
                fits into memory and the dataset is used multiple times, especially if expensive transforms
                are used. Warning: Do not enable if non-deterministic transforms are used.
        """
        super().__init__()
        # For now, always use all scf iterations.
        # Later, we can add a parameter to use only a subset, e.g. some slice
        assert isinstance(
            basis_info, BasisInfo
        ), f"basis_info must be of type BasisInfo, but is {type(basis_info)}"
        assert len(paths) > 0, "Paths must not be empty"
        self.paths = paths
        self.basis_info = basis_info
        self.of_data_kwargs = of_data_kwargs
        self.transforms = transforms

        # Find out how many samples (scf iterations) there are in total, and construct a list of indices that maps a
        # path index to the first sample index belonging to that path.
        if num_scf_iterations_per_path is None:
            num_scf_iterations_per_path = []
            for path in tqdm(
                self.paths, desc="Creating Dataset, counting scf iterations per path"
            ):
                # This works for zip or file system storage types
                with zarr.open(path, mode="r") as root:
                    if "n_scf_steps" in root["of_labels"]:
                        num_scf_iterations_per_path.append(root["of_labels/n_scf_steps"][()])
                    else:
                        raise KeyError(f" Key n_scf_steps not found in {path}")
        else:
            assert len(num_scf_iterations_per_path) == len(
                paths
            ), "Length of num_scf_iterations_per_path must match paths."
        self.limit_scf_iterations = limit_scf_iterations
        self.scf_iterations_per_path = self.configure_scf_iterations(
            num_scf_iterations_per_path, keep_initial_guess
        )
        num_scf_iterations_per_path = [len(scfs) for scfs in self.scf_iterations_per_path]
        self.path_indices = np.cumsum([0] + num_scf_iterations_per_path)
        self.n_samples = int(self.path_indices[-1])

        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            self.getitem = lru_cache(maxsize=None)(self.getitem)

    def configure_scf_iterations(
        self, num_scf_iterations_per_path: list[int], keep_initial_guess: bool = True
    ) -> list[np.ndarray]:
        """Determine for each path how many scf iterations are there, depending on scf_iterations.

        If scf_iterations>0 we use all scf iterations larger or equal to s, this could delete some geometries but only for large numbers.
        For larger numbers rather use the negative integer functionality. If scf_iterations<0 we use the s last iterations.
        When passing a list of ints, we use only the scf iterations in the list. This could also delete some geometries.

        Args:
            num_scf_iterations_per_path: List of number of scf iterations per path in the whole non-filtered dataset.
            keep_initial_guess: Whether to keep the initial guess in the dataset when filtering. Setting this to False
                will not remove the initial guess, limit_scf_iterations=1 and keep_initial_guess=False will remove them.

        Returns:
            scf_iterations_per_path: List of arrays containing the indices of the scf iterations to use for each path.
        """
        if self.limit_scf_iterations is None:
            if not keep_initial_guess:
                log.warning(
                    "keep_initial_guess is False but limit_scf_iterations is None, every label will be kept."
                )
            return [np.arange(scf_iterations) for scf_iterations in num_scf_iterations_per_path]
        scf_iterations_per_path = []
        if isinstance(self.limit_scf_iterations, int):
            # positive integer case
            if self.limit_scf_iterations >= 0:
                i = 0
                while i < len(self.paths):
                    # If keep initial guess we append an empty array and add 0 later
                    if (
                        num_scf_iterations_per_path[i] < self.limit_scf_iterations
                        and not keep_initial_guess
                    ):
                        num_scf_iterations_per_path.pop(i)
                        log.warning(
                            f"Path {self.paths[i]} has less than {self.limit_scf_iterations} scf iterations. Removing it from dataset."
                        )
                        self.paths.pop(i)
                    else:
                        scf_iterations_per_path.append(
                            np.arange(self.limit_scf_iterations, num_scf_iterations_per_path[i])
                        )
                        i += 1
            else:
                # negative integer case
                for num_scf_iterations in num_scf_iterations_per_path:
                    scf_iterations_per_path.append(
                        np.arange(
                            max(0, num_scf_iterations + self.limit_scf_iterations),
                            num_scf_iterations,
                        )
                    )
        elif isinstance(self.limit_scf_iterations, (list, ListConfig)):
            # list of integers case
            scf_set = set(self.limit_scf_iterations)
            i = 0
            while i < len(self.paths):
                existing_scf_iterations = set(range(num_scf_iterations_per_path[i]))
                negative_existing_scf_iterations = set(
                    range(-1, -num_scf_iterations_per_path[i] - 1, -1)
                )
                existing_scf_iterations.update(negative_existing_scf_iterations)
                used_scf_iterations = existing_scf_iterations.intersection(scf_set)
                # If keep initial guess we append an empty array and add 0 later
                if len(used_scf_iterations) == 0 and not keep_initial_guess:
                    num_scf_iterations_per_path.pop(i)
                    log.warning(
                        f"Path {self.paths[i]} has no scf iterations in {self.limit_scf_iterations}. Removing it from dataset."
                    )
                    self.paths.pop(i)
                else:
                    scf_iterations_per_path.append(np.fromiter(used_scf_iterations, int))
                    i += 1
        else:
            raise ValueError("scf_iterations must be int, list or None.")
        if keep_initial_guess:
            # add zero, save way to do it correctly
            for i in range(len(scf_iterations_per_path)):
                if 0 not in scf_iterations_per_path[i]:
                    scf_iterations_per_path[i] = np.insert(scf_iterations_per_path[i], 0, 0)
        return scf_iterations_per_path

    @classmethod
    def from_directory(cls, directory: str | Path, **kwargs) -> "OFDataset":
        """Initialize the dataset from a directory containing .zarr files.

        Args:
            directory: Path to the directory.
            **kwargs: Keyword arguments passed to the constructor.
        """
        directory = Path(directory)
        paths = sorted(directory.glob("*.zarr*"))
        if len(paths) == 0:
            raise FileNotFoundError(f"No .zarr files found in {directory}")
        return cls(paths=paths, **kwargs)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, item: int) -> OFData:
        """Returns the sample at the given index.

        Args:
            item: Index of the sample.
        """
        return self.getitem(item)

    def getitem(self, item: int) -> OFData:
        """Returns the sample at the given index.

        Note: This method is cached if cache_in_memory is True. As :meth:`__getitem__` cannot be overwritten at
        runtime, this method is used instead.

        Args:
            item: Index of the sample.

        Returns:
            sample: :class:`~mldft.ml.data.components.of_data.OFData` object.
        """
        geometry_idx = np.searchsorted(self.path_indices, item, side="right") - 1
        path = self.paths[geometry_idx]
        # numpy ints somehow don't batch correctly so we use python ints
        scf_iteration = int(
            self.scf_iterations_per_path[geometry_idx][item - self.path_indices[geometry_idx]]
        )

        sample = self.data_class.from_file(
            path=path,
            scf_iteration=scf_iteration,
            basis_info=self.basis_info,
            **self.of_data_kwargs,
        )

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


@dataclass(frozen=True)
class SourceSampleLocation:
    """Map a source file to its shard-local sample range."""

    shard_id: int
    mol_index: int
    sample_start: int
    n_scf_steps: int


class HDF5ShardReader:
    """Helper to keep per-shard HDF5 handles and datasets."""

    def __init__(self, path: Path):
        self.file = h5py.File(path, mode="r")
        self.samples = self.file["samples"]
        self.molecules = self.file["molecules"]
        self.source_keys = self.file["source_keys"]

        self.sample_mol_index = self.samples["mol_index"]
        self.sample_scf_iteration = self.samples["scf_iteration"]
        self.sample_energy = self.samples["energy"]
        self.sample_has_energy_label = self.samples["has_energy_label"]
        self.sample_bucket_id = self.samples["bucket_id"]
        self.sample_bucket_row = self.samples["bucket_row"]
        self.bucket_coeff_dim = self.file["sample_buckets"]["coeff_dim"]
        self.bucket_groups = self.file["sample_buckets"]["data"]
        self.bucket_coeff_grad = {
            int(k): self.bucket_groups[k]["coeff_grad"] for k in self.bucket_groups.keys()
        }

        self.mol_id = self.molecules["mol_id"]
        self.mol_n_basis = self.molecules["n_basis"]
        self.mol_n_channels = self.molecules["n_channels"]
        self.mol_atom_ptr = self.molecules["atom_ptr"]
        self.mol_atom_pos_flat = self.molecules["atom_pos_flat"]
        self.mol_atomic_numbers_flat = self.molecules["atomic_numbers_flat"]
        self.mol_basis_ptr = self.molecules["basis_ptr"]
        self.mol_basis_integrals_flat = self.molecules["basis_integrals_flat"]
        self.mol_gs_coeff_ptr = self.molecules["ground_state_coeff_ptr"]
        self.mol_gs_coeff_flat = self.molecules["ground_state_coeff_flat"]
        self.mol_ground_state_energies = self.molecules.get("ground_state_energies")
        self.mol_nuclear_gradients = self.molecules.get("nuclear_gradients")
        if self.mol_nuclear_gradients is None:
            self.mol_nuc_grad_available = None
            self.mol_nuc_grad_ptr = None
            self.mol_nuc_grad_flat = None
            self.mol_transform_grad_ptr = None
            self.mol_transform_grad_cols = None
            self.mol_transform_grad_flat = None
        else:
            self.mol_nuc_grad_available = self.mol_nuclear_gradients["available"]
            self.mol_nuc_grad_ptr = self.mol_nuclear_gradients["nuc_grad_label_ptr"]
            self.mol_nuc_grad_flat = self.mol_nuclear_gradients["nuc_grad_label_flat"]
            self.mol_transform_grad_ptr = self.mol_nuclear_gradients[
                "transform_grad_contracted_ptr"
            ]
            self.mol_transform_grad_cols = self.mol_nuclear_gradients[
                "transform_grad_contracted_cols"
            ]
            self.mol_transform_grad_flat = self.mol_nuclear_gradients[
                "transform_grad_contracted_flat"
            ]

    def close(self) -> None:
        self.file.close()


class HDF5ShardOFDataset(Dataset):
    """Dataset backed by converted HDF5 shards and a manifest.json file."""

    data_class = OFData

    def __init__(
        self,
        source_keys: List[str],
        basis_info: BasisInfo,
        hdf5_manifest_path: str | Path,
        transforms: MasterTransformation = None,
        limit_scf_iterations: int | list[int] | None = None,
        keep_initial_guess: bool = True,
        num_scf_iterations_per_path: List[int] | None = None,
        cache_in_memory: bool = False,
        hdf5_max_open_shards: int | None = None,
        label_path: bool = True,
        kohn_sham_path: bool = True,
        ground_state_coeffs: bool = True,
        **of_data_kwargs,
    ) -> None:
        """Initialize an OF dataset from sharded HDF5 data."""
        super().__init__()
        assert isinstance(
            basis_info, BasisInfo
        ), f"basis_info must be of type BasisInfo, but is {type(basis_info)}"
        assert len(source_keys) > 0, "source_keys must not be empty"

        self.source_keys = source_keys
        # Reuse OFDataset.configure_scf_iterations, which mutates self.paths for filtered entries.
        self.paths = [Path(k) for k in self.source_keys]
        self.basis_info = basis_info
        self.transforms = transforms
        self.limit_scf_iterations = limit_scf_iterations
        self.hdf5_max_open_shards = hdf5_max_open_shards
        self.load_label_path = bool(label_path)
        self.load_kohn_sham_path = bool(kohn_sham_path)
        self.load_ground_state_coeffs = bool(ground_state_coeffs)

        self.add_irreps = bool(of_data_kwargs.pop("add_irreps", False))
        of_data_kwargs.pop("energy_key", None)
        of_data_kwargs.pop("gradient_key", None)
        self.load_nuclear_gradients = bool(of_data_kwargs.pop("load_nuclear_gradients", False))
        self.nuclear_gradient_outlier_threshold = of_data_kwargs.pop(
            "nuclear_gradient_outlier_threshold", None
        )
        additional_scf = of_data_kwargs.pop("additional_keys_at_scf_iteration", None)
        additional_ground = of_data_kwargs.pop("additional_keys_at_ground_state", None)
        additional_geom = of_data_kwargs.pop("additional_keys_per_geometry", None)

        if additional_scf is not None or additional_geom is not None:
            raise NotImplementedError(
                "additional_keys_at_scf_iteration and additional_keys_per_geometry are not supported for HDF5 shard loading."
            )
        self.additional_ground = additional_ground or {}
        if of_data_kwargs:
            raise TypeError(
                f"Unsupported dataset kwargs for HDF5 shard loading: {list(of_data_kwargs.keys())}"
            )

        self.manifest_path = Path(hdf5_manifest_path)
        with self.manifest_path.open("r") as f:
            self.manifest = json.load(f)
        self.ground_state_only = bool(
            self.manifest.get("config", {}).get("ground_state_only", False)
        )

        base_dir = get_mldft_data_path()
        self.shard_paths = []
        for shard in self.manifest["shards"]:
            shard_path = Path(shard["output_file"])
            if not shard_path.is_absolute():
                shard_path = base_dir / shard_path
            self.shard_paths.append(shard_path)
        self.source_locations = self._build_source_index(self.shard_paths)

        if num_scf_iterations_per_path is None:
            num_scf_iterations_per_path = []
            for path_idx in range(len(self.paths)):
                source_key = self.source_keys[path_idx]
                _, location = self._resolve_source_location(source_key=source_key)
                num_scf_iterations_per_path.append(location.n_scf_steps)
        else:
            assert len(num_scf_iterations_per_path) == len(
                source_keys
            ), "Length of num_scf_iterations_per_path must match source_keys."

        original_paths = list(self.paths)
        original_source_keys = None if self.source_keys is None else list(self.source_keys)
        self.scf_iterations_per_path = OFDataset.configure_scf_iterations(
            self, list(num_scf_iterations_per_path), keep_initial_guess
        )
        if original_source_keys is not None and len(self.paths) != len(original_paths):
            # configure_scf_iterations can drop paths; realign source_keys to filtered paths.
            keys_by_path: dict[Path, list[str]] = {}
            for path, key in zip(original_paths, original_source_keys):
                keys_by_path.setdefault(path, []).append(key)
            rebuilt_keys = []
            for path in self.paths:
                keys = keys_by_path.get(path, [])
                if not keys:
                    raise RuntimeError(
                        f"Failed to realign source keys after SCF filtering for path {path}."
                    )
                rebuilt_keys.append(keys.pop(0))
            self.source_keys = rebuilt_keys

        self._build_sample_index()
        # Optional for improved performance
        if self.load_label_path or self.load_kohn_sham_path:
            self._build_path_metadata()
        else:
            self.path_label_paths = []
            self.path_kohn_sham_dirs = []

        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            self.getitem = lru_cache(maxsize=None)(self.getitem)

        self._open_shards: OrderedDict[int, HDF5ShardReader] = OrderedDict()

    @staticmethod
    def _decode_text(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8")
        return str(value)

    @classmethod
    def _build_source_index(cls, shard_paths: list[Path]) -> dict[str, SourceSampleLocation]:
        """Build O(1) source_key -> shard/local-range lookup for random access.

        A source_key is "<dataset>/<label_subdir>/<label_file_relpath>".
        """
        source_locations: dict[str, SourceSampleLocation] = {}
        for shard_id, shard_path in enumerate(shard_paths):
            with h5py.File(shard_path, mode="r") as h5f:
                source_keys = [cls._decode_text(s) for s in h5f["source_keys"][:]]
                mol_index = np.asarray(h5f["samples"]["mol_index"][:], dtype=np.int64)
            n_mol = len(source_keys)
            counts = np.bincount(mol_index, minlength=n_mol)
            starts = np.zeros(n_mol, dtype=np.int64)
            if n_mol > 1:
                starts[1:] = np.cumsum(counts[:-1])
            for mol_idx, source_key in enumerate(source_keys):
                source_locations[source_key] = SourceSampleLocation(
                    shard_id=shard_id,
                    mol_index=mol_idx,
                    sample_start=int(starts[mol_idx]),
                    n_scf_steps=int(counts[mol_idx]),
                )
        return source_locations

    def _build_sample_index(self) -> None:
        """Precompute global sample -> (shard_id, local_row, path_idx) indirection."""
        sample_shard_ids: list[int] = []
        sample_local_indices: list[int] = []
        sample_path_indices: list[int] = []
        path_source_keys: list[str] = []
        n_samples_per_path: list[int] = []

        for path_idx in range(len(self.paths)):
            source_key, location = self._resolve_source_location(
                source_key=self.source_keys[path_idx]
            )
            path_source_keys.append(source_key)
            selected_iterations = self.scf_iterations_per_path[path_idx]
            if self.ground_state_only:
                if len(selected_iterations) == 0:
                    raise ValueError(
                        f"ground_state_only shard requires at least one selected SCF iteration for {source_key}."
                    )
                if location.n_scf_steps < 1:
                    raise ValueError(
                        f"ground_state_only shard has no stored sample rows for {source_key}."
                    )
                sample_shard_ids.append(location.shard_id)
                sample_local_indices.append(location.sample_start)
                sample_path_indices.append(path_idx)
                n_samples_per_path.append(1)
                continue
            n_samples_for_path = 0
            for scf_iteration in selected_iterations:
                scf_iteration = int(scf_iteration)
                # Match OFData.from_file behavior: negative indices are counted from the end.
                if scf_iteration < 0:
                    scf_iteration += location.n_scf_steps
                if not (0 <= scf_iteration < location.n_scf_steps):
                    raise IndexError(
                        f"SCF iteration {scf_iteration} for {source_key} is out of range "
                        f"[0, {location.n_scf_steps})."
                    )
                sample_shard_ids.append(location.shard_id)
                sample_local_indices.append(location.sample_start + scf_iteration)
                sample_path_indices.append(path_idx)
                n_samples_for_path += 1
            n_samples_per_path.append(n_samples_for_path)

        self.sample_shard_ids = np.asarray(sample_shard_ids, dtype=np.int32)
        self.sample_local_indices = np.asarray(sample_local_indices, dtype=np.int64)
        self.sample_path_indices = np.asarray(sample_path_indices, dtype=np.int32)
        self.path_source_keys = path_source_keys
        self.path_indices = np.cumsum([0] + n_samples_per_path)
        self.n_samples = int(self.path_indices[-1])

    def _resolve_source_location(self, source_key: str) -> tuple[str, SourceSampleLocation]:
        """Resolve source key to its location."""
        location = self.source_locations[source_key]
        return source_key, location

    def _reshape_coeff_like(
        self, values_flat: np.ndarray, n_basis: int, n_channels: int
    ) -> np.ndarray:
        if n_channels != 1:
            raise ValueError("Only single-channel HDF5 samples are supported.")
        return values_flat.reshape(n_basis)

    def _get_shard_reader(self, shard_id: int) -> HDF5ShardReader:
        """Return shard reader from an LRU cache to cap open files per worker."""
        if shard_id in self._open_shards:
            reader = self._open_shards[shard_id]
            self._open_shards.move_to_end(shard_id)
            return reader

        reader = HDF5ShardReader(self.shard_paths[shard_id])
        self._open_shards[shard_id] = reader

        if (
            self.hdf5_max_open_shards is not None
            and len(self._open_shards) > self.hdf5_max_open_shards
        ):
            _, old_reader = self._open_shards.popitem(last=False)
            old_reader.close()
        return reader

    def _build_path_metadata(self) -> None:
        """Precompute optional string metadata used when building OFData samples."""
        self.path_label_paths: list[str] = []
        self.path_kohn_sham_dirs: list[str] = []
        for source_key in self.path_source_keys:
            self.path_label_paths.append(source_key)
            self.path_kohn_sham_dirs.append(str(Path(source_key).parent.parent / "kohn_sham"))

    def close(self) -> None:
        while self._open_shards:
            _, reader = self._open_shards.popitem(last=False)
            reader.close()

    def __getstate__(self):
        """Drop process-local h5 handles for pickling into DataLoader workers."""
        state = self.__dict__.copy()
        state["_open_shards"] = OrderedDict()
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> OFData:
        """Return one sample by global index.

        We intentionally delegate to :meth:`getitem` because this indirection is used
        for per-instance LRU caching when `cache_in_memory=True`.
        """
        return self.getitem(item)

    def getitem(self, item: int) -> OFData:
        """Resolve global sample index to HDF5 rows and build one OFData sample.

        Index flow:
        1. `item` -> (`sample_shard_ids`, `sample_local_indices`) via precomputed arrays.
        2. In that shard, `samples/*[local_sample_idx]` gives molecule id, bucket id, and bucket row.
        3. Coeff/grad are read from `sample_buckets/data/<bucket_id>/coeff_grad[bucket_row]`.
        4. Molecule-level arrays are read by pointer slices from `molecules/*_ptr`.
        """
        shard_id = int(self.sample_shard_ids[item])
        local_sample_idx = int(self.sample_local_indices[item])
        path_idx = int(self.sample_path_indices[item])
        reader = self._get_shard_reader(shard_id)

        mol_idx = int(reader.sample_mol_index[local_sample_idx])
        scf_iteration = int(reader.sample_scf_iteration[local_sample_idx])
        n_basis = int(reader.mol_n_basis[mol_idx])
        n_channels = int(reader.mol_n_channels[mol_idx])

        bucket_id = int(reader.sample_bucket_id[local_sample_idx])
        bucket_row = int(reader.sample_bucket_row[local_sample_idx])
        coeff_grad = np.asarray(reader.bucket_coeff_grad[bucket_id][bucket_row, :, :])
        coeff_flat = coeff_grad[0, :]
        grad_flat = coeff_grad[1, :]
        coeffs = self._reshape_coeff_like(coeff_flat, n_basis, n_channels)
        gradient_label = self._reshape_coeff_like(grad_flat, n_basis, n_channels)

        source_key = self.path_source_keys[path_idx]
        location = self.source_locations[source_key]
        atom_start = int(reader.mol_atom_ptr[mol_idx])
        atom_end = int(reader.mol_atom_ptr[mol_idx + 1])
        pos = np.asarray(reader.mol_atom_pos_flat[atom_start:atom_end])
        atomic_numbers = np.asarray(
            reader.mol_atomic_numbers_flat[atom_start:atom_end], dtype=np.uint8
        )

        basis_start = int(reader.mol_basis_ptr[mol_idx])
        basis_end = int(reader.mol_basis_ptr[mol_idx + 1])
        basis_integrals = np.asarray(reader.mol_basis_integrals_flat[basis_start:basis_end])

        mol_id = self._decode_text(reader.mol_id[mol_idx])
        n_scf_steps = self.source_locations[source_key].n_scf_steps
        label_path = self.path_label_paths[path_idx] if self.load_label_path else None
        if self.load_kohn_sham_path:
            kohn_sham_path = f"{self.path_kohn_sham_dirs[path_idx]}/{mol_id}.chk"
        else:
            kohn_sham_path = None

        ground_state_coeffs = None
        if self.load_ground_state_coeffs:
            gs_start = int(reader.mol_gs_coeff_ptr[mol_idx])
            gs_end = int(reader.mol_gs_coeff_ptr[mol_idx + 1])
            gs_flat = np.asarray(reader.mol_gs_coeff_flat[gs_start:gs_end])
            ground_state_coeffs = self._reshape_coeff_like(gs_flat, n_basis, n_channels)

        additional_keys = {}
        additional_representations = {}
        for key, representation in self.additional_ground.items():
            split = key.rsplit("/", 1)
            dataset_name = split[-1]
            if reader.mol_ground_state_energies is None:
                raise KeyError("Shard is missing molecules/ground_state_energies.")
            additional_keys[split[0] + "/ground_state_" + dataset_name] = np.asarray(
                reader.mol_ground_state_energies[dataset_name][mol_idx]
            )
            additional_representations[split[0] + "/ground_state_" + dataset_name] = representation

        is_ground_state = self.ground_state_only or scf_iteration == n_scf_steps - 1
        has_nuc_grad_label = None
        if self.load_nuclear_gradients:
            n_nuc_coords = 3 * int(pos.shape[0])
            has_nuc_grad_data = bool(is_ground_state)
            if has_nuc_grad_data:
                has_nuc_grad_data = reader.mol_nuc_grad_available is not None and bool(
                    int(reader.mol_nuc_grad_available[mol_idx])
                )
            if has_nuc_grad_data:
                nuc_grad_start = int(reader.mol_nuc_grad_ptr[mol_idx])
                nuc_grad_end = int(reader.mol_nuc_grad_ptr[mol_idx + 1])
                nuc_grad_label = np.asarray(reader.mol_nuc_grad_flat[nuc_grad_start:nuc_grad_end])
                if nuc_grad_label.shape != (n_nuc_coords,):
                    raise ValueError(
                        f"nuc_grad_label shape mismatch for molecule index {mol_idx}: "
                        f"{nuc_grad_label.shape} vs {(n_nuc_coords,)}"
                    )

                transform_grad_start = int(reader.mol_transform_grad_ptr[mol_idx])
                transform_grad_end = int(reader.mol_transform_grad_ptr[mol_idx + 1])
                transform_grad_cols = int(reader.mol_transform_grad_cols[mol_idx])
                if transform_grad_cols <= 0:
                    raise ValueError(
                        f"transform_grad_contracted_cols must be positive for molecule index "
                        f"{mol_idx}, got {transform_grad_cols}."
                    )
                transform_grad_flat = np.asarray(
                    reader.mol_transform_grad_flat[transform_grad_start:transform_grad_end]
                )
                expected_transform_size = n_nuc_coords * transform_grad_cols
                if transform_grad_flat.size != expected_transform_size:
                    raise ValueError(
                        f"transform_grad_contracted size mismatch for molecule index {mol_idx}: "
                        f"{transform_grad_flat.size} vs {expected_transform_size}"
                    )
                transform_grad_contracted = transform_grad_flat.reshape(
                    n_nuc_coords, transform_grad_cols
                )
                has_nuc_grad_label = [
                    not is_nuclear_gradient_outlier(
                        transform_grad_contracted,
                        self.nuclear_gradient_outlier_threshold,
                    )
                ]
                if not has_nuc_grad_label[0]:
                    nuc_grad_label = np.zeros((n_nuc_coords,), dtype=coeff_flat.dtype)
                    transform_grad_contracted = n_nuc_coords
            else:
                nuc_grad_label = np.zeros((n_nuc_coords,), dtype=coeff_flat.dtype)
                transform_grad_contracted = n_nuc_coords
                has_nuc_grad_label = [False]

            additional_keys["nuc_grad_label"] = nuc_grad_label
            additional_keys["transform_grad_contracted"] = transform_grad_contracted
            additional_representations["nuc_grad_label"] = Representation.NONE
            additional_representations["transform_grad_contracted"] = Representation.NONE

        sample = self.data_class.construct_new(
            basis_info=self.basis_info,
            add_irreps=self.add_irreps,
            mol_id=mol_id,
            source_key=source_key,
            scf_iteration=scf_iteration,
            is_ground_state=is_ground_state,
            pos=pos,
            atomic_numbers=atomic_numbers,
            coeffs=coeffs,
            ground_state_coeffs=ground_state_coeffs,
            gradient_label=gradient_label,
            energy_label=[float(reader.sample_energy[local_sample_idx])],
            label_path=label_path,
            kohn_sham_path=kohn_sham_path,
            has_energy_label=[bool(int(reader.sample_has_energy_label[local_sample_idx]))],
            has_nuc_grad_label=has_nuc_grad_label,
            basis_integrals=basis_integrals,
            additional_representations=additional_representations,
            **additional_keys,
        )

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
