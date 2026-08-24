"""PyTorch Lightning DataModule for all molecular datasets."""

import pickle
import platform
import random
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import lightning
import torch
from lightning.pytorch.utilities import CombinedLoader

from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import MasterTransformation
from mldft.ml.data.components.convert_transforms import (
    PrepareForDensityOptimization,
    ToTorch,
)
from mldft.ml.data.components.dataset import HDF5ShardOFDataset, OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import Representation
from mldft.ofdft import OFDFT_DTYPE
from mldft.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def worker_init_fn(worker_id: int, dtype: torch.dtype):
    """Helper function to set the default torch dtype inside the worker.

    Args:
        worker_id: The worker id, is not used here.
        dtype: The torch dtype to be used inside the data loader.
    """
    torch.set_default_dtype(dtype)


class OFDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        split_file: Path | str,
        data_dir: Path | str,
        transforms: MasterTransformation,
        basis_info: BasisInfo,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle_train: bool = True,
        shuffle_val: bool = True,
        shuffle_test: bool = False,
        use_cached_iterations: bool = True,
        do_denop_val: bool = False,  # Note: Controls denop inclusion for BOTH validation and test
        denop_val_select: str = "seeded_random",  # options: "last", "seeded_random"
        requires_grid: bool = False,
        grid_level: int | None = 3,
        grid_prune: str | None = "nwchem_prune",
        n_denop_mols: int | None = 10,
        dataset_kwargs: dict | None = None,
        dataloader_kwargs: dict | None = None,
    ):
        """Initialize the DataModule.

        Set up the parameters directory, basis info and transforms that will later be used in setup().

        Args:
            split_file: Name of the yaml file containing the train, val, test split
            data_dir: Path to the directory containing the data
            transforms: transforms for data augmentation
            basis_info: BasisInfo object describing which basis functions are used in the dataset
            batch_size: batch size
            num_workers: number of workers for the dataloader
            pin_memory: whether to pin memory
            shuffle_train: whether to shuffle the training set
            shuffle_val: whether to shuffle the validation set
            shuffle_test: whether to shuffle the test set
            use_cached_iterations: whether to load the number of scf iterations from the split file or recompute them
            do_denop_val: If True, adds a denop subset to validation and test dataloaders.
            denop_val_select: Strategy for selecting denop validation/test subset if do_denop_val is True.
            requires_grid: Whether denop samples require a grid.
            grid_level: Grid level for denop samples if requires_grid is True.
            grid_prune: Grid pruning scheme for denop samples if requires_grid is True.
            n_denop_mols: Number of samples to use for the denop subset (taken from the end). If None, use all.
            dataset_kwargs: Keyword arguments passed to :class:`OFDataset` for standard datasets.
            dataloader_kwargs: Keyword arguments passed to :class:`OFLoader`.
        """
        super().__init__()
        self.split_file = Path(split_file)
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.denop_transforms = None
        self.label_subdir = transforms.label_subdir
        self.basis_info = basis_info
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.use_cached_iterations = use_cached_iterations
        self.do_denop_val = do_denop_val
        self.denop_val_select = denop_val_select
        self.requires_grid = requires_grid
        self.grid_level = grid_level
        self.grid_prune = grid_prune
        self.n_denop_mols = n_denop_mols
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}

        # Datasets (initialized in setup)
        self.train_set: Optional[OFDataset] = None
        self.val_set: Optional[OFDataset] = None
        self.test_set: Optional[OFDataset] = None
        self.denop_val_set: Optional[OFDataset] = None
        self.denop_test_set: Optional[OFDataset] = None
        self.predict_set: Optional[OFDataset] = None  # predict_set handling remains unchanged

        # Denop specific configurations
        self.denop_dataset_kwargs = {}
        self._init_denop_config()

        # Worker initialization
        if platform.system() == "Darwin":
            self.worker_init_fn = partial(worker_init_fn, dtype=torch.get_default_dtype())
            log.warning(
                "Setting worker_init_fn for MacOS. This will break the seeding of the workers."
            )
        else:
            self.worker_init_fn = None

    def _init_denop_config(self):
        """Initialize denop_transforms and denop_dataset_kwargs."""
        # Create transforms for denop
        denop_pre_transforms = [
            PrepareForDensityOptimization(
                basis_info=self.basis_info,
                add_grid=self.requires_grid,
                grid_level=self.grid_level,
                grid_prune=self.grid_prune,
            )
        ] + [
            (
                ToTorch(float_dtype=OFDFT_DTYPE, device=transform.device)
                if isinstance(transform, ToTorch)
                else transform
            )
            for transform in self.transforms.pre_transforms
        ]
        denop_post_transforms = [
            (
                ToTorch(float_dtype=OFDFT_DTYPE, device=transform.device)
                if isinstance(transform, ToTorch)
                else transform
            )
            for transform in self.transforms.post_transforms
        ]
        all_basis_transforms = self.transforms.cached_basis_transforms[:]  # Make a copy
        all_basis_transforms.extend(self.transforms.basis_transforms)

        self.denop_transforms = MasterTransformation(
            self.transforms.name + "_denop",  # Give distinct name
            use_cached_data=False,
            pre_transforms=denop_pre_transforms,
            basis_transforms=all_basis_transforms,
            post_transforms=denop_post_transforms,
            add_transformation_matrix=True,
        )

        # Create dataset kwargs for denop
        self.denop_dataset_kwargs = self.dataset_kwargs.copy()
        self.denop_dataset_kwargs.pop("hdf5_manifest_path", None)
        denop_hdf5_manifest_path = self.denop_dataset_kwargs.pop("denop_hdf5_manifest_path", None)
        if denop_hdf5_manifest_path is not None:
            self.denop_dataset_kwargs["hdf5_manifest_path"] = denop_hdf5_manifest_path
        self.denop_dataset_kwargs["limit_scf_iterations"] = -1
        self.denop_dataset_kwargs["keep_initial_guess"] = False
        self.denop_dataset_kwargs["load_nuclear_gradients"] = False
        self.denop_dataset_kwargs["ground_state_coeffs"] = True
        self.denop_dataset_kwargs["additional_keys_at_ground_state"] = {
            "of_labels/energies/e_electron": Representation.SCALAR,
            "of_labels/energies/e_ext": Representation.SCALAR,
            "of_labels/energies/e_hartree": Representation.SCALAR,
            "of_labels/energies/e_kin": Representation.SCALAR,
            "of_labels/energies/e_kin_plus_xc": Representation.SCALAR,
            "of_labels/energies/e_kin_minus_apbe": Representation.SCALAR,
            "of_labels/energies/e_kinapbe": Representation.SCALAR,
            "of_labels/energies/e_xc": Representation.SCALAR,
            "of_labels/energies/e_tot": Representation.SCALAR,
        }

    def _load_split_data(self, split_name: str) -> List[Tuple[str, str, int]]:
        """Loads the raw split data for a given split name."""
        assert self.split_file.exists(), f"Split file {self.split_file} does not exist."
        with self.split_file.open("rb") as f:
            split_dict = pickle.load(f)
        if split_name not in split_dict:
            raise ValueError(f"Split name '{split_name}' not found in {self.split_file}.")
        return split_dict[split_name]

    def _select_denop_subset(
        self, split_data: List[Tuple[str, str, int]]
    ) -> List[Tuple[str, str, int]]:
        if self.n_denop_mols is None or len(split_data) < self.n_denop_mols:
            return split_data
        if self.denop_val_select == "last":
            return split_data[-self.n_denop_mols :]
        if self.denop_val_select == "seeded_random":
            return random.Random(0).sample(split_data, self.n_denop_mols)
        raise ValueError(
            f"denop_val_select '{self.denop_val_select}' not recognized. Use 'last' or 'seeded_random'."
        )

    def _create_dataset(
        self,
        split_data: List[Tuple[str, str, int]],
        transforms: MasterTransformation,
        dataset_kwargs: dict,
        label_subdir: str,
        split_name_for_log: str = "unknown",  # For logging purposes
    ) -> OFDataset | HDF5ShardOFDataset:
        """Helper function to create an OFDataset instance from preloaded split data."""
        dataset_kwargs = dataset_kwargs.copy()
        hdf5_manifest_path = dataset_kwargs.pop("hdf5_manifest_path", None)
        dataset_kwargs.pop("denop_hdf5_manifest_path", None)
        hdf5_max_open_shards = dataset_kwargs.pop("hdf5_max_open_shards", None)
        load_label_path = bool(dataset_kwargs.pop("label_path", True))
        load_kohn_sham_path = bool(dataset_kwargs.pop("kohn_sham_path", True))
        dataset_kwargs.pop("basis_integrals", None)
        load_ground_state_coeffs = bool(dataset_kwargs.pop("ground_state_coeffs", True))

        if not split_data:
            log.warning(f"Creating dataset for split '{split_name_for_log}' with empty data.")
            # Return an empty dataset to avoid errors downstream
            return OFDataset(
                paths=[], basis_info=self.basis_info, transforms=transforms, **dataset_kwargs
            )

        paths = [
            self.data_dir / dataset / label_subdir / label_path
            for dataset, label_path, _ in split_data
        ]
        iterations = None
        if self.use_cached_iterations:
            iterations = [scf_iterations for _, _, scf_iterations in split_data]

        log.info(
            f"Creating dataset for split '{split_name_for_log}' with {len(paths)} paths using labels from {label_subdir}."
        )
        if hdf5_manifest_path:
            source_keys = [
                f"{dataset}/{label_subdir}/{label_path}" for dataset, label_path, _ in split_data
            ]
            manifest_ref = str(hdf5_manifest_path).format(
                label_subdir=label_subdir,
            )
            # If manifest_ref is absolute, pathlib keeps it absolute even when joined.
            manifest_path = self.data_dir / Path(manifest_ref)
            if manifest_path.exists():
                log.info(f"Using HDF5 manifest for split '{split_name_for_log}': {manifest_path}")
                return HDF5ShardOFDataset(
                    source_keys=source_keys,
                    num_scf_iterations_per_path=iterations,
                    basis_info=self.basis_info,
                    transforms=transforms,
                    hdf5_manifest_path=manifest_path,
                    hdf5_max_open_shards=hdf5_max_open_shards,
                    label_path=load_label_path,
                    kohn_sham_path=load_kohn_sham_path,
                    ground_state_coeffs=load_ground_state_coeffs,
                    **dataset_kwargs,
                )
            log.info(
                f"HDF5 manifest not found for split '{split_name_for_log}' at {manifest_path}. "
                f"Falling back to zarr labels in '{label_subdir}'."
            )

        return OFDataset(
            paths=paths,
            num_scf_iterations_per_path=iterations,
            basis_info=self.basis_info,
            transforms=transforms,
            **dataset_kwargs,
        )

    def setup(self, stage: str):
        """Load the necessary datasets based on the stage."""

        if stage == "fit":
            if self.train_set is None:
                train_data = self._load_split_data("train")
                self.train_set = self._create_dataset(
                    train_data, self.transforms, self.dataset_kwargs, self.label_subdir, "train"
                )
            # Only setup val if needed for fitting stage
            if self.val_set is None:
                val_data = self._load_split_data("val")
                self.val_set = self._create_dataset(
                    val_data, self.transforms, self.dataset_kwargs, self.label_subdir, "val"
                )
                if self.do_denop_val and self.denop_val_set is None:
                    denop_val_subset = self._select_denop_subset(val_data)
                    self.denop_val_set = self._create_dataset(
                        denop_val_subset,
                        self.denop_transforms,
                        self.denop_dataset_kwargs,
                        "labels",
                        "denop_val",
                    )

        elif stage == "validate":
            if self.val_set is None:
                val_data = self._load_split_data("val")
                self.val_set = self._create_dataset(
                    val_data, self.transforms, self.dataset_kwargs, self.label_subdir, "val"
                )
                if self.do_denop_val and self.denop_val_set is None:
                    denop_val_subset = self._select_denop_subset(val_data)
                    self.denop_val_set = self._create_dataset(
                        denop_val_subset,
                        self.denop_transforms,
                        self.denop_dataset_kwargs,
                        "labels",
                        "denop_val",
                    )

        elif stage == "test":
            if self.test_set is None:
                test_data = self._load_split_data("test")
                self.test_set = self._create_dataset(
                    test_data, self.transforms, self.dataset_kwargs, self.label_subdir, "test"
                )
                if self.do_denop_val and self.denop_test_set is None:
                    denop_test_subset = self._select_denop_subset(test_data)
                    self.denop_test_set = self._create_dataset(
                        denop_test_subset,
                        self.denop_transforms,
                        self.denop_dataset_kwargs,
                        "labels",
                        "denop_test",
                    )

        elif stage == "predict":
            # Assuming predict uses the same data as test, adjust if different
            if self.predict_set is None:
                # For now, let's assume it's like test_set - adjust if needed
                predict_data = self._load_split_data("test")
                self.predict_set = self._create_dataset(
                    predict_data,
                    self.transforms,
                    self.dataset_kwargs,
                    self.label_subdir,
                    "predict",
                )
                log.warning(
                    "Predict stage setup assumes test set data. Adjust if prediction requires different data."
                )

        else:
            raise ValueError(
                f"Stage '{stage}' not supported. Must be 'fit', 'validate', 'test', or 'predict'."
            )

    def train_dataloader(self) -> OFLoader:
        """Return the training dataloader."""
        if self.train_set is None:
            self.setup("fit")  # Ensure dataset is loaded
        if self.train_set is None:
            raise RuntimeError("Training dataset failed to initialize.")

        return OFLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            drop_last=True,  # Important during training
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> OFLoader | CombinedLoader:
        """Return the validation dataloader.

        May be combined with denop samples.
        """
        if self.val_set is None:
            self.setup("validate")  # Ensure datasets are loaded
        if self.val_set is None:
            raise RuntimeError("Validation dataset failed to initialize.")

        val_loader = OFLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.dataloader_kwargs,
        )

        if self.do_denop_val:
            if self.denop_val_set is None:
                log.warning(
                    "do_denop_val is True, but denop_val_set is not initialized. Skipping denop loader."
                )
                return val_loader
            if len(self.denop_val_set) == 0:
                log.warning("denop_val_set is empty. Skipping denop loader.")
                return val_loader

            # Configure denop loader (single sample, potentially fewer workers)
            denop_workers = 0 if self.requires_grid or self.num_workers == 0 else 1
            denop_val_loader = OFLoader(
                self.denop_val_set,
                batch_size=1,  # Denop is processed sample by sample
                shuffle=self.shuffle_val,  # Usually same shuffle setting as main val
                num_workers=denop_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=self.worker_init_fn,
                **self.dataloader_kwargs,
            )
            combined_loader = CombinedLoader(
                {"val": val_loader, "denop_val": denop_val_loader},
                mode="sequential",
            )
            log.info("Returning combined validation dataloader (standard + denop).")
            return combined_loader
        else:
            return val_loader

    def test_dataloader(self) -> OFLoader | CombinedLoader:
        """Return the test dataloader.

        May be combined with denop samples.
        """
        if self.test_set is None:
            self.setup("test")  # Ensure datasets are loaded
        if self.test_set is None:
            raise RuntimeError("Test dataset failed to initialize.")

        test_loader = OFLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.dataloader_kwargs,
        )

        if self.do_denop_val:
            if self.denop_test_set is None:
                log.warning(
                    "do_denop_val is True, but denop_test_set is not initialized. Skipping denop loader."
                )
                return test_loader
            if len(self.denop_test_set) == 0:
                log.warning("denop_test_set is empty. Skipping denop loader.")
                return test_loader

            # Configure denop loader (single sample, potentially fewer workers)
            denop_workers = 0 if self.requires_grid or self.num_workers == 0 else 1
            denop_test_loader = OFLoader(
                self.denop_test_set,
                batch_size=1,
                shuffle=self.shuffle_test,
                num_workers=denop_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=self.worker_init_fn,
                **self.dataloader_kwargs,
            )
            combined_loader = CombinedLoader(
                {"test": test_loader, "denop_test": denop_test_loader},
                mode="sequential",
            )
            log.info("Returning combined validation dataloader (standard + denop).")
            return combined_loader
        else:
            return test_loader

    def predict_dataloader(self) -> OFLoader:
        """Return the prediction dataloader."""
        if self.predict_set is None:
            self.setup("predict")  # Ensure dataset is loaded
        if self.predict_set is None:
            raise RuntimeError("Predict dataset failed to initialize.")

        return OFLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=False,  # Typically no shuffle for predict
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
            **self.dataloader_kwargs,
        )
