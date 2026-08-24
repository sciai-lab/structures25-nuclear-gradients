from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from mldft.ml.data.components.convert_transforms import PrepareForDensityOptimization
from mldft.ml.data.components.dataset import HDF5ShardOFDataset, OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_data import BasisInfo, Representation
from mldft.ofdft.functional_factory import requires_grid
from mldft.utils.environ import get_mldft_data_path
from mldft.utils.omegaconf_resolvers import to_no_basis_transforms_dataset_statistics


def prepare_transforms(
    model_config: DictConfig | ListConfig,
    transforms: Any,
    basis_info: BasisInfo,
    negative_integrated_density_penalty_weight: float | None,
    grid_level: int,
    grid_prune: str,
):
    """Prepare transform pipeline for density-optimization inputs.

    Args:
        model_config (DictConfig | ListConfig): Model config containing target
            definitions.
        transforms (Any): Instantiated transform container from the datamodule
            config.
        basis_info (BasisInfo): Basis metadata used by preprocessing transforms.
        negative_integrated_density_penalty_weight (float): Penalty weight controlling
            grid requirement.
        grid_level (int): PySCF grid level for grid-dependent terms.
        grid_prune (str): Grid pruning strategy name.

    Returns:
        Any: Updated transform container with denop-specific preprocessing enabled.
    """
    # Side effects:
    # - Mutates ``transforms`` in place by inserting denop preprocessing and flags.

    add_grid = requires_grid(
        model_config.data.target_key, negative_integrated_density_penalty_weight
    )
    transforms.pre_transforms.insert(
        0, PrepareForDensityOptimization(basis_info, add_grid, grid_level, grid_prune)
    )
    transforms.add_transformation_matrix = True
    transforms.use_cached_data = False  # README: should this be the case?

    return transforms


def prepare_datasetkwargs(
    dataset_kwargs: dict, initialization: str, model_config: DictConfig | ListConfig
) -> tuple[str | Path, dict[str, Any]]:
    """Augment dataset kwargs and resolve dataset statistics path.

    Args:
        dataset_kwargs (dict): Base dataset kwargs instantiated from config.
        initialization (str): Initial guess mode used by denop.
        model_config (DictConfig | ListConfig): Model config with dataset statistics
            information.

    Returns:
        tuple[str | Path, dict[str, Any]]: Tuple of
        ``(dataset_statistics_path, dataset_kwargs)`` ready for dataset construction.
    """
    # Side effects:
    # - Mutates ``dataset_kwargs`` in place by injecting denop-specific dataset options.

    dataset_statistics_path = model_config.data.dataset_statistics.path
    if initialization == "sad":
        dataset_statistics_path = to_no_basis_transforms_dataset_statistics(
            dataset_statistics_path, model_config.data.transforms.name
        )
    dataset_kwargs.update(
        {
            "limit_scf_iterations": -1,
            "additional_keys_at_ground_state": {
                "of_labels/energies/e_electron": Representation.SCALAR,
                "of_labels/energies/e_ext": Representation.SCALAR,
                "of_labels/energies/e_hartree": Representation.SCALAR,
                "of_labels/energies/e_kin": Representation.SCALAR,
                "of_labels/energies/e_kin_plus_xc": Representation.SCALAR,
                "of_labels/energies/e_kin_minus_apbe": Representation.SCALAR,
                "of_labels/energies/e_kinapbe": Representation.SCALAR,
                "of_labels/energies/e_xc": Representation.SCALAR,
                "of_labels/energies/e_tot": Representation.SCALAR,
            },
        }
    )
    return dataset_statistics_path, dataset_kwargs


def prepare_dataset_and_loader(
    config: DictConfig,
    model_config: DictConfig | ListConfig,
    denop_cfg: DictConfig,
    paths: list[Path],
    num_scf_iterations_per_path: list[int],
    basis_info: BasisInfo,
    transforms: Any,
) -> tuple[str | Path, OFDataset | HDF5ShardOFDataset, OFLoader]:
    """Build denop dataset and dataloader from prepared paths/config.

    Args:
        config (DictConfig): Resolved runtime config.
        model_config (DictConfig | ListConfig): Training-time model config.
        denop_cfg (DictConfig): Density-optimization subsection of runtime config.
        paths (list[Path]): Molecule label paths selected for the run.
        num_scf_iterations_per_path (list[int]): Maximum SCF iterations per selected
            molecule.
        basis_info (BasisInfo): Basis metadata used by the dataset.
        transforms (Any): Prepared transform pipeline.

    Returns:
        tuple[str | Path, OFDataset | HDF5ShardOFDataset, OFLoader]: Tuple of
        ``(dataset_statistics_path, dataset, dataloader)``.
    """
    dataset_statistics_path, dataset_kwargs = prepare_datasetkwargs(
        instantiate(model_config.data.datamodule.dataset_kwargs),
        denop_cfg.get("initialization"),
        model_config,
    )

    local_dataset_kwargs = dataset_kwargs.copy()
    denop_hdf5_manifest_path = local_dataset_kwargs.pop("denop_hdf5_manifest_path", None)
    hdf5_manifest_path = local_dataset_kwargs.pop("hdf5_manifest_path", None)
    hdf5_max_open_shards = local_dataset_kwargs.pop("hdf5_max_open_shards", None)
    load_label_path = bool(local_dataset_kwargs.pop("label_path", True))
    load_kohn_sham_path = bool(local_dataset_kwargs.pop("kohn_sham_path", True))
    load_ground_state_coeffs = bool(local_dataset_kwargs.pop("ground_state_coeffs", True))

    dataset: OFDataset | HDF5ShardOFDataset
    manifest_ref = denop_hdf5_manifest_path
    if manifest_ref is None:
        manifest_ref = hdf5_manifest_path
    if manifest_ref is not None:
        manifest_ref = str(manifest_ref).format(label_subdir="labels")
        data_root = get_mldft_data_path()
        manifest_path = data_root / Path(manifest_ref)
        if manifest_path.exists() and all(path.is_relative_to(data_root) for path in paths):
            source_keys = [path.relative_to(data_root).as_posix() for path in paths]
            dataset = HDF5ShardOFDataset(
                source_keys=source_keys,
                num_scf_iterations_per_path=num_scf_iterations_per_path,
                basis_info=basis_info,
                transforms=transforms,
                hdf5_manifest_path=manifest_path,
                hdf5_max_open_shards=hdf5_max_open_shards,
                label_path=load_label_path,
                kohn_sham_path=load_kohn_sham_path,
                ground_state_coeffs=load_ground_state_coeffs,
                **local_dataset_kwargs,
            )
        else:
            dataset = OFDataset(
                paths=paths,
                num_scf_iterations_per_path=num_scf_iterations_per_path,
                basis_info=basis_info,
                transforms=transforms,
                **local_dataset_kwargs,
            )
    else:
        dataset = OFDataset(
            paths=paths,
            num_scf_iterations_per_path=num_scf_iterations_per_path,
            basis_info=basis_info,
            transforms=transforms,
            **local_dataset_kwargs,
        )

    list_keys = OmegaConf.select(config, "data.datamodule.dataloader_kwargs.list_keys")

    dataloader = OFLoader(
        dataset,
        batch_size=int(config.get("batch_size")),
        shuffle=False,
        num_workers=OmegaConf.select(
            config, "data.datamodule.dataloader_kwargs.num_workers", default=0
        ),
        pin_memory=OmegaConf.select(
            config, "data.datamodule.dataloader_kwargs.pin_memory", default=True
        ),
        list_keys=list_keys,
    )
    return dataset_statistics_path, dataset, dataloader
