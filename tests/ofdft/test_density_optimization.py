import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from mldft.ofdft.run_density_optimization import run_denop_processes
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.environ import parse_run_path, run_to_checkpoint_path
from mldft.utils.save_checkpoint import save_checkpoint_manually

TEST_DATASET = "QM9"
TEST_DATASET_SPLIT_PATH = Path(os.getenv("DFT_DATA")) / TEST_DATASET / "split.pkl"
TEST_DATATSET_STATISTICS_PATH = (
    Path(os.getenv("DFT_DATA"))
    / TEST_DATASET
    / "dataset_statistics"
    / "dataset_statistics_labels_no_basis_transforms_e_kin_plus_xc_without_bias.zarr"
)
TEST_DATA_DIR_EXISTS = (Path(os.getenv("DFT_DATA")) / TEST_DATASET).exists()
TEST_DATASET_SPLIT_PATH_EXISTS = TEST_DATASET_SPLIT_PATH.exists()
TEST_DATASET_STATISTICS_PATH_EXISTS = TEST_DATATSET_STATISTICS_PATH.exists()

TEST_DATASET_AVAILABLE = (
    TEST_DATA_DIR_EXISTS and TEST_DATASET_SPLIT_PATH_EXISTS and TEST_DATASET_STATISTICS_PATH_EXISTS
)


@pytest.fixture(scope="function")
def cfg_train(tmp_path: Path) -> DictConfig:
    """Provides a train config needed to store a valid model checkpoint.

    Note:
        We cannot simply use cfg_train of tests/ml/conftest.py as we want to make it dependent on
        the TEST_DATASET used. The two electron dataset used in the ml tests do not have a
        sufficient number of data points for the continued density optimization test.
    """

    with initialize(version_base="1.3", config_path="../../configs/ml"):
        cfg_train = compose(
            config_name="train.yaml",
            overrides=[
                f"data={TEST_DATASET.lower()}",
                "data/transforms=no_basis_transforms",
                "model=graphformer_small",
            ],
        )
    with open_dict(cfg_train):
        # Since we don't use hydra these need to be set manually
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)
        cfg_train.paths.work_dir = str(tmp_path)
        cfg_train.callbacks.model_checkpoint.dirpath = str(tmp_path / "checkpoints")

    return cfg_train


@pytest.fixture(scope="function")
def cfg_density_optimization(tmp_path: Path, initial_n_molecules: int) -> DictConfig:
    """Provides a valid ofdft config which should be sufficiently fast for testing purposes."""

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    with initialize(version_base=None, config_path="../../configs/ofdft"):
        cfg_ofdft = compose(config_name="density_optimization.yaml")
    with open_dict(cfg_ofdft):
        cfg_ofdft.run_path = str(tmp_path)
        cfg_ofdft.n_molecules = initial_n_molecules
        cfg_ofdft.device = device
        cfg_ofdft.density_optimizer.max_cycle = 5
        cfg_ofdft.metrics_grid_level = 0
        cfg_ofdft.compupte_grid_metrics = False
        cfg_ofdft.plot_every_n = 0
        cfg_ofdft.plot_summary = False
        cfg_ofdft.log_file = str(tmp_path / "ofdft.log")
        cfg_ofdft.continue_denop = False
        cfg_ofdft.denop_dir = str(tmp_path)
        cfg_ofdft.batch_size = 0
        cfg_ofdft.num_threads = 1
        cfg_ofdft.split_file_path = Path(os.getenv("DFT_DATA")) / TEST_DATASET / "split.pkl"

    return cfg_ofdft


@pytest.mark.skipif(
    not TEST_DATASET_AVAILABLE,
    reason=f"Test dataset {TEST_DATASET} or split file or statistics not available.",
)
@pytest.mark.parametrize(
    "initial_n_molecules, final_n_molecules",
    [
        (1, 3),
        (2, 2),
    ],
)
def test_density_optimization(
    tmp_path: Path,
    initial_n_molecules: int,
    final_n_molecules: int,
    cfg_train: DictConfig,
    cfg_density_optimization: DictConfig,
):
    """Test continued density optimization.

    The natural use case would be an interrupted density optimization or one with failed individual
    molecules. We simulate this here with increasing the number of molecules for a second run.
    """

    trainer = instantiate(cfg_train.trainer)
    model = instantiate(cfg_train.model)
    ckpt_path = cfg_train.callbacks.model_checkpoint.dirpath + "/last.ckpt"
    save_checkpoint_manually(ckpt_path, model, trainer)
    OmegaConf.save(cfg_train, str(tmp_path / "hparams.yaml"), resolve=False)

    run_density_optimization_wrapper(cfg_density_optimization)

    sample_dir = tmp_path / "sample_trajectories"
    assert sample_dir.is_dir(), f"Sample trajectories have not been saved into {sample_dir}"

    assert (
        sample_dir / "basis_info.pt"
    ).exists(), f"basis info has not been saved in {sample_dir}."
    initial_sample_paths = list(sample_dir.glob("sample*.pt"))
    assert len(initial_sample_paths) == initial_n_molecules

    with open_dict(cfg_density_optimization):
        cfg_density_optimization.continue_denop = True
        cfg_density_optimization.n_molecules = final_n_molecules

    run_density_optimization_wrapper(cfg_density_optimization)

    assert (
        sample_dir / "basis_info.pt"
    ).exists(), "basis info is not present after continued density optimization."
    final_sample_paths = list(sample_dir.glob("sample*.pt"))
    assert (
        len(final_sample_paths) == final_n_molecules
    ), "sample trajectories have not been added to the existing ones after continued density optimization."

    initial_sample_names = {path.name for path in initial_sample_paths}
    final_sample_names = {path.name for path in final_sample_paths}

    if initial_n_molecules == final_n_molecules:
        assert (
            initial_sample_names == final_sample_names
        ), "Samples should be identical when initial and final n_molecules are equal."
    else:
        assert initial_sample_names.issubset(
            final_sample_names
        ), "Initial samples should be present in final sample paths."
        assert len(final_sample_names - initial_sample_names) == (
            final_n_molecules - initial_n_molecules
        ), "The remaining samples should be different."


def run_density_optimization_wrapper(cfg: DictConfig) -> None:
    """Small wrapper around ``run_denop_processes``."""

    log_file = Path(cfg.get("log_file"))
    run_path = parse_run_path(cfg.run_path)
    denop_cfg = cfg.get("denop_settings")
    optimizer = instantiate(cfg.get("density_optimizer"))
    model_config_path = run_path / "hparams.yaml"
    model_config = OmegaConf.load(model_config_path)
    ckpt_path = run_to_checkpoint_path(run_path, cfg.ckpt_choice)

    run_denop_processes(
        ckpt_path=ckpt_path,
        model_config=model_config,
        optimizer=optimizer,
        split_file_path=cfg.get("split_file_path"),
        split=cfg.get("split"),
        guess_ckpt_path=None,
        device=cfg.get("device"),
        num_threads=cfg.get("num_threads"),
        num_devices=cfg.get("num_devices"),
        batch_size=cfg.get("batch_size"),
        batch_timeout=cfg.get("batch_timeout"),
        model_dtype=str_to_torch_float_dtype(cfg.get("model_dtype", torch.float64)),
        xc_functional=denop_cfg.get("xc_functional"),
        initialization=denop_cfg.get("initialization"),
        n_molecules=cfg.get("n_molecules"),
        molecule_choice=cfg.get("molecule_choice"),
        molecule_ids=cfg.get("molecule_ids"),
        seed=cfg.get("seed"),
        log_file=log_file,
        save_individual_plots=cfg.get("save_individual_plots"),
        plot_every_n=cfg.get("plot_every_n"),
        plot_summary=cfg.get("plot_summary"),
        swarm_plot_subsample=cfg.get("swarm_plot_subsample"),
        normalize_initial_guess=denop_cfg.get("normalize_initial_guess"),
        grid_level=denop_cfg.get("grid_level"),
        grid_prune=denop_cfg.get("grid_prune"),
        max_xc_memory=denop_cfg.get("max_xc_memory"),
        compute_grid_metrics=cfg.get("compute_grid_metrics"),
        metrics_grid_level=cfg.get("metrics_grid_level"),
        metrics_grid_prune=cfg.get("metrics_grid_prune"),
        negative_integrated_density_penalty_weight=denop_cfg.get(
            "negative_integrated_density_penalty_weight"
        ),
        convergence_criterion=denop_cfg.get("convergence_criterion"),
        fail_fast=cfg.get("fail_fast"),
        deterministic=cfg.get("deterministic"),
        save_coeff_interval=cfg.get("save_coeff_interval"),
        compute_final_hessian=cfg.get("compute_final_hessian"),
        continue_denop=cfg.get("continue_denop"),
        denop_dir=cfg.get("denop_dir"),
    )
