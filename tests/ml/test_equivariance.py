import copy
import os
import tempfile
from pathlib import Path

import pytest
import torch
from e3nn.o3 import rand_matrix
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

import mldft.utils.omegaconf_resolvers  # noqa: F401
from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.models.components.model_output import ModelOutput
from mldft.utils.utils import set_default_torch_dtype


def _download_qm9_test_mol():
    def complete_download(base_dir):
        dataset_path = base_dir / "sciai-test-mol"
        stats_path = (
            dataset_path
            / "dataset_statistics/dataset_statistics_labels_no_basis_transforms_e_kin_plus_xc.zarr"
        )
        return (dataset_path / "labels/0880476.zarr.zip").exists() and stats_path.exists()

    candidates = [Path(tempfile.gettempdir()) / "sciai-dft-hf-test"]
    if "DFT_DATA" in os.environ:
        dft_data = Path(os.environ["DFT_DATA"])
        try:
            dft_data.mkdir(parents=True, exist_ok=True)
            if complete_download(dft_data) or os.access(dft_data, os.W_OK):
                candidates.insert(0, dft_data)
        except OSError:
            pass

    last_error = None
    for base_dir in candidates:
        dataset_path = base_dir / "sciai-test-mol"
        if complete_download(base_dir):
            os.environ["DFT_DATA"] = str(base_dir)
            return dataset_path
        try:
            snapshot_download(
                repo_id="sciai-lab/structures25",
                allow_patterns=[
                    "sciai-test-mol/dataset_info.yaml",
                    "sciai-test-mol/labels/0880476.zarr.zip",
                    "sciai-test-mol/dataset_statistics/"
                    "dataset_statistics_labels_no_basis_transforms_e_kin_plus_xc.zarr/**",
                ],
                cache_dir=str(base_dir / ".hf_cache"),
                local_dir=str(base_dir),
                max_workers=4,
            )
        except Exception as exc:
            last_error = exc
            continue
        if complete_download(base_dir):
            os.environ["DFT_DATA"] = str(base_dir)
            return dataset_path

    pytest.skip(f"Could not download QM9 equivariance test data: {last_error}")


def _compose_cfg(model_name, dataset_path):
    config_dir = str(Path(__file__).parents[2] / "configs/ml")
    stats_path = (
        dataset_path
        / "dataset_statistics/dataset_statistics_labels_no_basis_transforms_e_kin_plus_xc.zarr"
    )
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"model={model_name}",
                "data=qm9_perturbed_fock",
                "data/transforms=no_basis_transforms",
                "data.dataset_name=sciai-test-mol",
                f"data.dataset_statistics.path={stats_path}",
            ],
        )

    net_cfg = cfg.model.net
    with open_dict(net_cfg):
        if model_name.startswith("esen"):
            net_cfg.backbone.deterministic = True
        else:
            net_cfg.deterministic = True
    return cfg


def _load_qm9_sample(cfg, dataset_path):
    dataset = OFDataset(
        paths=[dataset_path / "labels/0880476.zarr.zip"],
        basis_info=instantiate(cfg.data.basis_info),
        transforms=instantiate(cfg.data.transforms),
        limit_scf_iterations=-1,
        keep_initial_guess=False,
        add_irreps=True,
        energy_key="e_kin_plus_xc",
        gradient_key="grad_kin_plus_xc",
        cache_in_memory=False,
    )
    return dataset.basis_info, dataset[0]


def _make_rotated_sample(basis_info, sample):
    rotation = rand_matrix(dtype=torch.float64)
    # Rotate coefficient blocks with the same irreps used by the density/delta modules.
    coeff_rotation = torch.block_diag(
        *[
            basis_info.irreps_per_atom[int(atom_ind)].D_from_matrix(rotation)
            for atom_ind in sample.atom_ind.tolist()
        ]
    )
    rotated_sample = copy.deepcopy(sample)
    rotated_sample.pos = sample.pos @ rotation.T
    rotated_sample.coeffs = coeff_rotation @ sample.coeffs
    return rotated_sample, coeff_rotation


def _assert_model_equivariant(
    model,
    sample,
    rotated_sample,
    coeff_rotation,
    atol=1e-8,
    rtol=1e-8,
):
    model.eval()
    batch = OFBatch.from_data_list([sample], follow_batch=["coeffs"])
    rotated_batch = OFBatch.from_data_list([rotated_sample], follow_batch=["coeffs"])
    batch.coeffs.requires_grad_(True)
    rotated_batch.coeffs.requires_grad_(True)

    output = model(batch)
    rotated_output = model(rotated_batch)
    energy = output.pred_energy
    pred_delta = output.pred_diff
    rotated_energy = rotated_output.pred_energy
    rotated_pred_delta = rotated_output.pred_diff
    gradient = torch.autograd.grad(energy.sum(), batch.coeffs, retain_graph=True)[0]
    rotated_gradient = torch.autograd.grad(
        rotated_energy.sum(), rotated_batch.coeffs, retain_graph=True
    )[0]

    assert torch.allclose(
        coeff_rotation @ gradient, rotated_gradient, atol=atol, rtol=rtol
    ), "gradient is not equivariant"
    assert torch.allclose(energy, rotated_energy, atol=atol, rtol=rtol), "energy is not invariant"
    assert torch.allclose(
        coeff_rotation @ pred_delta, rotated_pred_delta, atol=atol, rtol=rtol
    ), "density delta is not equivariant"


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize(
    ("model_name", "tol"),
    [
        ("esen_small", 1e-8),
        # EquiformerV2 uses grid-space operations that might introduce
        # additional discretization noise, so we keep a looser tolerance here.
        ("equiformerv2_small", 5e-5),
    ],
)
def test_variational_model_equivariance(model_name, tol):
    torch.manual_seed(0)

    dataset_path = _download_qm9_test_mol()
    cfg = _compose_cfg(model_name, dataset_path)
    basis_info, sample = _load_qm9_sample(cfg, dataset_path)
    rotated_sample, coeff_rotation = _make_rotated_sample(basis_info, sample)

    _assert_model_equivariant(
        instantiate(cfg.model.net),
        sample,
        rotated_sample,
        coeff_rotation,
        atol=tol,
        rtol=tol,
    )


@set_default_torch_dtype(torch.float64)
def test_equivariance_check_works_for_equivariant_model():
    torch.manual_seed(0)

    dataset_path = _download_qm9_test_mol()
    cfg = _compose_cfg("esen_small", dataset_path)
    basis_info, sample = _load_qm9_sample(cfg, dataset_path)
    rotated_sample, coeff_rotation = _make_rotated_sample(basis_info, sample)

    class EquivariantModel(torch.nn.Module):
        def forward(self, batch):
            # Squared norm is invariant under orthogonal coefficient rotation.
            num_graphs = int(batch.coeffs_batch.max().item()) + 1
            energy = batch.coeffs.new_zeros(num_graphs)
            energy.index_add_(0, batch.coeffs_batch, batch.coeffs**2)
            return ModelOutput(pred_energy=energy, pred_diff=batch.coeffs)

    _assert_model_equivariant(
        EquivariantModel(),
        sample,
        rotated_sample,
        coeff_rotation,
    )


@set_default_torch_dtype(torch.float64)
def test_equivariance_check_fails_for_non_equivariant_energy():
    torch.manual_seed(0)

    dataset_path = _download_qm9_test_mol()
    cfg = _compose_cfg("esen_small", dataset_path)
    basis_info, sample = _load_qm9_sample(cfg, dataset_path)
    rotated_sample, coeff_rotation = _make_rotated_sample(basis_info, sample)

    class NonEquivariantEnergyModel(torch.nn.Module):
        def forward(self, batch):
            # Depends on absolute orientation in Cartesian frame.
            num_graphs = int(batch.batch.max().item()) + 1
            energy = batch.pos.new_zeros(num_graphs)
            energy.index_add_(0, batch.batch, batch.pos[:, 0])
            # Keep an autograd path to coeffs while preserving the same energy value.
            zero_coeff_term = batch.coeffs.new_zeros(num_graphs)
            zero_coeff_term.index_add_(0, batch.coeffs_batch, batch.coeffs * 0.0)
            energy = energy + zero_coeff_term
            return ModelOutput(pred_energy=energy, pred_diff=batch.coeffs)

    with pytest.raises(AssertionError, match="energy is not invariant"):
        _assert_model_equivariant(
            NonEquivariantEnergyModel(),
            sample,
            rotated_sample,
            coeff_rotation,
        )
