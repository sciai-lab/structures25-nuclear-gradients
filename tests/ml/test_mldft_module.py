from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from mldft.ml.data.components import of_data as of_data_module
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models import mldft_module as mldft_module_impl
from mldft.ml.models.components.loss_function import (
    CoefficientLoss,
    EnergyGradientLoss,
    EnergyLoss,
    WeightedLoss,
)
from mldft.ml.models.components.model_output import ModelOutput
from mldft.ml.models.components.toy_net import ToyNet
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.utils import set_default_torch_dtype


@pytest.fixture(autouse=True)
def patch_mldft_data_path(monkeypatch):
    """Keep dataset loading independent of a configured ``DFT_DATA`` environment."""
    monkeypatch.setattr(of_data_module, "get_mldft_data_path", lambda: Path("/tmp"))


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("k_neighbors", [3, 5])
def test_mldft_module(dummy_basis_info, dummy_dataset_torch, batch_size, k_neighbors):
    """Tests the model_step function of the MLDFTLitModule class."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0)

    batch = next(iter(dataloader))

    net = ToyNet(dummy_basis_info, k_neighbors)
    loss = WeightedLoss(
        energy_loss=dict(weight=0.3, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.3, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        coefficient_loss=dict(
            weight=0.3, loss=CoefficientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
    )

    model = MLDFTLitModule(
        net=net,
        optimizer=None,
        scheduler=None,
        loss_function=loss,
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
    )

    assert "val_metrics/mae_initial_guess" not in model.val_metrics

    test = model.forward(batch)

    assert test[0].shape == batch.energy_label.shape
    assert test[1].shape == batch.gradient_label.shape
    assert test[2].shape == batch.coeffs.shape


def test_sample_forward_adds_optional_saliency_outputs(dummy_basis_info, dummy_dataset_torch):
    """``sample_forward`` should persist optional saliency outputs onto the sample."""

    class DummySaliencyNet(nn.Module):
        def forward(self, sample):
            n_atom = sample.atomic_numbers.shape[0]
            return ModelOutput(
                pred_energy=sample.coeffs.new_zeros(1),
                pred_gradient=torch.zeros_like(sample.coeffs),
                pred_diff=torch.zeros_like(sample.coeffs),
                pred_energies_per_atom=sample.coeffs.new_full((n_atom, 1), 2.0),
                pred_energies_atom_ref=sample.coeffs.new_full((1,), 3.0),
            )

    dataloader = OFLoader(dummy_dataset_torch, batch_size=1, shuffle=False, num_workers=0)
    sample = next(iter(dataloader)).to_data_list()[0]

    model = MLDFTLitModule(
        net=DummySaliencyNet(),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=False,
        basis_info=dummy_basis_info,
        variational=False,
    )

    output_sample = model.sample_forward(sample)

    assert torch.equal(
        output_sample.pred_energies_per_atom,
        sample.coeffs.new_full((sample.n_atom, 1), 2.0),
    )
    assert torch.equal(
        output_sample.pred_energies_atom_ref,
        sample.coeffs.new_full((1,), 3.0),
    )


def _build_loss():
    return WeightedLoss(
        energy_loss=dict(weight=0.3, loss=EnergyLoss(loss_function=nn.L1Loss(reduction="none"))),
        gradient_loss=dict(
            weight=0.3, loss=EnergyGradientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
        coefficient_loss=dict(
            weight=0.3, loss=CoefficientLoss(loss_function=nn.L1Loss(reduction="none"))
        ),
    )


def test_init_builds_denop_runner(monkeypatch, dummy_basis_info):
    """Ensure denop validation setup creates a ``DenopRunnerEngine`` instance."""
    calls = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

    monkeypatch.setattr(mldft_module_impl, "DenopRunnerEngine", FakeRunner)

    model = MLDFTLitModule(
        net=ToyNet(dummy_basis_info, 3),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
        do_denop_val=True,
        dataset_statistics="dataset-statistics",
        density_optimizer="density-optimizer",
        denop_settings=OmegaConf.create(
            {
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
    )

    assert isinstance(model.denoprunner, FakeRunner)
    assert calls["kwargs"]["model"] is model
    assert calls["kwargs"]["optimizer"] == "density-optimizer"
    assert calls["kwargs"]["dataset_statistics"] == "dataset-statistics"
    assert calls["kwargs"]["batch_size"] == 1


@set_default_torch_dtype(torch.float64)
def test_denop_validation_step_uses_runner_and_logs_metrics(
    dummy_basis_info, dummy_dataset_torch, monkeypatch
):
    """Check denop validation uses ``DenopRunnerEngine`` and logs the computed metrics."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.basis_info = dummy_basis_info
    sample = batch.to_data_list()[0]
    sample.basis_info = dummy_basis_info
    logged = []
    calls = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.optimizer = kwargs["optimizer"]
            self.convergence_criterion = kwargs["cfg"].convergence_criterion
            self.func_factory = SimpleNamespace(
                get_energies_label=lambda sample, basis_info: "energies-label"
            )

    monkeypatch.setattr(mldft_module_impl, "DenopRunnerEngine", FakeRunner)

    model = MLDFTLitModule(
        net=ToyNet(dummy_basis_info, 3),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
        do_denop_val=True,
        dataset_statistics="dataset-statistics",
        density_optimizer=SimpleNamespace(max_cycle=5),
        denop_settings=OmegaConf.create(
            {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 1,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "compile_model": False,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
    )
    model._trainer = SimpleNamespace(
        sanity_checking=True,
        global_step=11,
        loggers=[],
    )
    model.compute_nuclear_gradient = True
    model.log_dict = lambda data, **kwargs: logged.append(dict(data))

    fake_runner = SimpleNamespace(
        optimizer=SimpleNamespace(max_cycle=5),
        convergence_criterion="last_iter",
        func_factory=SimpleNamespace(
            get_energies_label=lambda sample, basis_info: "energies-label"
        ),
    )

    def fake_run_samples(samples, progressbar):
        calls.setdefault("run_samples", []).append(
            {
                "samples": samples,
                "progressbar": progressbar,
                "compute_nuclear_gradient": model.compute_nuclear_gradient,
            }
        )
        return [SimpleNamespace(sample=samples[0], callback="callback")]

    fake_runner.run_samples = fake_run_samples
    model.denoprunner = fake_runner

    monkeypatch.setattr(
        mldft_module_impl,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: {
            "gs_gradient_norm": 0.2,
            "converged": True,
        },
    )
    result = model.denop_validation_step(batch, batch_idx=0)

    assert calls["run_samples"][0]["samples"][0].mol_id == sample.mol_id
    assert calls["run_samples"][0]["progressbar"] is False
    assert calls["run_samples"][0]["compute_nuclear_gradient"] is False
    assert model.compute_nuclear_gradient is True
    assert result["model_outputs"].mol_id == sample.mol_id
    assert logged[0]["gs_gradient_norm"] == 0.2
    assert logged[1]["val_denop_metrics/gs_gradient_norm"] == 0.2


@set_default_torch_dtype(torch.float64)
def test_denop_validation_step_aggregates_metrics_over_all_results(
    dummy_basis_info, dummy_dataset_torch, monkeypatch
):
    """Denop validation should aggregate metrics over all returned states."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.basis_info = dummy_basis_info
    samples = batch.to_data_list()
    for sample in samples:
        sample.basis_info = dummy_basis_info

    class FakeRunner:
        def __init__(self, **kwargs):
            self.optimizer = kwargs["optimizer"]
            self.convergence_criterion = kwargs["cfg"].convergence_criterion
            self.func_factory = SimpleNamespace(
                get_energies_label=lambda sample, basis_info: "energies-label"
            )

    monkeypatch.setattr(mldft_module_impl, "DenopRunnerEngine", FakeRunner)

    model = MLDFTLitModule(
        net=ToyNet(dummy_basis_info, 3),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
        do_denop_val=True,
        dataset_statistics="dataset-statistics",
        density_optimizer=SimpleNamespace(max_cycle=5),
        denop_settings=OmegaConf.create(
            {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 1,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "compile_model": False,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
    )
    model._trainer = SimpleNamespace(sanity_checking=True, global_step=0, loggers=[])
    logged = []
    model.log_dict = lambda data, **kwargs: logged.append(dict(data))

    captured = {}

    def fake_run_samples(run_samples, progressbar):
        captured["run_samples_len"] = len(run_samples)
        return [
            SimpleNamespace(sample=run_samples[0], callback=1.0),
            SimpleNamespace(sample=run_samples[1], callback=3.0),
        ]

    model.denoprunner = SimpleNamespace(
        optimizer=SimpleNamespace(max_cycle=5),
        convergence_criterion="last_iter",
        func_factory=SimpleNamespace(
            get_energies_label=lambda sample, basis_info: "energies-label"
        ),
        run_samples=fake_run_samples,
    )
    monkeypatch.setattr(
        mldft_module_impl,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: {
            "gs_gradient_norm": callback,
            "stopping_index": callback + 1.0,
        },
    )

    result = model.denop_validation_step(batch, batch_idx=0)

    assert captured["run_samples_len"] == 2
    assert result["model_outputs"].mol_id == samples[0].mol_id
    assert logged[0]["gs_gradient_norm"] == pytest.approx(2.0)
    assert logged[0]["stopping_index"] == pytest.approx(3.0)
    assert logged[1]["val_denop_metrics/gs_gradient_norm"] == pytest.approx(2.0)


@set_default_torch_dtype(torch.float64)
def test_denop_validation_step_plots_only_first_result_when_not_sanity_checking(
    dummy_basis_info, dummy_dataset_torch, monkeypatch
):
    """When plotting is enabled, only the first denop result should be rendered."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.basis_info = dummy_basis_info
    samples = batch.to_data_list()
    for sample in samples:
        sample.basis_info = dummy_basis_info

    class FakeRunner:
        def __init__(self, **kwargs):
            self.optimizer = kwargs["optimizer"]
            self.convergence_criterion = kwargs["cfg"].convergence_criterion
            self.func_factory = SimpleNamespace(
                get_energies_label=lambda sample, basis_info: "energies-label"
            )

    monkeypatch.setattr(mldft_module_impl, "DenopRunnerEngine", FakeRunner)

    model = MLDFTLitModule(
        net=ToyNet(dummy_basis_info, 3),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
        do_denop_val=True,
        dataset_statistics="dataset-statistics",
        density_optimizer=SimpleNamespace(max_cycle=5),
        denop_settings=OmegaConf.create(
            {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 1,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "compile_model": False,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
    )
    model._trainer = SimpleNamespace(sanity_checking=False, global_step=0, loggers=[])
    model.log_dict = lambda *args, **kwargs: None

    model.denoprunner = SimpleNamespace(
        optimizer=SimpleNamespace(max_cycle=5),
        convergence_criterion="last_iter",
        func_factory=SimpleNamespace(
            get_energies_label=lambda sample, basis_info: "energies-label"
        ),
        run_samples=lambda run_samples, progressbar: [
            SimpleNamespace(sample=run_samples[0], callback="cb-first"),
            SimpleNamespace(sample=run_samples[1], callback="cb-second"),
        ],
    )
    monkeypatch.setattr(
        mldft_module_impl,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: {"gs_gradient_norm": 1.0},
    )

    captured = {"plotted_mol_ids": []}

    plotted_sample = SimpleNamespace(
        ground_state_coeffs=torch.tensor([1.0], dtype=torch.float64),
        stopping_index=0,
    )
    plotted_sample.cpu = lambda: plotted_sample

    monkeypatch.setattr(
        mldft_module_impl,
        "add_density_optimization_trajectories_to_sample",
        lambda sample, callback, energies_label, save_coeff_interval: captured[
            "plotted_mol_ids"
        ].append(sample.mol_id)
        or plotted_sample,
    )
    monkeypatch.setattr(
        mldft_module_impl,
        "transform_tensor_with_sample",
        lambda sample, tensor, representation, invert: torch.tensor([1.0], dtype=torch.float64),
    )

    class _FakeFigure:
        def set_dpi(self, dpi):
            return None

    monkeypatch.setattr(
        mldft_module_impl, "plot_density_optimization", lambda *args, **kwargs: _FakeFigure()
    )

    model.denop_validation_step(batch, batch_idx=0)

    assert captured["plotted_mol_ids"] == [samples[0].mol_id]


@set_default_torch_dtype(torch.float64)
def test_denop_validation_step_uses_short_optimizer_during_sanity_check(
    dummy_basis_info, dummy_dataset_torch, monkeypatch
):
    """Ensure sanity checking temporarily shortens the runner optimizer."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.basis_info = dummy_basis_info
    batch.to_data_list()[0].basis_info = dummy_basis_info
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            self.optimizer = kwargs["optimizer"]
            self.convergence_criterion = kwargs["cfg"].convergence_criterion
            self.func_factory = SimpleNamespace(
                get_energies_label=lambda sample, basis_info: "energies-label"
            )

    monkeypatch.setattr(mldft_module_impl, "DenopRunnerEngine", FakeRunner)

    runner_optimizer = SimpleNamespace(max_cycle=7)
    model = MLDFTLitModule(
        net=ToyNet(dummy_basis_info, 3),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
        do_denop_val=True,
        dataset_statistics="dataset-statistics",
        density_optimizer=runner_optimizer,
        denop_settings=OmegaConf.create(
            {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 1,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "compile_model": False,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
    )
    model._trainer = SimpleNamespace(sanity_checking=True, global_step=0, loggers=[])
    model.log_dict = lambda *args, **kwargs: None

    def fake_run_samples(samples, progressbar):
        captured["max_cycle"] = model.denoprunner.optimizer.max_cycle
        return [SimpleNamespace(sample=samples[0], callback="callback")]

    model.denoprunner = SimpleNamespace(
        optimizer=runner_optimizer,
        convergence_criterion="last_iter",
        func_factory=SimpleNamespace(
            get_energies_label=lambda sample, basis_info: "energies-label"
        ),
        run_samples=fake_run_samples,
    )
    monkeypatch.setattr(
        mldft_module_impl,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: {"converged": True},
    )

    model.denop_validation_step(batch, batch_idx=0)

    assert captured["max_cycle"] == 2
    assert model.denoprunner.optimizer is runner_optimizer
    assert model.denoprunner.optimizer.max_cycle == 7


@set_default_torch_dtype(torch.float64)
def test_denop_validation_step_syncs_runner_device_before_run_samples(
    dummy_basis_info, dummy_dataset_torch, monkeypatch
):
    """Ensure denop runner uses the active module device for sample execution."""
    dataloader = OFLoader(dummy_dataset_torch, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch.basis_info = dummy_basis_info
    batch.to_data_list()[0].basis_info = dummy_basis_info

    class FakeRunner:
        def __init__(self, **kwargs):
            self.optimizer = kwargs["optimizer"]
            self.convergence_criterion = kwargs["cfg"].convergence_criterion
            self.func_factory = SimpleNamespace(
                get_energies_label=lambda sample, basis_info: "energies-label"
            )

    monkeypatch.setattr(mldft_module_impl, "DenopRunnerEngine", FakeRunner)

    model = MLDFTLitModule(
        net=ToyNet(dummy_basis_info, 3),
        optimizer=None,
        scheduler=None,
        loss_function=_build_loss(),
        target_key="kin",
        compile=True,
        basis_info=dummy_basis_info,
        do_denop_val=True,
        dataset_statistics="dataset-statistics",
        density_optimizer=SimpleNamespace(max_cycle=5),
        denop_settings=OmegaConf.create(
            {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 1,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "compile_model": False,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
    )
    model._trainer = SimpleNamespace(sanity_checking=True, global_step=0, loggers=[])
    model.log_dict = lambda *args, **kwargs: None

    captured = {}

    def fake_run_samples(samples, progressbar):
        captured["runner_device"] = model.denoprunner.device
        captured["module_device"] = model.device
        return [SimpleNamespace(sample=samples[0], callback="callback")]

    model.denoprunner = SimpleNamespace(
        device=torch.device("meta"),
        optimizer=SimpleNamespace(max_cycle=5),
        convergence_criterion="last_iter",
        func_factory=SimpleNamespace(
            get_energies_label=lambda sample, basis_info: "energies-label"
        ),
        run_samples=fake_run_samples,
    )
    monkeypatch.setattr(
        mldft_module_impl,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: {"converged": True},
    )

    model.denop_validation_step(batch, batch_idx=0)

    assert captured["runner_device"] == captured["module_device"]
