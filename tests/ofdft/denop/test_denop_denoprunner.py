import csv
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mldft.ofdft.denop import denoprunner as denoprunner_module
from mldft.ofdft.denop import denoprunner_engine as denoprunner_engine_module
from mldft.ofdft.run_density_optimization import run_denop_processes
from mldft.utils.conversions import str_to_torch_float_dtype
from mldft.utils.environ import parse_run_path, run_to_checkpoint_path


class FakeSample:
    """Minimal sample mock carrying the fields touched by runner methods."""

    def __init__(self, mol_id=1):
        """Create a fake sample with deterministic tensor-like attributes."""
        self.mol_id = mol_id
        self.mol = f"mol-{mol_id}"
        self.coeffs = torch.tensor([0.0, 0.0], dtype=torch.float64)
        self.basis_info = "basis-info"
        self.transformation_matrix = torch.eye(2, dtype=torch.float64)
        self.inv_transformation_matrix = torch.eye(2, dtype=torch.float64)
        self.ground_state_energy_total = 4.0
        self.trajectory_energy_total = torch.tensor([4.2, 4.1], dtype=torch.float64)
        self.trajectory_l2_norm = torch.tensor([0.5, 0.25], dtype=torch.float64)
        self.trajectory_gradient_norm = torch.tensor([0.2, 0.1], dtype=torch.float64)
        self.stopping_index = 1
        self.n_atom = 2
        self.n_electron = 4
        self.time = 0.0
        self.deleted_items = []
        self.added_items = []
        self.overlap_matrix = "overlap"
        self.coulomb_matrix = "coulomb"
        self.nuclear_attraction_vector = "nuclear"
        self.ao = "ao"

    def keys(self) -> list[str]:
        return [
            "overlap_matrix",
            "coulomb_matrix",
            "nuclear_attraction_vector",
            "ao",
        ]

    def to(self, device: str | torch.device):
        """Record the device assignment expected by runner loops."""
        self.device = device
        return self

    def add_item(self, key, value, representation):
        """Add an attribute and record the addition for test assertions."""
        self.added_items.append((key, value, representation))
        setattr(self, key, value)

    def delete_item(self, key):
        """Remove an attribute and record the deletion for test assertions."""
        self.deleted_items.append(key)
        delattr(self, key)

    def clone(self):
        """Create a shallow copy of the sample, mimicking the behavior of ``torch.clone`` used in
        the runner."""
        clone = FakeSample(self.mol_id)
        clone.coeffs = self.coeffs
        clone.basis_info = self.basis_info
        return clone


def _make_bare_runner(runner_cls, tmp_path: Path):
    """Build a partially initialized runner with lightweight test doubles."""
    runner = runner_cls.__new__(runner_cls)
    runner.device = "cpu"
    runner.optimizer = "optimizer"
    runner.func_factory = SimpleNamespace(
        get_energies_label=lambda sample, basis_info: "energies-label"
    )
    runner.bridge = SimpleNamespace()
    runner.proj_minao_module = "proj-module"
    runner.max_xc_memory = 1
    runner.initialization = "minao"
    runner.normalize_initial_guess = True
    runner.ks_basis = "sto-3g"
    runner.sad_guess_kwargs = {"sad": True}
    runner.convergence_criterion = "last_iter"
    runner.save_coeff_interval = 2
    runner.fail_fast = False
    runner.compute_final_hessian = False
    runner.plotter = None
    runner.batch_size = 2
    runner.step_count = 0
    runner.paths = ["path-1", "path-2"]
    runner.with_progress_bar = False
    runner.evaluator = SimpleNamespace(
        evaluate=lambda: None, finalize_sample=lambda **kwargs: None
    )
    runner.config = {"plot_n_processes": 1}
    runner.save_dir = tmp_path
    runner.xc_functional = "gga_x_pbe"
    runner.negative_integrated_density_penalty_weight = 0.0
    runner.compile_model = False
    runner._profiler = None
    runner.finalize_at_least_n = runner.batch_size
    return runner


@pytest.fixture
def bare_runner(tmp_path: Path):
    """Build a partially initialized ``DenopRunner`` with lightweight doubles."""
    return _make_bare_runner(denoprunner_module.DenopRunner, tmp_path)


@pytest.fixture
def bare_basic_runner(tmp_path: Path):
    """Build a partially initialized ``DenopRunnerEngine`` with lightweight doubles."""
    return _make_bare_runner(denoprunner_engine_module.DenopRunnerEngine, tmp_path)


def test_prepare_initial_guess_uses_tensor_directly(bare_runner):
    """Ensure tensor initialization is copied directly onto ``sample.coeffs``."""
    sample = FakeSample()
    bare_runner.initialization = torch.tensor([1.0, 2.0], dtype=torch.float64)

    bare_runner._prepare_initial_guess(sample)

    assert torch.equal(sample.coeffs, bare_runner.initialization)


def test_prepare_initial_guess_uses_helper(monkeypatch, bare_runner):
    """Verify non-tensor initialization delegates to ``initial_guess`` helper."""
    sample = FakeSample()
    bare_runner.initialization = "sad"
    captured = {}

    def fake_initial_guess(*args, **kwargs):
        return torch.tensor([3.0, 4.0], dtype=torch.float64)

    monkeypatch.setattr(denoprunner_engine_module, "initial_guess", fake_initial_guess)

    bare_runner._prepare_initial_guess(sample)

    assert torch.equal(sample.coeffs, torch.tensor([3.0, 4.0], dtype=torch.float64))


def test_denoprunner_engine_constructor_builds_empty_sad_kwargs_for_non_sad_init(
    monkeypatch,
):
    """Ensure non-SAD initialization leaves SAD helper kwargs empty."""
    monkeypatch.setattr(
        denoprunner_engine_module.FunctionalFactory,
        "from_module",
        lambda *args, **kwargs: "factory",
    )
    monkeypatch.setattr(
        denoprunner_engine_module,
        "BatchedFunctionalBridge",
        lambda *args, **kwargs: "bridge",
    )

    runner = denoprunner_engine_module.DenopRunnerEngine(
        cfg=OmegaConf.create(
            {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 3,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
        model="model",
        optimizer="optimizer",
        dataset_statistics="dataset-statistics",
        basis_info="basis-info",
        device="cpu",
        batch_size=2,
    )

    assert runner.sad_guess_kwargs == {}


def test_denoprunner_engine_constructor_builds_sad_guess_kwargs(monkeypatch):
    """Check the lightweight runner populates SAD initialization kwargs."""
    monkeypatch.setattr(
        denoprunner_engine_module.FunctionalFactory,
        "from_module",
        lambda *args, **kwargs: "factory",
    )
    monkeypatch.setattr(
        denoprunner_engine_module,
        "BatchedFunctionalBridge",
        lambda *args, **kwargs: "bridge",
    )

    runner = denoprunner_engine_module.DenopRunnerEngine(
        cfg=OmegaConf.create(
            {
                "initialization": "sad",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 3,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
            }
        ),
        model="model",
        optimizer="optimizer",
        dataset_statistics="dataset-statistics",
        basis_info="basis-info",
        device="cpu",
        batch_size=2,
    )

    assert runner.batch_size == 2
    assert runner.sad_guess_kwargs["dataset_statistics"] == "dataset-statistics"
    assert runner.sad_guess_kwargs["basis_info"] == "basis-info"


def test_run_dispatches_batches_and_swallows_evaluator_failures(monkeypatch, bare_runner, caplog):
    """Check non-legacy path runs batched execution and suppresses evaluator errors when not fail-
    fast."""
    dataset = [FakeSample(1), FakeSample(2)]
    calls = []
    bare_runner.dataset = dataset
    bare_runner.evaluator = SimpleNamespace(
        evaluate=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    monkeypatch.setattr(denoprunner_module, "is_legacy_optimizer", lambda optimizer: False)
    monkeypatch.setattr(
        bare_runner,
        "_run_batched",
        lambda incoming_dataset, progressbar=False: calls.append(incoming_dataset),
    )

    bare_runner.run()

    assert calls == [dataset]


def test_basic_run_samples_dispatches_legacy(monkeypatch, bare_basic_runner):
    """Ensure the shared ``run_samples`` path dispatches to legacy execution when needed."""
    samples = [FakeSample(1)]
    calls = []

    monkeypatch.setattr(denoprunner_engine_module, "is_legacy_optimizer", lambda optimizer: True)
    monkeypatch.setattr(
        bare_basic_runner,
        "_run_legacy",
        lambda incoming_samples: calls.append(incoming_samples) or ["legacy"],
    )

    results = bare_basic_runner.run_samples(samples)

    assert results == ["legacy"]
    assert calls == [[samples[0]]]
    assert samples[0].device == "cpu"
    assert bare_basic_runner.step_count == 0


def test_basic_run_samples_dispatches_lockstep(monkeypatch, bare_basic_runner):
    """Ensure the shared ``run_samples`` path dispatches to batched execution when supported."""
    samples = [FakeSample(1)]

    monkeypatch.setattr(denoprunner_module, "is_legacy_optimizer", lambda optimizer: False)
    monkeypatch.setattr(
        bare_basic_runner,
        "_run_batched",
        lambda incoming_samples, progressbar=False: ["batched", len(incoming_samples)],
    )

    results = bare_basic_runner.run_samples(samples)

    assert results == ["batched", 1]


def test_run_raises_when_evaluator_fails_in_fail_fast_mode(monkeypatch, bare_runner):
    """Ensure evaluator exceptions propagate when ``fail_fast`` is enabled."""
    sample = FakeSample(1)

    class FakeBatch:
        """Single-sample batch mock for the legacy branch."""

        def to_data_list(self):
            return [sample]

    bare_runner.fail_fast = True
    bare_runner.dataloader = [FakeBatch()]
    bare_runner.evaluator = SimpleNamespace(
        evaluate=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    monkeypatch.setattr(denoprunner_module, "is_legacy_optimizer", lambda optimizer: True)
    monkeypatch.setattr(bare_runner, "_run_legacy", lambda incoming_samples: [])

    with pytest.raises(RuntimeError, match="boom"):
        bare_runner.run()


def test_run_raises_when_batch_legacy_fails(monkeypatch, bare_runner):
    """Ensure legacy path exceptions from `_run_legacy` propagate through `run`."""

    class FakeBatch:
        def to_data_list(self):
            return [FakeSample(1)]

    bare_runner.dataloader = [FakeBatch()]
    monkeypatch.setattr(denoprunner_module, "is_legacy_optimizer", lambda optimizer: True)
    monkeypatch.setattr(
        bare_runner,
        "_run_legacy",
        lambda incoming_samples: (_ for _ in ()).throw(RuntimeError("legacy exploded")),
    )

    with pytest.raises(RuntimeError, match="legacy exploded"):
        bare_runner.run()


def test_run_raises_when_batched_run_fails(monkeypatch, bare_runner):
    """Ensure batched-path exceptions from `_run_batched` propagate through `run`."""
    bare_runner.dataset = [FakeSample(1)]
    monkeypatch.setattr(denoprunner_module, "is_legacy_optimizer", lambda optimizer: False)
    monkeypatch.setattr(
        bare_runner,
        "_run_batched",
        lambda incoming_dataset, progressbar=False: (_ for _ in ()).throw(
            RuntimeError("batched exploded")
        ),
    )

    with pytest.raises(RuntimeError, match="batched exploded"):
        bare_runner.run()


def test_run_batch_legacy_finalizes_results(monkeypatch, bare_runner):
    """Validate legacy per-sample execution forwards outputs to evaluator finalization."""
    sample = FakeSample(7)
    calls = {}

    monkeypatch.setattr(denoprunner_module, "ConvergenceCallback", lambda mol: f"callback:{mol}")

    def fake_density_optimization_with_label(**kwargs):
        return (
            {
                "gs_signed_energy_error": 0.1,
                "gs_l2_norm": 0.2,
                "gs_gradient_norm": 0.3,
                "stopping_index": 0,
            },
            "callback:mol-7",
            "energies-label",
            "energy-functional",
        )

    monkeypatch.setattr(
        denoprunner_module,
        "density_optimization_with_label",
        fake_density_optimization_with_label,
    )
    bare_runner.evaluator = SimpleNamespace(
        evaluate=lambda: None,
        finalize_sample=lambda **kwargs: calls.setdefault("finalize", kwargs),
    )

    bare_runner._run_legacy([sample])

    assert calls["finalize"]["sample"] is sample
    assert calls["finalize"]["callback"] == "callback:mol-7"
    assert calls["finalize"]["energies_label"] == "energies-label"
    assert calls["finalize"]["metric_dict"]["gs_gradient_norm"] == 0.3


def test_run_batch_legacy_respects_fail_fast(monkeypatch, bare_runner):
    """Check legacy execution swallows failures by default and raises in fail-fast mode."""
    sample = FakeSample(8)
    monkeypatch.setattr(denoprunner_module, "ConvergenceCallback", lambda mol: f"callback:{mol}")
    monkeypatch.setattr(
        denoprunner_module,
        "density_optimization_with_label",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("legacy failed")),
    )

    bare_runner._run_legacy([sample])

    bare_runner.fail_fast = True
    with pytest.raises(RuntimeError, match="legacy failed"):
        bare_runner._run_legacy([sample])


def test_run_batched_finalizes_each_state(monkeypatch, bare_runner):
    """Ensure batched execution prepares each sample and finalizes each finished state once."""
    samples = [FakeSample(1), FakeSample(2)]
    prepare_calls = []
    finalize_calls = []
    bare_runner.step_count = 0

    # TODO: this is all a little much mocking...
    bare_runner.bridge = SimpleNamespace(
        prepare=lambda sample: SimpleNamespace(energy_functional=f"energy:{sample.mol_id}"),
        evaluate_active=lambda active_samples, active_contexts: [
            SimpleNamespace(energies=f"e:{sample.mol_id}", gradient=f"g:{sample.mol_id}")
            for sample in active_samples
        ],
    )
    bare_runner.func_factory = SimpleNamespace(
        get_energies_label=lambda sample, basis_info: f"label:{sample.mol_id}"
    )

    monkeypatch.setattr(
        bare_runner,
        "_prepare_initial_guess",
        lambda sample: (prepare_calls.append(sample.mol_id), sample)[1],
    )

    monkeypatch.setattr(denoprunner_engine_module, "ConvergenceCallback", lambda mol: f"cb:{mol}")

    monkeypatch.setattr(
        denoprunner_engine_module,
        "build_optimizer_session",
        lambda optimizer, sample: SimpleNamespace(
            step=lambda sample, energy, gradient_vector, callback: SimpleNamespace(
                energy=energy,
                gradient_norm=float(sample.mol_id),
                finished=True,
                converged=True,
            )
        ),
    )

    monkeypatch.setattr(
        denoprunner_module,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: {
            "gs_signed_energy_error": 0.1,
            "gs_l2_norm": 0.2,
            "gs_gradient_norm": 0.3,
            "stopping_index": 0,
        },
    )

    bare_runner.evaluator = SimpleNamespace(
        evaluate=lambda: None,
        finalize_sample=lambda **kwargs: finalize_calls.append(kwargs),
    )

    bare_runner._run_batched(samples)

    assert prepare_calls == [1, 2]
    assert sorted(call["energies_label"] for call in finalize_calls) == [
        "label:1",
        "label:2",
    ]
    assert sorted(call["energy_functional"] for call in finalize_calls) == [
        "energy:1",
        "energy:2",
    ]


# FIXME: this overdoes the mocking  a bit
def test_run_batched_respects_fail_fast_on_finalize(monkeypatch, bare_runner):
    """Ensure finalize-time metric failures propagate when batched fail-fast is enabled."""
    sample = FakeSample(1)
    bare_runner.fail_fast = True
    bare_runner.batch_size = 1
    bare_runner.step_count = 0
    bare_runner.bridge = SimpleNamespace(
        prepare=lambda incoming_sample: SimpleNamespace(energy_functional="energy"),
        evaluate_active=lambda active_samples, active_contexts: [
            SimpleNamespace(energies="energy", gradient="gradient")
        ],
    )
    bare_runner.func_factory = SimpleNamespace(
        get_energies_label=lambda sample, basis_info: "label"
    )

    monkeypatch.setattr(
        bare_runner, "_prepare_initial_guess", lambda incoming_sample: incoming_sample
    )
    monkeypatch.setattr(denoprunner_engine_module, "ConvergenceCallback", lambda mol: "callback")
    monkeypatch.setattr(
        denoprunner_engine_module,
        "build_optimizer_session",
        lambda optimizer, sample: SimpleNamespace(
            step=lambda **kwargs: SimpleNamespace(
                energy="energy",
                gradient_norm=0.1,
                finished=True,
                converged=True,
            )
        ),
    )
    monkeypatch.setattr(
        denoprunner_module,
        "compute_density_optimization_metrics",
        lambda callback, energies_label, convergence_criterion: (_ for _ in ()).throw(
            RuntimeError("finalize failed")
        ),
    )

    with pytest.raises(RuntimeError, match="finalize failed"):
        bare_runner._run_batched([sample])


def test_exchange_samples_refills_then_reports_dataset_exhaustion(monkeypatch, bare_basic_runner):
    """Check the shared refill helper replaces finished slots and reports exhaustion."""
    active_states = [
        denoprunner_module.MoleculeState(
            sample=FakeSample(1),
            callback="cb-1",
            context=SimpleNamespace(non_model_grad="grad-1"),
            optimizer="opt-1",
            finished=True,
        ),
        denoprunner_module.MoleculeState(
            sample=FakeSample(2),
            callback="cb-2",
            context=SimpleNamespace(non_model_grad="grad-2"),
            optimizer="opt-2",
            finished=False,
        ),
    ]
    active_samples = [state.sample for state in active_states]
    active_contexts = [state.context for state in active_states]
    finished_states = []

    monkeypatch.setattr(
        bare_basic_runner,
        "_prepare_sample",
        lambda sample, sample_index: denoprunner_module.MoleculeState(
            sample=sample,
            callback=f"cb:{sample.mol_id}",
            context=SimpleNamespace(non_model_grad=f"grad:{sample.mol_id}"),
            optimizer=f"opt:{sample.mol_id}",
            sample_index=sample_index,
        ),
    )

    exhausted = bare_basic_runner._exchange_samples(
        iter([FakeSample(9)]),
        active_states,
        active_samples,
        active_contexts,
        finished_states,
    )

    assert exhausted is False
    assert len(finished_states) == 1
    assert finished_states[0].sample.mol_id == 1
    assert active_states[0].sample.mol_id == 9
    assert active_states[0].sample.device == "cpu"

    active_states[0].finished = True

    exhausted = bare_basic_runner._exchange_samples(
        iter([]), active_states, active_samples, active_contexts, finished_states
    )

    assert exhausted is True
    assert len(finished_states) == 2


def test_set_model_rebuilds_factory_and_bridge(monkeypatch, bare_basic_runner):
    """Verify ``set_model`` rebuilds the factory and bridge from the new model."""
    calls = {}

    def fake_from_module(model, xc_functional, negative_integrated_density_penalty_weight):
        calls["factory"] = (
            model,
            xc_functional,
            negative_integrated_density_penalty_weight,
        )
        return "factory"

    def fake_bridge(func_factory, max_xc_memory):
        calls["bridge"] = (func_factory, max_xc_memory)
        return "bridge"

    monkeypatch.setattr(
        denoprunner_engine_module.FunctionalFactory,
        "from_module",
        fake_from_module,
    )
    monkeypatch.setattr(
        denoprunner_engine_module,
        "BatchedFunctionalBridge",
        fake_bridge,
    )

    bare_basic_runner.set_model("new-model")

    assert calls["factory"] == ("new-model", "gga_x_pbe", 0.0)
    assert calls["bridge"] == (bare_basic_runner.func_factory, 1)


def test_finalize_run_invokes_evaluator_once(bare_runner):
    """Ensure post-run evaluation is always invoked exactly once."""
    calls = {"evaluate": 0}
    bare_runner.evaluator = SimpleNamespace(evaluate=lambda: calls.__setitem__("evaluate", 1))

    bare_runner._finalize_run()

    assert calls["evaluate"] == 1


def test_finalize_run_swallows_evaluator_failures_when_not_fail_fast(bare_runner):
    """Check evaluator failures are logged but not raised when ``fail_fast`` is false."""
    bare_runner.fail_fast = False
    bare_runner.evaluator = SimpleNamespace(
        evaluate=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    bare_runner._finalize_run()


def test_denoprunner_init_accepts_preexisting_save_dir(monkeypatch, tmp_path: Path):
    """Ensure ``DenopRunner`` init works when the target save dir already exists."""
    existing_dir = tmp_path / "existing_outputs"
    existing_dir.mkdir()
    (existing_dir / "sample_trajectories").mkdir()

    resolved_cfg = OmegaConf.create(
        {
            "denop_settings": {
                "initialization": "minao",
                "normalize_initial_guess": True,
                "convergence_criterion": "last_iter",
                "save_coeff_interval": 100,
                "ks_basis": "sto-3g",
                "max_xc_memory": 1,
                "xc_functional": "gga_x_pbe",
                "negative_integrated_density_penalty_weight": 0.0,
                "finalize_at_least_n": 1,
                "grid_level": 1,
                "grid_prune": "None",
            },
            "negative_integrated_density_penalty_weight": 0.0,
            "enable_grid_operations": True,
            "device": "cpu",
            "model_dtype": "float64",
            "deterministic": False,
            "denop_dir": str(existing_dir),
            "n_molecules": 1,
            "molecule_ids": [0],
            "molecule_choice": "first_n",
            "continue_denop": False,
            "split_file_path": "split.pkl",
            "split": "test",
            "seed": 0,
            "batch_size": 1,
            "density_optimizer": {"_target_": "dummy"},
            "plot_every_n": 0,
            "save_individual_plots": False,
            "enable_grid_plots": False,
            "fail_fast": False,
            "compute_grid_metrics": False,
            "compare_to_ks": False,
            "metrics_grid_level": 1,
            "metrics_grid_prune": "None",
            "swarm_plot_subsample": 1.0,
            "plot_summary": False,
            "compute_final_hessian": False,
            "save_coeff_interval": 100,
            "subplot_list": [],
        }
    )
    model_cfg = OmegaConf.create(
        {"data": {"basis_info": {"_target_": "dummy"}, "transforms": {"_target_": "dummy"}}}
    )

    monkeypatch.setattr(
        denoprunner_module.prepare_config,
        "prepare_config",
        lambda cfg: (resolved_cfg, model_cfg, Path("ckpt.pt"), None),
    )
    monkeypatch.setattr(
        denoprunner_module.prepare_runtime,
        "prepare_logger",
        lambda cfg: existing_dir / "ofdft.log",
    )
    monkeypatch.setattr(
        denoprunner_module.prepare_paths,
        "prepare_paths",
        lambda *args, **kwargs: ({}, ["path-1"], [2]),
    )
    monkeypatch.setattr(
        denoprunner_module.prepare_data,
        "prepare_transforms",
        lambda *args, **kwargs: "transforms",
    )
    monkeypatch.setattr(
        denoprunner_module.prepare_data,
        "prepare_dataset_and_loader",
        lambda *args, **kwargs: (
            Path("stats.pkl"),
            [FakeSample(1)],
            [SimpleNamespace(to_data_list=lambda: [FakeSample(1)])],
        ),
    )
    monkeypatch.setattr(denoprunner_module, "instantiate_model", lambda *args, **kwargs: "model")
    monkeypatch.setattr(denoprunner_module, "instantiate", lambda *args, **kwargs: "optimizer")
    monkeypatch.setattr(denoprunner_module, "DatasetStatistics", lambda *args, **kwargs: "stats")
    monkeypatch.setattr(denoprunner_module, "DenopEvaluator", lambda *args, **kwargs: "evaluator")
    monkeypatch.setattr(denoprunner_module.torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(denoprunner_module.OmegaConf, "save", lambda *args, **kwargs: None)

    def _fake_engine_init(
        self,
        cfg,
        model,
        optimizer,
        dataset_statistics,
        basis_info,
        device,
        batch_size,
        fail_fast,
    ):
        self.denop_config = cfg
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.save_coeff_interval = int(cfg.get("save_coeff_interval", 100))
        self.basis_info = basis_info
        self.func_factory = SimpleNamespace(get_energies_label=lambda sample, basis_info: "label")
        self.bridge = SimpleNamespace()
        self.proj_minao_module = model
        self.max_xc_memory = cfg.get("max_xc_memory")
        self.initialization = cfg.get("initialization")
        self.normalize_initial_guess = cfg.get("normalize_initial_guess")
        self.ks_basis = cfg.get("ks_basis")
        self.sad_guess_kwargs = {}
        self.convergence_criterion = cfg.get("convergence_criterion")
        self.fail_fast = fail_fast
        self.xc_functional = cfg.get("xc_functional")
        self.negative_integrated_density_penalty_weight = cfg.get(
            "negative_integrated_density_penalty_weight", 0.0
        )

    monkeypatch.setattr(denoprunner_module.DenopRunnerEngine, "__init__", _fake_engine_init)

    runner = denoprunner_module.DenopRunner(OmegaConf.create({}))

    assert runner.save_dir == existing_dir
    assert (existing_dir / "sample_trajectories").is_dir()


def test_reset_reinstantiates_optimizer(monkeypatch, bare_runner):
    """Ensure ``reset`` recreates the optimizer and resets lockstep step counter."""
    bare_runner.step_count = 7
    bare_runner.optimizer = "old-optimizer"
    bare_runner.config = {"density_optimizer": {"_target_": "x"}}

    monkeypatch.setattr(denoprunner_module, "instantiate", lambda cfg: "new-optimizer")

    bare_runner.reset()

    assert bare_runner.step_count == 0
    assert bare_runner.optimizer == "new-optimizer"


def test_cleanup_removes_known_artifacts(bare_runner, tmp_path: Path):
    """Ensure cleanup removes configured files and directories from the run output."""
    bare_runner.save_dir = tmp_path

    for file_name in [
        "hparams_resolved.yaml",
        "reduced_denop_data.csv",
        "denop_config.yaml",
        "density_optimization_metrics.yaml",
        "density_optimization_summary.pdf",
    ]:
        (tmp_path / file_name).write_text("x")
    for dir_name in ["sample_trajectories", "denop_log"]:
        (tmp_path / dir_name).mkdir()

    bare_runner.cleanup()

    assert not any(
        (tmp_path / name).exists()
        for name in [
            "sample_trajectories",
            "hparams_resolved.yaml",
            "reduced_denop_data.csv",
            "denop_config.yaml",
            "denop_log",
            "density_optimization_metrics.yaml",
            "density_optimization_summary.pdf",
        ]
    )


def test_run_denop_single_device_calls_seed_and_optional_cleanup(monkeypatch):
    """Verify wrapper seeds RNGs, runs the runner, and performs optional cleanup."""
    calls = []
    runner = SimpleNamespace(
        run=lambda progressbar=False: calls.append(("run", progressbar)),
        cleanup=lambda: calls.append("cleanup"),
    )

    monkeypatch.setattr(
        denoprunner_module, "seed_all_rngs", lambda seed: calls.append(("seed", seed))
    )
    monkeypatch.setattr(denoprunner_module, "DenopRunner", lambda cfg: runner)

    denoprunner_module.run_denop_single_device({"seed": 3}, cleanup=True)

    assert calls == [("seed", 3), ("run", False), "cleanup"]


def _compose_two_molecule_denop_cfg(
    generate_run_path: Path,
    output_dir: Path,
    *,
    batch_size: int,
) -> OmegaConf:
    """Create a small two-molecule density-optimization config for parity testing."""
    output_dir.mkdir(exist_ok=True)
    with initialize(version_base=None, config_path="../../../configs/ofdft"):
        cfg = compose(
            config_name="density_optimization.yaml",
            overrides=[
                f"run_path={generate_run_path}",
                f"log_file={output_dir / 'ofdft.log'}",
                f"config_file={output_dir / 'config.yaml'}",
                f"denop_dir={output_dir}",
                f"split_file_path={Path(os.getenv('DFT_DATA')) / 'TwoElectron' / 'split.pkl'}",
                "n_molecules=2",
                "molecule_ids=[0,1]",
                "molecule_choice=first_n",
                "device=cpu",
                "num_threads=1",
                f"batch_size={batch_size}",
                "model_dtype=float32",
                "plot_every_n=0",
                "plot_summary=False",
                "compute_grid_metrics=False",
                "compare_to_ks=False",
                "continue_denop=False",
                "density_optimizer.max_cycle=2",
                "denop_settings.initialization=minao",
                "+data.datamodule.dataloader_kwargs.batch_size=2",
                "+data.datamodule.dataloader_kwargs.num_workers=0",
                "+data.datamodule.dataloader_kwargs.pin_memory=False",
                "+data.datamodule.dataloader_kwargs.list_keys=['overlap_matrix','transform_grad_contracted','transformation_matrix','inv_transformation_matrix','coulomb_matrix']",
            ],
        )
    GlobalHydra.instance().clear()
    return cfg


def _run_legacy_density_optimization(cfg: OmegaConf) -> None:
    """Run the pre-denop-runner density-optimization workflow for one config."""
    log_file = Path(cfg.get("log_file"))
    run_path = parse_run_path(cfg.run_path)
    denop_cfg = cfg.get("denop_settings")
    optimizer = instantiate(cfg.get("density_optimizer"))
    model_config = OmegaConf.load(run_path / "hparams.yaml")
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
        compare_to_ks=cfg.get("compare_to_ks"),
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


def _read_reduced_denop_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    """Read reduced denop rows indexed by molecule id."""
    with open(csv_path, newline="") as handle:
        return {str(row["mol_id"]): row for row in csv.DictReader(handle)}


def _load_saved_samples(sample_dir: Path) -> dict[str, object]:
    """Load saved denop samples indexed by molecule id."""
    sample_map = {}
    for sample_path in sorted(sample_dir.glob("sample_*.pt")):
        sample = torch.load(sample_path, map_location="cpu", weights_only=False)
        sample_map[str(sample.mol_id)] = sample
    return sample_map


@pytest.mark.skipif(
    not (Path(os.getenv("DFT_DATA", "")) / "TwoElectron").exists(),
    reason="two_electron dataset not available",
)
def test_run_denop_single_device_integration_two_molecules(generate_run_path, tmp_path: Path):
    """Smoke-test two-molecule denop run and assert core output artifacts are produced."""
    with initialize(version_base=None, config_path="../../../configs/ofdft"):
        cfg = compose(
            config_name="density_optimization.yaml",
            overrides=[
                f"run_path={generate_run_path}",
                f"log_file={tmp_path / 'ofdft.log'}",
                f"config_file={tmp_path / 'config.yaml'}",
                f"denop_dir={tmp_path}",
                f"split_file_path={Path(os.getenv('DFT_DATA')) / 'TwoElectron' / 'split.pkl'}",
                "n_molecules=2",
                "molecule_ids=[0,1]",
                "molecule_choice=first_n",
                "device=cpu",
                "num_threads=1",
                "model_dtype=float64",
                "plot_every_n=0",
                "plot_summary=False",
                "continue_denop=False",
                "density_optimizer.max_cycle=2",
                "denop_settings.initialization=minao",
                "+data.datamodule.dataloader_kwargs.batch_size=2",
                "+data.datamodule.dataloader_kwargs.num_workers=0",
                "+data.datamodule.dataloader_kwargs.pin_memory=False",
                "+data.datamodule.dataloader_kwargs.list_keys=['overlap_matrix','transform_grad_contracted','transformation_matrix','inv_transformation_matrix','coulomb_matrix']",
            ],
        )
    try:
        denoprunner_module.run_denop_single_device(cfg, cleanup=False)
    finally:
        GlobalHydra.instance().clear()

    sample_dir = tmp_path / "sample_trajectories"
    saved_samples = sorted(sample_dir.glob("sample_*.pt"))

    assert (sample_dir / "basis_info.pt").exists()
    assert len(saved_samples) == 2
    assert (tmp_path / "reduced_denop_data.csv").exists()


# @pytest.mark.slow
@pytest.mark.skipif(
    not (Path(os.getenv("DFT_DATA", "")) / "TwoElectron").exists(),
    reason="two_electron dataset not available",
)
def test_denoprunner_matches_legacy_density_optimization_float32(
    generate_run_path, tmp_path: Path
):
    """Check new denop matches the legacy workflow for two molecules in float32."""
    legacy_dir = tmp_path / "legacy_outputs"
    runner_dir = tmp_path / "runner_outputs"
    legacy_cfg = _compose_two_molecule_denop_cfg(generate_run_path, legacy_dir, batch_size=0)
    runner_cfg = _compose_two_molecule_denop_cfg(generate_run_path, runner_dir, batch_size=2)

    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        _run_legacy_density_optimization(legacy_cfg)
        denoprunner_module.run_denop_single_device(runner_cfg, cleanup=False)
    finally:
        torch.set_float32_matmul_precision(previous_precision)

    legacy_rows = _read_reduced_denop_csv(legacy_dir / "reduced_denop_data.csv")
    runner_rows = _read_reduced_denop_csv(runner_dir / "reduced_denop_data.csv")
    assert len(legacy_rows) == len(runner_rows) == 2
    assert legacy_rows.keys() == runner_rows.keys()

    numeric_fields = [
        "ground_state_energy_total",
        "final_energy_total",
        "final_density_l2_norm",
        "final_gradient_norm",
        "total_energy_error_per_atom",
        "density_l2_error_per_electron",
    ]
    for mol_id in legacy_rows:
        legacy_row = legacy_rows[mol_id]
        runner_row = runner_rows[mol_id]
        assert legacy_row["sum_formula"] == runner_row["sum_formula"]
        assert int(legacy_row["iterations"]) == int(runner_row["iterations"])
        for field in numeric_fields:
            assert float(runner_row[field]) == pytest.approx(
                float(legacy_row[field]), rel=5e-3, abs=1e-3
            )

    legacy_samples = _load_saved_samples(legacy_dir / "sample_trajectories")
    runner_samples = _load_saved_samples(runner_dir / "sample_trajectories")
    assert len(legacy_samples) == len(runner_samples) == 2
    assert legacy_samples.keys() == runner_samples.keys()

    for mol_id in legacy_samples:
        legacy_sample = legacy_samples[mol_id]
        runner_sample = runner_samples[mol_id]
        assert legacy_sample.stopping_index == runner_sample.stopping_index
        assert (
            legacy_sample.trajectory_energy_total.shape
            == runner_sample.trajectory_energy_total.shape
        )
        assert legacy_sample.trajectory_l2_norm.shape == runner_sample.trajectory_l2_norm.shape
        assert (
            legacy_sample.trajectory_gradient_norm.shape
            == runner_sample.trajectory_gradient_norm.shape
        )
        assert torch.allclose(
            legacy_sample.coeffs,
            runner_sample.coeffs,
            rtol=1e-4,
            atol=1e-4,
        )
        assert torch.allclose(
            legacy_sample.trajectory_energy_total,
            runner_sample.trajectory_energy_total,
            rtol=5e-5,
            atol=1e-5,
        )
        assert torch.allclose(
            legacy_sample.trajectory_l2_norm,
            runner_sample.trajectory_l2_norm,
            rtol=5e-5,
            atol=1e-5,
        )
        assert torch.allclose(
            legacy_sample.trajectory_gradient_norm,
            runner_sample.trajectory_gradient_norm,
            rtol=5e-5,
            atol=1e-5,
        )


@pytest.mark.skipif(
    not (Path(os.getenv("DFT_DATA", "")) / "TwoElectron").exists(),
    reason="two_electron dataset not available",
)
def test_denoprunner_engine_integration_two_molecules_legacy(generate_run_path, tmp_path: Path):
    """Smoke-test ``DenopRunnerEngine`` on two molecules using the lockstep path."""
    with initialize(version_base=None, config_path="../../../configs/ofdft"):
        cfg = compose(
            config_name="density_optimization.yaml",
            overrides=[
                f"run_path={generate_run_path}",
                f"log_file={tmp_path / 'ofdft-basic.log'}",
                f"config_file={tmp_path / 'basic_outputs' / 'config.yaml'}",
                f"denop_dir={tmp_path / 'basic_outputs'}",
                f"split_file_path={Path(os.getenv('DFT_DATA')) / 'TwoElectron' / 'split.pkl'}",
                "n_molecules=2",
                "molecule_ids=[0,1]",
                "molecule_choice=first_n",
                "device=cpu",
                "num_threads=1",
                "model_dtype=float64",
                "plot_every_n=0",
                "plot_summary=False",
                "continue_denop=False",
                "density_optimizer=gradient_descent",
                "density_optimizer.max_cycle=2",
                "denop_settings.initialization=minao",
                "+data.datamodule.dataloader_kwargs.batch_size=2",
                "+data.datamodule.dataloader_kwargs.num_workers=0",
                "+data.datamodule.dataloader_kwargs.pin_memory=False",
                "+data.datamodule.dataloader_kwargs.list_keys=['overlap_matrix','transform_grad_contracted','transformation_matrix','inv_transformation_matrix','coulomb_matrix']",
            ],
        )
    try:
        runtime_runner = denoprunner_module.DenopRunner(cfg)
        denop_cfg = getattr(runtime_runner, "denop_config", None)
        basic_cfg = OmegaConf.create(OmegaConf.to_container(denop_cfg, resolve=False))
        basic_cfg.negative_integrated_density_penalty_weight = runtime_runner.config.get(
            "negative_integrated_density_penalty_weight", 0.0
        )
        basic_cfg.save_coeff_interval = runtime_runner.config.get("save_coeff_interval", 100)
        basic_runner = denoprunner_engine_module.DenopRunnerEngine(
            cfg=basic_cfg,
            model=runtime_runner.proj_minao_module,
            optimizer=runtime_runner.optimizer,
            dataset_statistics=denoprunner_module.DatasetStatistics(
                runtime_runner.dataset_statistics_path
            ),
            basis_info=runtime_runner.basis_info,
            device=runtime_runner.device,
            batch_size=runtime_runner.batch_size,
        )

        samples = next(iter(runtime_runner.dataloader)).to_data_list()
        results = basic_runner.run_samples(samples)
    finally:
        GlobalHydra.instance().clear()

    assert len(results) == 2
    assert all(state.finished for state in results)
    assert all(state.callback is not None for state in results)
    assert all(state.context is not None for state in results)
    assert all(state.optimizer is not None for state in results)
    assert all(hasattr(state.sample, "coeffs") for state in results)
