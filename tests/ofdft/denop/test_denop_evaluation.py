import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mldft.ofdft.denop import evaluation as evaluation_module


class FakeFinalizeSample:
    """Minimal sample mock carrying fields used by evaluator finalization."""

    def __init__(self, mol_id=1):
        self.mol_id = mol_id
        self.mol = f"mol-{mol_id}"
        self.transformation_matrix = torch.eye(2, dtype=torch.float64)
        self.inv_transformation_matrix = torch.eye(2, dtype=torch.float64)
        self.ground_state_energy_total = 4.0
        self.trajectory_energy_total = torch.tensor([4.2, 4.1], dtype=torch.float64)
        self.trajectory_l2_norm = torch.tensor([0.5, 0.25], dtype=torch.float64)
        self.trajectory_gradient_norm = torch.tensor([0.2, 0.1], dtype=torch.float64)
        self.stopping_index = 1
        self.n_atom = 2
        self.n_electron = 4
        self.overlap_matrix = "overlap"
        self.coulomb_matrix = "coulomb"
        self.nuclear_attraction_vector = "nuclear"
        self.ao = "ao"
        self.deleted_items = []
        self.added_items = []

    def add_item(self, key, value, representation):
        self.added_items.append((key, value, representation))
        setattr(self, key, value)

    def delete_item(self, key):
        self.deleted_items.append(key)
        delattr(self, key)

    def clone(self):
        clone = FakeFinalizeSample(self.mol_id)
        clone.ground_state_energy_total = self.ground_state_energy_total
        return clone


def test_denop_evaluator_saves_metrics_and_optional_summary(monkeypatch, tmp_path: Path):
    """Verify evaluator computes run data, saves metrics, and emits summary PDF when enabled."""
    calls = {}

    monkeypatch.setattr(
        evaluation_module,
        "get_runwise_density_optimization_data",
        lambda **kwargs: calls.setdefault("runwise", kwargs) or {"metric": 1},
    )
    monkeypatch.setattr(
        evaluation_module,
        "save_density_optimization_metrics",
        lambda **kwargs: calls.setdefault("metrics", kwargs),
    )
    monkeypatch.setattr(
        evaluation_module,
        "density_optimization_summary_pdf_plot",
        lambda **kwargs: calls.setdefault("summary", kwargs),
    )

    evaluator = evaluation_module.DenopEvaluator(
        save_dir=tmp_path,
        compute_grid_metrics=False,
        compare_to_ks=True,
        metrics_grid_level=4,
        metrics_grid_prune="treutler_prune",
        swarm_plot_subsample=0.25,
        plot_summary=True,
    )
    evaluator.evaluate()

    assert calls["runwise"]["sample_dir"] == tmp_path / "sample_trajectories"
    assert calls["runwise"]["compute_grid_metrics"] is False
    assert calls["runwise"]["compare_to_ks"] is True
    assert calls["runwise"]["metrics_grid_level"] == 4
    assert calls["runwise"]["metrics_grid_prune"] == "treutler_prune"
    assert calls["metrics"]["output_path"] == tmp_path / "density_optimization_metrics.yaml"
    assert calls["summary"]["out_pdf_path"] == tmp_path / "density_optimization_summary.pdf"
    assert calls["summary"]["subsample"] == 0.25


def test_denop_evaluator_skips_summary_when_disabled(monkeypatch, tmp_path: Path):
    """Ensure summary plotting is skipped when ``plot_summary=False``."""
    monkeypatch.setattr(
        evaluation_module, "get_runwise_density_optimization_data", lambda **kwargs: {"metric": 1}
    )
    monkeypatch.setattr(
        evaluation_module, "save_density_optimization_metrics", lambda **kwargs: None
    )
    monkeypatch.setattr(
        evaluation_module,
        "density_optimization_summary_pdf_plot",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("summary should not run")),
    )

    evaluation_module.DenopEvaluator(save_dir=tmp_path, plot_summary=False).evaluate()


def test_denop_plotter_filters_functions_in_init(monkeypatch, tmp_path: Path):
    """Check constructor delegates plotting-function filtering with config flags."""
    recorded = {}
    plotting_functions = [object()]
    monkeypatch.setattr(
        evaluation_module,
        "filter_plotting_functions",
        lambda plotting_functions, enable_grid_operations, compute_final_hessian: recorded.setdefault(
            "filtered",
            (
                list(plotting_functions),
                enable_grid_operations,
                compute_final_hessian,
            ),
        ),
    )

    plotter = evaluation_module.DenopPlotter(
        save_dir=tmp_path,
        plotting_functions=plotting_functions,
        enable_grid_plots=True,
        basis_info="basis",
        save_individual_plots=True,
        enable_grid_operations=True,
        compute_final_hessian=True,
    )

    assert plotter.plotting_functions == (plotting_functions, True, True)
    assert recorded["filtered"][1:] == (True, True)


def test_denop_plotter_plot_sets_start_method_and_maps_paths(monkeypatch, tmp_path: Path):
    """Validate plot orchestration sets spawn start-method and processes new paths once."""
    plot_dir = tmp_path / "plot_trajectories"
    plot_dir.mkdir()
    (plot_dir / "sample_0.pt").write_text("x")
    (plot_dir / "sample_1.pt").write_text("x")

    calls = {"plotted": [], "unlinked": []}

    class FakeContext:
        """Context stub that simulates an unset multiprocessing start method."""

        @staticmethod
        def get_start_method(allow_none=True):
            return None

    class FakePDF:
        """PDF context-manager stub used by the plot loop."""

        def __init__(self, out_pdf_path, individual_plot_directory):
            calls["pdf"] = (out_pdf_path, individual_plot_directory)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(evaluation_module.mp, "get_context", lambda: FakeContext())
    monkeypatch.setattr(
        evaluation_module.mp,
        "set_start_method",
        lambda method, force: calls.setdefault("start", (method, force)),
    )
    monkeypatch.setattr(evaluation_module, "HierarchicalPlotPDF", FakePDF)
    monkeypatch.setattr(
        evaluation_module,
        "filter_plotting_functions",
        lambda plotting_functions, enable_grid_operations, compute_final_hessian: plotting_functions,
    )
    monkeypatch.setattr(
        evaluation_module.DenopPlotter,
        "_plot_single_run",
        lambda self, plot_path, plotpdf: calls["plotted"].append(plot_path),
    )
    monkeypatch.setattr(
        evaluation_module.os, "unlink", lambda path: calls["unlinked"].append(path)
    )

    plotter = evaluation_module.DenopPlotter(
        save_dir=tmp_path,
        plotting_functions=["plot"],
        enable_grid_plots=False,
        basis_info="basis",
    )
    plotter.plot()
    plotter.plot()

    assert calls["start"] == ("spawn", True)
    assert sorted(calls["plotted"]) == sorted([plot_dir / "sample_0.pt", plot_dir / "sample_1.pt"])
    assert sorted(calls["unlinked"]) == sorted(
        [plot_dir / "sample_0.pt", plot_dir / "sample_1.pt"]
    )
    assert plotter._processed_plot_paths == {plot_dir / "sample_0.pt", plot_dir / "sample_1.pt"}


def test_denop_plotter_plot_single_run_loads_sample_and_plots(monkeypatch, tmp_path: Path):
    """Ensure single-run plotting loads sample data, invokes plotter, and writes PDF page."""
    plot_path = tmp_path / "sample_0.pt"
    plot_path.write_text("placeholder")
    added_items = []
    plot_calls = {}

    class FakeSample:
        """Sample stub exposing only attributes consumed by ``_plot_single_run``."""

        def __init__(self):
            self.ground_state_coeffs = torch.zeros(3, dtype=torch.float64)
            self.basis_function_ind = torch.tensor([0, 2], dtype=torch.long)
            self.stopping_index = 4
            self.mol_id = 17
            self.final_hessian_eigenvalues = torch.tensor([1.0, 2.0], dtype=torch.float64)

        def add_item(self, key, value, representation):
            added_items.append((key, value.shape, representation))
            setattr(self, key, value)

    class FakePlotPDF:
        """PDF backend stub that records saved figure labels."""

        def savefig(self, name):
            plot_calls["savefig"] = name

    monkeypatch.setattr(
        evaluation_module,
        "filter_plotting_functions",
        lambda plotting_functions, enable_grid_operations, compute_final_hessian: plotting_functions,
    )
    monkeypatch.setattr(evaluation_module.torch, "load", lambda *args, **kwargs: FakeSample())

    def fake_plot_density_optimization(*args, **kwargs):
        plot_calls["plot"] = (args, kwargs)
        return "figure"

    monkeypatch.setattr(
        evaluation_module,
        "plot_density_optimization",
        fake_plot_density_optimization,
    )
    monkeypatch.setattr(
        evaluation_module.plt,
        "close",
        lambda fig: plot_calls.setdefault("closed", fig),
    )

    plotter = evaluation_module.DenopPlotter(
        save_dir=tmp_path,
        plotting_functions=["plot"],
        enable_grid_plots=True,
        basis_info=SimpleNamespace(l_per_basis_func=torch.tensor([3, 4, 5])),
        save_individual_plots=True,
        enable_grid_operations=True,
        compute_final_hessian=True,
    )
    plotter._plot_single_run(plot_path, FakePlotPDF())

    assert added_items[0][0] == "transformation_matrix"
    assert added_items[1][0] == "inv_transformation_matrix"
    assert plot_calls["plot"][1]["basis_l"].tolist() == [3, 5]
    assert torch.equal(plot_calls["plot"][1]["final_hessian"], torch.tensor([1.0, 2.0]))
    assert plot_calls["savefig"] == "17"
    assert plot_calls["closed"] == "figure"


def test_denop_evaluator_finalize_sample_transforms_saves_hessian_and_plots(
    monkeypatch, tmp_path: Path
):
    """Verify finalize_sample performs cleanup, optional Hessian save, persistence, and
    plotting."""
    sample = FakeFinalizeSample(5)
    (tmp_path / "sample_trajectories").mkdir()
    append_calls = []
    trajectory_calls = []
    plotter = SimpleNamespace(plot=lambda: None)
    evaluator = evaluation_module.DenopEvaluator(
        save_dir=tmp_path,
        plot_summary=False,
        save_coeff_interval=2,
        compute_final_hessian=True,
        plotter=plotter,
        plot_every_n=1,
    )

    monkeypatch.setattr(
        evaluation_module,
        "ApplyBasisTransformation",
        lambda: (
            lambda incoming_sample, transformation_matrix, inv_transformation_matrix, invert: incoming_sample
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "add_density_optimization_trajectories_to_sample",
        lambda incoming_sample, callback, energies_label, save_coeff_interval: (
            trajectory_calls.append(save_coeff_interval) or incoming_sample
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "_append_reduced_csv_row",
        lambda incoming_sample: append_calls.append(incoming_sample),
    )
    monkeypatch.setattr(
        evaluation_module.torch, "save", lambda obj, path: append_calls.append(path)
    )
    monkeypatch.setattr(
        evaluation_module,
        "mole_to_sum_formula",
        lambda mol, use_subscript: f"sum-{mol}",
    )

    evaluator.finalize_sample(
        sample=sample,
        callback="callback",
        energies_label="energies-label",
        metric_dict={
            "gs_signed_energy_error": 0.001,
            "gs_l2_norm": 0.25,
            "gs_gradient_norm": 1e-3,
            "stopping_index": 1,
        },
        energy_functional=lambda incoming_sample, compute_hessian: (
            None,
            None,
            torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64),
        ),
    )

    added_item_keys = [item[0] for item in sample.added_items]
    assert "final_hessian_eigenvalues" in added_item_keys
    assert "final_untransformed_hessian_eigenvalues" in added_item_keys
    assert sample.deleted_items[:4] == [
        "overlap_matrix",
        "coulomb_matrix",
        "nuclear_attraction_vector",
        "ao",
    ]
    assert "transformation_matrix" in sample.deleted_items
    assert "inv_transformation_matrix" in sample.deleted_items
    assert append_calls[0] is sample
    assert append_calls[1] == tmp_path / "sample_trajectories" / "sample_5.pt"
    assert append_calls[2] == tmp_path / "plot_trajectories" / "sample_5.pt"
    assert trajectory_calls == [1, 2]


def test_denop_evaluator_evaluate_triggers_plotter(monkeypatch, tmp_path: Path):
    """Ensure evaluate delegates to plotter when configured for staged plotting."""
    calls = []
    monkeypatch.setattr(
        evaluation_module, "get_runwise_density_optimization_data", lambda **kwargs: {"metric": 1}
    )
    monkeypatch.setattr(
        evaluation_module, "save_density_optimization_metrics", lambda **kwargs: None
    )
    plotter = SimpleNamespace(plot=lambda: calls.append("plot"))
    evaluator = evaluation_module.DenopEvaluator(
        save_dir=tmp_path, plot_summary=False, plotter=plotter, plot_every_n=1
    )

    evaluator.evaluate()

    assert calls == ["plot"]


def test_denop_plotter_plot_returns_when_plot_directory_missing(monkeypatch, tmp_path: Path):
    """Ensure plotting exits cleanly when no staged plot trajectory directory exists."""
    calls = {"set_start_method": 0}

    class FakeContext:
        @staticmethod
        def get_start_method(allow_none=True):
            return "spawn"

    monkeypatch.setattr(evaluation_module.mp, "get_context", lambda: FakeContext())
    monkeypatch.setattr(
        evaluation_module.mp,
        "set_start_method",
        lambda method, force: calls.__setitem__("set_start_method", 1),
    )
    monkeypatch.setattr(
        evaluation_module,
        "filter_plotting_functions",
        lambda plotting_functions, enable_grid_operations, compute_final_hessian: plotting_functions,
    )

    plotter = evaluation_module.DenopPlotter(
        save_dir=tmp_path,
        plotting_functions=["plot"],
        enable_grid_plots=False,
        basis_info=SimpleNamespace(l_per_basis_func=torch.tensor([0], dtype=torch.long)),
    )

    plotter.plot()

    assert calls["set_start_method"] == 0
    assert plotter._processed_plot_paths == set()


def test_denop_plotter_plot_single_run_raises_without_plot_functions(monkeypatch, tmp_path: Path):
    """Check the explicit guard when no plotting functions survive filtering."""

    class FakeSample:
        def __init__(self):
            self.ground_state_coeffs = torch.zeros(2, dtype=torch.float64)
            self.basis_function_ind = torch.tensor([0, 1], dtype=torch.long)
            self.stopping_index = 1
            self.mol_id = 1

        def add_item(self, key, value, representation):
            setattr(self, key, value)

    class FakePlotPDF:
        def savefig(self, name):
            return None

    monkeypatch.setattr(
        evaluation_module,
        "filter_plotting_functions",
        lambda plotting_functions, enable_grid_operations, compute_final_hessian: [],
    )
    monkeypatch.setattr(evaluation_module.torch, "load", lambda *args, **kwargs: FakeSample())
    monkeypatch.setattr(
        evaluation_module,
        "plot_density_optimization",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plot_density_optimization should not be called")
        ),
    )

    plotter = evaluation_module.DenopPlotter(
        save_dir=tmp_path,
        plotting_functions=[],
        enable_grid_plots=False,
        basis_info=SimpleNamespace(l_per_basis_func=torch.tensor([0, 1], dtype=torch.long)),
    )

    with pytest.raises(ValueError, match="No plotting functions available"):
        plotter._plot_single_run(tmp_path / "sample_0.pt", FakePlotPDF())


def test_denop_evaluator_finalize_sample_handles_missing_optional_fields(
    monkeypatch, tmp_path: Path
):
    """Finalize should work when optional matrix/ao fields are already absent."""
    sample = FakeFinalizeSample(6)
    for key in ["overlap_matrix", "coulomb_matrix", "nuclear_attraction_vector", "ao"]:
        delattr(sample, key)
    (tmp_path / "sample_trajectories").mkdir()
    saved_paths = []
    appended = []

    evaluator = evaluation_module.DenopEvaluator(
        save_dir=tmp_path,
        plot_summary=False,
        compute_final_hessian=False,
        plotter=None,
        plot_every_n=0,
    )
    monkeypatch.setattr(
        evaluation_module,
        "ApplyBasisTransformation",
        lambda: (
            lambda incoming_sample, transformation_matrix, inv_transformation_matrix, invert: incoming_sample
        ),
    )
    monkeypatch.setattr(
        evaluation_module,
        "add_density_optimization_trajectories_to_sample",
        lambda incoming_sample, callback, energies_label, save_coeff_interval: incoming_sample,
    )
    monkeypatch.setattr(
        evaluator,
        "_append_reduced_csv_row",
        lambda incoming_sample: appended.append(incoming_sample),
    )
    monkeypatch.setattr(
        evaluation_module.torch, "save", lambda obj, path: saved_paths.append(path)
    )
    monkeypatch.setattr(evaluation_module, "mole_to_sum_formula", lambda mol, use_subscript: "X")

    evaluator.finalize_sample(
        sample=sample,
        callback="callback",
        energies_label="energies-label",
        metric_dict={
            "gs_signed_energy_error": 0.0,
            "gs_l2_norm": 0.0,
            "gs_gradient_norm": 0.0,
            "stopping_index": 0,
        },
        energy_functional=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("energy_functional should not be called when hessian is disabled")
        ),
    )

    assert appended == [sample]
    assert saved_paths == [tmp_path / "sample_trajectories" / "sample_6.pt"]
    assert "transformation_matrix" in sample.deleted_items
    assert "inv_transformation_matrix" in sample.deleted_items
    assert not any(
        key in sample.deleted_items
        for key in ["overlap_matrix", "coulomb_matrix", "nuclear_attraction_vector", "ao"]
    )


def test_denop_evaluator_append_reduced_csv_row_writes_header_once(monkeypatch, tmp_path: Path):
    """Check reduced CSV writer emits one header and appends rows across calls."""
    sample = FakeFinalizeSample(9)
    evaluator = evaluation_module.DenopEvaluator(save_dir=tmp_path, plot_summary=False)

    monkeypatch.setattr(
        evaluation_module,
        "mole_to_sum_formula",
        lambda mol, use_subscript: "H2",
    )

    evaluator._append_reduced_csv_row(sample)
    evaluator._append_reduced_csv_row(sample)

    with open(evaluator.csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["mol_id"] == "9"
    assert rows[0]["sum_formula"] == "H2"
