"""Tests for analytical force evaluation tasks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

import mldft.ofdft.evaluation.analytical_forces as eval_ng
import mldft.ofdft.evaluation.analytical_forces_at_gs as eval_gs


class DummyMol:
    def __init__(self, natm: int):
        self.natm = natm


class DummyModule(torch.nn.Module):
    def forward(self, *_args, **_kwargs):
        return torch.tensor(0.0, dtype=torch.float64)


def make_sample(
    *,
    natm: int = 2,
    mol_id: int = 7,
    pos: torch.Tensor | None = None,
    ground_state_coeffs: torch.Tensor | None = None,
    coeffs: torch.Tensor | None = None,
    trajectory_coeffs: torch.Tensor | None = None,
):
    sample = SimpleNamespace()
    sample.mol = DummyMol(natm)
    sample.n_atom = natm
    sample.mol_id = mol_id
    sample.atomic_numbers = torch.tensor([6] * natm)

    if pos is not None:
        sample.pos = pos
    if ground_state_coeffs is not None:
        sample.ground_state_coeffs = ground_state_coeffs
    if coeffs is not None:
        sample.coeffs = coeffs
    if trajectory_coeffs is not None:
        sample.trajectory_coeffs = trajectory_coeffs

    return sample


def write_reference_zarr(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zarr.ZipStore(str(path), mode="w") as store:
        root = zarr.open_group(store=store)
        for key, value in arrays.items():
            root.create_dataset(key, data=value)


def test_make_denop_transforms_prepares_density_optimization_transforms(monkeypatch):
    class DummyPrepareForDensityOptimization:
        def __init__(self, basis_info, add_grid, grid_level, grid_prune):
            self.basis_info = basis_info
            self.add_grid = add_grid
            self.grid_level = grid_level
            self.grid_prune = grid_prune

    stale_preparation = DummyPrepareForDensityOptimization(
        "old-basis",
        add_grid=True,
        grid_level=1,
        grid_prune="old_prune",
    )
    existing_transform = object()
    transforms = SimpleNamespace(
        pre_transforms=[stale_preparation, existing_transform],
        add_transformation_matrix=False,
        use_cached_data=True,
    )
    model_config = SimpleNamespace(
        data=SimpleNamespace(
            target_key="kin_plus_xc",
            transforms="transforms-config",
        )
    )

    monkeypatch.setattr(
        eval_gs,
        "PrepareForDensityOptimization",
        DummyPrepareForDensityOptimization,
    )
    monkeypatch.setattr(eval_gs, "instantiate", lambda config: transforms)
    monkeypatch.setattr(eval_gs, "requires_grid", lambda target_key, penalty: False)

    denop_transforms = eval_gs._make_denop_transforms(
        model_config=model_config,
        basis_info="basis-info",
        grid_level=3,
        grid_prune="nwchem_prune",
        negative_integrated_density_penalty_weight=0.0,
    )

    assert denop_transforms is transforms
    assert denop_transforms.pre_transforms[1:] == [existing_transform]
    prepared = denop_transforms.pre_transforms[0]
    assert isinstance(prepared, DummyPrepareForDensityOptimization)
    assert prepared.basis_info == "basis-info"
    assert prepared.add_grid is False
    assert prepared.grid_level == 3
    assert prepared.grid_prune == "nwchem_prune"
    assert denop_transforms.add_transformation_matrix is True
    assert denop_transforms.use_cached_data is False


def test_gradient_sample_converts_tensor_geometry_before_building_molecule(monkeypatch):
    sample = make_sample(natm=2)
    sample.pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    built = {}

    class DummySampleGenerator:
        def get_sample_from_mol(self, mol):
            assert mol == "mol"
            return "gradient-sample"

    def fake_build_molecule_np(charges, positions):
        built["charges"] = charges
        built["positions"] = positions
        assert isinstance(charges, np.ndarray)
        assert isinstance(positions, np.ndarray)
        assert charges.dtype.kind in {"i", "u"}
        assert positions.dtype == np.float64
        assert positions.shape == (2, 3)
        return "mol"

    monkeypatch.setattr(eval_ng, "build_molecule_np", fake_build_molecule_np)

    gradient_sample = eval_ng._gradient_sample(sample, DummySampleGenerator())

    assert gradient_sample == "gradient-sample"
    np.testing.assert_array_equal(built["charges"], np.array([6, 6]))
    np.testing.assert_allclose(built["positions"], sample.pos.numpy())


def test_reference_path_for_label_path_uses_sibling_nuclear_gradients_dir():
    label_path = Path("/data/QM9_perturbed_fock/labels/00000042.zarr.zip")

    reference_path = eval_ng.reference_path_for_label_path(
        label_path,
        nuclear_gradients_dirname="nuclear_gradients",
    )

    assert reference_path == Path("/data/QM9_perturbed_fock/nuclear_gradients/00000042.zarr.zip")


def test_reference_path_for_label_path_can_use_custom_reference_dirname():
    label_path = Path("/data/DATASET/labels/mol_1.zarr.zip")

    reference_path = eval_ng.reference_path_for_label_path(
        label_path,
        nuclear_gradients_dirname="nuclear_gradients_apbe",
    )

    assert reference_path == Path("/data/DATASET/nuclear_gradients_apbe/mol_1.zarr.zip")


def test_load_reference_gradients_reads_total_and_model_keys(tmp_path):
    reference_path = tmp_path / "00000001.zarr.zip"
    total = np.array([1.0, -2.0, 3.0, 4.0, -5.0, 6.0])
    model = np.array([-0.5, 0.25, 0.75, -1.0, 2.0, -3.0])
    write_reference_zarr(
        reference_path,
        {
            "nuc_grad_tot": total,
            "nuc_grad_kin_plus_xc": model,
        },
    )

    loaded_total, loaded_model = eval_ng.load_reference_gradients(
        reference_path,
        total_key="nuc_grad_tot",
        model_key="nuc_grad_kin_plus_xc",
    )

    np.testing.assert_allclose(loaded_total, total)
    np.testing.assert_allclose(loaded_model, model)


def test_load_reference_gradients_raises_clear_error_for_missing_key(tmp_path):
    reference_path = tmp_path / "00000001.zarr.zip"
    write_reference_zarr(reference_path, {"nuc_grad_tot": np.zeros(6)})

    with pytest.raises(KeyError, match="nuc_grad_kin_plus_xc"):
        eval_ng.load_reference_gradients(
            reference_path,
            total_key="nuc_grad_tot",
            model_key="nuc_grad_kin_plus_xc",
        )


def test_resolve_reference_model_key_uses_explicit_key_without_inspecting_config():
    model_config = SimpleNamespace(data=SimpleNamespace(target_key="kin_plus_xc"))

    key = eval_ng.resolve_reference_model_key(
        model_config=model_config,
        reference_model_key="nuc_grad_custom",
    )

    assert key == "nuc_grad_custom"


@pytest.mark.parametrize(
    ("target_key", "expected_reference_key"),
    [
        ("kin", "nuc_grad_kin"),
        ("kin_plus_xc", "nuc_grad_kin_plus_xc"),
        ("kin_minus_apbe", "nuc_grad_kin_minus_apbe"),
        ("kinapbe", "nuc_grad_kinapbe"),
        ("tot", "nuc_grad_tot"),
        ("grad_kin", "nuc_grad_kin"),
        ("grad_kin_plus_xc", "nuc_grad_kin_plus_xc"),
        ("grad_kin_minus_apbe", "nuc_grad_kin_minus_apbe"),
        ("grad_kinapbe", "nuc_grad_kinapbe"),
        ("grad_tot", "nuc_grad_tot"),
    ],
)
def test_resolve_reference_model_key_auto_maps_known_targets(
    target_key,
    expected_reference_key,
):
    model_config = SimpleNamespace(data=SimpleNamespace(target_key=target_key))

    key = eval_ng.resolve_reference_model_key(
        model_config=model_config,
        reference_model_key="auto",
    )

    assert key == expected_reference_key


def test_resolve_reference_model_key_auto_raises_for_unknown_target():
    model_config = SimpleNamespace(data=SimpleNamespace(target_key="unknown_target"))

    with pytest.raises(ValueError, match="unknown_target"):
        eval_ng.resolve_reference_model_key(
            model_config=model_config,
            reference_model_key="auto",
        )


def test_compute_force_error_statistics_reports_total_and_per_atom_metrics():
    predicted = np.array([1.0, 2.0, 4.0, -1.0, -2.0, -4.0])
    reference = np.array([0.0, 2.0, 1.0, -2.0, -2.0, -1.0])

    stats = eval_ng.compute_force_error_statistics(
        predicted=predicted,
        reference=reference,
        n_atoms=2,
        prefix="total",
    )

    error = predicted - reference

    assert stats["total_mae_ha_per_bohr"] == pytest.approx(np.mean(np.abs(error)))
    assert stats["total_rmse_ha_per_bohr"] == pytest.approx(np.sqrt(np.mean(error**2)))

    assert stats["total_mae_kcal_per_mol_angstrom"] == pytest.approx(
        np.mean(np.abs(error)) * eval_ng.HARTREE_TO_KCAL_PER_MOL / eval_ng.BOHR_TO_ANGSTROM
    )
    assert stats["total_rmse_kcal_per_mol_angstrom"] == pytest.approx(
        np.sqrt(np.mean(error**2)) * eval_ng.HARTREE_TO_KCAL_PER_MOL / eval_ng.BOHR_TO_ANGSTROM
    )

    assert stats["total_mae_mev_per_angstrom"] == pytest.approx(
        np.mean(np.abs(error)) * eval_ng.HARTREE_TO_MEV / eval_ng.BOHR_TO_ANGSTROM
    )
    assert stats["total_rmse_mev_per_angstrom"] == pytest.approx(
        np.sqrt(np.mean(error**2)) * eval_ng.HARTREE_TO_MEV / eval_ng.BOHR_TO_ANGSTROM
    )


def test_compute_force_error_statistics_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        eval_ng.compute_force_error_statistics(
            predicted=np.zeros(6),
            reference=np.zeros(3),
            n_atoms=2,
            prefix="total",
        )


def test_compute_force_error_statistics_rejects_non_3n_vector():
    with pytest.raises(ValueError, match="3 \\* n_atoms"):
        eval_ng.compute_force_error_statistics(
            predicted=np.zeros(5),
            reference=np.zeros(5),
            n_atoms=2,
            prefix="total",
        )


def test_evaluate_gradient_sample_uses_label_coeffs_and_reports_total_and_model_errors(tmp_path):
    label_path = tmp_path / "DATASET" / "labels" / "00000001.zarr.zip"
    label_path.parent.mkdir(parents=True)

    reference_path = tmp_path / "DATASET" / "nuclear_gradients" / "00000001.zarr.zip"
    reference_path.parent.mkdir(parents=True)

    ref_total = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    ref_model = np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5])
    write_reference_zarr(
        reference_path,
        {
            "nuc_grad_tot": ref_total,
            "nuc_grad_kin_plus_xc": ref_model,
        },
    )

    label_coeffs = torch.tensor([10.0, 20.0], dtype=torch.float64)
    sample = make_sample(
        natm=2,
        mol_id=1,
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64),
        ground_state_coeffs=label_coeffs,
        trajectory_coeffs=torch.tensor([[999.0, 999.0]], dtype=torch.float64),
    )

    seen_coeffs = []

    class DummySampleGenerator:
        def get_sample_from_mol(self, mol):
            return sample

    def total_gradient_fn(sample_arg, coeffs_arg):
        assert sample_arg is sample
        seen_coeffs.append(coeffs_arg.detach().clone())
        return torch.tensor([2.0, 1.0, 0.0, -2.0, -1.0, 0.0], dtype=torch.float64)

    def model_gradient_fn(sample_arg, coeffs_arg):
        assert sample_arg is sample
        seen_coeffs.append(coeffs_arg.detach().clone())
        return torch.tensor([0.25, 0.5, 0.75, -0.25, -0.5, -0.75], dtype=torch.float64)

    row = eval_ng.evaluate_gradient_sample(
        sample_generator=DummySampleGenerator(),
        sample=sample,
        label_path=label_path,
        total_gradient_fn=total_gradient_fn,
        model_gradient_fn=model_gradient_fn,
        nuclear_gradients_dirname="nuclear_gradients",
        reference_total_key="nuc_grad_tot",
        reference_model_key="nuc_grad_kin_plus_xc",
        report_forces=True,
    )

    assert len(seen_coeffs) == 2
    torch.testing.assert_close(seen_coeffs[0], label_coeffs)
    torch.testing.assert_close(seen_coeffs[1], label_coeffs)

    assert row["mol_id"] == 1
    assert row["n_atom"] == 2
    assert row["reference_total_key"] == "nuc_grad_tot"
    assert row["reference_model_key"] == "nuc_grad_kin_plus_xc"

    total_error = np.array([1.0, 0.0, -1.0, -1.0, 0.0, 1.0])
    model_error = np.array([-0.25, 0.0, 0.25, 0.25, 0.0, -0.25])

    assert row["total_mae_ha_per_bohr"] == pytest.approx(np.mean(np.abs(total_error)))
    assert row["model_mae_ha_per_bohr"] == pytest.approx(np.mean(np.abs(model_error)))


def test_summarize_metric_rows_reports_distribution_statistics():
    rows = [
        {"total_mae_ha_per_bohr": 1.0, "model_mae_ha_per_bohr": 0.5, "skipped": False},
        {"total_mae_ha_per_bohr": 3.0, "model_mae_ha_per_bohr": 1.5, "skipped": False},
        {"skipped": True, "skip_reason": "missing reference"},
    ]

    summary = eval_ng.summarize_metric_rows(rows)

    assert summary["n_total"] == 3
    assert summary["n_success"] == 2
    assert summary["n_skipped"] == 1


class DummyTrajectorySample:
    def __init__(self):
        self._keys = []

    def add_item(self, key, value, representation):
        setattr(self, key, value)
        if key not in self._keys:
            self._keys.append(key)

    def keys(self):
        return list(self._keys)

    def delete_item(self, key):
        self._keys.remove(key)
        delattr(self, key)


def test_ofdft_diagnostic_metadata_includes_abs_energy_error_and_convergence():
    metadata = eval_ng._ofdft_diagnostic_metadata(
        {
            "stopping_index": 3,
            "converged": False,
            "gs_signed_energy_error": -0.002,
            "gs_abs_energy_error": 0.004,
            "gs_l2_norm": 1.25,
            "gs_gradient_norm": 2.5,
        }
    )

    assert metadata["ofdft_iterations"] == 4
    assert metadata["ofdft_converged"] is False
    assert metadata["ofdft_abs_energy_error_mha"] == pytest.approx(4.0)
    assert metadata["ofdft_gradient_norm"] == pytest.approx(2.5)


def test_run_ofdft_for_sample_reuses_cached_sad_guess_kwargs(monkeypatch):
    cached_dataset_statistics = object()
    cached_sad_guess_kwargs = {"dataset_statistics": cached_dataset_statistics}
    optimizer = SimpleNamespace(
        convergence_tolerance=1e-4,
    )
    context = SimpleNamespace(
        optimizer=optimizer,
        denop_func_factory="factory",
        max_xc_memory=1,
        normalize_initial_guess=True,
        ks_basis="sto-3g",
        proj_minao_module="guess-model",
        initialization="sad",
        convergence_criterion="last_iter",
        sad_guess_kwargs=cached_sad_guess_kwargs,
    )
    sample = SimpleNamespace(mol="mol")
    seen_sad_guess_kwargs = []

    def fake_density_optimization_with_label(*_args, callback, sad_guess_kwargs, **_kwargs):
        seen_sad_guess_kwargs.append(sad_guess_kwargs)
        callback.stopping_index = 0
        callback.coeffs.append(torch.tensor([1.0], dtype=torch.float64))
        return {"gs_gradient_norm": 0}, callback, "energies", "functional"

    monkeypatch.setattr(
        eval_gs,
        "_sad_guess_kwargs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("SAD kwargs should be cached")
        ),
    )
    monkeypatch.setattr(
        eval_gs,
        "density_optimization_with_label",
        fake_density_optimization_with_label,
    )

    result = eval_gs.run_ofdft_for_sample(sample, context)

    assert torch.equal(result["coeffs"], torch.tensor([1.0], dtype=torch.float64))
    assert seen_sad_guess_kwargs == [{"dataset_statistics": cached_dataset_statistics}]
    assert seen_sad_guess_kwargs[0] is not cached_sad_guess_kwargs


def test_dataset_statistics_path_is_not_required_for_non_sad_initialization():
    class DataConfigWithoutDatasetStatistics:
        @property
        def dataset_statistics(self):
            raise AssertionError("dataset_statistics should not be read for minao")

    model_config = SimpleNamespace(data=DataConfigWithoutDatasetStatistics())

    assert eval_gs._dataset_statistics_path(model_config, "minao") is None


def test_evaluate_dataset_rows_skips_recoverable_errors_with_exception_type(monkeypatch):
    label_path = Path("/data/DATASET/labels/00000001.zarr.zip")
    sample = make_sample()

    monkeypatch.setattr(
        eval_ng,
        "_iter_samples_with_paths",
        lambda *_args, **_kwargs: iter([(label_path, sample)]),
    )
    monkeypatch.setattr(
        eval_ng,
        "evaluate_gradient_sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad reference")),
    )

    rows = eval_ng._evaluate_dataset_rows(
        dataset="dataset",
        paths=[label_path],
        sample_generator="sample-generator",
        total_gradient_fn=lambda *_args, **_kwargs: None,
        model_gradient_fn=lambda *_args, **_kwargs: None,
        nuclear_gradients_dirname="nuclear_gradients",
        reference_total_key="nuc_grad_tot",
        reference_model_key="nuc_grad_model",
        report_forces=True,
        fail_fast=False,
        torch_threads_per_worker=1,
        device="cpu",
    )

    assert rows[0]["skipped"] is True
    assert rows[0]["skip_reason"] == "bad reference"
    assert rows[0]["exception_type"] == "ValueError"


def test_evaluate_dataset_rows_propagates_unexpected_errors(monkeypatch):
    label_path = Path("/data/DATASET/labels/00000001.zarr.zip")
    sample = make_sample()

    monkeypatch.setattr(
        eval_ng,
        "_iter_samples_with_paths",
        lambda *_args, **_kwargs: iter([(label_path, sample)]),
    )
    monkeypatch.setattr(
        eval_ng,
        "evaluate_gradient_sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("programming bug")),
    )

    with pytest.raises(TypeError, match="programming bug"):
        eval_ng._evaluate_dataset_rows(
            dataset="dataset",
            paths=[label_path],
            sample_generator="sample-generator",
            total_gradient_fn=lambda *_args, **_kwargs: None,
            model_gradient_fn=lambda *_args, **_kwargs: None,
            nuclear_gradients_dirname="nuclear_gradients",
            reference_total_key="nuc_grad_tot",
            reference_model_key="nuc_grad_model",
            report_forces=True,
            fail_fast=False,
            torch_threads_per_worker=1,
            device="cpu",
        )


def test_raise_if_no_successful_rows_rejects_all_skipped_results():
    with pytest.raises(RuntimeError, match="No successful"):
        eval_ng._raise_if_no_successful_rows(
            [
                {"skipped": True, "skip_reason": "missing reference"},
                {"skipped": True, "skip_reason": "bad sample"},
            ]
        )
