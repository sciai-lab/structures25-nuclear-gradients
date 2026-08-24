"""Tests for gradient comparison evaluation task."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mldft.ofdft.evaluation.gradient_comparison as eval_cmp


class DummyMol:
    def __init__(self, natm: int):
        self.natm = natm


class DataConfigWithoutDatasetStatistics:
    @property
    def dataset_statistics(self):
        raise AssertionError("dataset_statistics should not be read for minao")


class DummyTransforms:
    pass


def make_sample(natm: int = 1):
    return SimpleNamespace(
        mol=DummyMol(natm),
        n_atom=natm,
        mol_id=7,
        atomic_numbers=torch.tensor([6] * natm),
    )


def test_make_gradient_methods_instantiates_requested_methods(monkeypatch):
    calls = []

    def fake_get_gradient_method(method, sample_generator, func_factory, *, device, **kwargs):
        calls.append((method, sample_generator, func_factory, device, kwargs))
        return f"gradient-{method}"

    monkeypatch.setattr(eval_cmp, "get_gradient_method", fake_get_gradient_method)
    context = SimpleNamespace(
        sample_generator="sample-generator",
        denop_func_factory="factory",
        optimizer="optimizer",
        max_xc_memory=7,
        normalize_initial_guess=False,
        ks_basis="sto-3g",
        proj_minao_module="guess-model",
        sad_guess_kwargs={"dataset_statistics": "stats"},
    )

    gradient_methods = eval_cmp.make_gradient_methods(
        context,
        methods=["analytical_generator", "numerical", "fully_numerical"],
        device=torch.device("cpu"),
        epsilon=1e-3,
        strict=True,
    )

    assert gradient_methods == {
        "analytical_generator": "gradient-analytical_generator",
        "numerical": "gradient-numerical",
        "fully_numerical": "gradient-fully_numerical",
    }
    assert [call[0] for call in calls] == [
        "analytical_generator",
        "numerical",
        "fully_numerical",
    ]
    fully_numerical_kwargs = calls[-1][4]
    assert fully_numerical_kwargs == {
        "optimizer": "optimizer",
        "epsilon": 1e-3,
        "max_xc_memory": 7,
        "normalize_initial_guess": False,
        "ks_basis": "sto-3g",
        "proj_minao_module": "guess-model",
        "sad_guess_kwargs": {"dataset_statistics": "stats"},
        "strict": True,
    }


def test_evaluate_gradient_comparison_sample_outputs_one_row_per_method(monkeypatch, tmp_path):
    sample = make_sample(natm=2)
    context = SimpleNamespace(sample_generator="sample-generator")
    gradient_sample = SimpleNamespace(mol=sample.mol)
    coeffs = torch.tensor([1.0], dtype=torch.float64)

    monkeypatch.setattr(
        eval_cmp,
        "run_ofdft_for_sample",
        lambda _sample, _context: {
            "coeffs": coeffs,
            "metric_dict": {"converged": True, "stopping_index": 2},
        },
    )
    monkeypatch.setattr(
        eval_cmp,
        "_gradient_sample",
        lambda _sample, _sample_generator: gradient_sample,
    )
    gradient_methods = {
        "analytical_generator": lambda _sample, _coeffs: np.array([1.0, 2.0, 2.0, 3.0, 0.0, 4.0]),
        "numerical": lambda _sample, _coeffs: np.array([2.0, 0.0, 0.0, 0.0, 0.0, 6.0]),
        "fully_numerical": lambda _sample, _coeffs: np.zeros(6),
    }

    rows = eval_cmp.evaluate_gradient_comparison_sample(
        sample=sample,
        label_path=Path("/data/DATASET/labels/00000007.zarr.zip"),
        context=context,
        gradient_methods=gradient_methods,
        reference_method="fully_numerical",
        gradient_save_dir=tmp_path,
    )

    gradient_path = tmp_path / "00000007.npz"
    assert gradient_path.exists()
    with np.load(gradient_path) as gradients:
        np.testing.assert_allclose(
            gradients["analytical_generator"],
            np.array([1.0, 2.0, 2.0, 3.0, 0.0, 4.0]),
        )
        np.testing.assert_allclose(
            gradients["numerical"],
            np.array([2.0, 0.0, 0.0, 0.0, 0.0, 6.0]),
        )
        np.testing.assert_allclose(gradients["fully_numerical"], np.zeros(6))

    rows_by_method = {row["method"]: row for row in rows}
    assert set(rows_by_method) == {"analytical_generator", "numerical", "fully_numerical"}

    def assert_error_metrics(method, prefix, mae, per_atom_mae):
        row = rows_by_method[method]
        assert row[f"{prefix}_mae_ha_per_bohr"] == pytest.approx(mae)
        assert row[f"{prefix}_per_cartesian_mae_ha_per_bohr"] == pytest.approx(mae)
        assert row[f"{prefix}_per_atom_mae_ha_per_bohr"] == pytest.approx(per_atom_mae)

    assert_error_metrics("analytical_generator", "vs_reference", 2.0, 4.0)
    assert_error_metrics("analytical_generator", "vs_numerical", 5 / 3, (3 + np.sqrt(13)) / 2)
    assert_error_metrics(
        "numerical",
        "vs_analytical_generator",
        5 / 3,
        (3 + np.sqrt(13)) / 2,
    )
    assert_error_metrics("numerical", "vs_reference", 4 / 3, 4.0)
    assert_error_metrics("fully_numerical", "vs_reference", 0.0, 0.0)
    assert rows_by_method["numerical"]["gradient_path"] == str(gradient_path)
    assert rows_by_method["numerical"]["gradient_key"] == "numerical"
    assert rows_by_method["numerical"]["evaluation_coefficients"] == "ofdft_optimized"
    assert rows_by_method["numerical"]["reference_method"] == "fully_numerical"
    assert rows_by_method["numerical"]["ofdft_iterations"] == 3


def test_gradient_comparison_rows_skip_recoverable_errors(monkeypatch):
    label_path = Path("/data/DATASET/labels/00000007.zarr.zip")
    sample = make_sample()

    def fake_evaluate_sample_rows(
        dataset,
        paths,
        desc,
        sample_evaluator,
        failure_row_factory,
        log_success,
        fail_fast,
        torch_threads_per_worker,
        device,
    ):
        assert dataset == "dataset"
        assert paths == [label_path]
        assert desc == "gradient comparison"
        assert fail_fast is False
        assert torch_threads_per_worker == 1
        assert device == "cpu"
        return [failure_row_factory(label_path, ValueError("bad reference"))]

    monkeypatch.setattr(eval_cmp, "_evaluate_sample_rows", fake_evaluate_sample_rows)

    rows = eval_cmp._evaluate_dataset_rows_from_gradient_comparison(
        dataset="dataset",
        paths=[label_path],
        context="context",
        gradient_methods={"fully_numerical": lambda *_args, **_kwargs: None},
        reference_method="fully_numerical",
        report_forces=True,
        fail_fast=False,
        torch_threads_per_worker=1,
        device="cpu",
    )

    assert rows == [
        {
            "mol_id": "00000007",
            "sum_formula": "",
            "n_atom": None,
            "n_heavy_atom": None,
            "quantity": "force",
            "skipped": True,
            "skip_reason": "bad reference",
            "exception_type": "ValueError",
            "reference_total_key": "fully_numerical",
            "reference_model_key": "fully_numerical",
            "label_path": str(label_path),
            "reference_path": str(
                eval_cmp.reference_path_for_label_path(label_path, "gradient_comparison")
            ),
        }
    ]


def test_run_gradient_comparison_writes_csv_and_summary(monkeypatch, tmp_path):
    metric_rows = [{"skipped": False, "n_atom": 1, "method": "numerical"}]
    summary = {"n_success": 1}
    written = {}

    monkeypatch.setattr(eval_cmp, "_make_denop_transforms", lambda **_kwargs: DummyTransforms())
    monkeypatch.setattr(eval_cmp, "_make_ofdft_context", lambda **_kwargs: "context")
    monkeypatch.setattr(
        eval_cmp,
        "_prepare_evaluation_dataset",
        lambda *args, **kwargs: (
            "dataset",
            [Path("/data/DATASET/labels/00000007.zarr.zip")],
        ),
    )
    monkeypatch.setattr(
        eval_cmp,
        "make_gradient_methods",
        lambda _context, methods, **_kwargs: {method: f"gradient-{method}" for method in methods},
    )
    captured_evaluation_kwargs = {}

    def fake_evaluate_dataset_rows(**kwargs):
        captured_evaluation_kwargs.update(kwargs)
        return metric_rows

    monkeypatch.setattr(
        eval_cmp,
        "_evaluate_dataset_rows_from_gradient_comparison",
        fake_evaluate_dataset_rows,
    )

    def fake_summarize_metric_rows(rows, metrics=True):
        return dict(summary)

    monkeypatch.setattr(eval_cmp, "summarize_metric_rows", fake_summarize_metric_rows)
    monkeypatch.setattr(
        eval_cmp,
        "_write_evaluation_outputs",
        lambda save_dir, csv_name, summary_name, rows, data: written.update(
            {
                "csv": (save_dir / csv_name, rows),
                "summary": (save_dir / summary_name, data),
            }
        ),
    )

    rows, run_summary = eval_cmp.run_gradient_comparison(
        ckpt_path="checkpoint.ckpt",
        model_config=SimpleNamespace(data=SimpleNamespace(target_key="kin_plus_xc")),
        optimizer="optimizer",
        split_file_path="split.npz",
        methods=["analytical_generator", "numerical", "fully_numerical"],
        reference_method="fully_numerical",
        save_dir=tmp_path,
        save_gradients=True,
    )

    assert rows is metric_rows
    assert run_summary["n_success"] == 1
    assert run_summary["reference_method"] == "fully_numerical"
    assert run_summary["methods"] == ["analytical_generator", "numerical", "fully_numerical"]
    assert (
        captured_evaluation_kwargs["gradient_save_dir"]
        == tmp_path / eval_cmp.GRADIENT_VECTORS_DIRNAME
    )
    assert written["csv"] == (tmp_path / eval_cmp.CSV_PER_MOLECULE, metric_rows)
    assert written["summary"][0] == tmp_path / eval_cmp.SUMMARY_YAML
