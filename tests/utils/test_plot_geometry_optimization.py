import numpy as np
import pytest

from mldft.utils.plotting import plot_geometry_optimization as plot_geom
from mldft.utils.plotting.plot_geometry_optimization import build_geometry_data_record


def _make_feature_dict() -> dict:
    """Create a minimal feature dictionary for geometry export tests."""
    return {
        "name": "test_01",
        "converged": True,
        "n_atoms": 2,
        "cycles": np.array([0, 1, 2]),
        "irmsd_val": 0.01,
        "relative_energy_error_mha": 0.2,
        "relative_energy_available": True,
        "ks_initial_energy": -1.5,
        "ks_final_energy": -1.6,
        "ml_initial_energy": -1.4,
        "ml_final_energy": -1.55,
        "ks_relaxation_mha": 100.0,
        "ml_relaxation_mha": 150.0,
        "ml_energy_at_ks_geometry": -1.56,
        "ml_energy_at_ks_geometry_converged": True,
        "geometry_energy_offset_mha": 10.0,
        "geometry_energy_offset_available": True,
        "ks_energy_at_ml_geometry": -1.2,
        "ks_energy_at_ml_geometry_converged": True,
        "ks_energy_at_ks_geometry": -1.25,
        "ks_geometry_energy_offset_mha": 50.0,
        "ks_geometry_energy_offset_available": True,
        "energies": np.array([0.01, 0.005]),
        "dist_mat1": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "dist_mat2": np.array([[0.0, 1.1], [1.1, 0.0]]),
        "bond_lengths": np.empty((0, 4)),
        "bond_angles": np.empty((0, 5)),
        "bond_dihedrals": np.empty((0, 6)),
    }


def test_build_geometry_data_record_includes_geometry_energy_offset_fields() -> None:
    """Geometry export should include the geometry-energy-offset fields."""
    data = _make_feature_dict()

    record = build_geometry_data_record(data)

    assert record["ml_energy_at_ks_geometry_ha"] == -1.56
    assert record["ml_energy_at_ks_geometry_converged"] is True
    assert record["geometry_energy_offset_mha"] == 10.0
    assert record["geometry_energy_offset_available"] is True
    assert record["ks_energy_at_ml_geometry_ha"] == -1.2
    assert record["ks_energy_at_ml_geometry_converged"] is True
    assert record["ks_energy_at_ks_geometry_ha"] == -1.25
    assert record["ks_geometry_energy_offset_mha"] == 50.0
    assert record["ks_geometry_energy_offset_available"] is True


def test_build_geometry_data_record_handles_missing_geometry_energy_offset_fields() -> None:
    """Geometry export should stay backward-compatible when fields are missing."""
    data = _make_feature_dict()
    data.pop("ml_energy_at_ks_geometry")
    data.pop("ml_energy_at_ks_geometry_converged")
    data.pop("geometry_energy_offset_mha")
    data.pop("geometry_energy_offset_available")
    data.pop("ks_energy_at_ml_geometry")
    data.pop("ks_energy_at_ml_geometry_converged")
    data.pop("ks_energy_at_ks_geometry")
    data.pop("ks_geometry_energy_offset_mha")
    data.pop("ks_geometry_energy_offset_available")

    record = build_geometry_data_record(data)

    assert record["ml_energy_at_ks_geometry_ha"] is None
    assert record["ml_energy_at_ks_geometry_converged"] is False
    assert record["geometry_energy_offset_mha"] is None
    assert record["geometry_energy_offset_available"] is False
    assert record["ks_energy_at_ml_geometry_ha"] is None
    assert record["ks_energy_at_ml_geometry_converged"] is False
    assert record["ks_energy_at_ks_geometry_ha"] is None
    assert record["ks_geometry_energy_offset_mha"] is None
    assert record["ks_geometry_energy_offset_available"] is False


def test_plot_single_geometry_optimization_rejects_non_scalar_present_offset_field(
    monkeypatch,
) -> None:
    """Present malformed scalar fields should raise instead of becoming NaN."""

    class DummyMol:
        natm = 2

        def atom_symbol(self, index):
            return ["H", "H"][index]

    class DummySample:
        mol = DummyMol()
        energy = -1.0
        geomopt_converged = True
        geomopt_cycle = 2
        geomopt_energies = [-0.5, -1.0]
        ml_energy_at_ks_geometry = [1.0, 2.0]
        ml_energy_at_ks_geometry_converged = True
        geometry_energy_offset = 0.0
        geometry_energy_offset_available = True

    def fake_get_geometry_features(_ks_mol, _of_mol, align=True):
        return (
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0, 1, 0.9, 1.0]]),
            np.empty((0, 5)),
            np.empty((0, 6)),
            object(),
            0.0,
        )

    monkeypatch.setattr(plot_geom, "get_geometry_features", fake_get_geometry_features)
    monkeypatch.setattr(
        plot_geom,
        "plot_molecule_image",
        lambda _rdkit_mol: np.zeros((1, 1, 3)),
    )

    with pytest.raises(ValueError, match="ml_energy_at_ks_geometry"):
        plot_geom.plot_single_geometry_optimization(
            DummySample(), DummyMol(), {"e_tot": -1.0}, return_features=True
        )


def test_plot_single_geometry_optimization_rejects_non_scalar_present_ks_offset_field(
    monkeypatch,
) -> None:
    """Present malformed KS scalar fields should raise instead of becoming NaN."""

    class DummyMol:
        natm = 2

        def atom_symbol(self, index):
            return ["H", "H"][index]

    class DummySample:
        mol = DummyMol()
        energy = -1.0
        geomopt_converged = True
        geomopt_cycle = 2
        geomopt_energies = [-0.5, -1.0]
        ks_energy_at_ml_geometry = [1.0, 2.0]
        ks_energy_at_ml_geometry_converged = True
        ks_energy_at_ks_geometry = -1.2
        ks_geometry_energy_offset = 0.1
        ks_geometry_energy_offset_available = True

    def fake_get_geometry_features(_ks_mol, _of_mol, align=True):
        return (
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0, 1, 0.9, 1.0]]),
            np.empty((0, 5)),
            np.empty((0, 6)),
            object(),
            0.0,
        )

    monkeypatch.setattr(plot_geom, "get_geometry_features", fake_get_geometry_features)
    monkeypatch.setattr(
        plot_geom,
        "plot_molecule_image",
        lambda _rdkit_mol: np.zeros((1, 1, 3)),
    )

    with pytest.raises(ValueError, match="ks_energy_at_ml_geometry"):
        plot_geom.plot_single_geometry_optimization(
            DummySample(), DummyMol(), {"e_tot": -1.0}, return_features=True
        )
