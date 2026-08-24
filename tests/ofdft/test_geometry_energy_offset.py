import numpy as np

from mldft.ofdft.geometry_energy_offset import (
    KS_ENERGY_AT_KS_GEOMETRY,
    KS_ENERGY_AT_ML_GEOMETRY,
    KS_ENERGY_AT_ML_GEOMETRY_CONVERGED,
    KS_GEOMETRY_ENERGY_OFFSET,
    KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE,
    KS_GEOMETRY_ENERGY_OFFSET_SCF_TIME,
    set_ks_geometry_energy_offset_defaults,
    set_ks_geometry_energy_offset_result,
)


class FakeSample:
    def add_item(self, key, value, *, representation):
        setattr(self, key, value)
        self.last_representation = representation


def test_set_ks_geometry_energy_offset_result_sets_sample_attrs():
    sample = FakeSample()

    set_ks_geometry_energy_offset_result(
        sample,
        ks_energy_at_ml_geometry=-1.2,
        ks_energy_at_ml_geometry_converged=True,
        ks_energy_at_ks_geometry=-1.4,
        scf_time=3.5,
    )

    assert getattr(sample, KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE) is True
    assert getattr(sample, KS_ENERGY_AT_ML_GEOMETRY) == -1.2
    assert getattr(sample, KS_ENERGY_AT_ML_GEOMETRY_CONVERGED) is True
    assert getattr(sample, KS_ENERGY_AT_KS_GEOMETRY) == -1.4
    assert np.isclose(getattr(sample, KS_GEOMETRY_ENERGY_OFFSET), 0.2)
    assert getattr(sample, KS_GEOMETRY_ENERGY_OFFSET_SCF_TIME) == 3.5


def test_set_ks_geometry_energy_offset_defaults_sets_nan_unavailable_attrs():
    sample = FakeSample()

    set_ks_geometry_energy_offset_defaults(sample)

    assert getattr(sample, KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE) is False
    assert np.isnan(getattr(sample, KS_ENERGY_AT_ML_GEOMETRY))
    assert getattr(sample, KS_ENERGY_AT_ML_GEOMETRY_CONVERGED) is False
    assert np.isnan(getattr(sample, KS_ENERGY_AT_KS_GEOMETRY))
    assert np.isnan(getattr(sample, KS_GEOMETRY_ENERGY_OFFSET))
    assert np.isnan(getattr(sample, KS_GEOMETRY_ENERGY_OFFSET_SCF_TIME))
