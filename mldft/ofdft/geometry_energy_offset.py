"""Helpers for storing geometry-energy-offset fields on samples."""

import numpy as np

from mldft.ml.data.components.of_data import Representation

GEOMETRY_ENERGY_OFFSET_AVAILABLE = "geometry_energy_offset_available"
ML_ENERGY_AT_KS_GEOMETRY_CONVERGED = "ml_energy_at_ks_geometry_converged"
ML_ENERGY_AT_KS_GEOMETRY = "ml_energy_at_ks_geometry"
GEOMETRY_ENERGY_OFFSET = "geometry_energy_offset"
GEOMETRY_ENERGY_OFFSET_DENOP_TIME = "geometry_energy_offset_denop_time"

KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE = "ks_geometry_energy_offset_available"
KS_ENERGY_AT_ML_GEOMETRY = "ks_energy_at_ml_geometry"
KS_ENERGY_AT_ML_GEOMETRY_CONVERGED = "ks_energy_at_ml_geometry_converged"
KS_ENERGY_AT_KS_GEOMETRY = "ks_energy_at_ks_geometry"
KS_GEOMETRY_ENERGY_OFFSET = "ks_geometry_energy_offset"
KS_GEOMETRY_ENERGY_OFFSET_SCF_TIME = "ks_geometry_energy_offset_scf_time"


def set_geometry_energy_offset_defaults(sample) -> None:
    """Mark geometry-energy-offset data as unavailable on a sample."""
    set_geometry_energy_offset_result(
        sample,
        ml_energy_at_ks_geometry=np.nan,
        ml_energy_at_ks_geometry_converged=False,
        denop_time=np.nan,
        geometry_energy_offset=np.nan,
        available=False,
    )


def set_geometry_energy_offset_result(
    sample,
    *,
    ml_energy_at_ks_geometry: float,
    ml_energy_at_ks_geometry_converged: bool,
    denop_time: float,
    geometry_energy_offset: float,
    available: bool | None = None,
) -> None:
    """Set geometry-energy-offset fields on a sample."""
    if available is None:
        available = bool(ml_energy_at_ks_geometry_converged)

    sample.add_item(
        GEOMETRY_ENERGY_OFFSET_AVAILABLE,
        bool(available),
        representation=Representation.NONE,
    )
    sample.add_item(
        ML_ENERGY_AT_KS_GEOMETRY_CONVERGED,
        bool(ml_energy_at_ks_geometry_converged),
        representation=Representation.NONE,
    )
    sample.add_item(
        ML_ENERGY_AT_KS_GEOMETRY,
        ml_energy_at_ks_geometry,
        representation=Representation.NONE,
    )
    sample.add_item(
        GEOMETRY_ENERGY_OFFSET,
        geometry_energy_offset,
        representation=Representation.NONE,
    )
    sample.add_item(
        GEOMETRY_ENERGY_OFFSET_DENOP_TIME,
        denop_time,
        representation=Representation.NONE,
    )


def set_ks_geometry_energy_offset_defaults(sample) -> None:
    """Mark KS geometry-energy-offset data as unavailable on a sample."""
    set_ks_geometry_energy_offset_result(
        sample,
        ks_energy_at_ml_geometry=np.nan,
        ks_energy_at_ml_geometry_converged=False,
        ks_energy_at_ks_geometry=np.nan,
        scf_time=np.nan,
        ks_geometry_energy_offset=np.nan,
        available=False,
    )


def set_ks_geometry_energy_offset_result(
    sample,
    *,
    ks_energy_at_ml_geometry: float,
    ks_energy_at_ml_geometry_converged: bool,
    ks_energy_at_ks_geometry: float,
    scf_time: float,
    ks_geometry_energy_offset: float | None = None,
    available: bool | None = None,
) -> None:
    """Set KS geometry-energy-offset fields on a sample."""
    if ks_geometry_energy_offset is None:
        ks_geometry_energy_offset = (
            ks_energy_at_ml_geometry - ks_energy_at_ks_geometry
            if ks_energy_at_ml_geometry_converged
            else np.nan
        )

    if available is None:
        available = bool(ks_energy_at_ml_geometry_converged)

    sample.add_item(
        KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE,
        bool(available),
        representation=Representation.NONE,
    )
    sample.add_item(
        KS_ENERGY_AT_ML_GEOMETRY_CONVERGED,
        bool(ks_energy_at_ml_geometry_converged),
        representation=Representation.NONE,
    )
    sample.add_item(
        KS_ENERGY_AT_ML_GEOMETRY,
        ks_energy_at_ml_geometry,
        representation=Representation.NONE,
    )
    sample.add_item(
        KS_ENERGY_AT_KS_GEOMETRY,
        ks_energy_at_ks_geometry,
        representation=Representation.NONE,
    )
    sample.add_item(
        KS_GEOMETRY_ENERGY_OFFSET,
        ks_geometry_energy_offset,
        representation=Representation.NONE,
    )
    sample.add_item(
        KS_GEOMETRY_ENERGY_OFFSET_SCF_TIME,
        scf_time,
        representation=Representation.NONE,
    )
