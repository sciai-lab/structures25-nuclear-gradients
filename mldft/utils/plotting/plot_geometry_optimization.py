import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from matplotlib.backends.backend_pdf import PdfPages
from rdkit.Chem import Draw, rdDepictor

from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.energies import Energies
from mldft.ofdft.geometry_energy_offset import (
    GEOMETRY_ENERGY_OFFSET,
    GEOMETRY_ENERGY_OFFSET_AVAILABLE,
    KS_ENERGY_AT_KS_GEOMETRY,
    KS_ENERGY_AT_ML_GEOMETRY,
    KS_ENERGY_AT_ML_GEOMETRY_CONVERGED,
    KS_GEOMETRY_ENERGY_OFFSET,
    KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE,
    ML_ENERGY_AT_KS_GEOMETRY,
    ML_ENERGY_AT_KS_GEOMETRY_CONVERGED,
)
from mldft.utils.conversions import as_scalar_bool, as_scalar_float, pyscf_to_rdkit
from mldft.utils.draw_molecule import cleanup_via_smiles
from mldft.utils.molecular_geometry import get_geometry_features
from mldft.utils.pyscf_calculations import load_scf_with_optional_data

CONVERGED_COLOR = "seagreen"
NOT_CONVERGED_COLOR = "lightcoral"


def angular_difference(a, b):
    """Smallest difference between two angles (in degrees)."""
    return ((a - b + 180) % 360) - 180


def format_geometry_stat(value, unit):
    """Format geometry statistics with more precision for Angstrom-scale errors."""
    is_angstrom = unit in {"Å", "A"}
    precision = 5 if is_angstrom else 3
    return f"{value:.{precision}f}{unit}"


def format_stats_block(values, label, unit):
    """Format summary statistics for an annotation block."""
    if values.size == 0:
        return f"{label}\nNo data"

    stats = compute_stats(values)
    return (
        f"{label}\n"
        f"MAE = {stats['mae']:.3f} {unit}\n"
        f"RMSD = {stats['rmsd']:.3f} {unit}\n"
        f"Max = {stats['max']:.3f} {unit}\n"
        f"Min = {stats['min']:.3f} {unit}"
    )


_MISSING = object()


def optional_sample_float(sample, key, default=np.nan):
    value = getattr(sample, key, _MISSING)
    if value is _MISSING:
        return default
    return as_scalar_float(value, name=key)


def optional_sample_bool(sample, key, default=False):
    value = getattr(sample, key, _MISSING)
    if value is _MISSING:
        return default
    return as_scalar_bool(value, name=key)


def plot_offset_summary(
    axes, values_mha, converged_mask, n_atoms, title, energy_label, scatter_title
):
    finite_mask = np.isfinite(values_mha)
    converged_offset_mask = converged_mask & finite_mask
    not_converged_offset_mask = (~converged_mask) & finite_mask
    converged_offsets = values_mha[converged_offset_mask]
    not_converged_offsets = values_mha[not_converged_offset_mask]
    finite_offsets = values_mha[finite_mask]

    bins = np.histogram_bin_edges(finite_offsets, bins=20) if finite_offsets.size > 1 else 20

    ax = axes[0]
    ax.set_title(title)
    ax.hist(
        converged_offsets,
        bins=bins,
        alpha=0.5,
        color=CONVERGED_COLOR,
        label="Converged",
    )
    ax.hist(
        not_converged_offsets,
        bins=bins,
        alpha=0.5,
        color=NOT_CONVERGED_COLOR,
        label="Not Converged",
    )
    ax.axvline(0.0, ls="--", color="black")
    ax.set_xlabel(energy_label)
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.annotate(
        format_stats_block(converged_offsets, "Converged", "mHa"),
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CONVERGED_COLOR),
    )
    ax.annotate(
        format_stats_block(not_converged_offsets, "Not Converged", "mHa"),
        xy=(0.98, 0.58),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NOT_CONVERGED_COLOR),
    )

    ax = axes[1]
    ax.scatter(
        n_atoms[converged_offset_mask],
        values_mha[converged_offset_mask],
        color=CONVERGED_COLOR,
        label="Converged",
        alpha=0.5,
    )
    ax.scatter(
        n_atoms[not_converged_offset_mask],
        values_mha[not_converged_offset_mask],
        color=NOT_CONVERGED_COLOR,
        label="Not Converged",
        alpha=0.5,
    )
    ax.axhline(0.0, ls="--", color="black")
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel(energy_label)
    ax.set_title(scatter_title)
    ax.legend(loc="upper left")


def format_irmsd_value(value):
    """Format iRMSD values for molecule overview labels."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.3f} Å"


def format_energy_mha_value(value):
    """Format energy values in mHa for molecule overview labels."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.2f} mHa"


def local_feature_plot(axis_main, axis_text, values, angular=False):
    for label, marker, pos, color, data in zip(
        ["Not Converged", "Converged"],
        ["x", "o"],
        [0.25, 0.75],
        [NOT_CONVERGED_COLOR, CONVERGED_COLOR],
        [values[1], values[0]],
    ):
        if len(data) == 0:
            continue
        x_value = data[:, -2]
        y_value = data[:, -1]
        axis_main.plot(x_value, y_value, ls="", color=color, marker=marker, label=label, alpha=0.5)

        diff = angular_difference(x_value, y_value) if angular else x_value - y_value

        mae = np.abs(diff).mean()
        rmsd = np.sqrt(np.mean(diff**2))
        max_diff = diff.max()
        min_diff = diff.min()

        unit = "°" if angular else "Å"
        text = (
            f"{label}\n"
            f"MAE  = {format_geometry_stat(mae, unit)}\n"
            f"RMSD = {format_geometry_stat(rmsd, unit)}\n"
            f"Max  = {format_geometry_stat(max_diff, unit)}\n"
            f"Min  = {format_geometry_stat(min_diff, unit)}"
        )

        axis_text.annotate(
            text,
            xy=(0, pos),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color),
        )


def separate_arrays_by_convergence(array_all, converged_mask):
    """Separate a list of arrays into converged and not converged based on a mask."""
    converged_mask = np.asarray(converged_mask, dtype=bool)

    if isinstance(array_all, np.ndarray):
        if array_all.ndim == 1:
            if array_all.shape[0] != converged_mask.size:
                raise ValueError(
                    "Expected 1D array length to match convergence mask size, got "
                    f"{array_all.shape[0]} and {converged_mask.size}."
                )
            return array_all[converged_mask], array_all[~converged_mask]

        if array_all.shape[0] == converged_mask.size:
            return array_all[converged_mask], array_all[~converged_mask]

        if array_all.shape[0] == 1 and array_all.shape[1] == converged_mask.size:
            return array_all[:, converged_mask].reshape(-1), array_all[:, ~converged_mask].reshape(
                -1
            )

    arrays = list(array_all)
    if len(arrays) != converged_mask.size:
        raise ValueError(
            "Expected one array per convergence flag, got "
            f"{len(arrays)} arrays and {converged_mask.size} flags."
        )

    converged_arrays = [
        array for array, flag in zip(arrays, converged_mask) if flag and array.size > 0
    ]
    not_converged_arrays = [
        array for array, flag in zip(arrays, converged_mask) if not flag and array.size > 0
    ]

    converged = np.concatenate(converged_arrays) if converged_arrays else np.array([])
    not_converged = np.concatenate(not_converged_arrays) if not_converged_arrays else np.array([])
    return converged, not_converged


# --- Main Plotting Functions ---


def plot_global_summary(
    geomopt_cycles,
    energies,
    rmsds,
    n_atoms,
    converged_mask,
    of_converged_distances,
    ks_converged_distances,
    of_not_converged_distances,
    ks_not_converged_distances,
    relative_energy_errors_mha=None,
    geometry_energy_offsets_mha=None,
    ks_geometry_energy_offsets_mha=None,
):
    """Plot a global summary of geometry optimization results."""
    if relative_energy_errors_mha is None:
        relative_energy_errors_mha = np.array([])
    relative_energy_errors_mha = np.asarray(relative_energy_errors_mha, dtype=float)
    has_relative_energy_data = relative_energy_errors_mha.size == converged_mask.size and np.any(
        np.isfinite(relative_energy_errors_mha[converged_mask])
    )
    if geometry_energy_offsets_mha is None:
        geometry_energy_offsets_mha = np.array([])
    geometry_energy_offsets_mha = np.asarray(geometry_energy_offsets_mha, dtype=float)
    has_geometry_offset_data = geometry_energy_offsets_mha.size == converged_mask.size and np.any(
        np.isfinite(geometry_energy_offsets_mha[converged_mask])
    )
    if ks_geometry_energy_offsets_mha is None:
        ks_geometry_energy_offsets_mha = np.array([])
    ks_geometry_energy_offsets_mha = np.asarray(ks_geometry_energy_offsets_mha, dtype=float)
    has_ks_geometry_offset_data = (
        ks_geometry_energy_offsets_mha.size == converged_mask.size
        and np.any(np.isfinite(ks_geometry_energy_offsets_mha[converged_mask]))
    )

    width, height = 8.27, 11.69
    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    n_subfigures = (
        3
        + int(has_relative_energy_data)
        + int(has_geometry_offset_data)
        + int(has_ks_geometry_offset_data)
    )
    subfigures_raw = fig.subfigures(n_subfigures, 1)
    if isinstance(subfigures_raw, np.ndarray):
        subfigures = subfigures_raw.flatten().tolist()
    else:
        subfigures = [subfigures_raw]

    # 1st Subfigure - Optimization Steps and Energy Error
    subfig = subfigures[0]
    axes = subfig.subplots(1, 2)
    max_geomopt_cycles = np.asarray([cycles[-1] for cycles in geomopt_cycles]) + 1
    # Optimization Steps Histogram
    ax = axes[0]
    ax.set_title("Optimization Steps")
    ax.set_xlabel("No. Steps")
    ax.hist(max_geomopt_cycles[converged_mask], bins=20, color=CONVERGED_COLOR, alpha=0.5)
    if len(max_geomopt_cycles[converged_mask]) > 0:
        mean_cycle = max_geomopt_cycles[converged_mask].mean()
        ax.axvline(mean_cycle, ls="--", color="black")
    else:
        mean_cycle = np.nan
    ax.annotate(
        f"Mean = {mean_cycle:.1f}\nVar = {max_geomopt_cycles.var():.1f}\nMax = {max_geomopt_cycles.max():.1f}\nMin = {max_geomopt_cycles.min():.1f}",
        xy=(0.7, 0.9),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )

    # Energy Error Histogram
    diff_energies = np.asarray([energy[-1] for energy in energies]) * 1000
    converged_energy_errors = diff_energies[converged_mask]
    not_converged_energy_errors = diff_energies[~converged_mask]
    ax = axes[1]
    ax.set_title("Total Energy Error")
    energy_bins = np.histogram_bin_edges(diff_energies, bins=20) if diff_energies.size > 1 else 20
    ax.hist(
        converged_energy_errors,
        bins=energy_bins,
        alpha=0.5,
        color=CONVERGED_COLOR,
        label="Converged",
    )
    ax.hist(
        not_converged_energy_errors,
        bins=energy_bins,
        alpha=0.5,
        color=NOT_CONVERGED_COLOR,
        label="Not Converged",
    )
    ax.set_xlabel(r"$\Delta E$ [mHa]")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.annotate(
        format_stats_block(converged_energy_errors, "Converged", "mHa"),
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CONVERGED_COLOR),
    )
    ax.annotate(
        format_stats_block(not_converged_energy_errors, "Not Converged", "mHa"),
        xy=(0.98, 0.58),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NOT_CONVERGED_COLOR),
    )

    next_subfigure_idx = 1
    if has_relative_energy_data:
        # Relative energy error from relaxation energies:
        # (E_init_ML - E_final_ML) - (E_init_KS - E_final_KS)
        subfig = subfigures[1]
        axes = subfig.subplots(1, 2)

        finite_rel_mask = np.isfinite(relative_energy_errors_mha)
        converged_rel_mask = converged_mask & finite_rel_mask
        not_converged_rel_mask = (~converged_mask) & finite_rel_mask
        converged_rel = relative_energy_errors_mha[converged_rel_mask]
        not_converged_rel = relative_energy_errors_mha[not_converged_rel_mask]
        rel_for_bins = relative_energy_errors_mha[finite_rel_mask]

        ax = axes[0]
        ax.set_title("Relative Energy Error")
        rel_bins = np.histogram_bin_edges(rel_for_bins, bins=20) if rel_for_bins.size > 1 else 20
        ax.hist(
            converged_rel,
            bins=rel_bins,
            alpha=0.5,
            color=CONVERGED_COLOR,
            label="Converged",
        )
        ax.hist(
            not_converged_rel,
            bins=rel_bins,
            alpha=0.5,
            color=NOT_CONVERGED_COLOR,
            label="Not Converged",
        )
        ax.axvline(0.0, ls="--", color="black")
        ax.set_xlabel(r"$\Delta E_{rel}$ [mHa]")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left")
        ax.annotate(
            format_stats_block(converged_rel, "Converged", "mHa"),
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CONVERGED_COLOR),
        )
        ax.annotate(
            format_stats_block(not_converged_rel, "Not Converged", "mHa"),
            xy=(0.98, 0.58),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NOT_CONVERGED_COLOR),
        )

        ax = axes[1]
        ax.scatter(
            n_atoms[converged_rel_mask],
            relative_energy_errors_mha[converged_rel_mask],
            color=CONVERGED_COLOR,
            label="Converged",
            alpha=0.5,
        )
        ax.scatter(
            n_atoms[not_converged_rel_mask],
            relative_energy_errors_mha[not_converged_rel_mask],
            color=NOT_CONVERGED_COLOR,
            label="Not Converged",
            alpha=0.5,
        )
        ax.axhline(0.0, ls="--", color="black")
        ax.set_xlabel("Number of Atoms")
        ax.set_ylabel(r"$\Delta E_{rel}$ [mHa]")
        ax.set_title("Relative Energy Error vs. Number of Atoms")
        ax.legend(loc="upper left")

        next_subfigure_idx = 2

    if has_geometry_offset_data:
        subfig = subfigures[next_subfigure_idx]
        axes = subfig.subplots(1, 2)
        plot_offset_summary(
            axes,
            geometry_energy_offsets_mha,
            converged_mask,
            n_atoms,
            "Geometry Energy Offset",
            r"$E_{ML}(A) - E_{ML}(B)$ [mHa]",
            "Geometry Energy Offset vs. Number of Atoms",
        )
        next_subfigure_idx += 1

    if has_ks_geometry_offset_data:
        subfig = subfigures[next_subfigure_idx]
        axes = subfig.subplots(1, 2)
        plot_offset_summary(
            axes,
            ks_geometry_energy_offsets_mha,
            converged_mask,
            n_atoms,
            "KS Geometry Energy Offset",
            r"$E_{KS}(ML geom) - E_{KS}(KS geom)$ [mHa]",
            "KS Geometry Energy Offset vs. Number of Atoms",
        )
        next_subfigure_idx += 1

    # 2nd Subfigure - Pairwise distances
    subfig = subfigures[next_subfigure_idx]
    axes = subfig.subplots(1, 2)

    max_dist = max(
        [
            array.max()
            for array in [
                of_converged_distances,
                ks_converged_distances,
                of_not_converged_distances,
                ks_not_converged_distances,
            ]
            if array.size > 0
        ]
    )
    padding = min(max_dist * 0.05, 0.1)
    lim_max = max_dist + padding

    ax = axes[0]
    ax.set_aspect("equal")
    ax.plot([0, lim_max], [0, lim_max], ls="-", color="black")
    ax.plot(
        of_not_converged_distances,
        ks_not_converged_distances,
        ls="",
        marker="x",
        color=NOT_CONVERGED_COLOR,
        label="Not Converged",
        alpha=0.5,
    )
    ax.plot(
        of_converged_distances,
        ks_converged_distances,
        ls="",
        marker="o",
        color=CONVERGED_COLOR,
        label="Converged",
        alpha=0.5,
    )
    ax.set_xlabel("OF Distances [Å]")
    ax.set_ylabel("KS Distances [Å]")
    # atom should not be closer than 0.5 in any geometry optimization
    ax.set_xlim(0.5, lim_max)
    ax.set_ylim(0.5, lim_max)
    ax.set_title("Pairwise Distances")
    ax.legend()

    ax = axes[1]

    ax.scatter(
        n_atoms[converged_mask],
        rmsds[converged_mask],
        color=CONVERGED_COLOR,
        label="Converged",
        alpha=0.5,
    )
    ax.scatter(
        n_atoms[~converged_mask],
        rmsds[~converged_mask],
        color=NOT_CONVERGED_COLOR,
        label="Not Converged",
        alpha=0.5,
    )
    ax.set_xlabel("Number of Atoms")
    ax.set_ylabel("iRMSD [Å]")
    ax.set_title("iRMSD vs. Number of Atoms")
    conv_rmsd, nconv_rmsd = separate_arrays_by_convergence(
        np.expand_dims(rmsds, 0), converged_mask
    )
    stats = {}
    stats["mean_conv_rmsd"] = conv_rmsd.mean() if conv_rmsd.size > 0 else np.nan
    stats["max_conv_rmsd"] = conv_rmsd.max() if conv_rmsd.size > 0 else np.nan
    stats["mean_nconv_rmsd"] = nconv_rmsd.mean() if nconv_rmsd.size > 0 else np.nan
    stats["max_nconv_rmsd"] = nconv_rmsd.max() if nconv_rmsd.size > 0 else np.nan
    ax.annotate(
        f"  C. Mean: {stats['mean_conv_rmsd']:3.3f}\n  C. Max  : {stats['max_conv_rmsd']:3.3f}\nNC. Mean: {stats['mean_nconv_rmsd']:3.3f}\nNC. Max  : {stats['max_nconv_rmsd']:3.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )

    # 3rd Subfigure - Optimization Trajectories
    subfig = subfigures[next_subfigure_idx + 1]
    axes = subfig.subplots(1, 2, width_ratios=[3, 1])

    ax = axes[0]
    ax.set_title("Optimization Trajectories")
    for cycle, energy in zip(geomopt_cycles, energies):
        ax.plot(np.asarray(energy, dtype=float) * 1000.0)
    ax.set_yscale("symlog", linthresh=0.1, linscale=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\Delta E$ [mHa]")

    ax = axes[1]
    ax.set_axis_off()
    no_samples = len(converged_mask)
    no_converged = np.count_nonzero(converged_mask)
    ax.annotate(
        f"No. Samples: {no_samples}\nConverged: {no_converged}\nNot Converged: {no_samples - no_converged}\nConverged [%]: {no_converged / no_samples * 100:.1f}",
        xy=(0.25, 0.5),
        xycoords="axes fraction",
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )

    return fig


def plot_local_geometry(bond_lengths, bond_angles, bond_dihedrals, converged_mask):
    """Plot local geometry features such as bond lengths, angles, and dihedrals."""
    width, height = 8.27, 11.69
    fig = plt.figure(constrained_layout=True, figsize=(width, height))
    subfig = fig.subfigures(1, 1)
    axes = subfig.subplots(3, 2, width_ratios=[3, 1])

    features = [
        ("Bond Lengths", bond_lengths, False),
        ("Bond Angles", bond_angles, True),
        ("Dihedrals", bond_dihedrals, True),
    ]

    for idx, (title, values, angular) in enumerate(features):
        ax = axes[idx, 0]
        non_empty_values = [value for value in values if value.size > 0]
        if not non_empty_values:
            ax.set_axis_off()
            ax.annotate(
                f"No {title} Found",
                xy=(0.5, 0.5),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
            )
            continue
        min_val = min(value[:, -2:].min() for value in non_empty_values)
        max_val = max(value[:, -2:].max() for value in non_empty_values)
        values = separate_arrays_by_convergence(values, converged_mask)

        ax.set_aspect("equal")
        ax.axline((min_val, min_val), slope=1, ls="-", color="black")
        if angular:
            ax.axline((min_val + 360, min_val), slope=1, ls="-", color="black")
            ax.axline((min_val - 360, min_val), slope=1, ls="-", color="black")

        local_feature_plot(ax, axes[idx, 1], values, angular=angular)
        ax.set_title(title)
        ax.set_xlabel(r"KS" + (r" [$^\circ$]" if angular else " [Å]"))
        ax.set_ylabel(r"OF" + (r" [$^\circ$]" if angular else " [Å]"))

        diff = max_val - min_val
        padding = 0.1
        lims = (min_val - padding * diff, max_val + padding * diff)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        axes[idx, 1].set_axis_off()

    return fig


def compute_stats(diff):
    """Compute statistics for the difference between two arrays."""
    return {
        "mae": np.abs(diff).mean(),
        "rmsd": np.sqrt(np.mean(diff**2)),
        "max": diff.max(),
        "min": diff.min(),
        "diff": diff,
    }


def _to_json_scalar(value):
    """Convert a numeric scalar to a JSON-safe float, mapping non-finite values to None."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _to_json_value(value):
    """Convert nested list/scalar values to JSON-safe built-in types."""
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist())
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return _to_json_scalar(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _stats_for_export(values):
    """Compute export-ready stats, ignoring non-finite values."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {}

    stats = compute_stats(finite_values)
    return {
        "mae": _to_json_scalar(stats["mae"]),
        "rmsd": _to_json_scalar(stats["rmsd"]),
        "max": _to_json_scalar(stats["max"]),
        "min": _to_json_scalar(stats["min"]),
    }


def _serialize_local_feature(feature_values, circular=False):
    """Serialize local geometry features into export-friendly data."""
    feature_values = np.asarray(feature_values)
    if feature_values.size == 0:
        return {
            "count": 0,
            "raw_rows": [],
            "ks_values": [],
            "of_values": [],
            "error_of_minus_ks": [],
            "error_stats": {},
        }

    ks_values = feature_values[:, -2]
    of_values = feature_values[:, -1]
    if circular:
        error = angular_difference(of_values, ks_values)
    else:
        error = of_values - ks_values

    return {
        "count": int(feature_values.shape[0]),
        "raw_rows": _to_json_value(feature_values),
        "ks_values": _to_json_value(ks_values),
        "of_values": _to_json_value(of_values),
        "error_of_minus_ks": _to_json_value(error),
        "error_stats": _stats_for_export(error),
    }


def build_geometry_data_record(data):
    """Build an export-data record for a single molecule."""
    dist_mat_ks = np.asarray(data["dist_mat1"])
    dist_mat_of = np.asarray(data["dist_mat2"])
    tri_indices = np.triu_indices_from(dist_mat_ks, k=1)
    ks_pairwise = dist_mat_ks[tri_indices]
    of_pairwise = dist_mat_of[tri_indices]
    pairwise_error = of_pairwise - ks_pairwise

    energy_error_mha = np.asarray(data["energies"], dtype=float) * 1000.0
    final_energy_error_mha = energy_error_mha[-1] if energy_error_mha.size > 0 else np.nan

    return {
        "name": data["name"],
        "converged": bool(data["converged"]),
        "n_atoms": int(data["n_atoms"]),
        "n_steps": int(np.asarray(data["cycles"]).size),
        "cycles": _to_json_value(np.asarray(data["cycles"], dtype=int)),
        "irmsd_angstrom": _to_json_scalar(data["irmsd_val"]),
        "relative_energy_error_mha": _to_json_scalar(
            data.get("relative_energy_error_mha", np.nan)
        ),
        "relative_energy_available": bool(data.get("relative_energy_available", False)),
        "ks_initial_energy_ha": _to_json_scalar(data.get("ks_initial_energy", np.nan)),
        "ks_final_energy_ha": _to_json_scalar(data.get("ks_final_energy", np.nan)),
        "ml_initial_energy_ha": _to_json_scalar(data.get("ml_initial_energy", np.nan)),
        "ml_final_energy_ha": _to_json_scalar(data.get("ml_final_energy", np.nan)),
        "ml_energy_at_ks_geometry_ha": _to_json_scalar(
            data.get("ml_energy_at_ks_geometry", np.nan)
        ),
        "ml_energy_at_ks_geometry_converged": bool(
            data.get("ml_energy_at_ks_geometry_converged", False)
        ),
        "ks_relaxation_mha": _to_json_scalar(data.get("ks_relaxation_mha", np.nan)),
        "ml_relaxation_mha": _to_json_scalar(data.get("ml_relaxation_mha", np.nan)),
        "geometry_energy_offset_mha": _to_json_scalar(
            data.get("geometry_energy_offset_mha", np.nan)
        ),
        "geometry_energy_offset_available": bool(
            data.get("geometry_energy_offset_available", False)
        ),
        "ks_energy_at_ml_geometry_ha": _to_json_scalar(
            data.get("ks_energy_at_ml_geometry", np.nan)
        ),
        "ks_energy_at_ml_geometry_converged": bool(
            data.get("ks_energy_at_ml_geometry_converged", False)
        ),
        "ks_energy_at_ks_geometry_ha": _to_json_scalar(
            data.get("ks_energy_at_ks_geometry", np.nan)
        ),
        "ks_geometry_energy_offset_mha": _to_json_scalar(
            data.get("ks_geometry_energy_offset_mha", np.nan)
        ),
        "ks_geometry_energy_offset_available": bool(
            data.get("ks_geometry_energy_offset_available", False)
        ),
        "energy_error_mha": {
            "trajectory": _to_json_value(energy_error_mha),
            "final": _to_json_scalar(final_energy_error_mha),
            "stats": _stats_for_export(energy_error_mha),
        },
        "pairwise_distance_angstrom": {
            "ks_values": _to_json_value(ks_pairwise),
            "of_values": _to_json_value(of_pairwise),
            "error_of_minus_ks": _to_json_value(pairwise_error),
            "error_stats": _stats_for_export(pairwise_error),
        },
        "bond_lengths_angstrom": _serialize_local_feature(data["bond_lengths"], circular=False),
        "bond_angles_degree": _serialize_local_feature(data["bond_angles"], circular=True),
        "bond_dihedrals_degree": _serialize_local_feature(data["bond_dihedrals"], circular=True),
    }


def save_geometry_data(data_for_plot: list, filename: str | Path) -> None:
    """Save per-molecule geometry optimization data for downstream plotting."""
    valid_data = [data for data in data_for_plot if data is not None]
    if not valid_data:
        logger.error("No valid data provided for geometry-data export.")
        return

    output_file = Path(filename)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "format": "geometry_opt_data",
            "version": 1,
            "error_convention": "error_of_minus_ks = OF - KS",
        },
        "molecules": [build_geometry_data_record(data) for data in valid_data],
    }

    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    logger.info(f"Saved geometry data to {output_file}")


def draw_comparison(ax, x, y, xlabel, ylabel, unit):
    diff = x - y
    stats = compute_stats(diff)
    lim = min(x.min(), y.min())
    ax.axline((lim, lim), slope=1, color="black")
    ax.plot(x, y, marker="x", linestyle="", alpha=0.5)
    ax.set_xlabel(f"{xlabel} [{unit}]")
    ax.set_ylabel(f"{ylabel} [{unit}]")
    ax.set_aspect("equal")

    text = (
        f"MAE  = {format_geometry_stat(stats['mae'], unit)}\n"
        f"RMSD = {format_geometry_stat(stats['rmsd'], unit)}\n"
        f"Max  = {format_geometry_stat(stats['max'], unit)}\n"
        f"Min  = {format_geometry_stat(stats['min'], unit)}"
    )
    ax.annotate(
        text,
        xy=(1, 0.3),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )


def draw_feature_panel(ax, x, y, xlabel, ylabel, title, unit, circular=False):
    if circular:
        diff = angular_difference(x, y)
    else:
        diff = y - x
    stats = compute_stats(diff)

    lim = min(x.min(), y.min())
    ax.axline((lim, lim), slope=1, color="black")
    if circular:
        ax.axline((lim + 360, lim), slope=1, color="black")
        ax.axline((lim - 360, lim), slope=1, color="black")

    ax.plot(x, y, ls="", marker="o", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    min_val, max_val = min(x.min(), y.min()), max(x.max(), y.max())
    padding = 0.1 * (max_val - min_val)
    ax.set_xlim(min_val - padding, max_val + padding)

    text = (
        f"MAE  = {format_geometry_stat(stats['mae'], unit)}    RMSD = {format_geometry_stat(stats['rmsd'], unit)}\n"
        f"Max  = {format_geometry_stat(stats['max'], unit)}    Min  = {format_geometry_stat(stats['min'], unit)}\n"
        f"N = {diff.size}"
    )
    ax.annotate(
        text,
        xy=(0.5, -0.35),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
    )


def plot_molecule_image(rdkit_mol):
    """Plots a molecule using RDKit as 2D in its actual conformation and returns the image."""
    rdDepictor.SetPreferCoordGen(True)
    size = 750
    d2d = Draw.MolDraw2DCairo(size, size * 2 // 3)
    opts = d2d.drawOptions()
    opts.setBackgroundColour((1, 1, 1, 0))
    opts.addAtomIndices = True
    opts.bondLineWidth = 4
    opts.minFontSize = 3
    opts.annotationFontScale = 0.7
    d2d.DrawMolecule(rdkit_mol)

    return Draw._drawerToImage(d2d)


def plot_molecule_image_for_summary(rdkit_mol, converged=False):
    """Plots a molecule using RDKit as 2D  Lewis structure cleaned up and returns the image."""
    size = 750
    mol = cleanup_via_smiles(rdkit_mol)
    if mol is None:
        mol = rdkit_mol
    d2d = Draw.MolDraw2DCairo(size, size * 2 // 3)
    opts = d2d.drawOptions()
    if converged:
        opts.setBackgroundColour((221 / 255, 238 / 255, 220 / 255, 0.5))
    else:
        opts.setBackgroundColour((1, 204 / 255, 203 / 255, 0.5))

    # opts.addAtomIndices = True
    opts.bondLineWidth = 4
    opts.minFontSize = 3
    opts.annotationFontScale = 0.7
    d2d.DrawMolecule(mol)
    return Draw._drawerToImage(d2d)


def plot_single_geometry_optimization(sample, ks_mol, ks_data, name="", return_features=False):
    """Plot geometry optimization results for a single sample.

    Args:
        sample (Sample): The sample object from a geometry optimization.
        ks_mol (Molecule): The KS molecule object against which to compare.
        ks_data (dict): The KS data containing energy and other scf properties.
        name (str): The name of the sample for labeling.
        return_features (bool): Whether to return the features used for plotting.

    Returns:
        fig (Figure): The matplotlib figure object.
        features (dict): A dictionary containing the features used for plotting.
    """
    of_mol = sample.mol
    n_atoms = of_mol.natm
    if isinstance(sample.energy, Energies):
        e_tot_of = sample.energy.total_energy
    else:
        e_tot_of = sample.energy
    e_tot_ks = ks_data["e_tot"]

    converged = as_scalar_bool(sample.geomopt_converged, name="geomopt_converged")
    cycles = np.arange(sample.geomopt_cycle)

    if isinstance(sample.geomopt_energies[0], Energies):
        geomopt_total_energies = np.asarray(
            [energy.total_energy for energy in sample.geomopt_energies], dtype=float
        )
    else:
        geomopt_total_energies = np.asarray(sample.geomopt_energies, dtype=float)
    energies = geomopt_total_energies - e_tot_ks

    initial_ml_energy = geomopt_total_energies[0] if geomopt_total_energies.size > 0 else np.nan
    final_ml_energy = float(e_tot_of)
    final_ks_energy = float(e_tot_ks)
    initial_ks_energy_raw = ks_data.get("initial_energy")
    if initial_ks_energy_raw is None:
        initial_ks_energy = np.nan
    else:
        initial_ks_energy = as_scalar_float(initial_ks_energy_raw, name="initial_energy")

    if np.isfinite(initial_ks_energy):
        ks_relaxation_mha = (initial_ks_energy - final_ks_energy) * 1000.0
        ml_relaxation_mha = (initial_ml_energy - final_ml_energy) * 1000.0
        relative_energy_error_mha = ml_relaxation_mha - ks_relaxation_mha
        relative_energy_available = True
    else:
        ks_relaxation_mha = np.nan
        ml_relaxation_mha = np.nan
        relative_energy_error_mha = np.nan
        relative_energy_available = False

    ml_energy_at_ks_geometry = optional_sample_float(sample, ML_ENERGY_AT_KS_GEOMETRY)
    ml_energy_at_ks_geometry_converged = optional_sample_bool(
        sample, ML_ENERGY_AT_KS_GEOMETRY_CONVERGED
    )
    geometry_energy_offset_mha = optional_sample_float(sample, GEOMETRY_ENERGY_OFFSET) * 1000.0
    geometry_energy_offset_available = optional_sample_bool(
        sample, GEOMETRY_ENERGY_OFFSET_AVAILABLE
    ) and np.isfinite(geometry_energy_offset_mha)

    ks_energy_at_ml_geometry = optional_sample_float(sample, KS_ENERGY_AT_ML_GEOMETRY)
    ks_energy_at_ml_geometry_converged = optional_sample_bool(
        sample, KS_ENERGY_AT_ML_GEOMETRY_CONVERGED
    )
    ks_energy_at_ks_geometry = optional_sample_float(sample, KS_ENERGY_AT_KS_GEOMETRY)
    ks_geometry_energy_offset_mha = (
        optional_sample_float(sample, KS_GEOMETRY_ENERGY_OFFSET) * 1000.0
    )
    ks_geometry_energy_offset_available = optional_sample_bool(
        sample, KS_GEOMETRY_ENERGY_OFFSET_AVAILABLE
    ) and np.isfinite(ks_geometry_energy_offset_mha)

    if not converged:
        geometry_energy_offset_available = False
        geometry_energy_offset_mha = np.nan
        ks_geometry_energy_offset_available = False
        ks_geometry_energy_offset_mha = np.nan

    (
        dist_mat1,
        dist_mat2,
        bond_lengths,
        bond_angles,
        bond_dihedrals,
        rdkit_mol,
        irmsd_val,
    ) = get_geometry_features(ks_mol, of_mol, align=True)
    if irmsd_val is None:
        irmsd_val = np.nan

    atom_labels = [of_mol.atom_symbol(i) for i in range(of_mol.natm)]
    atom_ticks = list(range(of_mol.natm))

    fig = plt.figure(constrained_layout=True, figsize=(8.27, 11.69))

    fig.suptitle(f"Geometry Optimization: {name}", fontsize=16)
    subfigures = fig.subfigures(3, 1, height_ratios=[1, 2, 1])

    # Molecule and Energy Plot
    ax_img, ax_energy = subfigures[0].subplots(1, 2, width_ratios=[1, 2])
    ax_img.imshow(plot_molecule_image(rdkit_mol))
    ax_img.set_axis_off()
    title = (
        f"Converged: {converged} dE = {(e_tot_of - e_tot_ks) * 1000:2.4f} mHa "
        f"iRMSD: {irmsd_val:2.3f} Å"
    )
    if geometry_energy_offset_available:
        title += f"\nML E_GEO = {geometry_energy_offset_mha:2.4f} mHa"
    if ks_geometry_energy_offset_available:
        title += f"\nKS E_GEO = {ks_geometry_energy_offset_mha:2.4f} mHa"
    ax_energy.set_title(title)
    ax_energy.plot(cycles, energies)

    # Distance Matrices
    subfig = subfigures[1]
    subfig.suptitle("Distance Matrices")
    (ax_ks, ax_of), (ax_diff, ax_scatter) = subfig.subplots(2, 2)

    im = ax_ks.matshow(dist_mat1)
    subfig.colorbar(im, ax=ax_ks, fraction=0.046, pad=0.04)
    ax_ks.set_title("KS")

    im = ax_of.matshow(dist_mat2)
    subfig.colorbar(im, ax=ax_of, fraction=0.046, pad=0.04)
    ax_of.set_title("OF")

    im = ax_diff.matshow(dist_mat1 - dist_mat2, cmap="bwr", norm=mpl.colors.CenteredNorm())
    subfig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
    ax_diff.set_title("Difference")

    for ax in [ax_ks, ax_of, ax_diff]:
        ax.set_xticks(atom_ticks, labels=atom_labels)
        ax.set_yticks(atom_ticks, labels=atom_labels)

    x_vals = dist_mat1[np.triu_indices_from(dist_mat1, k=1)]
    y_vals = dist_mat2[np.triu_indices_from(dist_mat2, k=1)]
    draw_comparison(ax_scatter, x_vals, y_vals, "KS", "OF", "A")
    ax_scatter.set_title("Distances")

    # Bond Features
    subfig = subfigures[2]
    subfig.suptitle("Bond Features")
    axes = subfig.subplots(1, 3)

    draw_feature_panel(
        axes[0],
        bond_lengths[:, -2],
        bond_lengths[:, -1],
        "KS [A]",
        "OF [A]",
        "Bond Lengths",
        "A",
    )

    if bond_angles.size > 0:
        draw_feature_panel(
            axes[1],
            bond_angles[:, -2],
            bond_angles[:, -1],
            r"KS [$^\circ$]",
            r"OF [$^\circ$]",
            "Bond Angles",
            "°",
            circular=True,
        )
    else:
        axes[1].set_axis_off()
        axes[1].annotate(
            "No Angles Found",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
        )

    if bond_dihedrals.size > 0:
        draw_feature_panel(
            axes[2],
            bond_dihedrals[:, -2],
            bond_dihedrals[:, -1],
            r"KS [$^\circ$]",
            r"OF [$^\circ$]",
            "Dihedrals",
            "°",
            circular=True,
        )
    else:
        axes[2].set_axis_off()
        axes[2].annotate(
            "No Dihedrals Found",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"),
        )

    if return_features:
        return (
            fig,
            {
                "name": name,
                "energies": energies,
                "n_atoms": n_atoms,
                "cycles": cycles,
                "converged": converged,
                "dist_mat1": dist_mat1,
                "dist_mat2": dist_mat2,
                "bond_lengths": bond_lengths,
                "bond_angles": bond_angles,
                "bond_dihedrals": bond_dihedrals,
                "rdkit_mol": rdkit_mol,
                "irmsd_val": irmsd_val,
                "relative_energy_error_mha": relative_energy_error_mha,
                "relative_energy_available": relative_energy_available,
                "ks_initial_energy": initial_ks_energy,
                "ks_final_energy": final_ks_energy,
                "ml_initial_energy": initial_ml_energy,
                "ml_final_energy": final_ml_energy,
                "ks_relaxation_mha": ks_relaxation_mha,
                "ml_relaxation_mha": ml_relaxation_mha,
                "ml_energy_at_ks_geometry": ml_energy_at_ks_geometry,
                "ml_energy_at_ks_geometry_converged": ml_energy_at_ks_geometry_converged,
                "geometry_energy_offset_mha": geometry_energy_offset_mha,
                "geometry_energy_offset_available": geometry_energy_offset_available,
                "ks_energy_at_ml_geometry": ks_energy_at_ml_geometry,
                "ks_energy_at_ml_geometry_converged": ks_energy_at_ml_geometry_converged,
                "ks_energy_at_ks_geometry": ks_energy_at_ks_geometry,
                "ks_geometry_energy_offset_mha": ks_geometry_energy_offset_mha,
                "ks_geometry_energy_offset_available": ks_geometry_energy_offset_available,
            },
        )
    return fig


def plot_summary_geometry_optimization(
    data_for_plot: list, filename: str | Path = "geometry_opt_summary.pdf"
):
    """Plots a summary of geometry optimization results.

    Args:
        data_for_plot: List of dictionaries containing data for each sample, generated via plot_single_geometry_optimization.
        filename: The name of the output PDF file.
    """
    # Filter out None values (this happens if an error occurred)
    data_for_plot = [data for data in data_for_plot if data is not None]
    if not data_for_plot:
        logger.error("No valid data provided for plotting.")
        return

    # Unpack data
    names = [data["name"] for data in data_for_plot]
    ks_rdkit_mols = [data["ks_rdkit_mol"] for data in data_for_plot]
    geomopt_cycles = [data["cycles"] for data in data_for_plot]
    converged_mask = np.asarray([data["converged"] for data in data_for_plot], dtype=bool)
    energies = [data["energies"] for data in data_for_plot]
    rmsds = np.asarray([data["irmsd_val"] for data in data_for_plot])
    n_atoms = np.asarray([data["n_atoms"] for data in data_for_plot])
    absolute_energy_errors_mha = np.asarray(
        [
            (
                np.abs(np.asarray(data["energies"], dtype=float)[-1]) * 1000.0
                if np.asarray(data["energies"], dtype=float).size > 0
                else np.nan
            )
            for data in data_for_plot
        ],
        dtype=float,
    )
    relative_energy_errors_mha = np.asarray(
        [data.get("relative_energy_error_mha", np.nan) for data in data_for_plot],
        dtype=float,
    )
    geometry_energy_offsets_mha = np.asarray(
        [
            (
                data.get("geometry_energy_offset_mha", np.nan)
                if data.get("geometry_energy_offset_available", False)
                else np.nan
            )
            for data in data_for_plot
        ],
        dtype=float,
    )
    ks_geometry_energy_offsets_mha = np.asarray(
        [
            (
                data.get("ks_geometry_energy_offset_mha", np.nan)
                if data.get("ks_geometry_energy_offset_available", False)
                else np.nan
            )
            for data in data_for_plot
        ],
        dtype=float,
    )

    if any(data["converged"] for data in data_for_plot):
        ks_converged_distances = np.concatenate(
            [
                data["dist_mat1"][np.triu_indices_from(data["dist_mat1"], k=1)]
                for data in data_for_plot
                if data["converged"]
            ]
        )
        of_converged_distances = np.concatenate(
            [
                data["dist_mat2"][np.triu_indices_from(data["dist_mat2"], k=1)]
                for data in data_for_plot
                if data["converged"]
            ]
        )
    else:
        ks_converged_distances = np.array([])
        of_converged_distances = np.array([])

    if any(not data["converged"] for data in data_for_plot):
        ks_not_converged_distances = np.concatenate(
            [
                data["dist_mat1"][np.triu_indices_from(data["dist_mat1"], k=1)]
                for data in data_for_plot
                if not data["converged"]
            ]
        )
        of_not_converged_distances = np.concatenate(
            [
                data["dist_mat2"][np.triu_indices_from(data["dist_mat2"], k=1)]
                for data in data_for_plot
                if not data["converged"]
            ]
        )
    else:
        ks_not_converged_distances = np.array([])
        of_not_converged_distances = np.array([])

    bond_lengths = [data["bond_lengths"] for data in data_for_plot]
    bond_angles = [data["bond_angles"] for data in data_for_plot]
    bond_dihedrals = [data["bond_dihedrals"] for data in data_for_plot]

    no_samples = converged_mask.size

    width, height = 8.27, 11.69
    with PdfPages(filename) as pdf:
        fig1 = plot_global_summary(
            geomopt_cycles,
            energies,
            rmsds,
            n_atoms,
            converged_mask,
            of_converged_distances,
            ks_converged_distances,
            of_not_converged_distances,
            ks_not_converged_distances,
            relative_energy_errors_mha=relative_energy_errors_mha,
            geometry_energy_offsets_mha=geometry_energy_offsets_mha,
            ks_geometry_energy_offsets_mha=ks_geometry_energy_offsets_mha,
        )
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2 = plot_local_geometry(bond_lengths, bond_angles, bond_dihedrals, converged_mask)
        pdf.savefig(fig2)
        plt.close(fig2)

        # Molecule overview: use fewer tiles per page to keep images and labels readable.
        overview_nrows = 4
        overview_ncols = 3

        counter = 0
        while counter < no_samples:
            fig = plt.figure(constrained_layout=True, figsize=(width, height))
            axes = fig.subplots(nrows=overview_nrows, ncols=overview_ncols)
            for i, ax in enumerate(axes.flat):
                if counter < no_samples:
                    rdkit_mol = ks_rdkit_mols[counter]
                    converged = converged_mask[counter]
                    name = names[counter]
                    irmsd = rmsds[counter]
                    abs_e = absolute_energy_errors_mha[counter]
                    rel_e = relative_energy_errors_mha[counter]
                    geo_e = geometry_energy_offsets_mha[counter]
                    ks_geo_e = ks_geometry_energy_offsets_mha[counter]

                    img = plot_molecule_image_for_summary(rdkit_mol, converged=converged)

                    ax.imshow(img)
                    ax.set_title(
                        "\n".join(
                            [
                                f"ID: {name}",
                                f"iRMSD: {format_irmsd_value(irmsd)}",
                                f"|ΔE|: {format_energy_mha_value(abs_e)}",
                                f"ΔE_rel: {format_energy_mha_value(rel_e)}",
                                f"ML E_GEO: {format_energy_mha_value(geo_e)}",
                                f"KS E_GEO: {format_energy_mha_value(ks_geo_e)}",
                            ]
                        ),
                        fontsize=10,
                    )

                    counter += 1

                ax.set_axis_off()

            pdf.savefig(fig)
            plt.close(fig)


def plot_data_from_path(
    ofdft_path,
    ks_path,
    summary_output=None,
    data_output=None,
):
    """Plot data from the specified OFDFT and KS calculation paths.

    Args:
        ofdft_path: Path to OF-DFT calculation outputs.
        ks_path: Path to KS calculation outputs.
        summary_output: Optional custom output path for the summary PDF.
        data_output: Optional custom output path for geometry data JSON.
    """
    ks_opt_path = ks_path / "ks-opt"
    ofdft_opt_path = ofdft_path / "geometry_opt_sample"
    (ofdft_path / "geometry_opt_plot").mkdir(exist_ok=True)
    map_location = torch.device("cpu")

    data_for_plot = []

    for of_file in sorted(ofdft_opt_path.glob("*.pt")):
        basename = of_file.stem
        ks_file = ks_opt_path / f"{basename}.chk"
        logger.info(f"Plotting {of_file.name} against {ks_file.name}")
        try:
            ks_mol, ks_data = load_scf_with_optional_data(
                ks_file,
                optional_keys=("initial_energy",),
            )
            sample = torch.load(
                of_file,
                weights_only=False,
                map_location=map_location,
            )
        except Exception as e:
            logger.error(f"Encountered exception {e} during loading. Skipping {of_file.name}")
            continue

        try:
            ks_rdkit_mol = pyscf_to_rdkit(ks_mol, sanitize=True)

            fig, features = plot_single_geometry_optimization(
                sample, ks_mol, ks_data, name=basename, return_features=True
            )
            features["ks_rdkit_mol"] = ks_rdkit_mol
            data_for_plot.append(features)

            plot_file = ofdft_path / "geometry_opt_plot" / f"{basename}.pdf"
            fig.savefig(plot_file, bbox_inches="tight", dpi=300)

            plt.close(fig)
        except Exception as e:
            logger.error(f"Encountered exception {e} during plotting. Skipping {of_file}")
            sample_xyz_file = ofdft_path / (basename + "_sample.xyz")
            ks_xyz_file = ofdft_path / (basename + "_ks.xyz")
            sample.mol.tofile(str(sample_xyz_file), format="xyz")
            ks_mol.tofile(str(ks_xyz_file), format="xyz")
            logger.error(
                f"Dumped sample geometry to {sample_xyz_file.name} and KS geometry to {ks_xyz_file.name}"
            )

    # Plot summary
    summary_file = summary_output or (ofdft_path / "geometry_opt_summary.pdf")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    plot_summary_geometry_optimization(data_for_plot, filename=summary_file)

    data_file = data_output or (ofdft_path / "geometry_opt_publication_data.json")
    save_geometry_data(data_for_plot, data_file)


def plot_cumulative_timing_all_samples(sample_files: list[Path]):
    """Plot cumulative times for all samples as stacked bars."""
    fig, ax = plt.subplots(figsize=(12, 6))
    map_location = torch.device("cpu")

    sample_names = []
    cumulative_denop = []
    cumulative_gradient = []
    cumulative_geometric = []
    total_times = []

    for sample_file in sample_files:
        sample = torch.load(
            sample_file,
            weights_only=False,
            map_location=map_location,
        )
        denop_times = sample.geomopt_denop_times.cpu().numpy()
        gradient_times = sample.geomopt_gradient_times.cpu().numpy()
        geometric_times = sample.geomopt_geometric_times.cpu().numpy()

        sample_names.append(sample_file.stem)
        cumulative_denop.append(denop_times.sum())
        cumulative_gradient.append(gradient_times.sum())
        cumulative_geometric.append(geometric_times.sum())
        total_times.append(sample.geomopt_time)

    x = np.arange(len(sample_names))
    width = 0.6

    # Create stacked bars
    ax.bar(
        x,
        cumulative_denop,
        width,
        label="Density Optimization",
        color="#1f77b4",
        alpha=0.8,
    )
    ax.bar(
        x,
        cumulative_gradient,
        width,
        bottom=cumulative_denop,
        label="Gradient Computation",
        color="#ff7f0e",
        alpha=0.8,
    )
    ax.bar(
        x,
        cumulative_geometric,
        width,
        bottom=np.array(cumulative_denop) + np.array(cumulative_gradient),
        label="geomeTRIC",
        color="#2ca02c",
        alpha=0.8,
    )

    # Add total time as horizontal lines spanning each bar
    for i, total_time in enumerate(total_times):
        xs = [x[i] - width / 2, x[i] + width / 2]
        ax.plot(xs, [total_time, total_time], "k--", linewidth=2, zorder=10)
    # Add one invisible line for the legend
    ax.plot([], [], "k--", linewidth=2, label="Total Time (geomopt_time)")

    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Cumulative Timing Breakdown per Sample", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sample_names, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_timing_per_iteration(sample: OFData, title: str = None):
    """Plot timing breakdown per iteration for a single sample."""
    denop_times = sample.geomopt_denop_times.cpu().numpy()
    gradient_times = sample.geomopt_gradient_times.cpu().numpy()
    geometric_times = sample.geomopt_geometric_times.cpu().numpy()

    n_steps = len(denop_times)
    iterations = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked bar plot
    ax.bar(
        iterations,
        denop_times,
        label="Density Optimization",
        color="#1f77b4",
        alpha=0.8,
    )
    ax.bar(
        iterations,
        gradient_times,
        bottom=denop_times,
        label="Gradient Computation",
        color="#ff7f0e",
        alpha=0.8,
    )
    ax.bar(
        iterations,
        geometric_times,
        bottom=denop_times + gradient_times,
        label="geomeTRIC",
        color="#2ca02c",
        alpha=0.8,
    )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Timing Breakdown per Iteration", fontsize=14)

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    return fig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process OFDFT and KS calculation files.")
    parser.add_argument("ofdft_path", type=Path, help="Path to the OFDFT calculation output.")
    parser.add_argument("ks_path", type=Path, help="Path to the KS calculation output.")
    parser.add_argument(
        "-o",
        "--summary-output",
        type=Path,
        default=None,
        help=(
            "Output path for the summary PDF. "
            "Defaults to <ofdft_path>/geometry_opt_summary.pdf."
        ),
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=None,
        help=(
            "Output path for geometry data JSON. "
            "Defaults to <ofdft_path>/geometry_opt_publication_data.json."
        ),
    )
    return parser.parse_args()


def main():
    """Allows plotting of already existing geometry optimization results."""
    args = parse_args()

    plot_data_from_path(
        args.ofdft_path,
        args.ks_path,
        summary_output=args.summary_output,
        data_output=args.data_output,
    )


if __name__ == "__main__":
    main()
