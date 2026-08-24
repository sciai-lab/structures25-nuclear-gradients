"""Plotting functions for density optimization to visualize the optimization process.

The central object used is the OFData object. It is expected to contain the following attributes:
- trajectory_coeffs: List of coefficient tensors for some/each optimization step.
- trajectory_energy_total: List of total energies for some/each optimization step.
- trajectory_energy_{energy_name}: List of energies for some/each optimization step for further energy contributions.
- trajectory_gradient_norm: List of gradient norms for some/each optimization step.
- trajectory_l2_norm: List of L2 norms of the density difference to the target density for some/each optimization step.
- mol: The PySCF molecule object.
- basis_integrals: The dual basis integrals tensor.
- final_untransformed_hessian_eigenvalues: The eigenvalues of the final Hessian before transformation. (If hessian is to be plotted)
"""

import functools
import time
import warnings
from pathlib import Path
from typing import Callable, List

import numpy as np
import pyvista as pv
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from pyscf import dft, gto

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.natural_reparametrization import (
    natural_reparametrization_matrices_torch,
)
from mldft.ofdft.basis_integrals import get_overlap_matrix, get_overlap_tensor
from mldft.ofdft.chemical_properties import get_dipole_moment
from mldft.utils.molecules import (
    build_molecule_np,
    build_molecule_ofdata,
    get_mol_view_link,
)
from mldft.utils.plotting.plot_density_slices import PlotDensitySlices
from mldft.utils.plotting.radial_density import plot_radial_densities
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.visualize_3d import visualize_orbital


def plot_density_optimization(
    sample: OFData,
    coeffs_label: torch.Tensor,
    plots: List[Callable] = None,
    stopping_index: int = None,
    basis_l: torch.Tensor = None,
    final_hessian: torch.Tensor | None = None,
    figure_path: Path | str = None,
    add_mol_view_link: bool = False,
    enable_grid_operations: bool = False,
) -> plt.Figure:
    """Plot the density optimization.

    If a figure path is given, the figure is saved to that path.

    Args:
        coeffs_label: The label coefficients.
        sample: The sample used for transformations.
        coeffs_label: The label coefficients.
        plots: A list of functions to plot. If None, a default subset is shown.
        stopping_index: The index at which the optimization stopped. Optional.
        basis_l: Angular momentum of basis functions. Optional.
        final_hessian: The final hessian to plot its eigenvalues. Optional.
        figure_path: The path to save the figure. If None, the figure is not saved.
        add_mol_view_link: If True, adds a MolView link to the plot.
        enable_grid_operations: Whether to allow plots requiring grid operations.
    """
    if plots is None:
        plots = [
            _add_all_energy_differences,
            _add_density_and_gradient,
            _add_coefficient_differences_lines,
            _add_dipole_moment,
        ]

        if enable_grid_operations:  # DEBUG: TODO
            plots.insert(2, _add_integrated_negative_density)
            plots.insert(-1, _add_radial_density)

        if "trajectory_grad_norm_ratio" in sample and "trajectory_grad_cos_sim" in sample:
            plots.insert(2, _add_grad_projection_metrics)
    # add hessian plot if a final hessian is given
    if final_hessian is not None:
        hessian_in_plots = any(
            plot == _add_final_hessian_eigenvalues
            or (
                isinstance(plot, functools.partial) and plot.func == _add_final_hessian_eigenvalues
            )
            for plot in plots
        )
        if not hessian_in_plots:
            plots.append(_add_final_hessian_eigenvalues)

    fig, axes = plt.subplots(
        len(plots),
        1,
        figsize=(9, 3 * len(plots)),
        gridspec_kw={"height_ratios": [1] * len(plots)},
        sharex=False,
    )
    if add_mol_view_link:
        url = get_mol_view_link(sample.mol)
        if url:
            axes[0].annotate("↗ MolView", xy=(0.9, 1.15), xycoords="axes fraction", url=url)

    coeffs_label = coeffs_label.detach()
    for plot, ax in zip(plots, axes):
        start_time = time.time()
        plot(
            ax=ax,
            sample=sample,
            coeffs_label=coeffs_label,
            basis_l=basis_l,
            stopping_index=stopping_index,
            fig=fig,
            final_hessian=final_hessian,
            enable_grid_operations=enable_grid_operations,
        )
        end_time = time.time()
        if end_time - start_time > 5:
            if isinstance(plot, functools.partial):
                plot_name = plot.func.__name__
            else:
                plot_name = plot
            logger.info(
                f"Running {str(plot_name)} took {end_time - start_time:.2f} seconds. You might want to reconsider running this for every structure."
            )

    plt.suptitle(
        f"Density Optimization for {sample.mol_id}, {mole_to_sum_formula(sample.mol, True)}"
    )
    # Removes the warning of tight_layout which happens when density slices or 3d are displayed.
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=(
            "This figure includes Axes that are not compatible with tight_layout, "
            "so results might be incorrect."
        ),
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    # fig.tight_layout()

    if figure_path is not None:
        fig.savefig(figure_path, bbox_inches="tight", dpi=150)
    return fig


def _add_grad_projection_metrics(
    ax: plt.Axes,
    sample: OFData,
    stopping_index: int = None,
    **kwargs,
):
    """Plot gradient projection diagnostics (ratio + cosine similarity).

    Requires both ``trajectory_grad_norm_ratio`` and ``trajectory_grad_cos_sim`` in ``sample``.
    """

    grad_norm_ratio = sample.trajectory_grad_norm_ratio
    grad_cos_sim = sample.trajectory_grad_cos_sim

    if grad_norm_ratio is None or grad_cos_sim is None:
        return

    x = torch.arange(len(grad_norm_ratio))
    ratio_top = max(1.0, 1.05 * torch.max(grad_norm_ratio).item())

    ax.plot(x, grad_norm_ratio, color="tab:blue", label=r"$\|g'\|/\|g\|$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|g'\|/\|g\|$")
    ax.set_ylim(0.0, ratio_top)
    ax.grid()

    ax_twin = ax.twinx()
    ax_twin.plot(x, grad_cos_sim, color="tab:orange", label=r"$\cos(g, g')$")
    ax_twin.set_ylabel(r"$\cos(g, g')$")
    ax_twin.set_ylim(top=1.0)
    if grad_cos_sim.min() >= 0:
        ax_twin.set_ylim(bottom=0.0)
    ax_twin.grid(False)

    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title("Gradient projection diagnostics")


def filter_plotting_functions(
    plotting_functions: List[Callable],
    enable_grid_operations: bool = False,
    compute_final_hessian: bool = False,
) -> List[Callable]:
    filtered_plots = []
    plots_requiring_grid = [
        "_add_integrated_negative_density",
        "_add_radial_density",
        "_add_plot_density_slices",
    ]
    final_hessian = ["_add_final_hessian_eigenvalues"]
    for plotting_function in plotting_functions:
        if isinstance(plotting_function, functools.partial):
            name = plotting_function.func.__name__
            if not enable_grid_operations and name in plots_requiring_grid:
                logger.warning(f"Skipping {name} as grid operations were disabled.")
                continue
            if not compute_final_hessian and name in final_hessian:
                logger.warning(f"Skipping {name} as computing the hessian was disabled.")
                continue
            filtered_plots.append(plotting_function)
        else:
            raise TypeError("Plotting function should be of type functools.partial.")
    return filtered_plots


def compute_ks_overlap_projection_eigendecomposition(sample: OFData):
    """Compute eigendecomposition of the KS overlap-projection matrix.

    This constructs the overlap tensor between the OFData basis and the
    Kohn-Sham (KS) orbital basis, forms the associated projection matrix in
    the natural-reparametrized basis, and computes its eigendecomposition.

    The overlap tensor is first built between the molecular AO basis and the
    KS orbital basis, then contracted to a projection matrix. This matrix is
    transformed using the natural reparametrization defined by the AO overlap
    matrix before diagonalization.

    Args:
        sample: OFData instance containing at least

            - ``mol``: PySCF molecule object.
            - ``ks_basis``: KS basis specification used to build the
              orbital-basis molecule.
            - ``trajectory_coeffs``: Tensor used to infer dtype and device
              for all constructed tensors.

    Returns:
        transformation_matrix: Tensor mapping from the overlap-matrix basis
            to the natural-reparametrized basis as returned by
            ``natural_reparametrization_matrices_torch``.
        eigenvalues: 1D tensor of eigenvalues of the KS overlap-projection
            matrix in the natural-reparametrized basis.
        eigenvectors: 2D tensor whose columns are the eigenvectors
            corresponding to ``eigenvalues``.
    """
    mol = sample.mol
    mol_orbital_basis = build_molecule_ofdata(sample, sample.ks_basis)
    overlap_tensor = get_overlap_tensor(mol, mol_orbital_basis)
    overlap_tensor = torch.tensor(
        overlap_tensor,
        dtype=sample.trajectory_coeffs.dtype,
        device=sample.trajectory_coeffs.device,
    )
    overlap_matrix = get_overlap_matrix(mol)
    overlap_matrix = torch.tensor(
        overlap_matrix,
        dtype=sample.trajectory_coeffs.dtype,
        device=sample.trajectory_coeffs.device,
    )
    (
        transformation_matrix,
        inv_transformation_matrix,
    ) = natural_reparametrization_matrices_torch(
        overlap_matrix,
    )
    A = torch.einsum("iab, jab -> ij", overlap_tensor, overlap_tensor)
    A = inv_transformation_matrix.T @ A @ inv_transformation_matrix

    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    return transformation_matrix, eigenvalues, eigenvectors


def _add_coeffs_over_eigenvalues(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    transformation_matrix, eigenvalues, eigenvectors = (
        compute_ks_overlap_projection_eigendecomposition(sample)
    )

    # transform coefficients
    init_coeffs = transformation_matrix @ sample.trajectory_coeffs[0]
    converged_coeffs = transformation_matrix @ sample.trajectory_coeffs[stopping_index]
    ground_truth_coeffs = transformation_matrix @ coeffs_label

    # transform
    init_eigen = torch.abs(eigenvectors.T @ init_coeffs).clamp_min(
        torch.finfo(sample.trajectory_coeffs.dtype).tiny
    )
    converged_eigen = torch.abs(eigenvectors.T @ converged_coeffs).clamp_min(
        torch.finfo(sample.trajectory_coeffs.dtype).tiny
    )
    ground_truth_coeffs_eigen = torch.abs(eigenvectors.T @ ground_truth_coeffs).clamp_min(
        torch.finfo(sample.trajectory_coeffs.dtype).tiny
    )

    # Sort by eigenvalues for plotting
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[sorted_indices].cpu()
    init_eigen_sorted = init_eigen[sorted_indices].cpu()
    converged_eigen_sorted = converged_eigen[sorted_indices].cpu()
    ground_truth_coeffs_eigen_sorted = ground_truth_coeffs_eigen[sorted_indices].cpu()
    ax.plot(
        eigenvalues_sorted,
        init_eigen_sorted,
        label="Initial Guess",
        marker="o",
        linestyle="None",
        markersize=1,
        alpha=0.3,
    )
    ax.plot(
        eigenvalues_sorted,
        converged_eigen_sorted,
        label="Converged",
        marker="o",
        linestyle="None",
        markersize=1,
        alpha=0.3,
    )
    ax.plot(
        eigenvalues_sorted,
        ground_truth_coeffs_eigen_sorted,
        label="Ground Truth",
        marker="o",
        linestyle="None",
        markersize=1,
        alpha=0.3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue of Overlap Matrix")
    ax.set_ylabel("|Coefficients| in Eigenbasis")
    ax.set_title("Coefficients over Eigenvalues")
    ax.legend()
    ax.grid()


def _add_coeffs_diff_over_eigenvalues(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    transformation_matrix, eigenvalues, eigenvectors = (
        compute_ks_overlap_projection_eigendecomposition(sample)
    )

    init_coeffs = transformation_matrix @ sample.trajectory_coeffs[0]
    converged_coeffs = transformation_matrix @ sample.trajectory_coeffs[stopping_index]
    ground_truth_coeffs = transformation_matrix @ coeffs_label

    init_delta = ground_truth_coeffs - init_coeffs
    converged_delta = ground_truth_coeffs - converged_coeffs

    # transform deltas to eigenbasis
    init_delta_eigen = eigenvectors.T @ init_delta
    converged_delta_eigen = eigenvectors.T @ converged_delta

    # Sort by eigenvalues for cumulative plot
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[sorted_indices].cpu()
    init_delta_eigen_sorted = torch.abs(init_delta_eigen[sorted_indices]).cpu()
    converged_delta_eigen_sorted = torch.abs(converged_delta_eigen[sorted_indices]).cpu()

    # Compute cumulative sums
    init_delta_cumsum = torch.cumsum(init_delta_eigen_sorted**2, dim=0).sqrt()
    converged_delta_cumsum = torch.cumsum(converged_delta_eigen_sorted**2, dim=0).sqrt()

    # Normalize to 100%
    init_delta_cumsum = (
        100
        * init_delta_cumsum
        / init_delta_cumsum[-1].clamp_min(torch.finfo(init_delta_cumsum.dtype).tiny)
    )
    converged_delta_cumsum = (
        100
        * converged_delta_cumsum
        / converged_delta_cumsum[-1].clamp_min(torch.finfo(converged_delta_cumsum.dtype).tiny)
    )

    # Split axis into two subplots
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    fig = ax.figure
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec(), wspace=0.3)
    ax.remove()
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # Left plot: individual coefficients
    ax_left.plot(
        eigenvalues_sorted,
        init_delta_eigen_sorted,
        marker="o",
        linestyle="None",
        markersize=1,
        alpha=0.3,
        label="Initial Guess",
    )
    ax_left.plot(
        eigenvalues_sorted,
        converged_delta_eigen_sorted,
        marker="o",
        linestyle="None",
        markersize=1,
        alpha=0.3,
        label="Converged",
    )
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_xlabel("Eigenvalue of Overlap Matrix")
    ax_left.set_ylabel(r"$|\Delta c|$")
    ax_left.set_title("Coefficient Differences in Overlap Eigenbasis")
    ax_left.grid()
    ax_left.legend()

    # Right plot: cumulative
    num_eigenvalues = len(eigenvalues_sorted)
    eigenvalue_percentages = 100 * torch.arange(1, num_eigenvalues + 1) / num_eigenvalues

    ax_right.plot(
        eigenvalues_sorted,
        init_delta_cumsum,
        linestyle="-",
        linewidth=2,
        alpha=0.7,
        label="Initial Guess",
    )
    ax_right.plot(
        eigenvalues_sorted,
        converged_delta_cumsum,
        linestyle="-",
        linewidth=2,
        alpha=0.7,
        label="Converged",
    )
    ax_right.set_xscale("log")
    ax_right.set_xlabel("Eigenvalue of Overlap Matrix")
    ax_right.set_ylabel(r"Cumulative $|\Delta c|$ [%]")
    ax_right.set_title("Cumulative Coefficient Differences")
    ax_right.grid()

    # Add twin axis for eigenvalue percentage
    ax_right_twin = ax_right.twinx()
    ax_right_twin.plot(
        eigenvalues_sorted,
        eigenvalue_percentages,
        linestyle="--",
        color="gray",
        label="Eigenvalue Percentage",
    )
    ax_right_twin.set_ylabel("Percentage of Eigenvalues [%]")
    ax_right_twin.set_yscale("linear")

    # Keep a single legend so we don't end up with overlapping legends
    # where the last-drawn (twin) legend covers the primary one.
    h1, l1 = ax_right.get_legend_handles_labels()
    h2, l2 = ax_right_twin.get_legend_handles_labels()
    ax_right.legend(h1 + h2, l1 + l2, loc="upper left")


def _add_coeffs_over_eigenvalues_2d(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    """Plot 2D histogram of coefficient changes over eigenvalues and iterations.

    Args:
        ax: The axis to add the plot to.
        sample: The denop sample.
        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    transformation_matrix, eigenvalues, eigenvectors = (
        compute_ks_overlap_projection_eigendecomposition(sample)
    )

    # Transform all trajectory coefficients
    ground_truth_coeffs = transformation_matrix @ coeffs_label
    trajectory_coeffs_transformed = torch.stack(
        [transformation_matrix @ c for c in sample.trajectory_coeffs]
    )

    # Compute deltas for all iterations
    deltas = trajectory_coeffs_transformed - ground_truth_coeffs  # shape: (n_iter, n_basis)

    # Transform to eigenbasis
    deltas_eigen = torch.einsum("ij,nj->ni", eigenvectors.T, deltas)  # shape: (n_iter, n_basis)
    deltas_eigen_abs = torch.abs(deltas_eigen).cpu().numpy()

    # Sort by eigenvalues
    sorted_indices = torch.argsort(eigenvalues)
    eigenvalues_sorted = eigenvalues[sorted_indices].cpu().numpy()
    deltas_eigen_abs_sorted = deltas_eigen_abs[:, sorted_indices.cpu().numpy()]

    # Normalize each row (iteration) independently
    row_norms = np.linalg.norm(deltas_eigen_abs_sorted, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1  # Avoid division by zero
    deltas_eigen_normalized = deltas_eigen_abs_sorted / row_norms

    # Create 2D histogram
    n_iterations = deltas_eigen_normalized.shape[0]
    iterations = np.arange(n_iterations)

    # Create index array for x-axis instead of eigenvalues
    eigenvalue_indices = np.arange(len(eigenvalues_sorted))

    # Plot
    im = ax.pcolormesh(
        eigenvalue_indices,
        iterations,
        deltas_eigen_normalized,
        cmap="hot",
        shading="nearest",
        norm=plt.matplotlib.colors.LogNorm(
            vmin=deltas_eigen_normalized[deltas_eigen_normalized > 0].min(),
            vmax=deltas_eigen_normalized.max(),
        ),
    )

    ax.set_xlabel("Eigenvalue Index (sorted by eigenvalue)")
    ax.set_ylabel("Iteration")
    ax.set_title("Normalized Coefficient Changes in Overlap Eigenbasis")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Normalized $|\Delta c|$")

    # Add twin x-axis showing actual eigenvalues
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    # Set ticks at regular intervals in index space
    n_ticks = 5
    tick_indices = np.linspace(0, len(eigenvalues_sorted) - 1, n_ticks, dtype=int)
    ax_twin.set_xticks(tick_indices)
    ax_twin.set_xticklabels([f"{eigenvalues_sorted[i]:.2e}" for i in tick_indices])
    ax_twin.set_xlabel("Eigenvalue of Overlap Matrix")

    if stopping_index is not None:
        ax.axhline(
            stopping_index, color="cyan", linestyle="--", linewidth=2, label="Stopping Index"
        )


def _add_density_difference(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    r"""Add the density difference to the given axis.

    The minimal density difference to the ground state achievable with the initial normalization
    :math:`N` is given by

    .. math::
        \min_{w^T p = N} \|\rho(p) - \rho_\mathrm{target}\|_2
        = \frac{|N - N_\mathrm{target}|}{\sqrt{w^T W^{-1} w}}.

    Args:
        ax: The axis to add the density difference to.
        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    # Calculate minimal achievable norm if overlap matrix is small enough
    label = r"\|\rho_\mathrm{pred} - \rho_\mathrm{target}\|_2"
    minimal_norm = None

    basis_integrals = sample.basis_integrals
    traj0 = sample.trajectory_coeffs[0]
    if not torch.is_tensor(traj0):
        traj0 = torch.as_tensor(traj0)
    n_elec_label = basis_integrals @ coeffs_label
    n_elec_init = basis_integrals @ traj0

    if coeffs_label.shape[0] <= 2048:
        overlap_matrix = torch.tensor(
            get_overlap_matrix(sample.mol),
            dtype=traj0.dtype,
            device=traj0.device,
        )
        inv_overlap = torch.linalg.inv(overlap_matrix)
        denominator = torch.sqrt(basis_integrals @ inv_overlap @ basis_integrals)

        minimal_norm = (torch.abs(n_elec_init - n_elec_label) / denominator).item()

    l2_norm = sample.trajectory_l2_norm
    if minimal_norm is not None:
        label = rf"${label} \geq {minimal_norm:.2g}$"
    ax.plot(l2_norm, label=label)
    ax.set_ylabel(r"$\Vert\rho_\mathrm{pred} - \rho_\mathrm{target}\Vert_2$ " "[electrons]")
    ax.set_yscale("log")
    ax.grid()
    title = "Density difference"
    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")
        title += f". Error at stopping index: {l2_norm[stopping_index]:.2g} electrons"
    ax.set_title(title)
    ax.set_xlabel("Iteration")


def _add_all_energy_differences(
    ax: plt.Axes,
    sample: OFData,
    stopping_index: int = None,
    **kwargs,
):
    """Add all energy differences to the given axis.

    Args:
        ax: The axis to add the energy differences to.
        sample: The denop sample.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    linthresh = 10

    total_energy = sample.trajectory_energy_total
    ax.plot(
        1e3 * (total_energy - sample.ground_state_energy_total), label=r"$\Delta E_\mathrm{tot}$"
    )
    # Get needed energy names and values purely from the sample object
    energy_names = []

    for key in sample.keys():
        if key.startswith("ground_state_energy_"):
            energy_name = key[len("ground_state_energy_") :]
            energy_names.append(energy_name)

    energy_names.remove("total")
    energy_names.remove("electronic")

    # available energies in sample
    for energy_name in energy_names:
        if energy_name == "nuclear_repulsion":
            continue
        try:
            energy_np = getattr(sample, f"trajectory_energy_{energy_name}")
        except KeyError:
            logger.warning(f"Energy 'trajectory_energy_{energy_name}' not found in sample")
            continue
        label = r"$\Delta E_\mathrm{" + energy_name.replace("_", r"\_") + "}$"
        ax.plot(
            1e3 * (energy_np - getattr(sample, f"ground_state_energy_{energy_name}")), label=label
        )

    ax.axhline(0, color="black")
    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--", label="stopping index")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta E$ [mHa]")
    ax.set_yscale("symlog", linthresh=linthresh)
    title = "Energy differences"
    if stopping_index is not None:
        err_mHa = float(1e3 * (total_energy[stopping_index] - sample.ground_state_energy_total))
        title += f". Energy error at stopping index: {err_mHa:.2f} mHa"
    ax.set_title(title)
    ax.legend()
    ax.grid(which="both")
    for h in torch.linspace(-linthresh, linthresh, 21)[1:-1]:
        ax.axhline(h, color="black", alpha=0.1, linestyle="--", linewidth=0.5)


def _add_density_and_gradient(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    """Add the density and gradient to the given axis as a pixel image.

    Args:
        ax: The axis to add the density and gradient to.
        sample: The sample used for transformations.

        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    _add_density_difference(
        ax,
        sample=sample,
        coeffs_label=coeffs_label,
        stopping_index=stopping_index,
        show_legend=False,
    )
    twin_ax = plt.twinx(ax)
    _add_gradient_norm(twin_ax, sample, stopping_index, show_legend=False)
    twin_ax.grid(False)  # don't show grid twice

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin_ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center")
    ax.set_xlabel("Iteration")


def _add_gradient_norm(
    ax: plt.Axes,
    sample: OFData,
    stopping_index: int = None,
    show_legend: bool = True,
    **kwargs,
):
    """Add the gradient norm to the given axis.

    Args:
        ax: The axis to add the gradient norm to.
        sample: The denop sample.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    ax.plot(sample.trajectory_gradient_norm, label="gradient norm", color="tab:orange")
    ax.set_ylabel("gradient norm")
    ax.set_xlabel("Iteration")
    ax.set_yscale("log")
    ax.grid()
    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")


def _add_coefficient_differences_pixels(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    **kwargs,
):
    """Add the coefficient differences to the given axis as a pixel image.

    Args:
        ax: The axis to add the coefficient difference to.
        sample: The denop sample object.
        coeffs_label: The label coefficients.
        **kwargs: Additional arguments for uniform plot interface.
    """
    coeffs = (
        sample.trajectory_coeffs
        if torch.is_tensor(sample.trajectory_coeffs)
        else torch.stack(sample.trajectory_coeffs)
    )
    coeffs_delta = (coeffs - coeffs_label).cpu()
    coeffs_delta_max = torch.max(torch.abs(coeffs_delta)).item()
    img = ax.imshow(
        coeffs_delta.T.detach().numpy(),
        aspect="auto",
        cmap="seismic",
        interpolation="none",
        vmin=-coeffs_delta_max,
        vmax=coeffs_delta_max,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"basis function")
    plt.colorbar(img, ax=ax, location="top")


def _add_coefficient_differences_lines(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    basis_l: torch.Tensor = None,
    stopping_index: int = None,
    **kwargs,
):
    """Add the coefficient differences to the given axis as a plot.

    Args:
        ax: The axis to add the coefficient difference to.
        sample: The denop sample object.
        coeffs_label: The label coefficients.
        basis_l: Angular momentum of basis functions.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    coeffs = (
        sample.trajectory_coeffs
        if torch.is_tensor(sample.trajectory_coeffs)
        else torch.stack(sample.trajectory_coeffs)
    )
    coeffs_delta = (coeffs - coeffs_label).cpu()

    iterations = torch.arange(coeffs_delta.shape[0]) * sample.save_coeff_interval

    if basis_l is None:
        ax.plot(iterations, coeffs_delta.detach().numpy(), lw=0.3)
    else:
        max_l = np.max(basis_l)
        orbital_labels = np.array(["s", "p", "d", "f", "g", "h"])
        l_colors = np.array(
            ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        )
        l_colors_handles = [
            Patch(facecolor=color, label=label)
            for color, label in zip(l_colors[: max_l + 1], orbital_labels)
        ]
        ax.legend(
            handles=l_colors_handles,
            title="Angular Momentum",
            fontsize="small",
            loc="upper center",
        )

        # shuffle z order
        np.random.seed(0)
        permutation = np.random.permutation(basis_l.shape[0])
        for i in permutation:
            ax.plot(
                iterations, coeffs_delta[:, i].detach().numpy(), lw=0.3, c=l_colors[basis_l[i]]
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\Delta p$")
    ax.set_title("Difference between predicted and ground state coefficients")
    ax.grid()

    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")


def _add_integrated_negative_density(
    ax: plt.Axes,
    sample: OFData,
    stopping_index: int = None,
    grid_level: int = 3,
    enable_grid_operations: bool = True,
    **kwargs,
):
    """Add the integrated negative density to the given axis.

    Args:
        ax: The axis to add the integrated negative density to.
        sample: The sample used for transformations.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    mol = sample.mol
    coeffs = sample.trajectory_coeffs

    grid = dft.Grids(mol)
    grid.level = grid_level
    grid.prune = None
    grid.build()

    weights = torch.as_tensor(grid.weights, dtype=torch.float64)
    # weights = torch.clip(weights, min=0, max=np.inf)
    ao = dft.numint.eval_ao(mol, grid.coords, deriv=0)
    ao = torch.as_tensor(ao, dtype=torch.float64, device=sample.coeffs.device)
    ao = transform_tensor_with_sample(sample, ao, Representation.AO)
    ao = ao.to("cpu")
    density = coeffs @ ao.T
    density[density > 0] = 0
    integrated_negative_density = density @ weights

    ax.plot(integrated_negative_density, label="integrated negative density")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("int. neg. density [electrons]")
    ax.grid()

    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")


def _add_radial_density(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    """Add the radial density to the given axis.

    Args:
        ax: The axis to add the radial density to.
        sample: The denop sample.
        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    plt.sca(ax)
    mol = sample.mol
    coeffs_label = transform_tensor_with_sample(
        sample, coeffs_label, Representation.VECTOR, invert=True
    )
    coeffs_init = sample.trajectory_coeffs[0]
    coeffs_init = transform_tensor_with_sample(
        sample, coeffs_init, Representation.VECTOR, invert=True
    )
    coeffs_dict = {
        "label": coeffs_label.detach().cpu().numpy(),
        # "init": coeffs_init.cpu().detach().numpy(),
    }

    if stopping_index is not None:
        coeffs_stop = sample.trajectory_coeffs[stopping_index]
        coeffs_stop = transform_tensor_with_sample(
            sample, coeffs_stop, Representation.VECTOR, invert=True
        )
        coeffs_dict["stop"] = coeffs_stop.cpu().detach().numpy()
        # coeffs_dict["diff * 1e2"] = 1e2 * (coeffs_dict["stop"] - coeffs_dict["label"])
    else:
        coeffs_last = sample.trajectory_coeffs[-1]
        coeffs_last = transform_tensor_with_sample(
            sample, coeffs_last, Representation.VECTOR, invert=True
        )
        coeffs_dict["last"] = coeffs_last.cpu().detach().numpy()

    plot_radial_densities(coeffs_dict, mol, center_atom_idx=0)


def add_inset_axes(rect, units="ax", ax_target=None, fig=None, projection=None, **kw):
    """
    Wrapper around `fig.add_axes` to achieve `ax.inset_axes` functionality
    that works also for insetting 3D plot on 2D ax/figures
    From https://stackoverflow.com/questions/36331478/in-matplotlib-how-do-i-include-a-3d-plot-as-an-inset
    """
    assert ax_target is not None or fig is not None, "`fig` or `ax_target` must be provided!"
    _units = {"ax", "norm2ax", "norm2fig"}
    assert {units} <= _units, f"`rect_units` not in {repr(_units)}"

    if ax_target is not None:
        # Inspired from:
        # https://stackoverflow.com/questions/14568545/convert-matplotlib-data-units-to-normalized-units
        bb_data = Bbox.from_bounds(*rect)
        trans = ax_target.transData if units == "ax" else ax_target.transAxes
        disp_coords = trans.transform(bb_data)
        fig = ax_target.get_figure()
        fig_coord = fig.transFigure.inverted().transform(disp_coords)
    elif fig is not None:
        if ax_target is not None and units != "norm2fig":
            bb_data = Bbox.from_bounds(*rect)
            trans = ax_target.transData if units == "ax" else ax_target.transAxes
            disp_coords = trans.transform(bb_data)
        else:
            fig_coord = Bbox.from_bounds(*rect)

    axin = fig.add_axes(Bbox(fig_coord), projection=projection, **kw)

    return axin


def _add_3d_atom_positions(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    rotate_molecule_mode: str = "pca",
    zoom: float = 0.5,
    **kwargs,
):
    """
    Args:
        ax: The axis to add the radial density to.
        sample: The denop sample.
        coeffs_label: The label coefficients.
        rotate_molecule_mode: The method used to rotate the molecule into its symmetry axis.
        **kwargs: Additional arguments for uniform plot interface.
    """
    parent_axis = ax
    mol = sample.mol
    pds0 = PlotDensitySlices.show_mol_atoms(mol, rotate_molecule_mode)
    pds0.scale = zoom
    pds0.sub_title_size = 12
    inside_axes = add_inset_axes(
        rect=[0.0, -0.5, 1.0, 1.0], units="ax", ax_target=parent_axis, projection="3d"
    )
    pds0.axes = [inside_axes]
    parent_axis.set_visible(False)
    pds0.plot_fig()


def _add_plot_density_slices(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    fig=None,
    stopping_index: int = None,
    n_pixel: int = 100,
    dpi: int = 50,
    zoom: float = 1.0,
    annotate_atoms: int = 2,
    initial_guess_index: int = 0,
    rotate_molecule_mode: str = "pca",
    mode: str = "density",
    labelmode: int = 0,
    calculate_max_range: bool = True,
    c_range: float = 0.5,
    title_left_plot="Density at stopping Criteria",
    title_right_plot="Density from label",
    **kwargs,
):
    """Plots density slices of the molecule.

    Args:
        ax: The axis to add the radial density to.
        sample: The denop sample.
        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        n_pixel: int = 100, The resolution of the plot in terms of pixels. Determines the density of points on the plot.
        dpi: int = 50, Dots per inch (DPI) for the figure, controlling the resolution of the plot. Higher values produce sharper images.
        zoom: float = 1., Zoom factor for the plot. A value of 1.0 means no zoom, while values smaller than 1.0 zoom in.
        annotate_atoms: int = 2,
        Specifies whether to annotate the atoms in the plot. The value represents the style of annotation.
        initial_guess_index: int = 0,
            Index of the initial guess used in the plot. Useful for comparisons with the final result.
        rotate_molecule_mode: str = "pca",
            Mode used for rotating the molecule in the plot. Options include "pca" (Principal Component Analysis) or other custom modes.
        mode: str = "density",
            The mode for plotting, determining what data to display. Common options are "density" or other data types.
        labelmode: int = 0,
            Specifies the labeling mode for the plot. Different values represent different label formatting or styling.
        calculate_max_range: bool = True,
            Whether to calculate the maximum range for the plot. Used to determine the scale of the plot axes.
        c_range: float = 0.5,
            Color range for the plot. Defines the range of values used for coloring the density map.
        title_left_plot: str = "Density at stopping Criteria",
            Title of the left plot, used to describe the content shown at the stopping criteria.
        title_right_plot: str = "Density from label",
            Title of the right plot, used to describe the labeled density being compared.
        **kwargs: Additional arguments for uniform plot interface.
    """
    parent_axis = ax
    mol = sample.mol
    mol = build_molecule_np(
        mol.atom_charges(),
        mol.atom_coords(),
        basis=mol.basis,
        unit="Bohr",
    )
    coeffs_label = (
        transform_tensor_with_sample(sample, coeffs_label, Representation.VECTOR, invert=True)
        .detach()
        .cpu()
        .numpy()
    )
    if stopping_index is not None:
        coeffs_stop = sample.trajectory_coeffs[stopping_index]
        coeffs_stop = transform_tensor_with_sample(
            sample, coeffs_stop, Representation.VECTOR, invert=True
        )
        coeffs_predicted = coeffs_stop.cpu().detach().numpy()
    else:
        coeffs_last = sample.trajectory_coeffs[-1]
        coeffs_last = transform_tensor_with_sample(
            sample, coeffs_last, Representation.VECTOR, invert=True
        )
        coeffs_predicted = coeffs_last.cpu().detach().numpy()
    pds = PlotDensitySlices.show_difference_of_densities(
        mol,
        coeffs_predicted,
        [mol],
        [coeffs_label],
        [title_right_plot],
    )
    pds.text_size = 12
    pds.sub_title_size = 12
    pds.n_pixel = n_pixel
    pds.mode_names = (mode,)
    pds.annote_atoms = annotate_atoms
    pds.initial_guess_index = initial_guess_index
    pds.rotate_molecule_mode = rotate_molecule_mode
    pds.labelmode = labelmode
    pds.figsize = (9, 4)
    pds.dpi = dpi
    pds.dark_mode = False
    pds.show_atoms = False
    pds.scale = zoom
    pds.fig = fig
    pds.name_comparison_mol = title_left_plot
    pds.mode_changed = calculate_max_range
    pds.c_range = c_range
    inside_axes = [
        add_inset_axes(
            rect=[-0.3, -0.5, 1.0, 1.0],
            units="ax",
            ax_target=parent_axis,
        ),
        add_inset_axes(
            rect=[0.3, -0.5, 1.0, 1.0],
            units="ax",
            ax_target=parent_axis,
        ),
    ]
    pds.axes = inside_axes
    parent_axis.set_visible(False)
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    asb = AnchoredSizeBar(
        inside_axes[0].transData,
        1.0,
        r"1 Bohr",
        loc="lower center",
        pad=0.1,
        borderpad=0.5,
        sep=5,
        frameon=False,
    )
    inside_axes[0].add_artist(asb)
    inside_axes[0].get_yaxis().set_visible(False)
    inside_axes[0].get_xaxis().set_visible(False)
    inside_axes[1].get_yaxis().set_visible(False)
    inside_axes[1].get_xaxis().set_visible(False)
    pds.plot_fig()


def _add_dipole_moment(
    ax: plt.Axes,
    sample: OFData,
    coeffs_label: torch.Tensor,
    stopping_index: int = None,
    **kwargs,
):
    """Add the dipole moment to the given axis.

    Args:
        ax: The axis to add the dipole moment to.
        sample: The denop sample.
        coeffs_label: The label coefficients.
        stopping_index: The index at which the optimization stopped. Optional.
        **kwargs: Additional arguments for uniform plot interface.
    """
    mol = sample.mol
    coeffs = sample.trajectory_coeffs
    # Handle coeffs being a tensor or list of tensors
    if not torch.is_tensor(coeffs):
        coeffs = torch.stack(coeffs)

    dipole_moments = get_dipole_moment(mol, coeffs)
    dipole_label = get_dipole_moment(mol, coeffs_label).cpu()
    delta_final = dipole_moments[-1] - dipole_label
    x_color, y_color, z_color = "#5eb517", "#175eb5", "#b5175e"

    ax.plot(dipole_moments[:, 0], label=rf"$\Delta_x$ = {delta_final[0]:.3f}", color=x_color)
    ax.plot(dipole_moments[:, 1], label=rf"$\Delta_y$ = {delta_final[1]:.3f}", color=y_color)
    ax.plot(dipole_moments[:, 2], label=rf"$\Delta_z$ = {delta_final[2]:.3f}", color=z_color)

    ax.axhline(dipole_label[0].item(), color=x_color, alpha=0.5)
    ax.axhline(dipole_label[1].item(), color=y_color, alpha=0.5)
    ax.axhline(dipole_label[2].item(), color=z_color, alpha=0.5)

    ax.set_title(f"Final difference norm of dipol moment: {torch.norm(delta_final).item():.4f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("dipole moment difference [au]")
    ax.grid()
    ax.legend(title="Final differences")

    if stopping_index is not None:
        ax.axvline(stopping_index, color="black", linestyle="--")


def _add_final_hessian_eigenvalues(
    ax: plt.Axes,
    sample: OFData,
    final_hessian: torch.Tensor,
    **kwargs,
):
    """Add the eigenvalues of the final Hessian to the given axis.

    Args:
        ax: The axis to add the eigenvalues to.
        sample: The denop sample.
        final_hessian: The final Hessian.
        **kwargs: Additional arguments for uniform plot interface.
    """
    if final_hessian is None:
        ax.set_title("No final Hessian was provided.")
        return
    eigenvalues = final_hessian
    eigenvalues = np.sort(eigenvalues.cpu().detach().numpy())
    mask = eigenvalues < -1e-6
    ax.set_title(
        f"Eigenvalues of the final Hessian, {np.sum(mask)}/{mask.shape[0]} smaller than -1e-6"
    )
    ax.scatter(np.arange(eigenvalues.shape[0]), eigenvalues, s=1, label="transformed")

    eigenvalues = sample.final_untransformed_hessian_eigenvalues
    eigenvalues = np.sort(eigenvalues.cpu().detach().numpy())
    ax.scatter(np.arange(eigenvalues.shape[0]), eigenvalues, s=1, label="untransformed")

    ax.legend()
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.axhline(0, c="k", lw=0.5)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(axis="y")


def plot_volume_and_isosurface(mol: gto.Mole, coeffs: np.ndarray, title: str = ""):
    """Plot the volume and isosurface of the given coefficients.

    Given the molecule and the coefficients, this function plots the volume and isosurface of the
    given coefficients. This is useful to visualize the density or the gradients.

    Args:
        mol: The molecule.
        coeffs: The coefficients.
        title: The title of the plot.
    """

    pv.set_jupyter_backend("static")

    images = []

    pl = pv.Plotter(window_size=[800, 600], off_screen=True, notebook=False)
    visualize_orbital(
        mol,
        coeff=coeffs,
        margin=5,
        resolution=0.2,
        plotter=pl,
    )
    pl.camera_position = (1, 2, -7)  # 'iso'
    pl.camera.zoom(3)
    images.append(pl.show(screenshot=True))

    pl = pv.Plotter(window_size=[800, 600], off_screen=True, notebook=False)
    visualize_orbital(
        mol,
        coeff=coeffs,
        margin=5,
        resolution=0.2,
        isosurface_quantile=0.8,
        plotter=pl,
    )
    pl.camera_position = (1, 2, -7)  # 'iso'
    pl.camera.zoom(1.5)
    images.append(pl.show(screenshot=True))
    pl.close()

    fig, axs = plt.subplots(1, len(images), figsize=(11, 4))
    for image, ax in zip(images, axs):
        ax.imshow(image)
        ax.set_yticks([])
        ax.set_xticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
