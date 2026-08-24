import logging
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence, Union

import numpy as np
import seaborn as sns
import torch
import yaml
from loguru import logger
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from torch import Tensor
from torch_scatter import scatter
from tqdm import tqdm

from mldft.datagen.methods.density_fitting import ksdft_density_matrix
from mldft.ml.data.components.of_data import BasisInfo, OFData
from mldft.ofdft import OFDFT_DTYPE
from mldft.utils.environ import get_mldft_data_path
from mldft.utils.grids import (
    compute_density_density_basis,
    compute_density_orbital_basis,
    grid_setup,
)
from mldft.utils.molecules import build_molecule_np, build_molecule_ofdata, load_scf
from mldft.utils.pdf_utils import HierarchicalPlotPDF
from mldft.utils.plotting.axes import format_basis_func_xaxis
from mldft.utils.plotting.symlog_locater import MinorSymLogLocator

logging.getLogger("matplotlib.category").setLevel(logging.WARNING)  # hide matplotlib.category info


def calculate_grid_metrics(
    sample: OFData,
    basis_info: BasisInfo,
    grid_level: int,
    grid_prune: str,
    compare_to_ks: bool = False,
) -> dict[str, Tensor]:
    """Calculates the L1 and L2 norm of the difference between the Kohn-Sham density and the OF
    density."""
    mol_of = sample.mol
    predicted_ground_state_coeffs = sample.predicted_ground_state_coeffs.numpy(force=True)
    ground_state_coeffs = sample.ground_state_coeffs.numpy(force=True)
    coeffs = np.stack([predicted_ground_state_coeffs, ground_state_coeffs], axis=0)
    grid = grid_setup(mol_of, grid_level, grid_prune)
    grid_weights = torch.as_tensor(grid.weights)
    rho_of_pred_np = compute_density_density_basis(mol_of, grid, predicted_ground_state_coeffs)
    rho_of_label_np = compute_density_density_basis(mol_of, grid, ground_state_coeffs)
    rho_of_pred = torch.as_tensor(rho_of_pred_np)
    rho_of_label = torch.as_tensor(rho_of_label_np)
    delta_rho_of = torch.dot(torch.abs(rho_of_pred - rho_of_label), grid_weights)
    result_dict = {"density_error_l1": delta_rho_of}

    if compare_to_ks:
        ks_subpath = Path(sample.kohn_sham_path)
        if ks_subpath is None:
            raise ValueError("Kohn-Sham path is not set in the sample.")
        dataset_str = ks_subpath.parts[0]
        with open(get_mldft_data_path() / dataset_str / "kohn_sham" / "config.yaml") as f:
            kohn_sham_config = yaml.safe_load(f)
        ks_basis_set = kohn_sham_config["basis"]
        mol_ks = build_molecule_np(
            mol_of.atom_charges(),
            mol_of.atom_coords(),
            ks_basis_set,
            unit=mol_of.unit,
            spin=mol_of.spin,
        )
        ks_file = get_mldft_data_path() / ks_subpath
        results, initialization, data_of_iteration = load_scf(ks_file)
        gamma = ksdft_density_matrix(
            results["molecular_coeffs_orbitals"], results["occupation_numbers_orbitals"]
        )
        rho_ks = compute_density_orbital_basis(mol_ks, grid, gamma)
        rho_of = torch.stack((rho_of_pred, rho_of_label))
        delta_rho = rho_of - rho_ks.unsqueeze(0)
        weighted_delta_rho = delta_rho * grid.weights
        l1_norm = weighted_delta_rho.abs().sum(dim=1)
        l2_norm = torch.sqrt((weighted_delta_rho**2).sum(dim=1))
        result_dict.update(
            {
                "density_error_l1_ks": l1_norm[0],
                "density_error_l2_ks": l2_norm[0],
                "df_error_l1_ks": l1_norm[1],
                "df_error_l2_ks": l2_norm[1],
            }
        )
    return result_dict


def get_runwise_density_optimization_data(
    sample_dir: Path | str,
    compute_grid_metrics: bool = True,
    compare_to_ks: bool = False,
    metrics_grid_level: int = 3,
    metrics_grid_prune: str = "nwchem_prune",
) -> dict:
    """Get the data for a set of molecules for the density optimization process from a directory of
    saved samples.

    Args:
        sample_dir: Path to the directory containing the samples. These samples are expected to
            result from the :mod:`mldft.ofdft.run_density_optimization.py` script.
        compute_grid_metrics: Whether to compute the grid based metrics.
        metrics_grid_level: Level of the grid to use for grid based metrics.
        metrics_grid_prune: Pruning scheme to use for the grid based metrics.

    Returns:
        run_data_dict: Dictionary containing the stopping indices, energy trajectories, gradient
            norms, coefficient trajectories and density differences for the set of molecules.
    """

    if isinstance(sample_dir, str):
        sample_dir = Path(sample_dir)
    assert (
        sample_dir.is_dir()
    ), f"Density Optimization sample directory does not exist at {sample_dir}"

    # Count the number of samples (molecules) in the directory
    sample_files = sorted(sample_dir.glob("sample*.pt"))
    n_molecules = len(sample_files)
    if n_molecules == 0:
        raise ValueError(f"No sample files found in {sample_dir}")

    stopping_indices = torch.zeros(n_molecules, dtype=torch.int64)
    num_atoms = torch.zeros(n_molecules, dtype=torch.int64)
    num_electrons = torch.zeros(n_molecules, dtype=torch.int64)

    (
        energy_trajectories_dict,
        energy_ground_state_dict,
    ) = initialize_energy_dicts(sample_dir=sample_dir, n_molecules=n_molecules)

    gradient_norms = []
    coeffs_trajectories = []
    coeffs_ground_state = []
    coeffs_pred_ground_state = []

    density_diffs_l1 = []
    density_diffs_l2 = []
    density_diffs_ks_l1 = []
    density_diffs_ks_l2 = []
    density_diffs_df_l1 = []
    density_diffs_df_l2 = []

    run_times = []

    basis_info_path = sample_dir / "basis_info.pt"
    assert basis_info_path.exists(), (
        f"Missing required basis info file: {basis_info_path}. "
        "This plotting pipeline expects basis_info to always be present."
    )
    basis_info: BasisInfo = torch.load(
        basis_info_path, map_location=torch.device("cpu"), weights_only=False
    )

    basis_function_indices = torch.as_tensor(
        np.concatenate(basis_info.atom_ind_to_basis_function_ind),
        dtype=torch.int32 if OFDFT_DTYPE == torch.float32 else torch.int64,
    )

    # Saved denop trajectories can come from float32 or float64 runs, so use a stable
    # accumulation dtype here instead of tying the buffer to the current global OFDFT_DTYPE.
    cumulative_coeff_error = torch.zeros_like(basis_function_indices, dtype=torch.float64)
    cumulative_counts = torch.zeros_like(
        basis_function_indices,
        dtype=torch.int32 if OFDFT_DTYPE == torch.float32 else torch.int64,
    )

    extra_per_sample_keys = ["final_hessian_eigenvalues"]
    run_data_dict = {key: [] for key in extra_per_sample_keys}

    for i, file_path in tqdm(
        enumerate(sample_files),
        desc="Loading samples",
        dynamic_ncols=True,
        total=len(sample_files),
    ):
        sample: OFData = torch.load(
            file_path, map_location=torch.device("cpu"), weights_only=False
        )
        assert sample.stopping_index is not None

        stopping_indices[i] = sample.stopping_index
        num_atoms[i] = sample.n_atom
        num_electrons[i] = sample.n_electron

        # energy names are set correctly in the dict initialization
        for energy_name in energy_trajectories_dict.keys():
            energy_trajectories_dict[energy_name].append(
                1e3 * (sample[f"trajectory_energy_{energy_name}"].detach().cpu())
            )
            if f"ground_state_energy_{energy_name}" in sample.keys():
                if isinstance(sample[f"ground_state_energy_{energy_name}"], torch.Tensor):
                    energy_ground_state_dict[energy_name][i] = 1e3 * (
                        sample[f"ground_state_energy_{energy_name}"].detach().cpu()
                    )
                elif isinstance(sample[f"ground_state_energy_{energy_name}"], float):
                    energy_ground_state_dict[energy_name][i] = torch.as_tensor(
                        1e3 * sample[f"ground_state_energy_{energy_name}"], dtype=torch.float64
                    )
            else:
                logger.warning(f"{energy_name} not in ground state energies but in trajectories.")

        for key in extra_per_sample_keys:
            if key in sample:
                run_data_dict[key].append(sample[key])
            else:
                run_data_dict[key].append(None)

        gradient_norms.append(sample.trajectory_gradient_norm.detach().cpu())

        # incorporate overlap matrix here as well to make difference meaningful?
        coeffs_trajectories.append(sample.trajectory_coeffs.detach().cpu())
        coeffs_ground_state.append(sample.ground_state_coeffs.detach().cpu())
        if hasattr(sample, "predicted_ground_state_coeffs"):
            coeffs_pred_ground_state.append(sample.predicted_ground_state_coeffs.detach().cpu())
        # old samples from run_density_optimization have sample.coeffs
        elif not hasattr(sample, "transformation_matrix"):
            coeffs_pred_ground_state.append(sample.coeffs.detach().cpu())
            # Add to sample for l1 norm calculation
            sample.add_item("predicted_ground_state_coeffs", sample.coeffs, "vector")
        # Samples from run_ofdft have the full trajectory_coeffs
        else:
            predicted_ground_state_coeffs = sample.trajectory_coeffs[sample.stopping_index]
            coeffs_pred_ground_state.append(predicted_ground_state_coeffs.detach().cpu())
            # Add to sample for l1 norm calculation
            sample.add_item(
                "predicted_ground_state_coeffs", predicted_ground_state_coeffs, "vector"
            )
        # done inside this loop as we need the samples basis_function_ind
        cumulative_coeff_error, cumulative_counts = cumulate_coeff_error(
            sample=sample,
            pred_ground_state_coeffs=coeffs_pred_ground_state[-1],
            cumulative_error=cumulative_coeff_error,
            cumulative_counts=cumulative_counts,
        )

        # get density difference right away as this needs the samples overlap matrix
        if hasattr(sample, "trajectory_l2_norm"):
            density_diffs_l2.append(sample.trajectory_l2_norm.detach().cpu())
        else:
            density_diffs_l2.append(get_density_difference_l2_norm(sample))
        if hasattr(sample, "time") and sample.time is not None:
            run_times.append(sample.time)
        else:
            run_times.append(np.nan)

        if compute_grid_metrics or compare_to_ks:
            grid_metrics = calculate_grid_metrics(
                sample, basis_info, metrics_grid_level, metrics_grid_prune, compare_to_ks
            )
            if compute_grid_metrics:
                density_diffs_l1.append(grid_metrics["density_error_l1"])
            if compare_to_ks:
                density_diffs_ks_l1.append(grid_metrics["density_error_l1_ks"])
                density_diffs_ks_l2.append(grid_metrics["density_error_l2_ks"])
                density_diffs_df_l1.append(grid_metrics["df_error_l1_ks"])
                density_diffs_df_l2.append(grid_metrics["df_error_l2_ks"])

    # sanity check
    assert n_molecules == len(gradient_norms), "Failed to load all samples correctly"

    run_data_dict = {
        "n_molecules": n_molecules,
        "basis_info": basis_info,
        "basis_function_indices": basis_function_indices,
        "sample_basis_function_ind": sample.basis_function_ind,  # only sensible for fixed number of aos
        "stopping_indices": stopping_indices,
        "energy_trajectories_dict": energy_trajectories_dict,
        "energy_ground_state_dict": energy_ground_state_dict,
        "gradient_norms": gradient_norms,
        "coeffs_trajectories": coeffs_trajectories,
        "coeffs_ground_state": coeffs_ground_state,
        "coeffs_pred_ground_state": coeffs_pred_ground_state,
        "density_differences_l2": density_diffs_l2,
        "cumulative_coeff_error": cumulative_coeff_error,
        "cumulative_counts": cumulative_counts,
        "num_atoms": num_atoms,
        "num_electrons": num_electrons,
        "run_times": run_times,
        **run_data_dict,
    }

    if compute_grid_metrics:
        run_data_dict["density_differences_l1"] = density_diffs_l1
    if compare_to_ks:
        run_data_dict["density_differences_ks_l1"] = density_diffs_ks_l1
        run_data_dict["density_differences_ks_l2"] = density_diffs_ks_l2
        run_data_dict["density_differences_df_l1"] = density_diffs_df_l1
        run_data_dict["density_differences_df_l2"] = density_diffs_df_l2

    return run_data_dict


def save_density_optimization_metrics(
    output_path: Path,
    run_data_dict: dict,
):
    """Save the density optimization metrics for a set of molecules to a file.

    Args:
        output_path: Path where to save the metrics yaml file.
        run_data_dict: Dictionary containing the n_molecules, basis_info, basis_function_indices,
            stopping indices, energy trajectories, gradient norms, coefficient trajectories and
            density differences for the set of molecules.
    """
    num_atoms = run_data_dict["num_atoms"]
    num_electrons = run_data_dict["num_electrons"]
    stopping_indices = run_data_dict["stopping_indices"]
    # These keys will be at the top of the yaml file
    benchmark_keys = [
        "n_molecules",
        "mean_total_energy_error[mHa]",
        "mean_per_atom_total_energy_error[mHa]",
        "mean_per_electron_density_error_l1[%]",
        "mean_gradient_norm",
        "nonconverged_molecules_ratio[%]",
    ]
    metric_dict = {key: np.nan for key in benchmark_keys}
    metric_dict["n_molecules"] = run_data_dict["n_molecules"]
    energy_errors, absolute_energy_errors = get_energy_ground_state_errors(
        energy_trajectories_dict=run_data_dict["energy_trajectories_dict"],
        energy_ground_state_dict=run_data_dict["energy_ground_state_dict"],
        stopping_indices=run_data_dict["stopping_indices"],
    )
    energy_key = "total"
    total_errors = absolute_energy_errors[energy_key]
    per_atom_errors = total_errors / num_atoms
    metric_dict.update(
        {
            f"mean_{energy_key}_energy_error[mHa]": total_errors.mean().item(),
            f"mean_per_atom_{energy_key}_energy_error[mHa]": per_atom_errors.mean().item(),
            f"median_total_{energy_key}_energy_error[mHa]": total_errors.median().item(),
            f"median_per_atom_{energy_key}_energy_error[mHa]": per_atom_errors.median().item(),
            f"ratio_of_molecules_below_1mHa_{energy_key}_energy_error": (
                (total_errors < 1).sum().item() / len(total_errors)
            ),
            f"ratio_of_molecules_below_1mHa_per_atom_{energy_key}_energy_error": (
                (per_atom_errors < 1).sum().item() / len(per_atom_errors)
            ),
            f"90th_percentile_{energy_key}_energy_error[mHa]": torch.quantile(
                total_errors, 0.9
            ).item(),
        }
    )

    def add_density_stats(metric_name: str, values: Tensor) -> None:
        metric_dict[f"mean_{metric_name}"] = values.mean().item()
        metric_dict[f"median_{metric_name}"] = values.median().item()
        per_electron = values / num_electrons
        metric_dict[f"mean_per_electron_{metric_name}[%]"] = per_electron.mean().item() * 100
        metric_dict[f"median_per_electron_{metric_name}[%]"] = per_electron.median().item() * 100

    stopping_density_differences_l2 = torch.stack(
        [
            tensor[index]
            for tensor, index in zip(run_data_dict["density_differences_l2"], stopping_indices)
        ]
    )
    add_density_stats("density_error_l2", stopping_density_differences_l2)

    if "density_differences_l1" in run_data_dict:
        add_density_stats("density_error_l1", torch.stack(run_data_dict["density_differences_l1"]))
    if "density_differences_ks_l1" in run_data_dict:
        density_ks_l1 = run_data_dict["density_differences_ks_l1"]
        assert all(
            d.ndim == 0 for d in density_ks_l1
        ), "Density differences must be 0D to compute mean L1 norm"
        add_density_stats("density_error_l1_ks", torch.stack(density_ks_l1))
    if "density_differences_ks_l2" in run_data_dict:
        add_density_stats(
            "density_error_l2_ks", torch.stack(run_data_dict["density_differences_ks_l2"])
        )
    if "density_differences_df_l1" in run_data_dict:
        add_density_stats(
            "df_error_l1_ks", torch.stack(run_data_dict["density_differences_df_l1"])
        )
    if "density_differences_df_l2" in run_data_dict:
        add_density_stats(
            "df_error_l2_ks", torch.stack(run_data_dict["density_differences_df_l2"])
        )

    stopping_gradient_norms = torch.stack(
        [tensor[index] for tensor, index in zip(run_data_dict["gradient_norms"], stopping_indices)]
    )
    metric_dict["mean_gradient_norm"] = torch.mean(stopping_gradient_norms).item()
    metric_dict["median_gradient_norm"] = torch.median(stopping_gradient_norms).item()
    metric_dict["nonconverged_molecules_ratio[%]"] = (
        torch.sum(stopping_gradient_norms > 1e-4).item()
        / len(stopping_gradient_norms)
        * 100  # TODO:why is the tolerance hardcoded here to 1e-4?
    )

    logger.info("Successfully computed metrics:")
    # Round values for nicer formatting
    for key, value in metric_dict.items():
        if isinstance(value, float):
            metric_dict[key] = round(value, 6)
        logger.info(f"  {key}: {metric_dict[key]}")

    with open(output_path, "w") as f:
        yaml.safe_dump(metric_dict, f, sort_keys=False)
    return


def density_optimization_summary_pdf_plot(
    out_pdf_path: Path | str,
    run_data_dict: dict,
    matplotlib_backend: str = "pdf",
    subsample: float = 1.0,
):
    """Create a summary page for the density optimization for a set of molecules.

    Args:
        out_pdf_path: Path to the output PDF file.
        run_data_dict: Dictionary containing the n_molecules, basis_info, basis_function_indices,
            stopping indices, energy trajectories, gradient norms, coefficient trajectories and
            density differences for the set of molecules.
        matplotlib_backend: Matplotlib backend to use for plotting. By default the pdf backend is
            used to create vectorized PDF plots, while for instance 'agg' could be used to create
            rasterized PNG plots.
        subsample: Fraction of molecules to plot in the swarm plots (individual molecule
            trajectories). By default (1.0), all molecules are plotted. Only takes effect, if
            n_molecules is larger than the number of molecules to plot and n_molecules > 2.
    """

    if matplotlib_backend is not None:
        plt.switch_backend(matplotlib_backend)
    with HierarchicalPlotPDF(
        out_pdf_path=out_pdf_path,
    ) as summary_pdf:
        plot_ofdft_run_summary(**run_data_dict)
        plt.suptitle("Density Optimization Run Summary")
        # Using constrained layout, we have to set the bbox_inches and pad_inches to replace rect
        summary_pdf.savefig(
            "Density Optimization Run Summary", bbox_inches="tight", pad_inches=0.8
        )
        plt.close()

        plot_ofdft_energy_distribution(**run_data_dict)
        plt.suptitle("Energy Error Distribution")
        summary_pdf.savefig("Energy Error Distribution")
        plt.close()

        plot_density_optimization_trajectory_means(**run_data_dict)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle("Density Optimization Mean Trajectories")
        summary_pdf.savefig("Density Optimization Mean Trajectories")
        plt.close()

        density_optimization_swarm_plot(**run_data_dict, subsample=subsample)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle("Individual Molecule Density Optimization Trajectories")
        summary_pdf.savefig("Individual Molecule Density Optimization Trajectories")
        plt.close()

        plot_energy_summary_scatter(**run_data_dict)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.suptitle("Stopping vs Initial Energy Errors")
        summary_pdf.savefig("Stopping vs Initial Energy Errors")
        plt.close()


def density_optimization_swarm_plot(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
    coeffs_trajectories: list[torch.Tensor] | torch.Tensor,
    coeffs_ground_state: list[torch.Tensor] | torch.Tensor,
    gradient_norms: list[torch.Tensor] | torch.Tensor,
    density_differences_l2: list[torch.Tensor] | torch.Tensor,
    stopping_indices: torch.Tensor,
    n_molecules: int | None = None,
    subsample: float = 1.0,
    **_,
):
    """Summarize the density optimization process for a set of molecules, by plotting a line for
    every molecule, showing the energy error, gradient norm and L2 norm of the density error."""

    if n_molecules is not None:
        assert n_molecules <= len(stopping_indices), (
            f"n_molecules to plot, {n_molecules}, exceeds the number of available molecules,"
            f" {len(stopping_indices)}"
        )
    else:
        n_molecules = len(stopping_indices)  # all molecules

    energy_errors_dict, energy_absolute_error_dict = get_energy_errors_dict(
        energy_trajectories_dict=energy_trajectories_dict,
        energy_ground_state_dict=energy_ground_state_dict,
    )

    energy_names = list(energy_errors_dict.keys())

    plots = [
        partial(
            energy_error_swarm_line_plot,
            energy_errors_dict=energy_errors_dict,
            energy_name=energy_name,
            subsample=subsample,
        )
        for energy_name in energy_names
    ] + [
        partial(gradient_norm_swarm_line_plot, gradient_norms=gradient_norms, subsample=subsample),
        partial(
            density_differences_swarm_line_plot,
            density_differences_l2=density_differences_l2,
            subsample=subsample,
        ),
    ]

    fig, axs = plt.subplots(len(plots), 1, figsize=(9, 5 * len(plots)), sharex=True)

    for ax, plot in zip(axs, plots):
        plot(ax=ax, stopping_indices=stopping_indices, n_molecules=n_molecules)
        ax.grid(True)
        # share x-axis with energy plot

    axs[-1].set_xlabel("Iteration")

    # Create a legend for the line styles
    solid_line = mlines.Line2D([], [], color="black", linestyle="-", label="Before stopping index")
    dashed_line = mlines.Line2D(
        [], [], color="black", linestyle="--", label="After stopping index"
    )
    axs[-1].legend(handles=[solid_line, dashed_line])
    fig.tight_layout()


def initialize_energy_dicts(
    sample_dir: Path,
    n_molecules: int,
):
    """Initialize the energy trajectories dictionary for a set of molecules.

    We load the first sample in the given directory. As of yet, this seems necessary in order to
    check for available energy names (ground state and trajectories) and not do it within the loop
    over the sample directory itself. If the energy names are not provided, all available energies
    are extracted from the first sample.
    """

    energy_trajectories: dict[str, list[torch.Tensor]] = {}
    energy_ground_state_dict: dict[str, torch.Tensor] = {}

    sample_files = sorted(sample_dir.glob("sample*.pt"))
    if not sample_files:
        raise ValueError(f"No sample files found in {sample_dir}")
    sample: OFData = torch.load(sample_files[0], map_location="cpu", weights_only=False)

    for key in sample.keys():
        if key == "trajectory_energy_nuclear_repulsion":
            continue
        if not key.startswith("trajectory_energy_"):
            continue

        energy_name = key[len("trajectory_energy_") :]
        if f"ground_state_energy_{energy_name}" not in sample.keys():
            logger.warning(
                f"{energy_name} ground-state energy not found in sample; skipping this energy."
            )
            continue

        energy_ground_state_dict[energy_name] = torch.zeros(n_molecules)
        energy_trajectories[energy_name] = []

    # move the total energy to the start of the dict keys
    if "total" in energy_trajectories:
        total = energy_trajectories.pop("total")
        energy_trajectories = {"total": total, **dict(sorted(energy_trajectories.items()))}

    if "total" in energy_ground_state_dict:
        total = energy_ground_state_dict.pop("total")
        energy_ground_state_dict = {
            "total": total,
            **dict(sorted(energy_ground_state_dict.items())),
        }

    return energy_trajectories, energy_ground_state_dict


def plot_density_optimization_trajectory_means(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
    coeffs_trajectories: list[torch.Tensor] | torch.Tensor,
    coeffs_ground_state: list[torch.Tensor] | torch.Tensor,
    gradient_norms: list[torch.Tensor] | torch.Tensor,
    density_differences_l2: list[torch.Tensor] | torch.Tensor,
    stopping_indices: torch.Tensor,
    num_electrons: torch.Tensor,
    n_molecules: int,
    **kwargs,
):
    """Summarize the density optimization process for a set of molecules, by plotting the mean
    energy difference, gradient norm and density difference."""

    energy_names = list(energy_ground_state_dict.keys())

    molecule_indices = torch.arange(n_molecules)  # simply a range of 0 to n_molecules

    if isinstance(density_differences_l2, torch.Tensor):
        stopping_density_differences_l2 = density_differences_l2[
            molecule_indices, stopping_indices
        ]
    else:
        try:
            density_differences_l2 = torch.stack(density_differences_l2)
            stopping_density_differences_l2 = density_differences_l2[
                molecule_indices, stopping_indices
            ]
            initial_density_differences_l2 = density_differences_l2[:, 0]

        except Exception as e:
            stopping_density_differences_l2 = torch.stack(
                [
                    tensor[stopping_index]
                    for tensor, stopping_index in zip(density_differences_l2, stopping_indices)
                ]
            )

    energy_errors_dict, energy_absolute_errors_dict = get_energy_errors_dict(
        energy_trajectories_dict=energy_trajectories_dict,
        energy_ground_state_dict=energy_ground_state_dict,
    )

    stopping_energy_absolute_errors_dict = {}
    for energy_name, energy_absolute_errors in energy_absolute_errors_dict.items():
        stopping_energy_absolute_errors_dict[energy_name] = torch.as_tensor(
            [
                molecule_energy_absolute_errors[stopping_index]
                for molecule_energy_absolute_errors, stopping_index in zip(
                    energy_absolute_errors, stopping_indices
                )
            ]
        )

    plots = [
        partial(
            _add_energy_mae,
            energy_maes=energy_absolute_errors_dict[energy_name],
            energy_stopping_maes=stopping_energy_absolute_errors_dict[energy_name],
            energy_name=energy_name,
        )
        for energy_name in energy_names
    ] + [
        partial(
            _add_mean_gradient_norm,
            gradient_norms=gradient_norms,
            stopping_indices=stopping_indices,
        ),
        partial(
            _add_mean_density_differences_l2,
            density_differences_l2=density_differences_l2,
            stopping_density_differences_l2=stopping_density_differences_l2,
        ),
        partial(_add_stopping_index_histogram, stopping_indices=stopping_indices),
    ]

    fig, axs = plt.subplots(len(plots), 1, figsize=(9, 4 * len(plots)), sharex=True)
    for plot, ax in zip(plots, axs):
        plot(ax)
        ax.axvline(
            torch.median(stopping_indices),
            color="black",
            linestyle="--",
            label=f"Median stopping index {torch.median(stopping_indices)}",
        )

        ax.grid(True)

    axs[-1].legend()
    plt.tight_layout()


def plot_ofdft_run_summary(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
    coeffs_trajectories: torch.Tensor | list[torch.Tensor],
    coeffs_ground_state: torch.Tensor | list[torch.Tensor],
    coeffs_pred_ground_state: torch.Tensor | list[torch.Tensor],
    gradient_norms: torch.Tensor | list[torch.Tensor],
    density_differences_l2: torch.Tensor | list[torch.Tensor],
    stopping_indices: torch.Tensor,
    n_molecules: int,
    basis_info: BasisInfo,
    basis_function_indices: torch.Tensor,
    sample_basis_function_ind: torch.Tensor,
    cumulative_coeff_error: torch.Tensor,
    cumulative_counts: torch.Tensor,
    final_hessian_eigenvalues: list[torch.Tensor | None] | None = None,
    **_,
):
    """Create a summary page for the density optimization process for a set of molecules."""

    # density differences are lists for now
    initial_density_differences_l2 = torch.stack([tensor[0] for tensor in density_differences_l2])
    stopping_density_differences_l2 = torch.stack(
        [tensor[index] for tensor, index in zip(density_differences_l2, stopping_indices)]
    )
    (
        stopping_energy_errors_dict,
        stopping_energy_absolute_errors_dict,
    ) = get_energy_ground_state_errors(
        energy_trajectories_dict,
        energy_ground_state_dict,
        stopping_indices,
    )

    # Get the last dimension of the first tensor in the list
    first_n_basis = coeffs_trajectories[0].shape[-1]
    if all(
        coeff_trajectory.shape[-1] == first_n_basis for coeff_trajectory in coeffs_trajectories
    ):
        # coeffs_trajectories are of shape (n_molecules, n_cycles, n_basis_functions)
        stopping_coeffs_errors = torch.stack(coeffs_pred_ground_state) - torch.stack(
            coeffs_ground_state
        )
        coeff_mae_scatter = partial(
            _single_shape_coeff_mae_scatter,
            coeff_errors=stopping_coeffs_errors,
            basis_info=basis_info,
            basis_function_ind=sample_basis_function_ind,
        )
    else:
        coeff_mae_scatter = partial(
            _multi_shape_coeff_mae_scatter,
            cumulative_coeff_errors=cumulative_coeff_error,
            cumulative_counts=cumulative_counts,
            basis_function_indices=basis_function_indices,
            basis_info=basis_info,
        )

    with_hessians = final_hessian_eigenvalues and any(
        [evs is not None for evs in final_hessian_eigenvalues]
    )
    if with_hessians:
        fig = plt.figure(figsize=(15, 25.1), layout="constrained")
        subfigs = fig.subfigures(
            4, 1, wspace=0.10, hspace=0.07, height_ratios=[0.80, 0.80, 0.55, 0.55]
        )
    else:
        fig = plt.figure(figsize=(15, 20), layout="constrained")
        subfigs = fig.subfigures(3, 1, wspace=0.10, hspace=0.07, height_ratios=[0.80, 0.80, 0.55])

    (energy_ax1, energy_ax2) = subfigs[0].subplots(1, 2)

    _add_energy_boxplot(
        energy_ax1,
        energy_errors_dict=stopping_energy_errors_dict,
        title="Signed Energy Contribution Errors",
    )
    _add_energy_boxplot(
        energy_ax2,
        energy_errors_dict=stopping_energy_absolute_errors_dict,
        title="Absolute Energy Contribution Errors",
    )

    (density_ax, gradient_ax) = subfigs[1].subplots(1, 2)
    _compare_initial_to_stopping_l2_norms(
        density_ax,
        initial_l2_norms=initial_density_differences_l2,
        stopping_l2_norms=stopping_density_differences_l2,
        ground_state_energy_dict=energy_ground_state_dict,
    )
    _compare_initial_to_stopping_gradient_norms(
        gradient_ax,
        gradient_norms=gradient_norms,
        stopping_indices=stopping_indices,
        ground_state_energy_dict=energy_ground_state_dict,
    )

    coeff_ax = subfigs[2].subplots(1, 1)
    coeff_mae_scatter(coeff_ax)

    if with_hessians:
        hessian_ax = subfigs[3].subplots(1, 1)
        assert final_hessian_eigenvalues is not None
        _add_hessian_eigenvalues_hist(hessian_ax, final_hessian_eigenvalues)


def plot_ofdft_energy_distribution(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
    stopping_indices: torch.Tensor,
    n_molecules: int,
    num_atoms: torch.Tensor,
    **_,
):
    """Create a energy distribution summary page for the density optimization process for a set of
    molecules."""

    energy_names = list(energy_ground_state_dict.keys())

    (
        stopping_energy_errors_dict,
        stopping_energy_absolute_errors_dict,
    ) = get_energy_ground_state_errors(
        energy_trajectories_dict,
        energy_ground_state_dict,
        stopping_indices,
    )

    fig = plt.figure(figsize=(15, 6 + len(energy_names) * 10), layout="constrained")

    # height ratios accustomed to variable number of energy names
    height_ratios = [0.55, 0.80] * len(energy_names)
    subfigs = fig.subfigures(
        2 * len(energy_names), 1, wspace=0.10, hspace=0.07, height_ratios=height_ratios
    )

    for i, energy_name in zip(np.arange(0, 2 * len(energy_names), 2), energy_names):
        eom_ax = subfigs[i].subplots(1, 1)
        _add_energy_error_over_n_atoms(
            eom_ax,
            num_atoms=num_atoms,
            energy_errors_dict=stopping_energy_absolute_errors_dict,
            energy_name=energy_name,
        )

        (energy_ax1, energy_ax2) = subfigs[i + 1].subplots(1, 2)

        _add_energy_histogram(
            energy_ax1,
            energy_errors_dict=stopping_energy_errors_dict,
            energy_name=energy_name,
            title="Signed Energy Contribution Errors",
        )
        _add_energy_histogram(
            energy_ax2,
            energy_errors_dict=stopping_energy_absolute_errors_dict,
            energy_name=energy_name,
            title="Absolute Energy Contribution Errors",
        )


def plot_energy_summary_scatter(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
    stopping_indices: torch.Tensor,
    n_molecules: int | None = None,
    **_,
):
    """Scatter plots of the stopping over initial energy errors for a set of molecules.

    Args:
        energy_trajectories_dict: Dictionary containing the energy trajectories for each molecule.
            Keys are energy names and values are tensors of shape (n_molecules, n_cycles).
        energy_ground_state_dict: Dictionary containing the ground state energies for each molecule.
            Keys are energy names and values are tensors of shape (n_molecules,).
        stopping_indices: Tensor containing the stopping indices for each molecule determined by
            the convergence criteria of the density optimization run.
        n_molecules: Number of molecules to be plotted.
    """

    if n_molecules is None:
        n_molecules = len(stopping_indices)

    energy_names = list(energy_ground_state_dict.keys())
    energy_errors_dict, energy_absolute_error_dict = get_energy_errors_dict(
        energy_trajectories_dict=energy_trajectories_dict,
        energy_ground_state_dict=energy_ground_state_dict,
    )

    initial_energy_absolute_errors_dict = {}
    stopping_energy_absolute_errors_dict = {}
    for energy_name, absolute_energy_errors in energy_absolute_error_dict.items():
        initial_energy_absolute_errors_dict[energy_name] = torch.as_tensor(
            [molecule_energy_errors[0] for molecule_energy_errors in absolute_energy_errors]
        )

        stopping_energy_absolute_errors_dict[energy_name] = torch.as_tensor(
            [
                absolute_molecule_energy_errors[stopping_index]
                for absolute_molecule_energy_errors, stopping_index in zip(
                    absolute_energy_errors, stopping_indices
                )
            ]
        )

    fig, axs = plt.subplots(len(energy_names), 1, figsize=(10, 8 * len(energy_names)))
    if len(energy_names) == 1:
        axs = [axs]

    for ax, energy_name in zip(axs, energy_names):
        initial_energy_error = initial_energy_absolute_errors_dict[energy_name]
        stopping_energy_error = stopping_energy_absolute_errors_dict[energy_name]
        ground_state_energy = energy_ground_state_dict[energy_name]
        scatter = ax.scatter(
            initial_energy_error,
            stopping_energy_error,
            c=ground_state_energy,  # set color by ground state energy
            cmap="seismic",
            s=40 if n_molecules < 50 else 10,
        )

        # add a diagonal line.
        min_energy_error = min(min(initial_energy_error), min(stopping_energy_error))
        max_energy_error = max(max(initial_energy_error), max(stopping_energy_error))
        ax.plot(
            [min_energy_error, max_energy_error],
            [min_energy_error, max_energy_error],
            color="grey",
            alpha=0.6,
        )

        ax.set_xlabel(
            rf"$ \vert \Delta E_\mathrm{{{energy_name}, initial}} - \Delta E_\mathrm{{{energy_name}, target}} \vert$ [mHa] (initial)",
            fontsize=14,
        )
        ax.set_ylabel(
            rf"$ \vert \Delta E_\mathrm{{{energy_name}, pred}} - \Delta E_\mathrm{{{energy_name}, target}} \vert$ [mHa] (stopping)",
            fontsize=14,
        )

        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label(rf"$E_\mathrm{{{energy_name}, target}}$ [mHa]", fontsize=12)

        title = f"Absolute {energy_name} energy error of initial and stopping state"
        ax.set_title(title, fontsize=15)


def _add_energy_boxplot(
    ax: plt.Axes,
    energy_errors_dict: dict[str, torch.Tensor],
    title="Energy Contribution Errors",
    plot_electronic: bool = False,
):
    """Add a boxplot of the energy errors of different contributions for a set of molecules."""

    if not plot_electronic and "electronic" in energy_errors_dict.keys():
        energy_errors_dict = energy_errors_dict.copy()
        del energy_errors_dict["electronic"]

    energy_names = list(energy_errors_dict.keys())

    if not plot_electronic and "electronic" in energy_names:
        energy_names = list(energy_names)
        energy_names.remove("electronic")

    if "total" in energy_errors_dict.keys() and "tot" in energy_errors_dict.keys():
        del energy_errors_dict["tot"]
        del energy_names[energy_names.index("tot")]

    # seaborn works with numpy arrays
    if isinstance(list(energy_errors_dict.values())[0], torch.Tensor):
        energy_errors_dict = {key: value.numpy() for key, value in energy_errors_dict.items()}

    # electronic error should be the same as total error
    sns.boxplot(
        data=[energy_errors_dict[energy_name] for energy_name in energy_names],
        ax=ax,
        flierprops=dict(marker=".", markersize=4),
    )

    ax.set_xticks([i for i in range(len(energy_names))], energy_names, fontsize=13)

    ax.set_ylabel(r" $\Delta E$ [mHa]", fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.axhline(0, color="gray", alpha=0.5)  # Add a line for 0 difference

    total_energy_error_mean = np.mean(np.abs(energy_errors_dict["total"]))
    linthresh = custom_round(total_energy_error_mean)

    # Calculate the minimum and maximum values in the data with some padding
    min_value = min(min(values) for values in energy_errors_dict.values()) - (linthresh * 10e-2)
    max_value = max(max(values) for values in energy_errors_dict.values()) + (linthresh * 10e-2)

    # restrict y_axis to data range
    ax.set_ylim(min_value, max_value)

    # when all data is positive, set y-axis to log scale
    if min_value > 0:
        # ax.set_ylim(0, max_value)
        ax.set_yscale("log")

    else:
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh))
        # Highlight linear region with solid lines
        ax.axhline(-linthresh, color="gray", linestyle="-", linewidth=0.7)
        ax.axhline(linthresh, color="gray", linestyle="-", linewidth=0.7)

    # custom gridlines
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    # ax.grid(True, which='minor', linestyle=':', linewidth=0.3)

    # Add mean as text labels next to mean y-value
    for i, (key, values) in enumerate(energy_errors_dict.items()):
        mean_value = float(np.mean(values))
        ax.text(
            x=i,
            y=mean_value + 0.001,
            s=f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            weight="bold",
            fontfamily="monospace",
            bbox=dict(
                facecolor="white", alpha=0.15, edgecolor="none", boxstyle="round,pad=0.1"
            ),  # Text background
        )
        # Draw a small horizontal line at the mean value
        ax.hlines(y=mean_value, xmin=i - 0.015, xmax=i + 0.015, color="darkgray", linewidth=2)
    # rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="center")


def _compare_initial_to_stopping_l2_norms(
    ax: plt.Axes,
    initial_l2_norms: torch.Tensor | list[torch.Tensor],
    stopping_l2_norms: torch.Tensor | list[torch.Tensor],
    ground_state_energy_dict: Mapping[str, torch.Tensor | float],
    energy_name: str = "total",
):
    """Plot the initial density difference against the stopping density difference measured by the
    L2 norm."""

    ground_state_energy = ground_state_energy_dict[energy_name]

    scatter = ax.scatter(
        initial_l2_norms,
        stopping_l2_norms,
        c=ground_state_energy,
        cmap="seismic",
        s=40 if len(initial_l2_norms) < 50 else 10,
    )

    # add a diagonal line.
    min_l2_norm = min(min(initial_l2_norms), min(stopping_l2_norms))
    max_l2_norm = max(max(initial_l2_norms), max(stopping_l2_norms))
    ax.plot([min_l2_norm, max_l2_norm], [min_l2_norm, max_l2_norm], color="grey", alpha=0.6)

    ax.set_xlabel(
        r"$\Vert\rho_\mathrm{pred} - \rho_\mathrm{target}\Vert_2$ [electrons] (initial)",
        fontsize=14,
    )
    ax.set_ylabel(
        r"$\Vert\rho_\mathrm{init} - \rho_\mathrm{target}\Vert_2$ [electrons] (final)", fontsize=14
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.008)
    cbar.set_label(rf"$E_\mathrm{{{energy_name}, target}}$ [mHa]", fontsize=13)
    title = "Initial vs. stopping L2 density difference"
    ax.set_title(title, fontsize=15)


def _compare_initial_to_stopping_gradient_norms(
    ax: plt.Axes,
    gradient_norms: torch.Tensor | list[torch.Tensor],
    stopping_indices: torch.Tensor,
    ground_state_energy_dict: Mapping[str, torch.Tensor | float],
    energy_name: str = "total",
):
    """Scatter plot the initial gradient norm against the stopping gradient norm."""

    n_molecules = len(stopping_indices)

    if not isinstance(gradient_norms, torch.Tensor):
        try:
            gradient_norms = torch.stack(gradient_norms)
            initial_gradient_norms = gradient_norms[:, 0]
            stopping_gradient_norms = gradient_norms[torch.arange(n_molecules), stopping_indices]
        except Exception as e:
            initial_gradient_norms = [gradient_norm[0] for gradient_norm in gradient_norms]
            stopping_gradient_norms = [
                gradient_norm[stopping_index]
                for gradient_norm, stopping_index in zip(gradient_norms, stopping_indices)
            ]
    else:
        initial_gradient_norms = gradient_norms[:, 0]
        stopping_gradient_norms = gradient_norms[torch.arange(n_molecules), stopping_indices]

    ground_state_energy = ground_state_energy_dict[energy_name]

    scatter = ax.scatter(
        initial_gradient_norms,
        stopping_gradient_norms,
        c=ground_state_energy,
        cmap="seismic",
        s=40 if n_molecules < 50 else 10,
    )

    # add a diagonal line.
    min_gradient_norm = min(min(initial_gradient_norms), min(stopping_gradient_norms))
    max_gradient_norm = max(max(initial_gradient_norms), max(stopping_gradient_norms))
    ax.plot(
        [min_gradient_norm, max_gradient_norm],
        [min_gradient_norm, max_gradient_norm],
        color="gray",
        alpha=0.6,
    )

    ax.set_xlabel(r"Initial gradient norm", fontsize=14)
    ax.set_ylabel(r"Stopping gradient norm", fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.008)
    cbar.set_label(rf"$E_\mathrm{{{energy_name}, target}}$ [mHa]", fontsize=13)
    title = "Initial vs. stopping gradient norm"
    ax.set_title(title, fontsize=15)


def _add_stopping_index_histogram(ax, stopping_indices):
    """Plot a histogram of the stopping indices."""

    # better handle with histplot_kwargs
    sns.histplot(stopping_indices, ax=ax)
    title = f"Histogram of stopping indices with median {torch.median(stopping_indices.detach().cpu())}"
    ax.set_title(title, fontsize=15)


def _add_energy_mae(
    ax: plt.Axes,
    energy_maes: torch.Tensor,
    energy_stopping_maes: torch.Tensor,
    energy_name: str = "total",
    color: str = "blue",
):
    """Add a plot of the mean energy absolute errors."""

    plot_quantiles_data(energy_maes, ax, color=color)
    ax.axhline(0, color="black")

    # add y-ticks
    ax.set_yticks([0, 1, 3, 6, 10, 100])

    ax.set_ylim(-1e-1, max([energy_mae.max() for energy_mae in energy_maes]) * 1.05)

    ax.set_ylabel(r"$\Delta E$ [mHa]", fontsize=14)
    linthresh = 10
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_minor_locator(MinorSymLogLocator(linthresh))
    title = (
        r"$\Delta E_\mathrm{"
        + energy_name.replace("_", r"\_")
        + "}$ with mean stopping index MAE "
        + f"{torch.mean(energy_stopping_maes).item():.4g} mHa"
    )
    ax.set_xlabel("Iteration", fontsize=14)

    ax.set_title(title, fontsize=15)


def _add_mean_gradient_norm(
    ax: plt.Axes,
    gradient_norms: torch.Tensor | list[torch.Tensor],
    stopping_indices: torch.Tensor,
    color="blue",
):
    """Add a plot of the mean gradient norms."""

    n_molecules = len(stopping_indices)

    if not isinstance(gradient_norms, torch.Tensor):
        try:
            gradient_norms = torch.stack(gradient_norms)
            stopping_gradient_norms = gradient_norms[torch.arange(n_molecules), stopping_indices]
        except Exception:
            stopping_gradient_norms = torch.stack(
                [
                    gradient_norm[int(stopping_index)]
                    for gradient_norm, stopping_index in zip(gradient_norms, stopping_indices)
                ]
            )
    else:
        stopping_gradient_norms = gradient_norms[torch.arange(n_molecules), stopping_indices]

    plot_quantiles_data(gradient_norms, ax, color=color)

    ax.set_ylabel("Gradient norm", fontsize=14)  # units?
    ax.set_yscale("log")
    # ax.set_ylim(bottom=gradient_norms.min()*0.9)
    title = (
        "Average stopping Gradient norm  " + f"{torch.mean(stopping_gradient_norms).item():.4g}"
    )
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_title(title, fontsize=15)


def _add_mean_density_differences_l2(
    ax: plt.Axes,
    density_differences_l2: torch.Tensor,
    stopping_density_differences_l2: torch.Tensor,
    color: str = "blue",
):
    """Add a plot of the mean density differences."""

    plot_quantiles_data(density_differences_l2, ax, color=color)

    ax.axhline(0, color="black")
    ax.set_ylabel(
        r"$\Vert\rho_\mathrm{pred} - \rho_\mathrm{target}\Vert_2$ [electrons]", fontsize=14
    )
    ax.set_yscale("log")
    ax.grid()
    # maybe better median
    title = (
        "Mean Density differences with mean stopping L2 difference "
        + f"{torch.mean(stopping_density_differences_l2).item():.4g} electrons"
    )
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_title(title, fontsize=15)


def _add_mean_relative_density_differences_l1(
    ax: plt.Axes,
    num_electrons: torch.Tensor,
    density_differences_l1: torch.Tensor,
    stopping_density_differences_l1: torch.Tensor,
    color: str = "blue",
):
    """Add a plot of the mean density difference l1 norm divided by the number of electrons."""
    if isinstance(density_differences_l1, torch.Tensor):
        relative_density_difference_l1 = density_differences_l1 / num_electrons.unsqueeze(-1)
    else:
        relative_density_difference_l1 = [
            l1_diff / e for l1_diff, e in zip(density_differences_l1, num_electrons)
        ]
    stopping_relative_density_difference_l1 = stopping_density_differences_l1 / num_electrons
    plot_quantiles_data(relative_density_difference_l1, ax, color=color)

    ax.axhline(0, color="black")
    ax.set_ylabel(
        r"$\Vert\rho_\mathrm{pred} - \rho_\mathrm{target}\Vert_1$ [electrons]", fontsize=14
    )
    ax.set_yscale("log")
    ax.grid()
    # maybe better median
    title = (
        "Mean Density differences l1 norm over number of electrons, stop. ind: "
        + f"{torch.mean(stopping_relative_density_difference_l1).item():.4g} electrons"
    )
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_title(title, fontsize=15)


def stopping_index_line_plot(
    ax: plt.Axes,
    data_array: torch.Tensor | np.ndarray,
    stopping_index: int,
    color="blue",
    **kwargs,
):
    """Plot the data array as solid line up to the stopping index and dashed line afterwards."""

    ax.plot(
        range(stopping_index + 1),
        data_array[: stopping_index + 1],
        linestyle="-",
        color=color,
        **kwargs,
    )

    ax.plot(
        range(stopping_index, len(data_array)),
        data_array[stopping_index:],
        linestyle="--",
        color=color,
    )


def get_density_difference_l2_norm(sample: OFData):
    """Get the L2 norm of the difference between the predicted and target densities for a single
    sample."""

    coeffs_delta = (
        sample.trajectory_coeffs.detach().cpu() - sample.ground_state_coeffs.detach().cpu()
    )

    # assume that the overlap matrix is correctly transformed
    if hasattr(sample, "overlap_matrix"):
        overlap = sample.overlap_matrix.detach().cpu()
        l2_norm = torch.sqrt(torch.einsum("ni,ij,nj->n", coeffs_delta, overlap, coeffs_delta))
        return l2_norm
    else:
        logger.warning(
            "Sample does not have an overlap matrix. Density difference plotting is skipped"
        )
        return


def get_density_difference_l1_norm(
    sample: OFData,
    basis_info: BasisInfo,
    l1_grid_level: int = 3,
    l1_grid_prune: str = "nwchem_prune",
) -> torch.Tensor:
    """Get the L1 norm of the difference between the predicted and target densities for a single
    sample."""
    coeffs_delta = (
        sample.predicted_ground_state_coeffs.detach().cpu()
        - sample.ground_state_coeffs.detach().cpu()
    )
    mol = build_molecule_ofdata(sample, basis_info.basis_dict)
    # build grid
    grid = grid_setup(mol, l1_grid_level, l1_grid_prune)

    density_difference = compute_density_density_basis(mol, grid, coeffs_delta)
    grid_weights = torch.as_tensor(grid.weights, dtype=OFDFT_DTYPE)
    l1_norm_difference = torch.dot(torch.abs(density_difference), grid_weights)
    return l1_norm_difference


def _single_shape_coeff_mae_scatter(
    ax: plt.Axes,
    coeff_errors: torch.Tensor,
    basis_info: BasisInfo,
    basis_function_ind: torch.Tensor,
):
    """Plots the mean absolute density coefficient error over basis dimensions.

    This simple version can be called, if all molecules display the same coefficient shape, e.g.
    for MD17.
    """
    coeffs_absolute_error = torch.abs(coeff_errors)
    # median returns a tuple of val and indices
    coeffs_mae = torch.median(coeffs_absolute_error, dim=0)[0]

    coeffs_mae_lower_bound = []
    coeffs_mae_upper_bound = []

    # Did not find a way to make this work without a loop
    for i in range(coeffs_absolute_error.shape[1]):
        sorted_errors = coeffs_absolute_error[:, i].sort().values
        lower_bound = sorted_errors[int(0.1 * sorted_errors.shape[0])]
        upper_bound = sorted_errors[int(0.9 * sorted_errors.shape[0])]
        coeffs_mae_lower_bound.append(lower_bound)
        coeffs_mae_upper_bound.append(upper_bound)

    coeffs_mae_lower_bound = torch.stack(coeffs_mae_lower_bound)
    coeffs_mae_upper_bound = torch.stack(coeffs_mae_upper_bound)

    x = basis_function_ind.detach().cpu() + 0.5
    # ax.errorbar(x, coeffs_mae, yerr=[coeffs_mae_below_std, coeffs_mae_above_std], fmt='o', **scatter_kwargs, label="Ground State Coeffs")

    # Use fill_between with sorted data
    ax.errorbar(
        x,
        coeffs_mae,
        yerr=[coeffs_mae - coeffs_mae_lower_bound, coeffs_mae_upper_bound - coeffs_mae],
        fmt=".",
        markersize=4,
        elinewidth=0.5,
        ecolor="C1",
        label="Mean Absolute Error",
    )

    handles = [
        Patch(color="C0", label="Median Absolute Error"),
        Patch(color="C1", label="Interquantile Range (10-90)"),
    ]
    ax.legend(handles=handles, loc="upper right")

    format_basis_func_xaxis(ax, basis_info)
    title = "Mean Absolute Error of Coefficients"
    ax.set_ylabel(
        r"$\left \langle \vert  \rho_{\mathrm{pred}} - \rho_{\mathrm{target}} \vert \right \rangle$ [electrons]",
        fontsize=14,
    )
    ax.set_yscale("log")

    # horizontal line for 0 error
    ax.axhline(0, color="grey", alpha=0.5)
    ax.set_title(title, fontsize=15)


def _multi_shape_coeff_mae_scatter(
    ax: plt.Axes,
    cumulative_coeff_errors: torch.Tensor,
    cumulative_counts: torch.Tensor,
    basis_function_indices: torch.Tensor | np.ndarray,
    basis_info: BasisInfo,
):
    """Plots the mean absolute coefficient error over basis dimensions for possibly varying number
    of basis dimensions.

    Args:
        cumulative_coeff_errors (torch.Tensor): Cumulative sum of coefficient absolute errors retrieved by
            :func:`cumulate_coeff_error`.
        cumulative_counts (torch.Tensor): Counts of basis dimension appearances retrieved by
            :func:`cumulate_coeff_error`.
        basis_function_indices (torch.Tensor | np.ndarray): A tensor of all basis dimensions present in the
            basis_info. Differs from a sample.basis_function_ind.
        basis_info (BasisInfo): The dataset's basis_info.
    """

    coeff_maes = cumulative_coeff_errors / cumulative_counts

    x = torch.as_tensor(basis_function_indices).detach().cpu() + 0.5

    scatter_kwargs = dict(s=40, alpha=0.6, marker=".", color="C0")
    ax.scatter(x, coeff_maes, **scatter_kwargs)
    format_basis_func_xaxis(ax, basis_info)
    title = "Mean Absolute Error of Coefficients"
    ax.set_ylabel(
        r"$\left \langle \vert  \rho_{\mathrm{pred}} - \rho_{\mathrm{target}} \vert \right \rangle$ [electrons]",
        fontsize=14,
    )
    ax.set_yscale("log")

    # horizontal line for 0 error
    ax.axhline(0, color="grey", alpha=0.5)
    ax.set_title(title, fontsize=15)


def plot_mean_and_fill_between(data: torch.Tensor, ax: plt.Axes, color: str = "blue"):
    """Plot the mean of the data and fill the area between mean - std and mean + std."""

    # Calculate mean and standard deviation
    if isinstance(data, torch.Tensor):
        data_means = torch.mean(data, dim=0)
        data_stds = torch.std(data, dim=0)

    elif isinstance(data, np.ndarray):
        data_means = np.mean(data, axis=0)
        data_stds = np.std(data, axis=0)
    else:
        try:
            data = torch.stack(data)
            data_means = torch.mean(data, dim=0)
            data_stds = torch.std(data, dim=0)
        except Exception as e:
            data_means, data_stds = get_overlapping_mean(data=data)

    ax.plot(data_means, color=color)

    # Plot confidence interval
    ax.fill_between(
        range(len(data_means)),
        data_means - data_stds,
        data_means + data_stds,
        color=color,
        alpha=0.2,
    )


def plot_quantiles_data(
    data: torch.Tensor | np.ndarray | Sequence[torch.Tensor] | Sequence[np.ndarray],
    ax: plt.Axes,
    color: str = "blue",
    quantiles=[0.0, 0.1, 0.9, 1.0],
    quantile_colors=["lightgreen", "blue", "lightcoral"],
):
    """Plot ensemble data on a given Axes object.

    Parameters:
    - data: 2D array of shape (num_datasets, num_points)
    - ax: matplotlib Axes object to plot on
    - color: color for the mean line (default: 'blue')
    - quantiles: list of quantiles to plot (default: [0.,0.1,0.9,1.])
    - quantile_colors: list of colors for quantile areas (default: ["lightgreen","blue", "lightcoral"])
    """
    # Calculate mean and quantiles
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            data = data.cpu()
        data_y = data.detach().numpy()
        mean_y = np.mean(data_y, axis=0)
        quantile_y = np.quantile(data_y, quantiles, axis=0)
    elif isinstance(data, np.ndarray):
        data_y = data
        mean_y = np.mean(data_y, axis=0)
        quantile_y = np.quantile(data_y, quantiles, axis=0)
    else:
        try:
            data_y = np.asarray(data)
            mean_y = np.mean(data_y, axis=0)
            quantile_y = np.quantile(data_y, quantiles, axis=0)
        except Exception as e:
            mean_y, quantile_y = get_overlapping_quantiles(data=data, quantiles=quantiles)
            mean_y = mean_y.detach().numpy()
            quantile_y = quantile_y.detach().numpy()

    mean_color = color
    # Create the plot
    x = np.arange(len(mean_y))
    # Plot mean line
    ax.plot(x, mean_y, label="Mean", color=mean_color, linewidth=2)

    # Plot quantile areas
    for i in range(len(quantiles) - 1):
        ax.fill_between(
            x,
            quantile_y[i],
            quantile_y[i + 1],
            alpha=0.3,
            color=quantile_colors[i],
            label=f"{quantiles[i] * 100}% - {quantiles[(i + 1)] * 100}% Quantile",
        )

    # set y-limit to the data and quantile range
    min_y = np.min(quantile_y[0] * 0.9)
    max_y = np.max(quantile_y[-1] * 1.1)
    ax.set_ylim(min_y, max_y)
    ax.legend()
    ax.grid(True, alpha=0.3)


def get_energy_error(sample: OFData, energy_name: str):
    """Get the (signed) energy error of 'energy_name' for a single sample in mHa."""

    predicted_energy = sample[f"trajectory_energy_{energy_name}"].detach().cpu()
    energy_label = sample[f"ground_state_energy_{energy_name}"]

    energy_error = 1e3 * (predicted_energy - energy_label)

    return energy_error


def get_energy_errors_dict(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, list], dict[str, list]]:
    """Calculate the (signed) energy errors for a set of molecules from the energy trajectories and
    their ground state values."""

    energy_errors_dict = {}
    energy_absolute_errors_dict = {}

    for energy_name in energy_ground_state_dict.keys():
        predicted_energies = energy_trajectories_dict[energy_name]
        try:
            energy_labels = energy_ground_state_dict[energy_name]
        except KeyError:
            logger.warning(
                f"ground state energy {energy_name} not found in sample."
                f"Energy errors for {energy_name} will not be plotted"
            )
            continue

        # predicted_energies is a list of length n_molecules with energies of possibly different
        # shape, allowing the number of cycles to vary
        energy_errors, energy_absolute_errors = [], []
        for predicted_energy, energy_label in zip(predicted_energies, energy_labels):
            error = predicted_energy - energy_label
            energy_errors.append(error)
            energy_absolute_errors.append(torch.abs(error))
        energy_errors_dict[energy_name] = energy_errors
        energy_absolute_errors_dict[energy_name] = energy_absolute_errors

    return energy_errors_dict, energy_absolute_errors_dict


def get_energy_ground_state_errors(
    energy_trajectories_dict: dict[str, list[torch.Tensor]],
    energy_ground_state_dict: dict[str, torch.Tensor],
    stopping_indices: torch.Tensor,
):
    energy_ground_state_errors, energy_ground_state_absolute_errors = {}, {}
    for energy_name in energy_ground_state_dict.keys():
        energy_ground_state_errors[energy_name] = torch.empty(
            len(energy_ground_state_dict[energy_name]), dtype=torch.float64
        )
        energy_ground_state_absolute_errors[energy_name] = torch.empty(
            len(energy_ground_state_dict[energy_name]), dtype=torch.float64
        )
        for i, (predicted_energy, energy_label) in enumerate(
            zip(energy_trajectories_dict[energy_name], energy_ground_state_dict[energy_name])
        ):
            stopping_index = stopping_indices[i]
            ground_state_error = predicted_energy[stopping_index] - energy_label
            energy_ground_state_errors[energy_name][i] = ground_state_error.item()
            energy_ground_state_absolute_errors[energy_name][i] = torch.abs(
                ground_state_error
            ).item()

    return energy_ground_state_errors, energy_ground_state_absolute_errors


def energy_error_swarm_line_plot(
    ax: plt.Axes,
    stopping_indices: torch.Tensor,
    energy_name: str,
    energy_errors_dict: dict[str, list[torch.Tensor]] | None = None,
    energy_errors: list[torch.Tensor] | None = None,
    n_molecules: int | None = None,
    subsample: float = 1.0,
):
    """Plot the energy error for a set of molecules as a swarm plot with a line for each
    molecule."""

    if n_molecules is None:
        n_molecules = len(stopping_indices)

    if energy_errors_dict is None and energy_errors is None:
        raise ValueError("Either energy_errors_dict or energy_errors must be provided.")
    if energy_errors_dict is not None and energy_errors is not None:
        raise ValueError("Only one of energy_errors_dict or energy_errors must be provided.")
    elif energy_errors_dict is not None:
        energy_errors = energy_errors_dict[energy_name]
    # else energy_errors is already provided

    assert energy_errors is not None

    ax.axhline(0, color="black")  # 0 line for reference
    ax.set_ylabel(r"$\Delta E$ [mHa]", fontsize=14)
    ax.set_yscale("symlog", linthresh=10)
    title = r"$\Delta E_\mathrm{" + energy_name.replace("_", r"\_") + "}$"
    ax.set_title(title, fontsize=15)

    ax.grid(True)

    # We do not use the subsample function as we sample in dependence of max and min error
    if subsample < 1.0 and n_molecules > 2:
        n_samples = len(energy_errors)
        # definitely get the trajectory with the max and min energy error
        # energy_errors is a list of length n_molecules, with tensors of varying denop iterations
        mean_energy_errors = torch.zeros(len(energy_errors))
        for i, mol_error_trajectory in enumerate(energy_errors):
            mean_energy_errors[i] = torch.mean(mol_error_trajectory)
        max_mean_energy_error_idx = torch.argmax(mean_energy_errors)
        min_mean_energy_error_idx = torch.argmin(mean_energy_errors)

        # plot min and max first
        stopping_index_line_plot(
            ax=ax,
            data_array=energy_errors[max_mean_energy_error_idx],
            stopping_index=int(stopping_indices[max_mean_energy_error_idx]),
            color="darkred",
            label="Max mean energy error",
        )

        stopping_index_line_plot(
            ax=ax,
            data_array=energy_errors[min_mean_energy_error_idx],
            stopping_index=int(stopping_indices[min_mean_energy_error_idx]),
            color="darkgreen",
            label="Min mean energy error",
        )

        ax.legend()

        # Then subsample the rest omitting the max and min
        subsample_indices = torch.randperm(n_samples)
        # remove the max and min indices
        subsample_indices = subsample_indices[
            (subsample_indices != max_mean_energy_error_idx)
            & (subsample_indices != min_mean_energy_error_idx)
        ]
        subsample_indices = subsample_indices[: int(subsample * n_samples - 2)]

        energy_errors = [energy_errors[subsample_idx] for subsample_idx in subsample_indices]
        stopping_indices = stopping_indices[subsample_indices]

        # overwrite n_molecules if subsampled
        n_molecules = len(subsample_indices)

    for i, energy_error_trajectory in enumerate(energy_errors[: n_molecules + 1]):
        stopping_index = int(stopping_indices[i])
        # energy_ground_state = energy_ground_state_dict[energy_name][i]
        # maybe relate the groundstate energy to the color

        color = plt.get_cmap("rainbow")(i / (n_molecules - 1 + 1e-6))
        stopping_index_line_plot(
            ax=ax,
            data_array=energy_error_trajectory,
            stopping_index=stopping_index,
            color=color,
        )


def gradient_norm_swarm_line_plot(
    ax: plt.Axes,
    stopping_indices: torch.Tensor,
    gradient_norms: torch.Tensor | list[torch.Tensor],
    n_molecules: int | None = None,
    subsample: float = 1.0,
):
    """Plot the gradient norms for a set of molecules as a swarm plot with a line for each
    molecule."""

    if n_molecules is None:
        n_molecules = len(stopping_indices)

    ax.set_yscale("log")
    ax.set_title("Gradient norm", fontsize=15)
    ax.grid(True)

    if subsample < 1.0 and n_molecules > 2:
        gradient_norms, stopping_indices, n_molecules = subsample_swarm(
            ax=ax,
            n_molecules=n_molecules,
            subsample=subsample,
            data_array=gradient_norms,
            stopping_indices=stopping_indices,
            label="gradient norm",
        )

    for i, gradient_norm_trajectory in enumerate(gradient_norms[: n_molecules + 1]):
        stopping_index = int(stopping_indices[i])
        # energy_ground_state = energy_ground_state_dict[energy_name][i]
        # maybe relate the groundstate energy to the color
        color = plt.get_cmap("rainbow")(i / (n_molecules - 1 + 1e-6))
        stopping_index_line_plot(
            ax=ax,
            data_array=gradient_norm_trajectory,
            stopping_index=stopping_index,
            color=color,
        )


def density_differences_swarm_line_plot(
    ax: plt.Axes,
    stopping_indices: torch.Tensor,
    density_differences_l2: torch.Tensor | list[torch.Tensor],
    n_molecules: int | None = None,
    subsample: float = 1.0,
):
    """Plot the energy error for a set of molecules as a swarm plot with a line for each
    molecule."""

    if n_molecules is None:
        n_molecules = len(stopping_indices)

    ax.axhline(0, color="black")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("Density difference", fontsize=15)
    ax.set_ylabel(
        r"$\Vert\rho_\mathrm{pred} - \rho_\mathrm{target}\Vert_2$ [electrons]", fontsize=14
    )
    ax.grid(True)

    if subsample < 1.0 and n_molecules > 2:
        density_differences_l2, stopping_indices, n_molecules = subsample_swarm(
            ax=ax,
            n_molecules=n_molecules,
            subsample=subsample,
            data_array=density_differences_l2,
            stopping_indices=stopping_indices,
            label="density difference",
        )

    for i, density_difference_trajectory in enumerate(density_differences_l2[: n_molecules + 1]):
        stopping_index = int(stopping_indices[i])
        # energy_ground_state = energy_ground_state_dict[energy_name][i]
        # maybe relate the groundstate energy to the color
        color = plt.get_cmap("rainbow")(i / (n_molecules - 1 + 1e-6))
        stopping_index_line_plot(
            ax=ax,
            data_array=density_difference_trajectory,
            stopping_index=stopping_index,
            color=color,
        )


def subsample_swarm(
    ax: plt.Axes,
    n_molecules: int,
    subsample: float,
    data_array: torch.Tensor | list[torch.Tensor] | np.ndarray,
    stopping_indices: torch.Tensor,
    label: str = "error",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Subsample the data array and stopping indices and plot the min and max on given axis."""

    if isinstance(data_array, np.ndarray):
        data_array = torch.from_numpy(data_array)
    elif not isinstance(data_array, torch.Tensor):
        data_array = torch.stack(list(data_array))

    n_samples = len(data_array)
    if subsample < 1.0 and n_molecules > 2:
        # definitely get the trajectory with the max and min energy error
        max_mean_data_idx = torch.argmax(data_array.mean(dim=1))
        min_mean_data_idx = torch.argmin(data_array.mean(dim=1))

        # plot min and max first
        stopping_index_line_plot(
            ax=ax,
            data_array=data_array[max_mean_data_idx],
            stopping_index=int(stopping_indices[max_mean_data_idx]),
            color="darkred",
            label=f"Max mean {label}",
        )

        stopping_index_line_plot(
            ax=ax,
            data_array=data_array[min_mean_data_idx],
            stopping_index=int(stopping_indices[min_mean_data_idx]),
            color="darkgreen",
            label=f"Min mean {label}",
        )

        ax.legend()

        # Then subsample the rest omitting the max and min
        subsample_indices = torch.randperm(n_samples)
        # remove the max and min indices
        subsample_indices = subsample_indices[
            (subsample_indices != max_mean_data_idx) & (subsample_indices != min_mean_data_idx)
        ]
        subsample_indices = subsample_indices[: int(subsample * n_samples - 2)]

        data_array = data_array[subsample_indices]
        stopping_indices = stopping_indices[subsample_indices]

        # overwrite n_molecules if subsampled
        n_molecules = len(subsample_indices)

        return data_array, stopping_indices, n_molecules

    return data_array, stopping_indices, n_molecules


def get_overlapping_mean(
    data: Sequence[Union[torch.Tensor, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the mean of a list of 1D tensors/arrays with different length for overlapping
    regions."""

    sorted_data = [torch.as_tensor(t) for t in sorted(data, key=len)]  # ascending in length

    overlapping_mean = torch.zeros(size=(len(sorted_data[-1]),))
    overlapping_std = torch.zeros(size=(len(sorted_data[-1]),))

    tensor_lengths = torch.tensor([len(tensor) for tensor in sorted_data])

    previous_ending_index = 0

    for i, length in enumerate(tensor_lengths):
        query_tensors = torch.stack(
            [tensor[previous_ending_index:length] for tensor in sorted_data[i:]]
        )
        overlapping_mean[previous_ending_index:length] = torch.mean(query_tensors, dim=0)
        overlapping_std[previous_ending_index:length] = torch.std(query_tensors, dim=0)

        previous_ending_index = length

    return overlapping_mean, overlapping_std


def get_overlapping_quantiles(
    data: Sequence[Union[torch.Tensor, np.ndarray]],
    quantiles: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the quantiles of a list of 1D tensors/arrays with different length for overlapping
    regions."""

    sorted_data = [torch.as_tensor(t) for t in sorted(data, key=len)]  # ascending in length

    overlapping_quantiles = torch.zeros(size=(len(quantiles), len(sorted_data[-1])))
    overlapping_mean = torch.zeros(size=(len(sorted_data[-1]),))

    tensor_lengths = torch.tensor([len(tensor) for tensor in sorted_data])

    previous_ending_index = 0

    for i, length in enumerate(tensor_lengths):
        if length - previous_ending_index == 0:
            continue
        query_tensors = torch.stack(
            [tensor[previous_ending_index:length] for tensor in sorted_data[i:]]
        )
        overlapping_mean[previous_ending_index:length] = torch.mean(query_tensors, dim=0)
        overlapping_quantiles[:, previous_ending_index:length] = torch.quantile(
            query_tensors, torch.tensor(quantiles, dtype=query_tensors.dtype), dim=0
        )

        previous_ending_index = length

    return overlapping_mean, overlapping_quantiles


def cumulate_coeff_error(
    sample: OFData,
    pred_ground_state_coeffs: torch.Tensor,
    cumulative_error: torch.Tensor,
    cumulative_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulates the absolute error of coefficients by adding the (stopping) basis_function wise
    error onto the cumulative error.

    This is necessary if density coefficient dimensions vary across molecules. The cumulative
    counts are returned as well in order to build the average afterwards.
    """
    coeffs_ground_state = sample.ground_state_coeffs.detach().cpu()
    absolute_error = torch.abs(pred_ground_state_coeffs - coeffs_ground_state).to(
        dtype=cumulative_error.dtype
    )
    # For compatibility with old samples saved from the M-OFDFT code
    if isinstance(sample.basis_function_ind, np.ndarray):
        sample.basis_function_ind = torch.tensor(
            sample.basis_function_ind,
            dtype=torch.int32 if OFDFT_DTYPE == torch.float32 else torch.int64,
        )
    counts = torch.ones_like(
        sample.basis_function_ind,
        dtype=torch.int32 if OFDFT_DTYPE == torch.float32 else torch.int64,
    )
    scatter(
        src=absolute_error, index=sample.basis_function_ind, out=cumulative_error, reduce="sum"
    )
    scatter(src=counts, index=sample.basis_function_ind, out=cumulative_counts, reduce="sum")

    return cumulative_error, cumulative_counts


def custom_round(value):
    # Determine the order of magnitude
    order_of_magnitude = 10 ** np.floor(np.log10(value))
    # Find the next highest integer in that magnitude
    next_highest = np.ceil(value / order_of_magnitude) * order_of_magnitude
    # Add one integer of that magnitude
    rounded_value = next_highest + order_of_magnitude
    return rounded_value


def _add_energy_histogram(
    ax: plt.Axes,
    energy_errors_dict: dict,
    energy_name="total",
    title="Stopping Energy Error Distribution",
    cut_off=10**3,
):
    """Add a histogram of the total energy error for a set of molecules."""

    energy_errors_dict = {key: value.numpy() for key, value in energy_errors_dict.items()}

    filtered_data = energy_errors_dict[energy_name]

    # if all values are positive, we can use a log scale
    if np.min(filtered_data) > 0:
        if cut_off:
            lower_cutoff = max(1 / cut_off, np.min(filtered_data))
            upper_cutoff = min(cut_off, np.max(filtered_data))

            # map all outliers to the cutoffs
            filtered_data = [
                lower_cutoff if x < lower_cutoff else upper_cutoff if x > upper_cutoff else x
                for x in filtered_data
            ]
            # count the data points outside the cutoffs
            outside_count = len(
                [
                    x
                    for x in energy_errors_dict[energy_name]
                    if x < lower_cutoff or x > upper_cutoff
                ]
            )

        total_energy_error_mean = np.mean(np.abs(filtered_data))
        if upper_cutoff - lower_cutoff > 1e-10:
            ax.set_xlim(left=lower_cutoff, right=upper_cutoff)
        linthresh = custom_round(total_energy_error_mean)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
        ax.axvline(linthresh, color="gray", linestyle="-", linewidth=0.7)
        sns.histplot(
            filtered_data,
            ax=ax,
            label=f"{energy_name} (outside range: {outside_count})",
            stat="density",
            bins="auto",
        )
    else:
        if cut_off:
            lower_cutoff = max(-cut_off, np.min(filtered_data))
            upper_cutoff = min(cut_off, np.max(filtered_data))
            if upper_cutoff - lower_cutoff > 1e-10:
                ax.set_xlim(lower_cutoff, upper_cutoff)
            # map all outliers to the cutoffs
            filtered_data = [
                lower_cutoff if x < lower_cutoff else upper_cutoff if x > upper_cutoff else x
                for x in filtered_data
            ]
            # count the data points outside the cutoffs
            outside_count = len(
                [
                    x
                    for x in energy_errors_dict[energy_name]
                    if x < lower_cutoff or x > upper_cutoff
                ]
            )

        total_energy_error_mean = np.mean(np.abs(filtered_data))
        linthresh = custom_round(total_energy_error_mean)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.xaxis.set_minor_locator(MinorSymLogLocator(linthresh))
        # Highlight linear region with solid lines
        ax.axvline(-linthresh, color="gray", linestyle="-", linewidth=0.7)
        ax.axvline(linthresh, color="gray", linestyle="-", linewidth=0.7)

        sns.histplot(
            filtered_data,
            ax=ax,
            label=f"{energy_name} (outside range: {outside_count})",
            stat="density",
            bins="auto",
        )
    ax.set_xlabel(r"$\Delta E$ [mHa]", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title(f"{energy_name} " + title, fontsize=14)
    ax.legend()


def _add_energy_error_over_n_atoms(
    ax: plt.Axes,
    energy_errors_dict: dict,
    num_atoms: torch.Tensor,
    energy_name="total",
    title="Absolute Stopping Energy Error vs Number of Atoms",
):
    """Add a scatter plot of the total energy error vs the number of atoms in the molecule."""

    energy_errors = energy_errors_dict[energy_name].numpy()
    num_atoms = num_atoms.cpu().numpy()

    # Add a small amount of horizontal jitter to the num_atoms values
    jitter_strength = 0.05  # Adjust this value as needed to get the desired amount of jitter
    jitter = np.random.uniform(-jitter_strength, jitter_strength, num_atoms.size)
    num_atoms_jittered = num_atoms + jitter

    ax.scatter(num_atoms_jittered, energy_errors, alpha=0.5, s=40 if len(num_atoms) < 50 else 10)
    # # use stripe plot instead
    # sns.stripplot(x=num_atoms, y=energy_errors, ax=ax, alpha=0.6, s=4)

    # ax.set_xscale("symlog", linthresh=custom_round(np.mean(np.abs(energy_errors))))
    ax.set_ylabel(r"$\Delta E$ [mHa]", fontsize=14)
    ax.set_xlabel(r"Number of Atoms", fontsize=14)
    ax.set_title(f"{energy_name} Absolute Energy Error vs Number of Atoms", fontsize=15)
    # set x_ticks as integers for number of atoms
    # get x-tick labels for all n_atoms in the dataset
    x_tick_labels = np.unique(num_atoms)

    ax.set_xticks(x_tick_labels)
    ax.set_yscale("log")


def _add_hessian_eigenvalues_hist(ax: plt.Axes, final_hessian_eigenvalues: list):
    """Add a histogram of the final hessian eigenvalues."""
    eigenvalues = [eig for eig in final_hessian_eigenvalues if eig is not None]
    eigenvalues = np.concatenate(eigenvalues)
    mask = eigenvalues < -1e-6
    bins = np.linspace(min(eigenvalues), max(eigenvalues), 150)
    ax.hist(eigenvalues[mask], color="r", bins=bins)
    ax.hist(eigenvalues[~mask], color="g", bins=bins)
    ax.set_title(
        f"Eigenvalues of the Hessian, {np.sum(mask)}/{mask.shape[0]} smaller than -1e-6",
        fontsize=14,
    )
    ax.set_xlabel("eigenvalue [Ha]", fontsize=14)
    ax.set_ylabel("count", fontsize=14)
    ax.set_yscale("log")
