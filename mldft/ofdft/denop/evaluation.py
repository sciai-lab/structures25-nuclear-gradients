from __future__ import annotations

import csv
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable

import torch
from loguru import logger
from matplotlib import pyplot as plt
from torch_geometric.data.data import BaseData

from mldft.ml.data.components.basis_transforms import (
    ApplyBasisTransformation,
    transform_tensor_with_sample,
)
from mldft.ml.data.components.of_data import BasisInfo, OFData, Representation
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.utils.pdf_utils import HierarchicalPlotPDF
from mldft.utils.plotting.plot_density_optimization import (
    filter_plotting_functions,
    plot_density_optimization,
)
from mldft.utils.plotting.summary_density_optimization import (
    density_optimization_summary_pdf_plot,
    get_runwise_density_optimization_data,
    save_density_optimization_metrics,
)
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula
from mldft.utils.sample import add_density_optimization_trajectories_to_sample


class DenopEvaluator:
    """Evaluate and summarize denop outputs saved in ``save_dir``."""

    def __init__(
        self,
        save_dir: Path | str,
        n_molecules: int | None = None,
        energy_names: list[str] | None = None,
        compute_grid_metrics: bool = True,
        compare_to_ks: bool = False,
        metrics_grid_level: int = 3,
        metrics_grid_prune: str = "nwchem_prune",
        swarm_plot_subsample: float = 1.0,
        plot_summary: bool = True,
        save_coeff_interval: int = 100,
        compute_final_hessian: bool = False,
        plotter: DenopPlotter | None = None,
        plot_every_n: int = 0,
    ) -> None:
        """Initialize evaluator configuration for one denop output directory.

        Args:
            save_dir (Path | str): Directory containing denop outputs.
            n_molecules (int | None): Optional limit of molecules to include.
            energy_names (list[str] | None): Optional energy keys to evaluate.
            plot_l1_norm (bool): Whether to compute and include L1 norm metrics.
            l1_grid_level (int): Grid level used for L1 norm evaluation.
            l1_grid_prune (str): Grid pruning strategy used for L1 norm evaluation.
            swarm_plot_subsample (float): Fraction of points to include in swarm plots.
            plot_summary (bool): Whether to render summary PDF plots.
            save_coeff_interval (int): Interval used when exporting coefficient trajectories.
            compute_final_hessian (bool): Whether to compute/store final Hessian eigenvalues.
            plotter (DenopPlotter | None): Optional per-sample plotter invoked after finalization.
        """
        # Side effects:
        # - Normalizes and stores evaluator configuration/state for subsequent finalization calls.
        # - Defines output file locations (for example reduced CSV path) under ``save_dir``.

        self.save_dir = Path(save_dir)
        self.n_molecules = n_molecules
        self.energy_names = energy_names
        self.swarm_plot_subsample = swarm_plot_subsample
        self.plot_summary = plot_summary
        self.save_coeff_interval = save_coeff_interval
        self.compute_final_hessian = compute_final_hessian
        self.plotter = plotter
        self.plot_every_n = plot_every_n
        self._finalized_sample_count = 0
        self.csv_path = self.save_dir / "reduced_denop_data.csv"
        self.compute_grid_metrics = compute_grid_metrics
        self.compare_to_ks = compare_to_ks
        self.metrics_grid_level = metrics_grid_level
        self.metrics_grid_prune = metrics_grid_prune
        if self.plotter is not None and self.plot_every_n > 0:
            (self.save_dir / "plot_trajectories").mkdir(exist_ok=True)

    def evaluate(self) -> None:
        """Evaluate saved denop trajectories, persist summary metrics, and optionally plot."""
        run_data_dict = get_runwise_density_optimization_data(
            sample_dir=self.save_dir / "sample_trajectories",
            compute_grid_metrics=self.compute_grid_metrics,
            compare_to_ks=self.compare_to_ks,
            metrics_grid_level=self.metrics_grid_level,
            metrics_grid_prune=self.metrics_grid_prune,
        )
        save_density_optimization_metrics(
            output_path=self.save_dir / "density_optimization_metrics.yaml",
            run_data_dict=run_data_dict,
        )
        if self.plot_summary:
            logger.info("Saving density optimization summary plot.")
            density_optimization_summary_pdf_plot(
                out_pdf_path=self.save_dir / "density_optimization_summary.pdf",
                run_data_dict=run_data_dict,
                subsample=self.swarm_plot_subsample,
            )
        if self.plotter is not None and self.plot_every_n > 0:
            self.plotter.plot()

    def finalize_sample(
        self,
        sample: OFData | BaseData,
        callback: ConvergenceCallback,
        energies_label,
        metric_dict: dict,
        energy_functional,
        sample_index: int | None = None,
    ) -> None:
        """Finalize one optimized sample and persist output artifacts.

        Args:
            sample (OFData | BaseData): The sample to finalize, containing trajectory and metadata.
            callback (ConvergenceCallback): Callback instance used during optimization, providing access to trajectory data.
            energies_label (_type_): The energy corresponding to the energy values in the trajectory, used for plotting and summary.
            metric_dict (dict): The dictionary of metrics to be computed.
            energy_functional (_type_): The energy functional used for computing the final hessian.
            sample_index (int | None, optional): The index of the sample. Defaults to None.
        """
        if self.compute_final_hessian:
            hess_sample = sample.clone()
            hessian = energy_functional(hess_sample, compute_hessian=True)[2]
            eigenvalues, _ = torch.linalg.eigh(hessian)

            # TODO: this should go away if possible.
            sample.add_item(
                "final_hessian_eigenvalues",
                eigenvalues,
                representation=Representation.SCALAR,
            )
            untransformed_hessian = transform_tensor_with_sample(
                sample,
                hessian,
                Representation.BILINEAR_FORM,
                invert=True,
            )
            untransformed_eigenvalues, _ = torch.linalg.eigh(untransformed_hessian)

            # TODO: this should go away if possible.
            sample.add_item(
                "final_untransformed_hessian_eigenvalues",
                untransformed_eigenvalues,
                representation=Representation.SCALAR,
            )

        # TODO: this should go away if possible
        if hasattr(sample, "overlap_matrix"):
            sample.delete_item("overlap_matrix")
        if hasattr(sample, "coulomb_matrix"):
            sample.delete_item("coulomb_matrix")
        if hasattr(sample, "nuclear_attraction_vector"):
            sample.delete_item("nuclear_attraction_vector")
        if hasattr(sample, "ao"):
            sample.delete_item("ao")

        sample = ApplyBasisTransformation()(
            sample,
            sample.transformation_matrix,
            sample.inv_transformation_matrix,
            invert=True,
        )
        sample.delete_item("transformation_matrix")
        sample.delete_item("inv_transformation_matrix")

        stage_index = sample_index
        if stage_index is None:
            stage_index = self._finalized_sample_count
        should_stage_plot = (
            self.plotter is not None
            and self.plot_every_n > 0
            and stage_index % self.plot_every_n == 0
        )
        plot_sample = None
        if should_stage_plot:
            plot_sample = add_density_optimization_trajectories_to_sample(
                sample.clone(),
                callback,
                energies_label,
                save_coeff_interval=1,
            )

        sample = add_density_optimization_trajectories_to_sample(
            sample, callback, energies_label, self.save_coeff_interval
        )

        self._append_reduced_csv_row(sample)
        sample_path = self.save_dir / "sample_trajectories" / f"sample_{sample.mol_id}.pt"
        torch.save(sample, sample_path)
        if plot_sample is not None:
            plot_path = self.save_dir / "plot_trajectories" / f"sample_{sample.mol_id}.pt"
            torch.save(plot_sample, plot_path)

        logger.info(
            f"{sample.mol_id:<15} {mole_to_sum_formula(sample.mol, use_subscript=True):<15}"
            f"ΔE={metric_dict['gs_signed_energy_error'] * 1000:6.2g} mHa, "
            f"|Δρ|={metric_dict['gs_l2_norm']:6.2g}, "
            f"|g|={metric_dict['gs_gradient_norm']:7.2e}, "
            f"iters={metric_dict['stopping_index'] + 1:<4}"
        )
        self._finalized_sample_count += 1

    def _append_reduced_csv_row(self, sample: OFData | BaseData) -> None:
        """Append one summary row for a sample to the reduced CSV output."""
        stopping_index = sample.stopping_index
        sample_path = self.save_dir / "sample_trajectories" / f"sample_{sample.mol_id}.pt"
        saved_data = {
            "mol_id": sample.mol_id,
            "sum_formula": mole_to_sum_formula(sample.mol, True),
            "ground_state_energy_total": sample.ground_state_energy_total,
            "final_energy_total": sample.trajectory_energy_total[stopping_index].item(),
            "final_density_l2_norm": sample.trajectory_l2_norm[stopping_index].item(),
            "final_gradient_norm": sample.trajectory_gradient_norm[stopping_index].item(),
            "total_energy_error_per_atom": torch.abs(
                sample.trajectory_energy_total[stopping_index] - sample.ground_state_energy_total
            ).item()
            / sample.n_atom,
            "density_l2_error_per_electron": sample.trajectory_l2_norm[stopping_index].item()
            / sample.n_electron,
            "iterations": stopping_index,
            "path_to_sample_tra": sample_path,
        }

        file_exists = self.csv_path.exists()
        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=saved_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(saved_data)


class DenopPlotter:
    """Create per-sample denop trajectory plots from saved run artifacts."""

    def __init__(
        self,
        save_dir: Path | str,
        plotting_functions: list[Callable],
        enable_grid_plots: bool,
        basis_info: BasisInfo,
        save_individual_plots: bool = False,
        fail_fast: bool = False,
        enable_grid_operations: bool = False,
        compute_final_hessian: bool = False,
    ):
        """Initialize plotting helper for saved denop trajectories.

        Args:
            save_dir (Path | str): Denop output directory.
            plotting_functions (list[Callable]): Candidate plotting callables from
                config.
            enable_grid_plots (bool): Whether grid-dependent plot operations are
                allowed.
            basis_info (BasisInfo): Basis metadata used for angular-momentum
                annotations.
            save_individual_plots (bool): Save each figure as separate files in
                addition to PDF.
            fail_fast (bool): Reserved flag for strict failure behavior.
            enable_grid_operations (bool): Whether to keep grid-dependent plotting
                functions.
            compute_final_hessian (bool): Whether final-hessian-dependent plots may
                be emitted.
        """
        # Side effects:
        # - Filters and stores plotting function configuration for later plotting calls.
        # - Initializes plotter runtime state tied to ``save_dir``.

        self.save_dir = Path(save_dir)
        self.plotting_functions = filter_plotting_functions(
            plotting_functions,
            enable_grid_operations=enable_grid_operations,
            compute_final_hessian=compute_final_hessian,
        )
        self.basis_info = basis_info
        self.enable_grid_plots = enable_grid_plots
        self.save_individual_plots = save_individual_plots
        self.fail_fast = fail_fast
        self._processed_plot_paths: set[Path] = set()

    def plot(self) -> None:
        """Generate trajectory PDFs for all saved plot samples.

        Args:
            n_processes (int): Worker-process count for parallel plotting.
        """
        # Side effects:
        # - May set the global multiprocessing start method to ``spawn``.
        # - Spawns worker processes and dispatches plotting jobs across them.
        logger.info("Starting denop trajectory plotting.")
        if mp.get_context().get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing start method to 'spawn'.")
        plot_root = self.save_dir / "plot_trajectories"
        if not plot_root.exists():
            return

        plot_paths = sorted(plot_root.glob("*.pt"))
        new_plot_paths = [path for path in plot_paths if path not in self._processed_plot_paths]
        if len(new_plot_paths) == 0:
            return

        with HierarchicalPlotPDF(
            out_pdf_path=self.save_dir / "density_optimization.pdf",
            individual_plot_directory=(
                self.save_dir / "individual_results" if self.save_individual_plots else None
            ),
        ) as plotpdf:
            logger.info(f"Rendering {len(new_plot_paths)} new sample trajectories.")
            for plot_path in new_plot_paths:
                self._plot_single_run(plot_path, plotpdf)
                os.unlink(plot_path)

        self._processed_plot_paths.update(new_plot_paths)

    def _plot_single_run(
        self,
        plot_path: Path | str,
        plotpdf: HierarchicalPlotPDF,
    ) -> None:
        """Render one sample trajectory into the hierarchical PDF backend.

        Args:
            plot_path (Path | str): Path to one serialized plotting sample.
            individual_plot_directory (Path | str): Directory to save individual plot files.
            plotpdf (HierarchicalPlotPDF): PDF backend for hierarchical plotting.
        """
        # Side effects:
        # - Loads sample data from disk and creates plotting artifacts.
        # - Writes PDF/optional individual plot files and closes matplotlib figures.

        # Load plotting data
        sample = torch.load(plot_path, map_location="cpu", weights_only=False)
        coeff_dim = sample.ground_state_coeffs.shape[0]
        sample.add_item(
            "transformation_matrix",
            torch.eye(coeff_dim, dtype=torch.float64),
            representation=Representation.VECTOR,
        )
        sample.add_item(
            "inv_transformation_matrix",
            torch.eye(coeff_dim, dtype=torch.float64),
            representation=Representation.VECTOR,
        )

        basis_l = self.basis_info.l_per_basis_func[sample.basis_function_ind]
        final_hessian_eigenvalues = None
        if hasattr(sample, "final_hessian_eigenvalues"):
            final_hessian_eigenvalues = sample.final_hessian_eigenvalues

        plots = self.plotting_functions if len(self.plotting_functions) > 0 else None
        if plots is None:
            raise ValueError(
                "No plotting functions available after filtering. Check configuration."
            )
        fig = plot_density_optimization(
            sample,
            sample.ground_state_coeffs,
            plots=plots,
            stopping_index=sample.stopping_index,
            basis_l=basis_l,
            final_hessian=final_hessian_eigenvalues,
            add_mol_view_link=True,
        )
        plotpdf.savefig(f"{sample.mol_id}")
        plt.close(fig)
