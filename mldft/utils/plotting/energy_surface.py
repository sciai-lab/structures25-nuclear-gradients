import argparse
from functools import partial
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pyscf import dft
from pyscf.gto import Mole
from tqdm import tqdm

from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ml.models.components.loss_function import project_gradient
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft import basis_integrals
from mldft.ofdft.functional_factory import FunctionalFactory
from mldft.utils.environ import parse_run_path
from mldft.utils.grids import grid_setup
from mldft.utils.instantiators import instantiate_datamodule
from mldft.utils.molecules import build_molecule_ofdata


def visualize_energy_surface(
    run_path: Path | str,
    mol_idx: int = 0,
    *,
    scale: float = 1e4,
    resolution: int = 31,
    use_3d=True,
    device="cuda",
):
    """Visualize the energy surface of a run.

    Args:
        run_path: The path to the run directory.
        mol_idx: The index of the molecule to visualize.
        scale: The scaling factor for the grid.
        resolution: The resolution of the grid in each direction.
        use_3d: Whether to use a 3D plot or two 2D plots.
        device: The device to run the visualization on.
    """
    run_path = parse_run_path(run_path)
    lightning_module = MLDFTLitModule.load_from_checkpoint(
        run_path / "checkpoints/last.ckpt", map_location=device
    )
    lightning_module.eval()
    lightning_module.to(torch.float64)
    model_config = OmegaConf.load(run_path / ".hydra/config.yaml")
    basis_info = instantiate(model_config.data.basis_info)
    datamodule = instantiate_datamodule(model_config, limit_scf_iterations=-3)
    datamodule.setup(stage="validate")
    dataset = datamodule.val_set

    mol_slices = list(pairwise(dataset.path_indices))
    samples = []
    for i in range(*mol_slices[mol_idx]):
        sample = dataset[i]
        sample = sample.to(device)
        samples.append(sample)

    mol = build_molecule_ofdata(samples[0], basis_info.basis_dict)
    functional_sample = build_functional(lightning_module, samples[0], mol, device)

    functional = partial(evaluate_functional, functional=functional_sample, sample=samples[0])

    coeffs_gs = samples[-1].coeffs
    coeffs_1 = samples[-2].coeffs
    coeffs_2 = samples[-3].coeffs

    delta_1 = coeffs_1 - coeffs_gs
    delta_2 = coeffs_2 - coeffs_gs

    # make second delta orthogonal to first and scale it to have the same norm
    delta_2 -= delta_1 * delta_1 @ delta_2 / (delta_1 @ delta_1)
    delta_2 *= torch.linalg.norm(delta_1) / torch.linalg.norm(delta_2)

    # evaluate energy on a grid
    xs = scale * np.linspace(-1, 1, resolution)
    ys = xs.copy()

    energies = np.zeros((len(xs), len(ys)))
    gradient_norms = np.zeros_like(energies)
    proj_gradient_norms = np.zeros_like(energies)

    for i, x in enumerate(tqdm(xs)):
        for j, y in enumerate(ys):
            coeffs = coeffs_gs + x * delta_1 + y * delta_2
            energy, gradient = functional(coeffs)
            energies[i, j] = (
                energy.electronic_energy - samples[0]["of_labels/energies/ground_state_e_electron"]
            )
            proj_gradient = project_gradient(gradient, samples[0])
            gradient_norms[i, j] = torch.linalg.norm(gradient)
            proj_gradient_norms[i, j] = torch.linalg.norm(proj_gradient)

    # plotting
    if use_3d:
        x_grid, y_grid = np.meshgrid(xs, ys)
        plt.figure(figsize=(8, 12))

        ax = plt.subplot(211, projection="3d")
        ax.plot_surface(x_grid, y_grid, 1e3 * energies, cmap="inferno_r", edgecolor="none")
        ax.set_title("Electronic energy")
        ax.set_xlabel(r"$\Delta_1$")
        ax.set_ylabel(r"$\Delta_2$")
        ax.set_zlabel("Energy [mHa]")

        ax = plt.subplot(212, projection="3d")
        ax.plot_surface(x_grid, y_grid, proj_gradient_norms, cmap="inferno_r", edgecolor="none")
        ax.set_title("Projected gradient norm")
        ax.set_xlabel(r"$\Delta_1$")
        ax.set_ylabel(r"$\Delta_2$")
        ax.set_zlabel("Gradient norm")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        img = axes[0].imshow(
            energies * 1e3,
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            origin="lower",
            cmap="inferno_r",
        )
        axes[0].set_title("Electronic energy")
        axes[0].set_xlabel("difference ground state to second to last SCF iteration")
        axes[0].set_ylabel("orthogonal direction in plane with three last SCF iterations")
        axes[0].plot(0, 0, "ro")
        plt.colorbar(img, label="mHa", ax=axes[0])

        img = axes[1].imshow(
            proj_gradient_norms,
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            origin="lower",
            cmap="inferno_r",
        )
        axes[1].set_title("Projected gradient norm")
        axes[1].set_xlabel("difference ground state to second to last SCF iteration")
        axes[1].set_ylabel("orthogonal direction in plane with three last SCF iterations")
        axes[1].plot(0, 0, "ro")
        plt.colorbar(img)

    plt.suptitle(run_path.name)
    plt.tight_layout()
    plt.show()


def build_functional(lightning_module: MLDFTLitModule, sample: OFData, mol: Mole, device: str):
    """Builds the functional for a given sample.

    Args:
        lightning_module: The lightning module.
        sample: The OFData sample.
        mol: The molecule object.
        device: The device to run the functional on.
    """

    func_factory = FunctionalFactory.from_module(lightning_module, "PBE")

    coulomb_matrix = basis_integrals.get_coulomb_matrix(mol)
    coulomb_matrix = torch.as_tensor(coulomb_matrix, dtype=torch.float64, device=device)
    coulomb_matrix = transform_tensor_with_sample(
        sample, coulomb_matrix, Representation.BILINEAR_FORM
    )

    nuclear_attraction = basis_integrals.get_nuclear_attraction_vector(mol)
    nuclear_attraction = torch.as_tensor(nuclear_attraction, dtype=torch.float64, device=device)
    nuclear_attraction = transform_tensor_with_sample(
        sample, nuclear_attraction, Representation.DUAL_VECTOR
    )

    grid = grid_setup(mol)

    ao = dft.numint.eval_ao(mol, grid.coords, deriv=func_factory.max_derivative)
    ao = transform_tensor_with_sample(
        sample,
        torch.as_tensor(ao, dtype=torch.float64, device=device),
        Representation.AO,
    )

    return func_factory.construct(
        mol=mol,
        coulomb_matrix=coulomb_matrix,
        nuclear_attraction_vector=nuclear_attraction,
        grid=grid,
        ao=ao,
    )


def evaluate_functional(coeffs, functional, sample):
    """Evaluate the functional for a given sample.

    Args:
        coeffs: The coefficients.
        functional: The functional.
        sample: The OFData sample.
    """
    sample.coeffs = torch.as_tensor(coeffs, dtype=torch.float64, device=sample.coeffs.device)
    energy, gradient = functional(sample)
    return energy, gradient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", type=str, help="Name or path of the run")
    parser.add_argument(
        "--mol_idx", type=int, help="Index of the molecule to visualize", default=0
    )
    parser.add_argument("--scale", type=float, help="Scaling factor for the grid", default=1e4)
    parser.add_argument("--resolution", type=int, help="Resolution of the grid", default=31)
    parser.add_argument("--plot_type", type=str, help="Plot type: either 2d or 3d", default="2d")
    parser.add_argument("--device", type=str, help="Device of the computation", default="cuda")
    args = parser.parse_args()

    assert args.plot_type in ["2d", "3d"], "plot_type must be either 2d or 3d"
    use_3d_plot = True if args.plot_type == "3d" else False

    visualize_energy_surface(
        args.run_path,
        args.mol_idx,
        scale=args.scale,
        resolution=args.resolution,
        use_3d=use_3d_plot,
        device=args.device,
    )
