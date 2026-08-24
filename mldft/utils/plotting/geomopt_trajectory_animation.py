from functools import partial

import imageio
import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

from mldft.ml.models.components.local_frames_module import LocalBasisModule
from mldft.utils.cube_files import DataCube
from mldft.utils.visualize_3d import (
    get_local_frames_mesh_dict,
    get_sticks_mesh_dict,
    plot_density,
)


def _eval_of_density(coords: np.ndarray, mol, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate OF density at given coordinates using density coefficients.

    Args:
        coords: The coordinates to evaluate the density at, shape (n, 3).
        mol: The molecule.
        coeffs: The OF density coefficients.

    Returns:
        The density at the given coordinates, shape (n).
    """
    ao = mol.eval_gto("GTOval", coords)
    rho = np.einsum("...i,i->...", ao, coeffs)
    return rho


def animate_geomopt_trajectory(
    sample,
    filename: str = "geomopt_animation.gif",
    fps: float = 10,
    subsample_points: int = None,
    show_local_frames: bool = False,
    show_density: bool = True,
    ignore_hydrogen: bool = True,
    axes_radius_scale: float = 0.04,
    scale: float = 1.5,
    opacity: float = 0.5,
    density_resolution: float = 0.3,
    density_margin: float = 3.0,
    density_isosurface_quantile: float = 0.9,
    density_opacity: float = 0.4,
):
    """Generate and save an animation of a geometry optimization trajectory.

    Args:
        sample: The sample containing geomopt_trajectory and mol.
        filename: Output filename for the animation.
        resolution: Resolution for rendering atoms/bonds.
        fps: Frames per second for the animation.
        subsample_points: Number of frames to subsample.
        show_local_frames: Whether to show local coordinate frames.
        show_density: Whether to show ground state density.
        ignore_hydrogen: Whether to ignore hydrogen in local frames.
        axes_radius_scale: Scale for local frame axes.
        scale: Scale for local frame arrows.
        opacity: Opacity for sticks/bonds.
        density_resolution: Resolution for density cube.
        density_margin: Margin for density cube.
        density_isosurface_quantile: Quantile for density isosurface.
        density_opacity: Opacity for density isosurface.
    """
    coords_traj = sample.geomopt_trajectory.cpu().numpy()
    coeffs_traj = sample.ground_state_coeffs_trajectory.cpu().numpy()
    assert len(coords_traj) == len(coeffs_traj), "Trajectory lengths do not match."
    mol = sample.mol
    atomic_numbers = torch.as_tensor(mol.atom_mass_list())

    # Subsample trajectory if needed
    if subsample_points is not None:
        step = max(1, len(coords_traj) // subsample_points)
        coords_traj = coords_traj[::step]
        coeffs_traj = coeffs_traj[::step]

    frames = []
    for coords, coeffs in tqdm(zip(coords_traj, coeffs_traj), total=len(coords_traj)):
        pl = pv.Plotter(off_screen=True, window_size=[1000, 1000], image_scale=2)
        mol.set_geom_(coords, unit="Bohr")

        # Add density visualization
        if show_density:
            cube = DataCube.from_function(
                mol=mol,
                func=partial(_eval_of_density, mol=mol, coeffs=coeffs),
                resolution=density_resolution,
                margin=density_margin,
            )
            plot_density(
                cube,
                plotter=pl,
                mode="isosurface",
                isosurface_quantile=density_isosurface_quantile,
                isosurface_opacity=density_opacity,
                plot_molecule=False,
            )

        # Add molecule
        pl.add_mesh(**get_sticks_mesh_dict(mol, resolution=10), opacity=opacity)

        # Add local frames if requested
        if show_local_frames:
            local_bases = LocalBasisModule(ignore_hydrogen=ignore_hydrogen)(
                torch.as_tensor(coords), atomic_numbers
            )
            pl.add_mesh(
                **get_local_frames_mesh_dict(
                    coords,
                    local_bases.numpy(),
                    scale=scale,
                    axes_radius_scale=axes_radius_scale,
                    resolution=10,
                )
            )

        pl.camera_position = "iso"
        frame = pl.screenshot(return_img=True)
        frames.append(frame)
        pl.close()

    imageio.mimsave(filename, frames, fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Animate geometry optimization trajectory.")
    parser.add_argument("sample_path", type=str, help="Path to the sample file.")
    parser.add_argument("output_filename", type=str, help="Output filename for the animation.")
    parser.add_argument("--subsample_points", type=int, default=None, help="Subsample frames.")
    parser.add_argument(
        "--show_local_frames", action="store_true", help="Show local coordinate frames."
    )
    parser.add_argument("--no_density", action="store_true", help="Hide ground state density.")
    parser.add_argument(
        "--ignore_hydrogen", action="store_true", help="Ignore hydrogen atoms in local frames."
    )
    parser.add_argument("--axes_radius_scale", type=float, default=0.04, help="Axes radius scale.")
    parser.add_argument("--scale", type=float, default=1.5, help="Arrow scale.")
    parser.add_argument("--opacity", type=float, default=0.5, help="Bond opacity.")
    parser.add_argument(
        "--density_resolution", type=float, default=0.3, help="Density cube resolution."
    )
    parser.add_argument("--density_margin", type=float, default=3.0, help="Density cube margin.")
    parser.add_argument(
        "--density_isosurface_quantile",
        type=float,
        default=0.9,
        help="Density isosurface quantile.",
    )
    parser.add_argument(
        "--density_opacity", type=float, default=0.4, help="Density isosurface opacity."
    )
    parser.add_argument("--fps", type=float, default=10, help="Frames per second.")

    args = parser.parse_args()
    sample = torch.load(args.sample_path, map_location=torch.device("cpu"), weights_only=False)
    animate_geomopt_trajectory(
        sample,
        args.output_filename,
        fps=args.fps,
        subsample_points=args.subsample_points,
        show_local_frames=args.show_local_frames,
        show_density=not args.no_density,
        ignore_hydrogen=args.ignore_hydrogen,
        axes_radius_scale=args.axes_radius_scale,
        scale=args.scale,
        opacity=args.opacity,
        density_resolution=args.density_resolution,
        density_margin=args.density_margin,
        density_isosurface_quantile=args.density_isosurface_quantile,
        density_opacity=args.density_opacity,
    )
    print(f"Animation saved to {args.output_filename}")
