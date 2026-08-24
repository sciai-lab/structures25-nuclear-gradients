import imageio
import numpy as np
import pyvista as pv
import torch
from pyscf import gto

from mldft.utils.molecules import build_molecule_ofdata
from mldft.utils.visualize_3d import visualize_orbital


def animate_orbital_trajectory(
    mol: gto.Mole,
    coeff_trajectory: list[np.ndarray],
    filename: str,
    mode: str = "isosurface",
    isosurface_quantile: float = 0.9,
    resolution: float = 0.3,
    margin: float = 3.0,
    fps: int = 10,
    subsample_points: int = 300,
    **kwargs,
):
    """Generate and save an animation of a trajectory from density optimization.

    Args:
        mol (gto.Mole): The molecule object.
        coeff_trajectory (list[np.ndarray]): List of orbital coefficients for each frame.
        filename (str): Output filename for the animation.
        mode (str): Visualization mode ('volume', 'isosurface', 'nested_isosurfaces').
        isosurface_quantile (float): Quantile for isosurface visualization.
        resolution (float): Resolution of the cube file.
        margin (float): Margin of the cube file.
        fps (int): Frames per second for the animation.
        subsample_points (int): Number of points to subsample in the trajectory.
        **kwargs: Additional keyword arguments for visualization.
    """

    if subsample_points is not None:
        # take every nth point in the trajectory to get subsample_points
        step = max(1, len(coeff_trajectory) // subsample_points)
        coeff_trajectory = coeff_trajectory[::step]

    frames = []
    for coeff in coeff_trajectory:
        # Create an off-screen plotter for each frame
        pl = pv.Plotter(off_screen=True)

        # Generate the orbital visualization
        visualize_orbital(
            mol,
            coeff,
            plotter=pl,
            mode=mode,
            isosurface_quantile=isosurface_quantile,
            resolution=resolution,
            margin=margin,
            **kwargs,
        )

        # Capture the frame as a numpy array
        frame = pl.screenshot(return_img=True)
        frames.append(frame)

        # Close the plotter to free memory
        pl.close()

    # Save the animation using imageio
    imageio.mimsave(filename, frames, fps=fps)


if __name__ == "__main__":
    # using argparse to allow command line execution
    import argparse

    parser = argparse.ArgumentParser(description="Animate orbital trajectory.")
    parser.add_argument(
        "sample_path",
        type=str,
        help="Path to the sample file containing the orbital coefficients.",
    )
    parser.add_argument("output_filename", type=str, help="Output filename for the animation.")
    parser.add_argument(
        "--mode",
        type=str,
        default="isosurface",
        help="Visualization mode ('volume', 'isosurface', 'nested_isosurfaces').",
    )
    parser.add_argument(
        "--isosurface_quantile",
        type=float,
        default=0.9,
        help="Quantile for isosurface visualization.",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.3, help="Resolution of the cube file."
    )
    parser.add_argument("--margin", type=float, default=3.0, help="Margin of the cube file.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the animation.")
    parser.add_argument(
        "--subsample_points", type=int, default=300, help="Number of points to subsample."
    )
    parser.add_argument(
        "--plot_diff",
        action="store_true",
        help="If set, plot the difference from the ground state coefficients.",
    )

    args = parser.parse_args()

    sample = torch.load(args.sample_path)
    basis_info = sample.basis_info
    mol = build_molecule_ofdata(sample, basis_info.basis_dict)
    coeff_trajectory = sample.trajectory_coeffs.cpu().numpy()
    if args.plot_diff:
        coeff_trajectory = coeff_trajectory - sample.ground_state_coeffs.cpu().numpy()[None, :]
    print("coeff_trajectory shape:", coeff_trajectory.shape)

    animate_orbital_trajectory(
        mol,
        coeff_trajectory,
        args.output_filename,
        mode=args.mode,
        isosurface_quantile=args.isosurface_quantile,
        resolution=args.resolution,
        margin=args.margin,
        fps=args.fps,
        subsample_points=300,  # Adjust as needed
    )
    print(f"Animation saved to {args.output_filename}")
