import os
import subprocess
import sys
from typing import cast

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import pytest
import pyvista as pv
from pyscf import gto

from mldft.utils.visualize_3d import plot_molecule, visualize_density, visualize_orbital

mols_image_sizes = [(gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"), (200, 200))]


def off_screen_rendering_available():
    """Check VTK off-screen rendering in an isolated process."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pyvista as pv; "
                "plotter = pv.Plotter(off_screen=True); "
                "plotter.show(screenshot=True); plotter.close()",
            ],
            env={**os.environ, "PYVISTA_OFF_SCREEN": "true"},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


headless = (
    os.name != "posix"  # display check probably only works on linux
    or "DISPLAY" not in os.environ
    or not os.environ["DISPLAY"]
    or os.environ["DISPLAY"].startswith("localhost")
)
rendering_unavailable = not off_screen_rendering_available()


def run_in_isolated_process(request: pytest.FixtureRequest) -> bool:
    """Run rendering test in a separate process to contain VTK crashes."""
    if os.environ.get("PYVISTA_TEST_CHILD") == "1":
        return False

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "not slow", request.node.nodeid, "-q"],
            cwd=os.getcwd(),
            env={**os.environ, "PYVISTA_TEST_CHILD": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("VTK rendering did not finish in an isolated process")

    if result.returncode < 0:
        pytest.skip("VTK rendering crashed in an isolated process")

    assert result.returncode == 0, result.stdout
    return True


@pytest.mark.skipif(
    headless or rendering_unavailable,
    reason="Cannot confirm display or VTK off-screen rendering available",
)
@pytest.mark.parametrize("mol, image_size", mols_image_sizes)
def test_plot_molecule(mol, image_size, request: pytest.FixtureRequest):
    """Test whether plot_molecule produces images."""
    if run_in_isolated_process(request):
        return

    pl = pv.Plotter(window_size=image_size, off_screen=True, notebook=False)
    plot_molecule(mol, plotter=pl)
    image = cast(np.ndarray, pl.show(screenshot=True))

    # make sure image is not empty
    assert np.max(image) != np.min(image), "Output image is constant."
    assert image.shape == image_size + (
        3,
    ), f"Incorrect images size: {image.shape} != {image_size + (3,)}"


mols_image_sizes_vis_kwargs = [
    (gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"), (100, 100), dict(mode="volume")),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="volume", plot_molecule="false"),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="isosurface"),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="nested_isosurfaces"),
    ),
]


@pytest.mark.skipif(
    headless or rendering_unavailable,
    reason="Cannot confirm display or VTK off-screen rendering available",
)
@pytest.mark.parametrize("mol, image_size, visualize_orbital_kwargs", mols_image_sizes_vis_kwargs)
def test_visualize_orbital(
    mol, image_size, visualize_orbital_kwargs, request: pytest.FixtureRequest
):
    """Test whether visualize_orbital produces images."""
    if run_in_isolated_process(request):
        return

    pl = pv.Plotter(window_size=image_size, off_screen=True, notebook=False)
    visualize_orbital(mol, coeff=np.ones(mol.nao), plotter=pl, **visualize_orbital_kwargs)
    image = cast(np.ndarray, pl.show(screenshot=True))

    # make sure image is not empty
    assert np.max(image) != np.min(image), "Output image is constant."
    assert image.shape == image_size + (
        3,
    ), f"Incorrect images size: {image.shape} != {image_size + (3,)}"


mols_image_sizes_vis_kwargs = [
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="isosurface"),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="isosurface", plot_molecule=False),
    ),
    (
        gto.M(atom="C 0 0 0, O 0 0 1, O 0 1 0", basis="cc-pvdz"),
        (100, 100),
        dict(mode="nested_isosurfaces"),
    ),
]


@pytest.mark.skipif(
    headless or rendering_unavailable,
    reason="Cannot confirm display or VTK off-screen rendering available",
)
@pytest.mark.parametrize("mol, image_size, visualize_density_kwargs", mols_image_sizes_vis_kwargs)
def test_visualize_density(
    mol: gto.Mole,
    image_size: tuple[int, int],
    visualize_density_kwargs: dict,
    request: pytest.FixtureRequest,
):
    """Test whether visualize_density produces images."""
    if run_in_isolated_process(request):
        return

    pl = pv.Plotter(window_size=image_size, off_screen=True, notebook=False)
    visualize_density(mol, density_matrix=np.eye(mol.nao), plotter=pl, **visualize_density_kwargs)
    image = cast(np.ndarray, pl.show(screenshot=True))

    # make sure image is not empty
    assert np.max(image) != np.min(image), "Output image is constant."
    assert image.shape == image_size + (
        3,
    ), f"Incorrect images size: {image.shape} != {image_size + (3,)}"
