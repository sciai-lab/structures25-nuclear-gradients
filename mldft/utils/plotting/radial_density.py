import numpy as np
from matplotlib import pyplot as plt
from pyscf import gto
from pyscf.data.elements import ELEMENTS

from mldft.utils.grids import get_radial_densities


def plot_radial_densities(coeffs_dict: dict, mol: gto.Mole, center_atom_idx: int = 0):
    """Plot the radial densities of a set of basis functions around a given atom.

    Atoms with charge > 1 are marked with the element symbol, while hydrogen atoms are marked with
    a half-transparent line and no element symbol.

    Args:
        coeffs_dict: dictionary mapping labels to coefficients of the (untransformed!) basis
            functions
        mol: the Molecule object in the density basis
        center_atom_idx: index of the atom around which to plot the radial densities
    """

    center = mol.atom_coords()[center_atom_idx]
    heavy_atom_mask = mol.atom_charges() > 1
    hydrogen_mask = mol.atom_charges() == 1
    distances = np.linalg.norm(mol.atom_coords() - center, axis=1)
    atom_charges = mol.atom_charges()
    radii = np.linspace(0, 1.2 * max(distances), 500)

    # biggest spherical grid is needed for decent accuracy around nuclei
    n_spherical_points = 5810
    max_density = 0

    radial_densities = get_radial_densities(
        mol, center, np.array(list(coeffs_dict.values())), radii, n_spherical_points
    )

    for label, radial_density in zip(coeffs_dict, radial_densities):
        max_density = max(max_density, radial_density.max())
        plt.plot(radii, radial_density, label=label, lw=1)

    plot_kwargs = {"color": "black", "lw": 0.5, "zorder": -2, "ymin": 0, "ymax": max_density}
    plt.vlines(distances[heavy_atom_mask], **plot_kwargs)
    plt.vlines(distances[hydrogen_mask], **plot_kwargs, alpha=0.3)

    for distance, charge in zip(distances[heavy_atom_mask], atom_charges[heavy_atom_mask]):
        plt.text(distance, -0.7, ELEMENTS[charge], horizontalalignment="center", zorder=-1)

    plt.legend()
    plt.ylabel("radial density [Bohr$^{-1}$]")
    plt.xlabel("distance from center atom [Bohr]")
    plt.ylim(-1)


if __name__ == "__main__":
    from mldft.utils.grids import LEBEDEV_NGRID
    from mldft.utils.molecules import build_mol_with_even_tempered_basis

    mol = gto.M(atom="O 0 0 -1; C 0 0 0; O 0 0 1")
    mol = build_mol_with_even_tempered_basis(mol)
    coeffs = np.random.randn(mol.nao)

    plt.figure(figsize=(10, 6))

    radii = np.linspace(0, 5, 500)

    cmap = plt.get_cmap("rainbow")
    for n_points in LEBEDEV_NGRID:
        radial_densities = get_radial_densities(mol, mol.atom_coords()[0], coeffs, radii, n_points)
        radial_density = radial_densities
        plt.plot(
            radii,
            radial_density,
            color=cmap(-np.log(n_points / LEBEDEV_NGRID[-1])),
            label=n_points,
        )

    plt.legend(ncols=3, title="n_points")
    plt.show()
