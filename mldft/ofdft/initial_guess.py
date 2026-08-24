"""Module for generating initial guesses for the electronic structure
(see [M-OFDFT]_, Section `4.4 <https://arxiv.org/html/2309.16578v2#S4.SS4>`_ and
Supplementary Section `B.5.2 <https://arxiv.org/html/2309.16578v2#alg3>`_)."""

from typing import TYPE_CHECKING, Callable, Literal, Optional

import numpy as np
import torch
import torch.nn
from loguru import logger
from pyscf import gto
from pyscf.scf.hf import init_guess_by_huckel, init_guess_by_minao

from mldft.datagen.methods.density_fitting import density_fitting_mol
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft import OFDFT_DTYPE

if TYPE_CHECKING:
    from mldft.ml.models.mldft_module import MLDFTLitModule

from mldft.utils.molecules import build_molecule_np
from mldft.utils.sad_guesser import SADGuesser


def initial_guess_from_last_simple(sample: OFData, coeffs: torch.Tensor) -> torch.Tensor:
    """Approximates the initial guess from the solution of the last step.

    Args:
        sample: OFData object containing the ground state coeffs.
        coeffs: Density coefficients in the AO basis

    Returns:
        Initial guess for the coefficients in the transformed density basis.
    """
    transformed_coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
    return transformed_coeffs


def initial_guess_from_last_safe(sample: OFData, coeffs: torch.Tensor) -> torch.Tensor:
    """Approximates the initial guess from the solution of the last step, only uses the s type
    basis functions.

    Args:
        sample: OFData object containing the ground state coeffs.
        coeffs: Density coefficients in the AO basis

    Returns:
        Initial guess for the coefficients in the transformed density basis.
    """
    mol = sample.mol
    mask = torch.zeros(mol.nao, dtype=torch.bool)
    for id in range(mol.natm):
        s_start = mol.ao_loc[mol.search_shell_id(id, 0)]
        s_end = mol.ao_loc[mol.search_shell_id(id, 1)]
        mask[s_start:s_end] = True
    masked_coeffs = torch.zeros_like(coeffs)
    masked_coeffs[mask] = coeffs[mask]
    transformed_coeffs = transform_tensor_with_sample(sample, masked_coeffs, Representation.VECTOR)
    return transformed_coeffs


def label_guess(sample: OFData):
    """Return the ground truth label of the ground state as the initial guess.

    Args:
        sample: OFData object containing the ground state coeffs.

    Returns:
        Tensor: Ground state coefficients.
    """
    # clone() is necessary to avoid in-place operations on the ground state coefficients
    return sample.ground_state_coeffs.clone()


def perturbed_label_guess(sample: OFData, scale: float):
    """Return the perturbed ground truth label of the ground state as the initial guess.

    Args:
        sample: OFData object containing the ground state coeffs.
        scale: Perturbation to apply to the ground state coefficients.

    Returns:
        Tensor: Ground state coefficients.
    """
    coeffs = sample.ground_state_coeffs
    guess = coeffs + scale * torch.randn_like(coeffs)
    guess *= sample.n_electron / (sample.basis_integrals @ guess)

    # trajectories are saved via mol id, so we need to change it to not overwrite the original
    sample.mol_id = sample.mol_id + "_" + str(hash(guess))

    return guess


def hueckel_guess_np(mol: gto.Mole, ks_basis: str = "6-31G(2df,p)") -> np.ndarray:
    """Return the Hückel initial guess.

        The Hückel initial guess is performed for the density matrix in the
        orbital basis. This is then transformed into the density basis using density
        fitting.

    Args:
        mol: Molecule object in the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the Hückel guess is performed.
    """
    mol_orbital = build_molecule_np(
        mol.atom_charges(), mol.atom_coords(), basis=ks_basis, mol_charge=mol.charge
    )
    dm = init_guess_by_huckel(mol_orbital)
    # density fitting into density basis
    coeffs = density_fitting_mol(dm, mol_orbital, mol)
    return coeffs


def hueckel_guess(
    sample: OFData,
    mol: gto.Mole,
    ks_basis: str = "6-31G(2df,p)",
    transform: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the Hückel initial guess.

    The Hückel initial guess is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting. Finally, the coefficients are rescaled to ensure the correct number
    of electrons.

    Args:
        sample: OFData object containing transformed basis_integrals and coeffs.
        mol: Molecule object containing information on the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the Hückel guess is performed.
        normalize: Whether to rescale the coefficients to ensure the correct number of electrons.

    Returns:
        Initial guess of the coefficients in the transformed density basis.
    """
    coeffs = hueckel_guess_np(mol, ks_basis)
    coeffs = torch.as_tensor(coeffs, dtype=OFDFT_DTYPE, device=sample.coeffs.device)
    if transform:
        coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
    if normalize:  # rescale to correct number of electrons
        n_guess = sample.basis_integrals @ coeffs
        coeffs *= sample.n_electron / n_guess
    return coeffs


def minao_guess_np(mol: gto.Mole, ks_basis: str = "6-31G(2df,p)") -> np.ndarray:
    """Return the MINAO initial guess.

    The MINAO initial guess is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting.

    Args:
        mol: Molecule object containing information on the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the MINAO guess is performed.
    """
    mol_orbital = build_molecule_np(
        mol.atom_charges(), mol.atom_coords(), basis=ks_basis, mol_charge=mol.charge
    )
    dm = init_guess_by_minao(mol_orbital)
    # density fitting into density basis
    coeffs = density_fitting_mol(dm, mol_orbital, mol)
    return coeffs


def minao_guess(
    sample: OFData,
    mol: gto.Mole,
    ks_basis: str = "6-31G(2df,p)",
    transform: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the optionally transformed and normalized MINAO initial guess.

    The MINAO initial guess is performed for the density matrix in the
    orbital basis. It is transformed into the density basis using density
    fitting. Finally, the coefficients can optionally be transformed according to the samples

    Args:
        sample: OFData object containing transformed basis_integrals and coeffs
        mol: Molecule object containing information on the density (OF) basis.
        ks_basis: Orbital (KS) basis in which the MINAO guess is performed.
        transform: Whether to transform the coefficients into the density basis.
        normalize: Whether to rescale the coefficients to ensure the correct number of electrons.

    Returns:
        Initial guess of the coefficients in the transformed density basis.
    """
    coeffs = minao_guess_np(mol, ks_basis)
    coeffs = torch.as_tensor(coeffs, dtype=OFDFT_DTYPE, device=sample.coeffs.device)
    if transform:
        coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
    if normalize:  # rescale to correct number of electrons
        n_guess = sample.basis_integrals @ coeffs
        coeffs *= sample.n_electron / n_guess
    return coeffs


def proj_minao_guess(
    sample: OFData,
    mol: gto.Mole,
    module: torch.nn.Module,
    ks_basis: str = "6-31G(2df,p)",
    normalize: bool = True,
) -> torch.Tensor:
    """Return the "projected" MINAO initial guess.

    First, the `py:meth:minao_guess` is performed for the density matrix in the
    orbital basis. This is then transformed into the density basis using density
    fitting. Finally, the coefficients are rescaled to ensure the correct number
    of electrons.

    Args:
        sample: OFData object containing transformed basis_integrals and coeffs.
        mol: Molecule object containing information on the density (OF) basis.
        module: Module used to predict the correction to / projection of the initial guess.
        ks_basis: Orbital (KS) basis in which the MINAO guess is performed.
        normalize: Whether to rescale the coefficients to ensure the correct number of electrons.

    Returns:
        Initial guess of the coefficients in the transformed density basis.
    """
    # Transformation already happens in minao_guess
    coeffs_minao = minao_guess(sample, mol, ks_basis)
    sample_for_proj = to_torch(sample.detach().clone(), float_dtype=module.dtype)
    sample_for_proj.coeffs = coeffs_minao.type(module.dtype)
    coeffs_delta = module(sample_for_proj)[2]
    coeffs_delta = coeffs_delta.type(torch.float64)
    coeffs = coeffs_minao - coeffs_delta
    if normalize:
        n_guess = sample.basis_integrals @ coeffs
        coeffs *= sample.n_electron / n_guess
    # check_rho_on_grid(mol, coeffs)
    return coeffs


def hartree_external_guess(sample: OFData, mol: gto.Mole) -> torch.Tensor:
    r"""Return the minimum of the Hartree and external energy with correct number of electrons.

    Minimizes the sum of the Hartree and external energy :math:`E = E_\text{H} + E_\text{ext}`
    under the constraint :math:`w^T p = N`, where :math:`w` are the dual basis integrals. This
    leads to

    .. math::
        p = C^{-1} \left( \frac{N + w^T C^{-1} v}{w^T C^{-1} w} w - v \right)

    where :math:`C` is the Coulomb matrix and :math:`v` is the nuclear attraction vector.

    Args:
        sample: OFData object containing the coefficients in the density basis.
        mol: Molecule object containing information on the density (OF) basis.

    Returns:
        Tensor: Initial guess for the coefficients in the density basis.
    """
    inverse_coulomb = sample.coulomb_matrix.inverse()
    w = sample.basis_integrals
    numerator = mol.nelectron + w @ inverse_coulomb @ sample.nuclear_attraction_vector
    lambd = numerator / (w @ inverse_coulomb @ w)
    coeffs = inverse_coulomb @ (lambd * w - sample.nuclear_attraction_vector)
    return coeffs


def dumb_guess(sample: OFData, mol: gto.Mole, transform: bool = True) -> torch.Tensor:
    """Generate a dumb initial guess for the electronic structure calculation.

    The dumb guess sets the coefficients of the atomic orbitals (AOs) to 1 for
    AOs with a positive normalization constant (norm > 0). The coefficients are
    then scaled to ensure the correct number of electrons. This currently only
    works for the density basis.

    Args:
        sample: OFData object containing the basis_integrals.
        mol: Molecule object containing information about the atomic
            structure and basis set.

    Returns:
        np.ndarray: Initial guess for the coefficients of the atomic orbitals.
    """

    coeffs = torch.zeros(mol.nao, dtype=torch.float64, device=sample.coeffs.device)
    l_per_basis_func = torch.tensor(
        sample.basis_info.l_per_basis_func, device=sample.coeffs.device
    )
    coeffs[l_per_basis_func[sample.basis_function_ind] == 0] = 1.0
    if transform:
        if hasattr(sample, "transformation_matrix"):
            coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
        else:
            logger.warning(
                "Sample does not have a transformation matrix, returning coefficients in the AO basis."
            )
    n_guess = sample.basis_integrals @ coeffs
    coeffs *= mol.nelectron / n_guess

    return coeffs


def one_atom_guess(
    sample: OFData,
    atom_id: int,
    mol: gto.Mole,
) -> torch.Tensor:
    """Generate a bad initial guess for testing stability of the method. All density is placed on
    one atom.

    Args:
        sample: OFData object
        atom_id: The ID of the atom to place all density on.

    Returns:
        np.ndarray: Initial guess for the coefficients of the atomic orbitals.
    """
    coeffs = dumb_guess(sample, mol, transform=False)
    coeffs[sample.coeff_ind_to_node_ind != atom_id] = 0.0
    coeffs = transform_tensor_with_sample(sample, coeffs, Representation.VECTOR)
    n_guess = sample.basis_integrals @ coeffs
    coeffs *= mol.nelectron / n_guess
    return coeffs


def initial_guess(
    sample: OFData,
    mol: gto.Mole,
    initial_guess_str: str,
    normalize_initial_guess: bool = True,
    ks_basis: str = "6-31G(2df,p)",
    proj_minao_module: Optional["MLDFTLitModule"] = None,
    sad_guess_kwargs: dict | None = None,
) -> torch.Tensor:
    """Return the initial guess for the coefficients
    (see [M-OFDFT]_, Section `4.4 <https://arxiv.org/html/2309.16578v2#S4.SS4>`_ and
    Supplementary Section `B.5.2 <https://arxiv.org/html/2309.16578v2#alg3>`_).

    Args:
        sample: The OFData object.
        mol: The molecule.
        initial_guess_str: The type of initial guess to use. Available options are:

            - ``dumb``: Dumb guess (all s orbitals set to 1).
            - ``label``: Use the ground truth label as initial guess.
            - ``perturbed_label_<scale>``: Use the ground truth label plus Gaussian noise
              with standard deviation ``<scale>``.
            - ``hueckel``: Hückel initial guess.
            - ``minao``: MINAO initial guess.
            - ``proj_minao``: Projected MINAO initial guess using a provided module.
            - ``sad``: Superposition of atomic densities (SAD) initial guess using dataset
              statistics.
            - ``sad_default``: SAD initial guess using default QM9 statistics.
            - ``sad_transformed``: SAD initial guess computed in the (potentially transformed)
              basis of the sample.
            - ``one_atom_<atom_id>``: Place all density on one atom with ID ``<atom_id>``.
            - ``hartree_external``: Initial guess minimizing the sum of Hartree and external
              energy.

        normalize_initial_guess: Whether to normalize the initial guess to the correct number of
            electrons.
        ks_basis: The basis set to use for orbital-based initial guesses (hueckel, minao, proj_minao).
        proj_minao_module: The module to use for the proj_minao initial guess.
        sad_guess_kwargs: Additional keyword arguments for the SAD guesser.

    Returns:
        torch.Tensor: The initial guess for the coefficients.

    Raises:
        KeyError: If the initial guess is not in INITIAL_GUESS (e.g. implemented guess).
    """
    if initial_guess_str == "dumb":
        return dumb_guess(sample, mol)
    elif initial_guess_str == "label":
        return label_guess(sample)
    elif initial_guess_str.startswith("perturbed_label"):
        scale = float(initial_guess_str.split("_")[-1])
        return perturbed_label_guess(sample, scale)
    elif initial_guess_str == "hueckel":
        return hueckel_guess(sample, mol, ks_basis, normalize=normalize_initial_guess)
    elif initial_guess_str == "minao":
        return minao_guess(sample, mol, ks_basis, normalize=normalize_initial_guess)
    elif initial_guess_str == "proj_minao":
        assert proj_minao_module is not None, "A lightning Module must be provided for proj_minao."
        return proj_minao_guess(
            sample,
            mol,
            proj_minao_module,
            ks_basis,
            normalize=normalize_initial_guess,
        )
    elif initial_guess_str == "sad" or initial_guess_str == "sad_default":
        if initial_guess_str == "sad":
            guesser = SADGuesser.from_dataset_statistics(**sad_guess_kwargs, mol=sample)
        else:
            guesser = SADGuesser.from_default_qm9_stats()
        guesser = guesser.to(device=sample.coeffs.device)

        # this is a bit of a hack, but we need the untransformed basis integrals, and this is one way to get them
        untransformed_sample = sample.clone()
        untransformed_sample.basis_integrals = transform_tensor_with_sample(
            untransformed_sample,
            untransformed_sample.basis_integrals,
            Representation.DUAL_VECTOR,
            invert=True,
        )
        coeffs_guess = guesser(untransformed_sample)
        coeffs_guess = transform_tensor_with_sample(sample, coeffs_guess, Representation.VECTOR)
        return coeffs_guess
    elif initial_guess_str == "sad_transformed":
        # compute the SAD guess in the (potentially transformed) basis of the sample
        # this seems to work well, except for global natural reparametrization
        guesser = SADGuesser.from_dataset_statistics(**sad_guess_kwargs, mol=sample)
        guesser = guesser.to(device=sample.coeffs.device)
        return guesser(sample)
    elif initial_guess_str.startswith("one_atom_"):
        atom_id = int(initial_guess_str.split("_")[-1])
        return one_atom_guess(sample, int(atom_id), mol)
    elif initial_guess_str == "hartree_external":
        return hartree_external_guess(sample, mol)
    else:
        raise KeyError(f"Initial guess {initial_guess_str} not recognized.")


def get_initial_guess_estimation(name: Literal["simple", "safe"] | None) -> Callable | None:
    """Get the initial guess estimation function based on the provided name.

    These functions provide an initial guess from the last density optimization in methods that
    perform several of these optimizations for the same molecule (with the same number of electrons),
    e.g. in geometry optimizations.

    Args:
        name: The name of the initial guess estimation method. Can be "simple" or "safe".

    Returns:
        Callable: The initial guess estimation function.
    """
    if name is not None:
        if name == "simple":
            initial_guess_estimation = initial_guess_from_last_simple
        elif name == "safe":
            initial_guess_estimation = initial_guess_from_last_safe
        else:
            raise ValueError(f"Unknown initial guess from last method {name}")
    else:
        initial_guess_estimation = None

    return initial_guess_estimation
