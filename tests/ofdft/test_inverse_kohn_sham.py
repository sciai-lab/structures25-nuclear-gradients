import torch
from pyscf import scf

from mldft.ofdft.basis_integrals import get_overlap_matrix
from mldft.ofdft.inverse_kohn_sham import single_ks_step


def test_single_ks_step(molecule_small):
    """Test the single Kohn-Sham step using different methods and against pyscf."""
    mol = molecule_small
    overlap_orbital = torch.as_tensor(get_overlap_matrix(mol), dtype=torch.float64)
    fock = torch.rand_like(overlap_orbital, dtype=torch.float64)
    fock = (fock + fock.T) / 2

    gamma_scipy = single_ks_step(fock, overlap_orbital, mol.nelectron, "scipy")
    gamma_eigh = single_ks_step(fock, overlap_orbital, mol.nelectron, "eigh")
    torch.testing.assert_close(gamma_scipy, gamma_eigh)

    # lobpcg is too inaccurate and sometimes fails to converge
    # gamma_lobpcg = single_ks_step(fock, overlap_orbital, mol.nelectron, "lobpcg")
    # torch.testing.assert_close(gamma_scipy, gamma_lobpcg)

    mf = scf.HF(mol)
    mf.get_fock = lambda *args, **kwargs: fock
    mf.run(max_cycle=0, verbose=0)
    gamma_mf = torch.as_tensor(mf.make_rdm1(), dtype=torch.float64)
    torch.testing.assert_close(gamma_eigh, gamma_mf)
