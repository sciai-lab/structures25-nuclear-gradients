Data Format
============

We store the data for each molecule in a `.zarr` file
(A molecule here refers to a certain set of atoms, and their positions in space).

The ``.zarr`` file contains the following arrays:

.. code-block:: yaml

    geometry:
        mol_id: str
        atom_pos: float array of shape (n_atoms, 3)
        atomic_number: int array of shape (n_atoms)
    of_labels:
        basis: str, relative path to basis info file
        n_scf_steps: int, number of scf steps
        energies:
            kinetic: float array of shape (n_scf_steps)
            exchange_correlation: float array of shape (n_scf_steps)
            hartree: float array of shape (n_scf_steps)
            nucelar_electron_attraction: float array of shape (n_scf_steps)
            nuclear_nuclear_repulsion: float (since it is the same for all scf steps)
        spatial:
            coeffs: float array of shape (n_scf_steps, n_basis)
            kinetic_gradient: float array of shape (n_scf_steps, n_basis)
            xc_gradient: float array of shape (n_scf_steps, n_basis)
    ks_labels:
        basis: string, basis used in KS part of data generation
        energies:
            kinetic: float array of shape (n_scf_steps)
            exchange_correlation: float array of shape (n_scf_steps)
            hartree: float array of shape (n_scf_steps)
            nucelar_electron_attraction: float array of shape (n_scf_steps)
            nuclear_nuclear_repulsion: float, since it is the same for all scf steps

.. currentmodule:: mldft.ml.data.components

``.zarr`` files with these structure can be read using :py:meth:`of_data.OFData.from_file`.

The basis info file should be a text file in nwchem format.
The function to read the basis info file is implemented at :py:func:`basis_info.BasisInfo.from_nwchem`.

For details on the structure of ``Data`` and ``Batch`` objects used in the models,
see :py:mod:`~of_data.OFData`.
