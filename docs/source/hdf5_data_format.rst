HDF5 Label Format and Workflow
=================================

This document describes:

1. How the new HDF5 label format is structured internally.
2. How to convert a normal dataset or a merged dataset.
3. How to use the format for training.

1. Internal format
------------------

Each HDF5 shard (``*_shard_XXXXX.h5``) stores molecule-static data once and sample-level data separately.

Root attributes
^^^^^^^^^^^^^^^

Important root attributes:

- ``format = "mldft_h5_sharded_v2_bucketed_row"``
- ``energy_key`` (for example ``e_kin_plus_xc``)
- ``gradient_key`` (for example ``grad_kin_plus_xc``)
- ``output_dtype``
- ``sample_bucket_layout = "coeff_grad_colocated_v1"``
- ``bucket_chunk_rows = 1``
- ``n_samples``, ``n_molecules``, ``n_source_files``, ``n_buckets``

Root datasets
^^^^^^^^^^^^^

- ``source_relpaths``: one entry per molecule/source file in this shard.
- ``source_keys``: stable key per source file, format:
  ``<split_dataset_name>/<label_subdir>/<label_relpath>``

These keys are what the datamodule uses to match split entries to shard rows.

``/samples`` group
^^^^^^^^^^^^^^^^^^

Sample-level arrays (length ``n_samples``):

- ``mol_index``
- ``scf_iteration``
- ``energy``
- ``has_energy_label``
- ``bucket_id``
- ``bucket_row``

Each sample points to:

- the molecule via ``mol_index``
- the coeff/grad row via (``bucket_id``, ``bucket_row``)

``/sample_buckets`` group
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``coeff_dim``: coefficient dimension per bucket id.
- ``data/<bucket_id>/coeff_grad``: shape ``(n_rows_in_bucket, 2, coeff_dim)``

  - axis 1 index ``0`` = coeffs
  - axis 1 index ``1`` = gradients

This is the colocated coeff+grad layout.

``/molecules`` group
^^^^^^^^^^^^^^^^^^^^

Molecule-level arrays:

- metadata: ``mol_id``, ``source_key``, ``source_relpath``, ``n_atom``, ``n_basis``, ``n_channels``
- atom data: ``atom_ptr``, ``atom_pos_flat``, ``atomic_numbers_flat``
- basis data: ``basis_ptr``, ``basis_integrals_flat``
- final-SCF coeffs: ``ground_state_coeff_ptr``, ``ground_state_coeff_flat``
- optional nuclear gradients: ``nuclear_gradients/available``,
  ``nuclear_gradients/nuc_grad_label_ptr``, ``nuclear_gradients/nuc_grad_label_flat``,
  ``nuclear_gradients/transform_grad_contracted_ptr``,
  ``nuclear_gradients/transform_grad_contracted_cols``,
  ``nuclear_gradients/transform_grad_contracted_flat``

``*_ptr`` arrays are CSR-style pointers into flattened arrays.
If a matching sibling file exists in ``<dataset>/nuclear_gradients/<label_filename>``
and nuclear gradients are enabled (via
``--include-nuclear-gradients`` or
``ConversionConfig.include_nuclear_gradients``), the converter stores the
corresponding nuclear-gradient labels. The HDF5 loader exposes the same
``nuc_grad_label``/``transform_grad_contracted`` fields as the zarr loader
only when called with ``load_nuclear_gradients=True``.

2. Shard planning behavior
--------------------------

Planning is **split-aware** (``train``, ``val``, ``test``) and done independently per split.

For each source file, scanning computes:

- ``n_scf_steps``
- ``coeff_dim``
- ``n_basis``
- ``n_atom``
- estimated bytes

Then, within each split:

1. Files are globally regrouped by shape: sorted by ``(coeff_dim, estimated_bytes desc)``.
2. Files are packed into shards by estimated size budget.

Default shard target is ``3 GiB`` (``--target-shard-size-gb 3``).

Compression guidance
^^^^^^^^^^^^^^^^^^^^

Use ``--zstd-level 0`` by default.

In our benchmarks, compression was significantly slower, especially on network filesystems:
it increased per-sample load time and reduced training/loader throughput. Only use compression
for very large datasets on local disks where storage pressure dominates over I/O throughput.
Defaults are ``--num-workers 8`` and ``--scan-workers 8``, which are usually safer for NFS.

3. Convert a normal dataset
---------------------------

Assume dataset root:

- ``${DFT_DATA}/QM9_perturbed_fock``
- labels in ``${DFT_DATA}/QM9_perturbed_fock/labels``
- split file ``${DFT_DATA}/QM9_perturbed_fock/split.pkl``

Run:

.. code-block:: bash

   python scripts/convert_labels_to_hdf5_shards.py \
     --split-file ${DFT_DATA}/QM9_perturbed_fock/split.pkl \
     --label-subdir labels \
     --target-key kin_plus_xc \
     --target-shard-size-gb 3 \
     --num-workers 8 \
     --scan-workers 8 \
     --zstd-level 0

Notes:

- Output defaults to ``${split_file.parent}/${label_subdir}_h5_${target_key}``.
  Example: ``.../labels_h5_kin_plus_xc``
- Keep ``--zstd-level 0`` unless you explicitly optimize for storage size on local disks.

4. Convert and use a merged dataset
-----------------------------------

Recommended flow:

1. Create/refresh merged split file.
2. Convert that split directly into a new physical shard set.

4.1 Create merged split
^^^^^^^^^^^^^^^^^^^^^^^

For supported merged naming schemes (for example ``QMUGSBin0QM9*``):

.. code-block:: bash

   python mldft/utils/create_custom_splits.py QMUGSBin0QM9_perturbed_fock --override

4.2 Convert merged split directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The converter reads the dataset names stored in the split entries and pulls the
corresponding label files from ``${DFT_DATA}/<dataset>/<label_subdir>/``.

.. code-block:: bash

   python scripts/convert_labels_to_hdf5_shards.py \
     --split-file ${DFT_DATA}/QMUGSBin0QM9_perturbed_fock/split.pkl \
     --label-subdir labels \
     --target-key kin_plus_xc \
     --target-shard-size-gb 3 \
     --num-workers 8 \
     --scan-workers 8 \
     --zstd-level 0

This writes:

- split file: ``${DFT_DATA}/QMUGSBin0QM9_perturbed_fock/split.pkl``
- manifest: ``${DFT_DATA}/QMUGSBin0QM9_perturbed_fock/labels_h5_kin_plus_xc/manifest.json``
- shard files physically stored under the merged dataset directory

5. Use HDF5 labels for training
-------------------------------

The hdf5 labels are configured via the ``hdf5_manifest_path`` in datamodule dataset kwargs. By default,
this is set to: ``${data.dataset_name}/{label_subdir}_h5_${data.target_key}/manifest.json`` and will look
for this path under ``DFT_DATA``. The ``hdf5_manifest_path`` supports the ``{label_subdir}`` placeholder, which in code
is replaced by the datamodule's ``label_subdir``.
The ``manifest.json`` entries for shard files (``shards[*].output_file``) are written relative to
``DFT_DATA`` by the converter and merge script.
