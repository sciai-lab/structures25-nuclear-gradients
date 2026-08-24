Installation
============


Install using Conda/Mamba/Micromamba

For conda or mamba replace `micromamba` with `conda` or `mamba` below.

.. code-block:: bash

   # create mamba environment with GPU [or only CPU] support
   micromamba env create -f environment[_cpu].yaml -n mldft
   # Or
   micromamba env create -f environment_cpu.yaml -n mldft
   # activate environment
   micromamba activate mldft
   # (Optional but will become mandatory) Install as an editable package
   pip install -e .
