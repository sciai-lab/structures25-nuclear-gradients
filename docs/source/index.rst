.. MLDFT documentation master file, created by
   sphinx-quickstart on Mon Nov 13 17:40:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MLDFT's documentation!
=================================

.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation.rst

.. toctree::
   :maxdepth: 1
   :caption: Notes

   data_format.rst
   hdf5_data_format.rst
   model_overview.rst

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   about.rst

.. autosummary::
   :toctree: reference
   :template: module_template.rst
   :recursive:

   mldft.datagen
   mldft.ml
   mldft.ofdft
   mldft.utils
   mldft.utils.gpu_allocation
   mldft.launchers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

This package started as the reimplementation of [M-OFDFT]_.

References
==========

.. [M-OFDFT] Zhang, H.; Liu, S.; You, J.; Liu, C.; Zheng, S.; Lu, Z.; Wang, T.; Zheng, N.;
    Shao, B. M-OFDFT: "Overcoming the Barrier of Orbital-Free Density Functional
    Theory for Molecular Systems Using Deep Learning". arXiv September 28, 2023.
    http://arxiv.org/abs/2309.16578.

.. [Graphormer] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di
    He, Yanming Shen, Tie-Yan Liu: "Do Transformers really perform badly for graph
    representation?". Advances in Neural Information Processing Systems, 34:28877–28888, 2021.

.. [STRUCTURES25] Roman Remme, Tobias Kaczun, Tim Ebert, Christof A. Gehrig, Dominik Geng,
    Gerrit Gerhartz, Marc K. Ickler, Manuel V. Klockow, Peter Lippmann, Johannes S. Schmidt, Simon Wagner,
    Andreas Dreuw, and Fred A. Hamprecht: "Stable and Accurate Orbital-Free Density Functional Theory
    Powered by Machine Learning". Journal of the American Chemical Society 2025 147 (32), 28851-28859.
    DOI: 10.1021/jacs.5c06219

.. [EquiformerV2] Yi-Lun Liao, Brandon Wood, Abhishek Das, and Tess Smidt:
      "EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations". arXiv
      March 6, 2024.
      https://arxiv.org/abs/2306.12059

.. [TensorNet] Guillem Simeon and Gianni de Fabritiis:
      "TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials". arXiv
      October 30, 2023.
      https://arxiv.org/abs/2306.06482

.. [Tensorframes] Peter Lippmann, Gerrit Gerhartz, Roman Remme and Fred A. Hamprecht:
      "Beyond Canonicalization: How Tensorial Messages Improve Equivariant Message Passing". arXiv
      March 5, 2025.
      https://arxiv.org/abs/2405.15389


.. note::

   Please note that equation numbering in the arXiv HTML version may not match that of the
   corresponding PDF. To ensure consistency, all equation references in the docstrings and
   documentation refer to the numbering used in the **HTML version** of the papers.

..
    ADIIS is no longer supported which is why it is commented out.
    [ADIIS] Hu, Xiangqian, and Weitao Yang: "Accelerating self-consistent field convergence with
    the augmented Roothaan–Hall energy function". The Journal of chemical physics 132.5 (2010).
