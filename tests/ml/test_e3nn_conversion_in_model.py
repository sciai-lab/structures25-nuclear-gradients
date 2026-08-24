import copy

import torch
import torch.testing

from mldft.ml.data.components.basis_transforms import ToE3NNConvention
from mldft.ml.data.components.convert_transforms import to_torch
from mldft.ml.models.components.equiformerv2_components.density_embedding import (
    DensityEmbedding,
)


@torch.no_grad()
def test_density_embedding_pyscf_to_e3nn_equivalence(dummy_sample, dummy_basis_info):
    """The pyscf->e3nn convention conversion done inside DensityEmbedding must produce the same
    output as using the ToE3NNConvention transform and disabling the in-model rotation."""
    density_embedding_lmax_list = [2]
    n_channels = 4

    # Use float64 throughout: ToE3NNConvention builds float64 Wigner matrices and the
    # matmul fails if the sample tensors are float32.
    sample_raw = to_torch(dummy_sample, float_dtype=torch.float64)

    # Model A: in-model convention conversion
    model_new = DensityEmbedding(
        density_embedding_lmax_list=density_embedding_lmax_list,
        n_channels=n_channels,
        basis_info=dummy_basis_info,
        apply_pyscf_to_e3nn_conversion=True,
    ).double()

    # Model B: same weights, but convention conversion is disabled
    model_old = DensityEmbedding(
        density_embedding_lmax_list=density_embedding_lmax_list,
        n_channels=n_channels,
        basis_info=dummy_basis_info,
        apply_pyscf_to_e3nn_conversion=False,
    ).double()
    model_old.load_state_dict(model_new.state_dict(), strict=False)

    # Pre-convert the sample with the transform
    sample_converted = copy.deepcopy(sample_raw)
    to_e3nn = ToE3NNConvention(dummy_basis_info)
    sample_converted = to_e3nn(sample_converted)

    output_new = model_new(sample_raw)
    output_old = model_old(sample_converted)

    torch.testing.assert_close(output_new, output_old)
