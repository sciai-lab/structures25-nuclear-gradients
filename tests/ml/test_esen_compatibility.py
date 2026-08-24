import copy

import torch

from mldft.ml.data.components.convert_transforms import AddFullEdgeIndex
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.models.components.atom_ref import AtomRef
from mldft.ml.models.components.dimension_wise_rescaling import DimensionWiseRescaling
from mldft.ml.models.components.equiformerv2_components.density_embedding import (
    DensityEmbedding,
)
from mldft.ml.models.components.esen_components.esen import (
    eSEN,
    eSEN_Backbone,
    eSEN_Fast,
    eSEN_Fast_Backbone,
)
from mldft.utils.utils import set_default_torch_dtype


def _make_batch(dummy_sample_torch):
    sample = AddFullEdgeIndex()(copy.deepcopy(dummy_sample_torch))
    return OFBatch.from_data_list([sample], follow_batch=["coeffs", "atomic_numbers"])


def _shared_modules(dummy_basis_info, dummy_dataset_statistics, n_channels):
    return {
        "density_embedding": DensityEmbedding([2], n_channels, dummy_basis_info),
        "atom_ref_module": AtomRef.from_dataset_statistics(
            dummy_dataset_statistics,
            basis_info=dummy_basis_info,
            weigher_key="constant",
            mode="scalar_only",
        ),
        "initial_guess_delta_module": None,
        "gradient_module": None,
        "feature_wise_rescaling_module": DimensionWiseRescaling.from_dataset_statistics(
            dummy_dataset_statistics,
            basis_info=dummy_basis_info,
            weigher_key="constant",
            equivariant=True,
            s_coeff=50,
            s_grad=0.05,
            epsilon=1e-8,
        ),
    }


@set_default_torch_dtype(torch.float64)
def test_esen_matches_esen_fast_with_shared_weights(
    dummy_basis_info,
    dummy_dataset_statistics,
    dummy_sample_torch,
):
    torch.manual_seed(0)

    n_channels = 8
    backbone_kwargs = {
        "max_num_elements": 10,
        "sphere_channels": n_channels,
        "lmax": 2,
        "mmax": 2,
        "otf_graph": True,
        "max_neighbors": 32,
        "use_pbc": False,
        "use_pbc_single": False,
        "cutoff": 6.0,
        "edge_channels": 8,
        "distance_function": "gaussian",
        "num_distance_basis": 32,
        "regress_forces": False,
        "regress_stress": False,
        "num_layers": 2,
        "hidden_channels": 8,
        "norm_type": "rms_norm_sh",
        "act_type": "gate",
        "mlp_type": "spectral",
        "use_envelope": True,
        "activation_checkpointing": False,
        "deterministic": True,
    }
    modules = _shared_modules(dummy_basis_info, dummy_dataset_statistics, n_channels)

    model = eSEN(
        backbone=eSEN_Backbone(use_quaternion_wigner=True, **backbone_kwargs),
        **modules,
    )
    fast_model = eSEN_Fast(
        backbone=eSEN_Fast_Backbone(**backbone_kwargs),
        **modules,
    )

    assert set(model.state_dict()) == set(fast_model.state_dict())
    fast_model.load_state_dict(model.state_dict())

    batch = _make_batch(dummy_sample_torch)
    model.eval()
    fast_model.eval()
    with torch.no_grad():
        model_output = model(batch)
        fast_output = fast_model(batch)

    assert torch.allclose(model_output.pred_energy, fast_output.pred_energy)
    assert torch.allclose(model_output.pred_diff, fast_output.pred_diff)
