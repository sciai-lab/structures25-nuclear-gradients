import numpy as np
import pytest
import torch

from mldft.ml.data.components.convert_transforms import AddCliqueNodes, ToTorch
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.components.virtual_nodes import VirtualFeatureAdder


@pytest.fixture
def sample_pair(dummy_basis_info) -> tuple[OFData, OFData]:
    n_basis = 28
    d_edge_attr = 12

    # Graph 1: 6 nodes
    #   Clique A: {0,1,2} (triangle)
    #   Clique B: {2,3}   (overlaps A via node 2)
    #   Clique C: {4,5}   (disjoint from A and B)
    # 3 cliques in total, 1 disjoint
    num_nodes1 = 6
    edge_index1 = torch.tensor(
        [
            [0, 1, 2, 1, 2, 0, 2, 3, 3, 2, 4, 5],
            [1, 2, 0, 0, 1, 2, 3, 2, 2, 3, 5, 4],
        ]
    )

    # Graph 2: 4 nodes
    #   Clique D: {0,1}   (edge)
    #   Clique E: {1,2,3} (triangle, overlaps D via node 1)
    # 2 cliques in total, overlap
    num_nodes2 = 4
    edge_index2 = torch.tensor(
        [
            [0, 1, 1, 2, 3, 2, 3, 1],
            [1, 0, 2, 1, 2, 3, 1, 3],
        ]
    )

    pos1 = torch.randn((num_nodes1, 3))
    pos2 = torch.randn((num_nodes2, 3))

    atomic_numbers1 = np.random.choice(dummy_basis_info.atomic_numbers, size=num_nodes1)
    atomic_numbers2 = np.random.choice(dummy_basis_info.atomic_numbers, size=num_nodes2)

    coeffs1 = torch.randn(n_basis)
    coeffs2 = torch.randn(n_basis)
    basis_integrals1 = torch.randn(n_basis)
    basis_integrals2 = torch.randn(n_basis)

    sample1 = OFData.construct_new(
        basis_info=dummy_basis_info,
        pos=pos1,
        atomic_numbers=atomic_numbers1,
        coeffs=coeffs1,
        basis_integrals=basis_integrals1,
        num_nodes=num_nodes1,
    )
    sample2 = OFData.construct_new(
        basis_info=dummy_basis_info,
        pos=pos2,
        atomic_numbers=atomic_numbers2,
        coeffs=coeffs2,
        basis_integrals=basis_integrals2,
        num_nodes=num_nodes2,
    )

    sample1.edge_index = edge_index1
    sample2.edge_index = edge_index2

    edge_attr1 = torch.randn((edge_index1.size(1), d_edge_attr))
    edge_attr2 = torch.randn((edge_index2.size(1), d_edge_attr))
    sample1.edge_attr = edge_attr1
    sample2.edge_attr = edge_attr2

    to_t = ToTorch()
    sample1, sample2 = to_t(sample1), to_t(sample2)

    return sample1, sample2


# Graph 1: 3 cliques ({0,1,2}, {2,3}, {4,5})
# Graph 2: 2 cliques ({0,1}, {1,2,3})
# (n_min, n_max) -> (expected_graph_cliques_1, expected_graph_cliques_2)
EXPECTED_CLIQUE_COUNTS = {
    (1, 1): (1, 1),
    (1, 2): (2, 1),
    (1, 3): (2, 1),
    (1, 4): (2, 1),
    (2, 2): (2, 2),
    (2, 3): (2, 2),
    (3, 3): (3, 2),
    (4, 6): (3, 2),
}


@pytest.mark.parametrize(
    "n_min_cliques,n_max_cliques",
    list(EXPECTED_CLIQUE_COUNTS.keys()),
)
def test_number_of_cliques(sample_pair, n_min_cliques, n_max_cliques):
    """Test clique number selection behaves as expected."""
    s1, s2 = sample_pair
    n_expected_1, n_expected_2 = EXPECTED_CLIQUE_COUNTS[(n_min_cliques, n_max_cliques)]

    cl_transform = AddCliqueNodes(n_min_cliques, n_max_cliques)
    s1_t = cl_transform(s1.clone())
    s2_t = cl_transform(s2.clone())

    def _count_virtual_nodes(sample: OFData) -> int:
        return sample.num_nodes_mixed - sample.num_nodes

    assert _count_virtual_nodes(s1_t) == n_expected_1
    assert _count_virtual_nodes(s2_t) == n_expected_2


@pytest.mark.parametrize(
    "n_min_cliques, n_max_cliques",
    [(1, 1), (2, 3)],
)
def test_clique_node_transform(sample_pair, n_min_cliques, n_max_cliques):
    """Assert that real graph remains invariant, and batching of transformed samples works as
    expected."""

    s1, s2 = sample_pair
    ofbatch = OFBatch.from_data_list([s1, s2], follow_batch=["coeffs"])
    cl_transform = AddCliqueNodes(n_min_cliques, n_max_cliques)
    s1_t = cl_transform(s1.clone())
    s2_t = cl_transform(s2.clone())
    ofbatch_t = OFBatch.from_data_list([s1_t, s2_t], follow_batch=["coeffs"])

    # real graph should remain invariant
    for key in {
        "edge_index",
        "batch",
        "num_nodes",
        "coeffs",
        "atomic_numbers",
    }:
        want = getattr(ofbatch, key)
        have = getattr(ofbatch_t, key)
        assert torch.allclose(
            torch.as_tensor(want), torch.as_tensor(have)
        ), f"Batch {key} changed after clique node transform"
        for s, s_t in zip(ofbatch.to_data_list(), ofbatch_t.to_data_list()):
            if key == "batch":
                continue
            assert torch.allclose(
                torch.as_tensor(getattr(s, key)),
                torch.as_tensor(getattr(s_t, key)),
            ), f"Sample {key} changed after clique node transform"

    want_batch_mixed = torch.cat(
        [
            torch.zeros(
                s1_t.num_nodes_mixed,
                dtype=ofbatch_t.batch_mixed.dtype,
            ),
            torch.ones(
                s2_t.num_nodes_mixed,
                dtype=ofbatch_t.batch_mixed.dtype,
            ),
        ],
        dim=0,
    ).to(ofbatch_t.batch_mixed.device)
    assert torch.equal(want_batch_mixed, ofbatch_t.batch_mixed)

    # virtual nodes only in mixed edge_index
    for s_t in ofbatch_t.to_data_list():
        v_nodes = torch.arange(s_t.num_nodes, s_t.num_nodes_mixed)

        assert (
            s_t.is_virtual_node[: s_t.num_nodes].sum().item() == 0
        ), "Real nodes must not be marked virtual."
        assert torch.all(
            s_t.is_virtual_node[s_t.num_nodes :]
        ), "All appended nodes must be marked virtual."

        assert not torch.isin(v_nodes, s_t.edge_index).any(), "Virtual indices in real edge_index."
        assert torch.isin(
            v_nodes, s_t.edge_index_mixed
        ).any(), "Virtual indices missing from edge_index_mixed."

    assert torch.equal(
        ofbatch_t.edge_index_mixed,
        torch.cat(
            [
                ofbatch_t[0].edge_index_mixed,
                ofbatch_t[1].edge_index_mixed + ofbatch_t[0].num_nodes_mixed,
            ],
            dim=1,
        ),
    ), "edge_index_mixed batch increment is wrong."


@pytest.mark.parametrize("node_scale", [-0.5, 1.0])
@pytest.mark.parametrize("edge_scale", [-0.3, 0.0])
@pytest.mark.parametrize(
    "n_min_cliques, n_max_cliques",
    [(1, 1), (2, 3)],
)
def test_virtual_feature_adder(sample_pair, node_scale, edge_scale, n_min_cliques, n_max_cliques):
    s1, s2 = sample_pair
    n_real_nodes = s1.num_nodes + s2.num_nodes

    cl_transform = AddCliqueNodes(n_min_cliques, n_max_cliques)
    s1_t = cl_transform(s1.clone())
    s2_t = cl_transform(s2.clone())
    batch = OFBatch.from_data_list([s1_t, s2_t], follow_batch=["coeffs"])

    n_mixed_nodes = batch.num_nodes_mixed
    n_mixed_edges = batch.edge_index_mixed.size(1)

    feature_dim = torch.randint(low=1, high=64, size=(1,)).item()
    x_real = torch.randn((n_real_nodes, feature_dim))  # we simply use coeffs as x
    edge_attr_real = batch.edge_attr.clone()
    length_real = torch.rand((batch.edge_attr.size(0), 1)).clone()

    adder = VirtualFeatureAdder(node_scale=node_scale, edge_attr_scale=edge_scale)
    x_mixed, edge_attr_mixed, length_mixed = adder(
        x=x_real,
        edge_attr=batch.edge_attr,
        length=length_real,
        is_virtual_node=batch.is_virtual_node,
        is_virtual_edge=batch.is_virtual_edge,
    )

    assert n_real_nodes == (batch.is_virtual_node == 0).sum()
    assert x_mixed.shape == (n_mixed_nodes, x_real.size(-1))
    assert edge_attr_mixed.shape == (
        n_mixed_edges,
        s1_t.edge_attr.size(-1),
    )
    assert length_mixed.shape == (n_mixed_edges, 1)

    assert torch.allclose(
        x_mixed[~batch.is_virtual_node], x_real
    ), "Real node features must be preserved."
    assert torch.allclose(
        edge_attr_mixed[~batch.is_virtual_edge], edge_attr_real
    ), "Real edge attrs must be preserved."
    assert torch.allclose(
        length_mixed[~batch.is_virtual_edge], length_real
    ), "Real lengths must be preserved."

    assert torch.all(
        x_mixed[batch.is_virtual_node] == node_scale
    ), "Virtual node features must equal node_scale."
    assert torch.all(
        edge_attr_mixed[batch.is_virtual_edge] == edge_scale
    ), "Virtual edge attrs must equal edge_scale."
    assert torch.all(length_mixed[batch.is_virtual_edge] == 0), "Virtual lengths must be zero."
