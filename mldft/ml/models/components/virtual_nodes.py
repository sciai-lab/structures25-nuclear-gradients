import networkx as nx
import torch
from torch import Tensor


class VirtualFeatureAdder:
    """Class for injecting virtual features into a batch of graphs that are already equipped with
    virtual nodes and edge_index."""

    def __init__(
        self,
        node_scale: float,
        edge_attr_scale: float,
    ):
        """Handling tensor initialization logic.

        Args:
            node_scale: Float with which virtual node features will be initialized.
            edge_scale: Float with which virtual edge attributes will be initialized.

        Note:
            In this implementation, `node_scale` and `edge_scale` are both constant and
            not learnt.
        """
        self.node_scale = node_scale
        self.edge_scale = edge_attr_scale

    def __call__(
        self,
        x: Tensor,
        edge_attr: Tensor,
        length: Tensor,
        is_virtual_node: Tensor,
        is_virtual_edge: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Updates node features `x`, `edge_attr` and `length` by adding virtual node features,
        edge attributes and lengths.

        Args:
            x: Node features of the real graph.
            edge_attr: Edge attribute of the real graph.
            length: Lengths of edges.
            is_virtual_node: Marking virtual nodes among real and virtual nodes.
            is_virtual_edge: Marking virtual edges among real and virtual edges.

        Returns:
            x_out: Combined node features with virtual nodes inserted at designated indices.
            edge_attr_out: Combined edge attributes with virtual features inserted at designated indices.
            length_out: Combined lengths with zero lengths added for virtual edges.
        """
        x_out = torch.empty(is_virtual_node.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        n_v_nodes = is_virtual_node.sum()
        x_out[~is_virtual_node] = x
        x_out[is_virtual_node] = self.node_scale * torch.ones(
            (n_v_nodes, x.size(-1)), dtype=x.dtype, device=x.device
        )

        edge_attr_out = torch.empty(
            is_virtual_edge.size(0),
            edge_attr.size(-1),
            device=edge_attr.device,
            dtype=edge_attr.dtype,
        )
        n_v_edges = is_virtual_edge.sum()
        edge_attr_out[~is_virtual_edge] = edge_attr
        edge_attr_out[is_virtual_edge] = self.edge_scale * torch.ones(
            (n_v_edges, edge_attr.size(-1)), dtype=edge_attr.dtype, device=edge_attr.device
        )

        length_out = torch.zeros(
            (is_virtual_edge.size(0), 1), dtype=length.dtype, device=length.device
        )
        length_out[~is_virtual_edge] = length

        return x_out, edge_attr_out, length_out


def find_max_cliques(edge_index: Tensor) -> set[frozenset[int]]:
    """From a graph edge index find the maximal cliques, i.e. fully connected subgraphs that are
    not a subset of another fully connected subgraph."""

    graph_nx = nx.Graph()
    graph_nx.add_edges_from(
        edge_index.t().tolist(),
    )
    cliques = {frozenset(clique) for clique in nx.find_cliques(graph_nx)}
    return cliques


def pick_initial_dissim_set(sets: list[set] | set[frozenset]) -> frozenset:
    """From a collection of sets, pick the set that minimizes the sum of overlaps to other sets.

    As a tiebreak for equal scores we choose the largest set.
    """
    scores = []
    for S in sets:
        total = sum(len(S & T) for T in sets if T is not S)
        scores.append((total, S))
    min_score = min(scores)[0]
    candidates = [candidate for score, candidate in scores if score == min_score]
    initial_pick = max(candidates, key=len)
    return initial_pick


def pick_min_overlapping_sets(sets: set[frozenset], k_min: int = 3, k_max: int = 6) -> list[list]:
    """From a collection of sets, pick the sets that minimize the maximal overlap to all other
    sets.

    Sets up to `k_min` will be chosen regardless if there is overlap between them. When `k_min`
    sets are collected, up to a limit of `k_max` total sets can be gathered, if they do not overlap
    with any previously chosen. After the collection of `k_min` sets, the function exits early
    when a candidate set would add overlap, since these are ordered ascending in overlap to
    already collected ones.
    We always return a list of lists for easy torch handling.
    """

    if len(sets) <= k_min:
        return [list(fr_set) for fr_set in sets]

    initial_set = pick_initial_dissim_set(sets)
    selected = {initial_set}
    remaining = sets - selected

    while remaining and len(selected) < k_max:
        best = min(remaining, key=lambda s: max(len(s & t) for t in selected))
        if len(selected) >= k_min:
            for already_selected in selected:
                if len(best & already_selected) > 0:
                    selected = [list(fr_set) for fr_set in selected]
                    return selected  # early exit
        selected.add(best)
        remaining.remove(best)

    selected = [list(fr_set) for fr_set in selected]

    return selected
