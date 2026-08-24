from dataclasses import dataclass

from torch import Tensor


@dataclass
class ModelOutput:
    """A dataclass to hold different possible outputs of the OFDFT model."""

    pred_energy: Tensor
    pred_gradient: Tensor | None = None
    pred_diff: Tensor | None = None
    pred_energies_per_atom: Tensor | None = None
    pred_energies_atom_ref: Tensor | None = None
    projected_gradient_difference: Tensor | None = None
    pred_nuc_grad_net: Tensor | None = None

    def __iter__(self):
        """Provide tuple-like unpacking compatibility.

        This keeps older call sites working while the codebase migrates from tuple outputs to the
        typed dataclass.
        """
        yield self.pred_energy
        yield self.pred_gradient
        yield self.pred_diff

    def __getitem__(self, index: int):
        """Provide index-based compatibility with historical tuple outputs."""
        return tuple(self)[index]
