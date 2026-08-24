from torch import Tensor, nn

from mldft.ml.models.components.model_output import ModelOutput


class MLDFTOracle(nn.Module):
    """Oracle module "predicting" the ground-truth energy, gradient and initial guess delta."""

    def __init__(self):
        """Initializes the oracle module."""
        super().__init__()
        # create a parameter to make the optimizer happy
        self.dummy_param = nn.Parameter(Tensor([0.0]), requires_grad=True)

    def forward(self, batch) -> ModelOutput:
        """Returns the ground-truth energy, gradient and initial guess delta.

        Args:
            batch (OFData): The batch / data object containing the input data and labels.

        Returns:
            ModelOutput: The ground-truth energy, gradient and the initial guess delta.
        """
        initial_guess_delta = batch.coeffs - batch.ground_state_coeffs

        initial_guess_delta = initial_guess_delta.clone().requires_grad_(True)
        energy = batch.energy_label.clone().requires_grad_(True)
        gradient = batch.gradient_label.clone().requires_grad_(True)

        return ModelOutput(
            pred_energy=energy, pred_gradient=gradient, pred_diff=initial_guess_delta
        )
