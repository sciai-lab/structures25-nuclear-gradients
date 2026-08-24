"""This module contains classes for weighing samples in a batch.

Currently, this is used in loss functions.
"""

import torch
from torch import Tensor

from mldft.ml.data.components.of_data import OFData


def get_ground_state_samples_in_batch(batch: OFData) -> Tensor:
    """Return per-sample ground-state flags from batch metadata.

    Returns:
        Tensor of shape [n_graphs] with boolean ground-state flags.
    """
    if not hasattr(batch, "is_ground_state"):
        raise AttributeError(
            "Batch is missing `is_ground_state`. Ensure dataset samples set this field."
        )
    is_ground_state = batch.is_ground_state
    if is_ground_state.ndim == 0:
        is_ground_state = is_ground_state.reshape(1)
    if is_ground_state.ndim > 1:
        is_ground_state = is_ground_state.reshape(-1)
    if is_ground_state.shape[0] != batch.batch_size:
        raise ValueError(
            f"Expected `is_ground_state` shape ({batch.batch_size},), got {tuple(is_ground_state.shape)}."
        )
    return is_ground_state.bool()


class SampleWeigher:
    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch.

        Args:
            batch: The batch containing the data.

        Returns:
            weights: The weights for the samples in the given batch.
        """
        raise NotImplementedError

    def __mul__(self, other):
        return ProductSampleWeigher(self, other)

    def __str__(self):
        """Returns a string representation of the sample weigher."""
        return f"{self.__class__.__name__}({self.summary_string()})"

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        raise NotImplementedError


class ProductSampleWeigher(SampleWeigher):
    """Sample weigher that combines multiple sample weighers by multiplying their weights."""

    def __init__(self, *sample_weighers: SampleWeigher):
        """Initializes the product sample weigher.

        Args:
            sample_weighers: The sample weighers to be combined.
        """
        super().__init__()
        self.sample_weighers = sample_weighers

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        weights = torch.ones_like(batch.energy_label)
        for weigher in self.sample_weighers:
            weights *= weigher.get_weights(batch)
        return weights

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return "_times_".join(str(weigher) for weigher in self.sample_weighers)


class ConstantSampleWeigher(SampleWeigher):
    def __init__(self, weight: float = 1):
        """Initializes the constant sample weigher.

        Args:
            weight: The constant weight to be used for all samples.
        """
        super().__init__()
        self.weight = weight

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        return (torch.ones_like(batch.energy_label) * self.weight).float()

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return f"constant_{self.weight}" if self.weight != 1 else "constant"


class InitialGuessOnlySampleWeigher(SampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples with ``scf_iteration == 0`` and 0
    otherwise."""

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        return (batch.scf_iteration == 0).float()

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return "initial_guess_only"


class MinSCFIterationSampleWeigher(SampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples with ``scf_iteration >=
    min_scf_iteration`` and 0 otherwise."""

    def __init__(self, min_scf_iteration: int):
        """Initializes the sample weigher.

        Args:
            min_scf_iteration: The minimum SCF iteration to assign a weight of 1.
        """
        super().__init__()
        self.min_scf_iteration = min_scf_iteration

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        return (batch.scf_iteration >= self.min_scf_iteration).float()

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return f"min_scf_iteration_{self.min_scf_iteration}"


class ListSCFIterationSampleWeigher(SampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples with ``scf_iteration`` in the given
    list and 0 otherwise."""

    def __init__(self, scf_iterations: list[int], include_ground_state: bool = False):
        """Sample weigher that assigns a weight of 1 to samples corresponding to specific SCF
        iterations (as defined in the `scf_iterations` attribute) and optionally to the ground
        state sample.

        Args:
            scf_iterations: The SCF iterations to assign a weight of 1.
            include_ground_state: Whether to assign a weight of 1 to the ground state sample.
        """
        super().__init__()
        self.scf_iterations = torch.as_tensor(scf_iterations)
        self.include_ground_state = include_ground_state

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        weights = torch.isin(
            batch.scf_iteration, self.scf_iterations.to(batch.scf_iteration.device)
        )
        if self.include_ground_state:
            weights = torch.logical_or(weights, get_ground_state_samples_in_batch(batch))
        return weights

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return f"scf_iteration_{self.scf_iterations.tolist()}"


class DefaultPerturbedIterationSampleWeigher(ListSCFIterationSampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples that are perturbed with the default
    perturbation settings (6 to 26 scf iterations plus the ground state) and 0 otherwise."""

    def __init__(self):
        scf_iterations = list(range(6, 27))
        super().__init__(scf_iterations=scf_iterations, include_ground_state=True)

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return "default_perturbed_with_ground_state"


class HasEnergyLabelSampleWeigher(SampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples with an energy label and 0
    otherwise."""

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        return batch.has_energy_label.float()

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return "has_energy_label"


class HasNucGradSampleWeigher(SampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples with a nuclear gradient label and 0
    otherwise."""

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        return batch.has_nuc_grad_label.float()

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return "has_nuc_grad_label"


class GroundStateOnlySampleWeigher(SampleWeigher):
    """Sample weigher that assigns a weight of 1 to the samples that the ground state and 0
    otherwise."""

    def get_weights(self, batch: OFData) -> Tensor:
        """Returns the weights for the given batch."""
        return get_ground_state_samples_in_batch(batch).float()

    def summary_string(self):
        """Returns a summary string of the sample weigher, used e.g. as key in the dataset
        statistics."""
        return "ground_state_only"
