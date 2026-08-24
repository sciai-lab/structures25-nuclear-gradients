import pytest
import torch

from mldft.ml.data.components.loader import OFLoader
from mldft.ml.models.components.loss_function import (
    EnergyLoss,
    WeightedLoss,
    project_gradient_difference,
)
from mldft.ml.models.components.toy_net import ToyNet
from mldft.ml.models.components.training_metrics import MAEGradient
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils.utils import set_default_torch_dtype


@set_default_torch_dtype(torch.float64)
@pytest.mark.parametrize("batch_sizes", [[2, 5, 6, 32]])
@pytest.mark.parametrize("k_neighbors", [3, 5])
@pytest.mark.parametrize(
    "metric", [MAEGradient(mode="per molecule"), MAEGradient(mode="per electron")]
)
def test_metrics(metric, dummy_basis_info, dummy_dataset_torch, batch_sizes, k_neighbors):
    """Tests if the shape of the toy net output is correct."""
    loss = torch.zeros(len(batch_sizes), dtype=torch.float64)
    net = ToyNet(dummy_basis_info, k_neighbors)
    loss_function = WeightedLoss(energy_loss={"weight": 1.0, "loss": EnergyLoss()})
    module = MLDFTLitModule(
        net,
        basis_info=dummy_basis_info,
        optimizer=None,
        scheduler=None,
        loss_function=loss_function,
        target_key="kin",
        compile=False,
    )
    # Test that the metric is independent of the batchsize
    for i, batch_size in enumerate(batch_sizes):
        dataloader = OFLoader(
            dummy_dataset_torch, batch_size=batch_size, shuffle=False, num_workers=0
        )
        metric.reset()
        for batch in dataloader:
            model_output = module.forward(batch)
            model_output.projected_gradient_difference = project_gradient_difference(
                model_output.pred_gradient, batch
            )
            metric.update(batch, model_output)
        loss[i] = metric.compute()
    assert pytest.approx(loss) == loss[0] * torch.ones(len(batch_sizes))
    # check if the shapes are right
    assert model_output.pred_energy.shape == batch.energy_label.shape
    assert model_output.pred_gradient.shape == batch.gradient_label.shape
