import torch
from pyscf import dft, gto
from torch_geometric.data import Batch

from mldft.datagen.methods.density_fitting import density_fitting_mofdft_torch
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import transform_tensor_with_sample
from mldft.ml.data.components.of_data import Representation
from mldft.ml.models.components.loss_function import (
    PerSampleWeightedPerCoeffLossFunction,
)
from mldft.ml.models.components.model_output import ModelOutput
from mldft.ml.models.components.sample_weighers import SampleWeigher
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_nuclear_attraction_matrix,
    get_nuclear_attraction_vector,
)
from mldft.ofdft.inverse_kohn_sham import get_gamma_from_v_eff, get_v_eff_from_gradient
from mldft.utils.molecules import build_molecule_ofdata


class GradientConsistencyLoss(PerSampleWeightedPerCoeffLossFunction):
    """Loss function to enforce the consistency of the predicted gradients with the input coeffs.

    The loss function performs a single Kohn-Sham step where the effective potential is obtained
    from the gradient of the model output. The resulting density coefficients are compared to the
    input coefficients of the model.
    """

    def __init__(
        self,
        target_key: str,
        basis_info: BasisInfo,
        loss_function: torch.nn.Module = None,
        sample_weigher: SampleWeigher = None,
        reduction: str = "mean",
    ):
        """Initialize the loss function.

        Args:
            target_key: Target key of the model.
            basis_info: BasisInfo object containing the basis dictionary.
            loss_function: Loss function to compare the coefficients.
        """
        super().__init__(loss_function, sample_weigher, reduction)
        self.target_key = target_key
        self.basis_info = basis_info

    def get_loss(
        self,
        batch: Batch,
        model_output: ModelOutput,
        **_,
    ) -> torch.Tensor:
        """Compute the loss value.

        Args:
            batch: Batch containing the OFData samples.
            model_output: Typed model outputs.

        Returns:
            Loss value.
        """
        # TODO: Respect basis transformations and think about the projection of the gradient
        deltas = []
        target_key = self.target_key
        sample_list: list = batch.to_data_list()
        slices = batch._slice_dict["coeffs"]
        gradients = torch.split(model_output.pred_gradient, list(torch.diff(slices)))

        for sample, gradient in zip(sample_list, gradients):
            mol = build_molecule_ofdata(sample, self.basis_info.basis_dict)
            mol_orbital = gto.M(atom=mol.atom, basis="6-31G(2df,p)", unit="bohr")
            if target_key != "kin":
                tensor_type = dict(dtype=sample.coeffs.dtype, device=sample.coeffs.device)
                grid = dft.grid.Grids(mol).build()
                aos = dft.numint.eval_ao(mol, grid.coords, deriv=1)
                grid_weights = torch.as_tensor(grid.weights, **tensor_type)
                aos = torch.as_tensor(aos, **tensor_type)
                aos = transform_tensor_with_sample(sample, aos, Representation.AO)
            else:
                aos = grid_weights = None
            v_eff = get_v_eff_from_gradient(gradient, target_key, sample.coeffs, aos, grid_weights)
            gamma = get_gamma_from_v_eff(v_eff, mol, mol_orbital)

            post_ks_coeffs = density_fitting_mofdft_torch(
                W_coulomb=get_coulomb_matrix(mol),
                L_coulomb=get_coulomb_tensor(mol, mol_orbital),
                v_external_p=get_nuclear_attraction_vector(mol),
                v_external_C=get_nuclear_attraction_matrix(mol_orbital),
                gamma=gamma,
            )
            deltas.append(sample.coeffs - post_ks_coeffs)

        delta = torch.cat(deltas)
        return self.loss_function(delta, torch.zeros_like(delta))


if __name__ == "__main__":
    from hydra import compose, initialize
    from hydra.utils import instantiate

    from mldft.utils.instantiators import instantiate_datamodule

    with initialize(version_base=None, config_path="../../../../configs/ml"):
        cfg = compose(config_name="train.yaml")
    basis_info_ = instantiate(cfg.data.basis_info)
    datamodule = instantiate_datamodule(cfg)
    datamodule.setup(stage="fit")
    sample_ = datamodule.train_set[0]

    loss = GradientConsistencyLoss("kin", basis_info_, torch.nn.MSELoss())

    batch_ = Batch.from_data_list([sample_])
    torch.manual_seed(0)
    gradient_ = torch.rand_like(sample_.coeffs)
    gradient_.requires_grad = True
    loss_value = loss(batch_, gradient_)
    loss_value.backward()
    print(gradient_.grad)
