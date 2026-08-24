from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pyscf import gto

from mldft.ml.data.components.basis_info import calculate_basis_size
from mldft.ml.data.components.convert_transforms import (
    PrepareForDensityOptimization,
    ToTorch,
    to_torch,
)
from mldft.ml.data.components.of_data import OFData
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.functional_factory import FunctionalFactory, requires_grid
from mldft.utils.environ import CheckpointChoice, parse_run_path, run_to_checkpoint_path
from mldft.utils.instantiators import instantiate_model
from mldft.utils.molecules import check_atom_types


# This class has to be kept in a different file than `mldft/utils/sample.py` to avoid
# circular imports (due to the import of the `add_density_optimization_trajectories_to_sample`
# function) in the MLDFTLitModule file. (This is also somehow linked to the active training in
# the datageneration part of the code)
class SampleGenerator:
    """Class to generate samples from the model configuration.

    Attributes:
        model_config: The model configuration.
        model: The model.
        transforms: The transforms.
        basis_info: The basis information.
        negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
    """

    def __init__(
        self,
        model_config: DictConfig,
        model: "MLDFTLitModule",
        negative_integrated_density_penalty_weight: float = 0.0,
        transform_device: str | torch.device = "cpu",
        xc_functional: str | None = None,
    ) -> None:
        """Initialize the SampleGenerator.

        Args:
            model_config: The model configuration.
            model: The model.
            negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
            xc_functional: XC functional to use for models that are not trained on XC.
        """
        basis_info = instantiate(model_config.data.basis_info)
        self.basis_info = basis_info

        self.transforms = instantiate(model_config.data.transforms)

        for i, transform in enumerate(self.transforms.pre_transforms):
            if isinstance(transform, ToTorch):
                self.transforms.pre_transforms[i] = ToTorch(
                    device=transform_device, float_dtype=OFDFT_DTYPE
                )

        self.model = model
        self.model_config = model_config

        self.set_functional(
            xc_functional=xc_functional,
            negative_integrated_density_penalty_weight=negative_integrated_density_penalty_weight,
        )

        self.transforms.add_transformation_matrix = True
        self.transforms.use_cached_data = False

        self.negative_integrated_density_penalty_weight = (
            negative_integrated_density_penalty_weight
        )

        self.transforms.device = transform_device

    @classmethod
    def from_run_path(
        cls,
        run_path: str | Path,
        device: str | torch.device = "cuda",
        transform_device: str | torch.device = "cpu",
        negative_integrated_density_penalty_weight: float = 0.0,
        ckpt_choice: CheckpointChoice = "last",
        xc_functional: str | None = None,
    ) -> "SampleGenerator":
        """Create a SampleGenerator from a run path.

        Args:
            run_path: The run path.
            device: The device to load the model on.
            transform_device: The device to apply the transforms on.
            negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
            ckpt_choice: the checkpoint file to use.
            xc_functional: XC functional to use for models that are not trained on XC.

        Returns:
            The instantiated SampleGenerator
        """

        run_path = parse_run_path(run_path)
        checkpoint_path = run_to_checkpoint_path(run_path, ckpt_choice)
        model_config_path = run_path / "hparams.yaml"
        model_config = OmegaConf.load(model_config_path)
        model = instantiate_model(checkpoint_path, device)

        return cls(
            model_config,
            model,
            negative_integrated_density_penalty_weight,
            transform_device=transform_device,
            xc_functional=xc_functional,
        )

    def get_sample_from_mol(self, mol: gto.Mole) -> OFData:
        """Get a sample from a molecule with the appropriate transforms applied.

        Args:
            mol: The molecule.

        Returns:
            The OFData sample.
        """
        # check that the molecule only contains allowed atom types by comparing to the
        # basis info of the model
        check_atom_types(mol, self.basis_info.atomic_numbers)

        n_basis = calculate_basis_size(mol, self.basis_info)

        sample = OFData.construct_new(
            basis_info=self.basis_info,
            pos=mol.atom_coords(unit="Bohr"),
            atomic_numbers=mol.atom_charges(),
            coeffs=np.zeros(n_basis),
            basis_integrals="infer_from_basis",
            add_irreps=True,
        )
        sample = self.transforms.forward(sample)
        sample.mol.charge = mol.charge
        sample = to_torch(sample, device=self.model.device, float_dtype=OFDFT_DTYPE)
        return sample

    def set_functional(
        self,
        xc_functional: str | None = None,
        negative_integrated_density_penalty_weight: float | None = None,
    ) -> None:
        """Set the functional to be used with this model and adapt the pre_transforms accordingly.

        Args:
            negative_integrated_density_penalty_weight: The weight for the negative integrated density penalty.
            xc_functional: XC functional to use for models that are not trained on XC.
        """
        if negative_integrated_density_penalty_weight is None:
            negative_integrated_density_penalty_weight = (
                self.negative_integrated_density_penalty_weight
            )
        else:
            self.negative_integrated_density_penalty_weight = (
                negative_integrated_density_penalty_weight
            )

        self.xc_functional = xc_functional
        self.requires_grid = requires_grid(
            self.model_config.data.target_key, negative_integrated_density_penalty_weight
        )

        # check if a PrepareForDensityOptimization instance is already present,
        # if so remove it as it might be the wrong one for the new functional
        for i, transform in enumerate(self.transforms.pre_transforms):
            if isinstance(transform, PrepareForDensityOptimization):
                self.transforms.pre_transforms.pop(i)
                break

        self.transforms.pre_transforms.insert(
            0, PrepareForDensityOptimization(self.basis_info, add_grid=self.requires_grid)
        )

    def get_functional_factory(self) -> FunctionalFactory:
        """Get a functional factory for the model and its config.

        Args:
            xc_functional: The XC functional to use.

        Returns:
            The functional factory.
        """

        return FunctionalFactory.from_module(
            self.model,
            self.xc_functional,
            negative_integrated_density_penalty_weight=self.negative_integrated_density_penalty_weight,
        )
