import torch

from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft import OFDFT_DTYPE
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.energies import Energies


def add_density_optimization_trajectories_to_sample(
    sample: OFData,
    callback: ConvergenceCallback,
    energies_label: Energies,
    save_coeff_interval: int = 100,
):
    """Add the density optimization trajectories of energies and coefficients to the sample."""

    sample.add_item("stopping_index", callback.stopping_index, representation=Representation.NONE)
    sample.add_item(
        "trajectory_gradient_norm",
        torch.as_tensor(callback.gradient_norm, dtype=OFDFT_DTYPE),
        representation=Representation.SCALAR,
    )

    # Optional diagnostics (only available for some optimizers / configurations).
    if len(callback.grad_norm_ratio) > 0 and len(callback.grad_cos_sim) > 0:
        if len(callback.grad_norm_ratio) != len(callback.gradient_norm) or len(
            callback.grad_cos_sim
        ) != len(callback.gradient_norm):
            raise ValueError(
                "Callback grad projection metrics must match the number of iterations. "
                f"Got len(grad_norm_ratio)={len(callback.grad_norm_ratio)}, "
                f"len(grad_cos_sim)={len(callback.grad_cos_sim)}, "
                f"len(gradient_norm)={len(callback.gradient_norm)}."
            )
        sample.add_item(
            "trajectory_grad_norm_ratio",
            torch.as_tensor(callback.grad_norm_ratio, dtype=OFDFT_DTYPE),
            representation=Representation.SCALAR,
        )
        sample.add_item(
            "trajectory_grad_cos_sim",
            torch.as_tensor(callback.grad_cos_sim, dtype=OFDFT_DTYPE),
            representation=Representation.SCALAR,
        )
    sample.add_item(
        "trajectory_l2_norm",
        torch.as_tensor(callback.l2_norm, dtype=OFDFT_DTYPE),
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "trajectory_energy_electronic",
        torch.as_tensor([e.electronic_energy for e in callback.energy], dtype=OFDFT_DTYPE),
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "trajectory_energy_total",
        torch.as_tensor([e.total_energy for e in callback.energy], dtype=OFDFT_DTYPE),
        representation=Representation.SCALAR,
    )
    # callback.energy is a list of Energies objects. Assume every iteration has the same keys.
    for energy_name in callback.energy[0].energies_dict.keys():
        sample.add_item(
            "trajectory_energy_" + energy_name,
            torch.as_tensor([e[energy_name] for e in callback.energy], dtype=OFDFT_DTYPE),
            representation=Representation.SCALAR,
        )
    # ground state energies are already present in the sample under the keys
    # of_labels/energies/ground_state_e_<energy_name>. These are rather inconvenient as key names
    # and less consistent. Thus, we remove them and substitute with more convenient keys.
    for key in sample.keys():
        if key.startswith("of_labels/energies/ground_state_e_"):
            sample.delete_item(key)
    sample.add_item(
        "ground_state_energy_total",
        energies_label.total_energy,  # float64?
        representation=Representation.SCALAR,
    )
    sample.add_item(
        "ground_state_energy_electronic",
        energies_label.electronic_energy,
        representation=Representation.SCALAR,
    )
    for energy_name, ground_state_energy in energies_label.energies_dict.items():
        sample.add_item(
            "ground_state_energy_" + energy_name,
            ground_state_energy,
            representation=Representation.SCALAR,
        )

    stopping_index = int(callback.stopping_index)
    if save_coeff_interval < 1:
        coeff_indices = [stopping_index]
    else:
        coeff_indices = range(0, len(callback.coeffs), int(save_coeff_interval))
    sample.add_item("save_coeff_interval", save_coeff_interval, representation=Representation.NONE)
    sample.add_item(
        "trajectory_coeffs",
        torch.stack([callback.coeffs[index] for index in coeff_indices]),
        representation=Representation.VECTOR,
    )
    sample.add_item(
        "predicted_ground_state_coeffs",
        callback.coeffs[stopping_index].detach().clone(),
        representation=Representation.VECTOR,
    )
    return sample
