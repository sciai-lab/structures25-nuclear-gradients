"""Convert a trusted Lightning checkpoint into portable inference files."""

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from mldft.ml.models.mldft_module import MLDFTLitModule

BASIS_INFO_TARGET = "mldft.ml.data.components.basis_info.BasisInfo.from_dataset_info_yaml"


def to_builtin(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def replace_basis_info(value, basis_config):
    if isinstance(value, dict):
        if value.get("_target_") == BASIS_INFO_TARGET:
            return basis_config
        return {key: replace_basis_info(item, basis_config) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_basis_info(item, basis_config) for item in value]
    return value


def tensor_config(tensor):
    return {"_target_": "torch.tensor", "_args_": [to_builtin(tensor)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    module = MLDFTLitModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    basis_info = module.basis_info
    basis_config = {
        "_target_": "mldft.ml.data.components.basis_info.BasisInfo",
        "atomic_numbers": to_builtin(basis_info.atomic_numbers),
        "basis_dict": to_builtin(basis_info.basis_dict),
        "irreps_per_atom": [str(irreps) for irreps in basis_info.irreps_per_atom],
        "integrals": to_builtin(basis_info.integrals),
    }

    resolved = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    model_config = replace_basis_info(resolved["model"], basis_config)
    model_config["do_denop_val"] = False
    model_config["dataset_statistics"] = None
    model_config.pop("restricted", None)
    for loss_config in model_config["loss_function"].values():
        if isinstance(loss_config, dict) and "loss" in loss_config:
            loss_config["loss"].pop("restricted", None)

    net = module.net
    net_config = model_config["net"]
    atom_ref = net.atom_ref_module
    net_config["atom_ref_module"] = {
        "_target_": "mldft.ml.models.components.atom_ref.AtomRef",
        "t_z": to_builtin(atom_ref.t_z),
        "t_global": atom_ref.t_global,
        "g_z": to_builtin(atom_ref.g_z),
    }

    if hasattr(net, "dimension_wise_rescaling_module"):
        rescaling_key = "dimension_wise_rescaling_module"
        initial_guess_key = "initial_guess_module"
    else:
        rescaling_key = "feature_wise_rescaling_module"
        initial_guess_key = "initial_guess_delta_module"

    rescaling = getattr(net, rescaling_key)
    net_config[rescaling_key] = {
        "_target_": ("mldft.ml.models.components.dimension_wise_rescaling.DimensionWiseRescaling"),
        "coeff_biases": tensor_config(rescaling.coeff_biases),
        "coeff_stds": tensor_config(rescaling.coeff_stds),
        "max_grads": tensor_config(rescaling.max_grads),
        "s_coeff": rescaling.s_coeff,
        "s_grad": rescaling.s_grad,
        "epsilon": rescaling.epsilon,
    }

    initial_guess = getattr(net, initial_guess_key)
    if initial_guess is None:
        net_config[initial_guess_key] = None
    else:
        net_config[initial_guess_key]["dataset_statistics"] = None

    OmegaConf.save(OmegaConf.create(model_config), output_dir / "model.yaml")
    state_dict = {
        name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()
    }
    output_path = output_dir / args.checkpoint.with_suffix(".safetensors").name
    save_file(state_dict, output_path, metadata={"format": "pt"})
    print(output_path)


if __name__ == "__main__":
    main()
