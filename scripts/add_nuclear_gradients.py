import argparse
import multiprocessing
from functools import partial
from pathlib import Path

import numpy as np
import torch
import zarr
from loguru import logger
from pyscf import dft, grad
from pyscf.gto import Mole
from pyscf.lib import chkfile

from mldft.datagen.methods.density_fitting import ksdft_density_matrix
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.basis_transforms import ToE3NNConvention
from mldft.ml.data.components.convert_transforms import ToTorch
from mldft.ml.data.components.of_data import OFData, Representation
from mldft.ofdft.nuclear_gradients import (
    _get_constant_post_basis_transform_matrix,
    global_symmetric_natrep_gradient_generator,
    hartree_basisfunction_gradient,
    hellmann_gradient_torch,
    local_frames_global_symmetric_natrep_gradient_lf_sparse_generator,
    no_basis_transforms_gradient_generator,
    nuclear_attraction_basisfunction_gradient,
)
from mldft.utils.environ import get_mldft_data_path
from mldft.utils.molecules import build_molecule_ofdata, load_scf
from mldft.utils.multiprocess import set_num_threads
from mldft.utils.pyscf_pretty_print import mole_to_sum_formula

BASIS_TRANSFORM_GRADIENT_GENERATORS = {
    "no_basis_transforms": no_basis_transforms_gradient_generator,
    "local_frames_global_symmetric_natrep": local_frames_global_symmetric_natrep_gradient_lf_sparse_generator,
    "global_symmetric_natrep": global_symmetric_natrep_gradient_generator,
}
BASIS_TRANSFORM_POST_TRANSFORMS = {
    "global_symmetric_natrep": lambda basis_info: [ToE3NNConvention(basis_info=basis_info)],
}
SUPPORTED_BASIS_TRANSFORMS = tuple(BASIS_TRANSFORM_GRADIENT_GENERATORS.keys())
DEFAULT_BASIS_TRANSFORMS = (
    "no_basis_transforms",
    "local_frames_global_symmetric_natrep",
    "global_symmetric_natrep",
)
DEVICE = torch.device("cpu")
SUPPORTED_TARGET_GRADIENT_KEYS = (
    "grad_kin",
    "grad_kin_plus_xc",
    "grad_kin_minus_apbe",
    "grad_tot",
    "grad_kinapbe",
)
DEFAULT_TARGET_GRADIENT_KEYS = ("grad_kin_plus_xc", "grad_tot")
CLASSICAL_GRADIENT_CONTRIBUTIONS = (
    hellmann_gradient_torch,
    hartree_basisfunction_gradient,
    nuclear_attraction_basisfunction_gradient,
)
NUC_GRADIENT_KEYS_REQUIRING_CLASSICAL_CONTRIBUTIONS = {
    "nuc_grad_kin",
    "nuc_grad_kin_plus_xc",
    "nuc_grad_kin_minus_apbe",
    "nuc_grad_kinapbe",
}
APBE_XC_FUNCTIONAL = "GGA_K_APBE"


def compute_ks_nuclear_gradient(mol_ks: Mole, gamma: np.ndarray, xc_functional: str) -> np.ndarray:
    """Compute KS nuclear gradients for a fixed density matrix and xc functional."""
    ks = dft.RKS(mol_ks, xc=xc_functional)
    ks.max_cycle = 0
    ks.kernel(gamma)

    grad_obj = grad.RKS(ks)
    return grad_obj.kernel().reshape(-1)


def compute_classical_nuclear_gradient(sample: OFData) -> np.ndarray:
    """Compute classical nuclear gradient terms to subtract from KS gradients."""
    classical_gradient = np.zeros(3 * sample.mol.natm)
    for contribution in CLASSICAL_GRADIENT_CONTRIBUTIONS:
        classical_gradient += contribution(sample, sample.coeffs, DEVICE).numpy(force=True)
    return classical_gradient


def decode_text(value: object) -> str:
    """Convert bytes or str to plain str."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected str or bytes, got {type(value)}")


def get_existing_quantities(output_file: Path) -> set[str]:
    """Check which datasets already exist in a zarr file."""
    if not output_file.exists():
        return set()
    try:
        with zarr.ZipStore(str(output_file), mode="r") as zipstore:
            root = zarr.group(store=zipstore)
            return set(root.keys())
    except Exception as e:
        logger.warning(f"Could not read {output_file}: {e}.")
        return set()


def get_post_basis_transform_matrix(
    sample: OFData, basis_info: BasisInfo, transform_name: str
) -> torch.Tensor | None:
    """Return constant post basis transforms used by training for a precomputed transform."""
    post_transform_factory = BASIS_TRANSFORM_POST_TRANSFORMS.get(transform_name)
    if post_transform_factory is None:
        return None
    return _get_constant_post_basis_transform_matrix(sample, post_transform_factory(basis_info))


def apply_post_basis_transform_to_gradient(
    grad_basis_transform: torch.Tensor, post_transform_matrix: torch.Tensor | None
) -> torch.Tensor:
    """Apply constant post basis transforms to ``dX / dR`` before contraction."""
    if post_transform_matrix is None:
        return grad_basis_transform
    post_transform_matrix = post_transform_matrix.to(
        device=grad_basis_transform.device,
        dtype=grad_basis_transform.dtype,
    )
    if grad_basis_transform.ndim == 2:
        return post_transform_matrix @ grad_basis_transform
    if grad_basis_transform.ndim == 3:
        return torch.einsum("ij,xjk->xik", post_transform_matrix, grad_basis_transform)
    raise ValueError(f"Unexpected gradient dimension: {grad_basis_transform.dim()}")


def contract_transform_gradient(
    sample: OFData,
    gradient_generator,
    post_transform_matrix: torch.Tensor | None = None,
) -> np.ndarray:
    """Contract a basis-transform gradient generator with the sample coefficients."""
    grad_shape = (3 * sample.mol.natm, sample.mol.nao)
    grad_contracted = torch.zeros(grad_shape, dtype=torch.float64)
    for i, g in enumerate(gradient_generator(sample, device=DEVICE)):
        g = apply_post_basis_transform_to_gradient(g, post_transform_matrix)
        if g.dim() == 2:  # (n_basis, n_basis) per nuclear coordinate
            grad_contracted[i] = g @ sample.coeffs
        elif g.dim() == 3:  # (n_coords, n_basis, n_basis) per generator step
            n_coords = g.shape[0]
            grad_contracted[i * n_coords : (i + 1) * n_coords] = g @ sample.coeffs
        else:
            raise ValueError(f"Unexpected gradient dimension: {g.dim()}")
    return grad_contracted.numpy(force=True)


def save_loss_components(
    label_path: Path,
    ks_dir: Path,
    ks_file_prefix: str,
    output_path: Path,
    basis_info: BasisInfo,
    num_threads: int = 1,
    allowed_ids: set | None = None,
    target_gradient_keys: set[str] | None = None,
    basis_transforms: list[str] | None = None,
):
    """Save the components for the nuclear gradient loss in a zarr file.

    Args:
        label_path: Path to the label file.
        ks_dir: Directory containing Kohn-Sham files.
        ks_file_prefix: Prefix for the Kohn-Sham file names.
        output_path: Directory to save the output zarr files.
        basis_info: Basis information for the molecules.
        num_threads: Number of threads to use.
        allowed_ids: Optional set of allowed label IDs to process.
        target_gradient_keys: Optional target gradient keys to compute and store as
            ``nuc_<gradient_key>``.
        basis_transforms: Optional list of basis transforms for which contracted
            transform gradients are computed.
    """
    torch.set_default_dtype(torch.float64)
    set_num_threads(num_threads)

    label_no = label_path.stem.split(".")[0]
    output_file = output_path / f"{label_no}.zarr.zip"
    if allowed_ids is not None and int(label_no) not in allowed_ids:
        logger.info(f"Skipped {label_path.name} as its ID is not in the allowed IDs.")
        return

    # load KS molecule and OF sample
    ks_path = ks_dir / f"{ks_file_prefix}{label_no}.chk"
    mol_ks_data = chkfile.load(ks_path, "mol")
    mol_ks = Mole().loads(decode_text(mol_ks_data))
    mol_ks.build()
    if sum(mol_ks.atom_charges() > 1) < 3:
        logger.info(
            f"Skipped {label_path.name} ({mole_to_sum_formula(mol_ks, True)}) as it has less than 3 heavy atoms."
        )
        return
    sample = OFData.from_file(label_path, -1, basis_info, add_irreps=True)
    mol_of = build_molecule_ofdata(sample, basis_info.basis_dict)
    sample.add_item("mol", mol_of, representation=Representation.NONE)
    sample = ToTorch()(sample)

    existing_quantities = get_existing_quantities(output_file)
    additional_quantities = dict()
    if target_gradient_keys is None:
        target_gradient_keys = set(DEFAULT_TARGET_GRADIENT_KEYS)
    if basis_transforms is None:
        basis_transforms = list(DEFAULT_BASIS_TRANSFORMS)

    basis_transforms = list(dict.fromkeys(basis_transforms))

    requested_nuc_gradient_keys = {f"nuc_{key}" for key in target_gradient_keys}
    missing_nuc_gradient_keys = requested_nuc_gradient_keys.difference(existing_quantities)

    try:
        if missing_nuc_gradient_keys:
            results, _, iterations = load_scf(ks_path)
            xc = decode_text(results["name_xc_functional"])
            mo_coeffs = iterations[-1]["molecular_coeffs_orbitals"]
            mo_occ = iterations[-1]["occupation_numbers_orbitals"]
            gamma = ksdft_density_matrix(mo_coeffs, mo_occ)
            needs_classical_nuclear_gradient = bool(
                missing_nuc_gradient_keys.intersection(
                    NUC_GRADIENT_KEYS_REQUIRING_CLASSICAL_CONTRIBUTIONS
                )
            )
            classical_nuclear_gradient = None
            if needs_classical_nuclear_gradient:
                classical_nuclear_gradient = compute_classical_nuclear_gradient(sample)

            nuc_grad_kin = None
            nuc_grad_ks = None

            if {
                "nuc_grad_kin_plus_xc",
                "nuc_grad_tot",
            }.intersection(missing_nuc_gradient_keys):
                nuc_grad_ks = compute_ks_nuclear_gradient(mol_ks, gamma, xc)

                if "nuc_grad_tot" in missing_nuc_gradient_keys:
                    additional_quantities["nuc_grad_tot"] = nuc_grad_ks

                if "nuc_grad_kin_plus_xc" in missing_nuc_gradient_keys:
                    assert classical_nuclear_gradient is not None
                    nuc_grad_kin_plus_xc = nuc_grad_ks - classical_nuclear_gradient
                    additional_quantities["nuc_grad_kin_plus_xc"] = nuc_grad_kin_plus_xc

            if {
                "nuc_grad_kin",
                "nuc_grad_kin_minus_apbe",
                "nuc_grad_kinapbe",
            }.intersection(missing_nuc_gradient_keys):
                assert classical_nuclear_gradient is not None
                nuc_grad_ks_no_xc = compute_ks_nuclear_gradient(mol_ks, gamma, "")
                nuc_grad_kin = nuc_grad_ks_no_xc - classical_nuclear_gradient

                if "nuc_grad_kin" in missing_nuc_gradient_keys:
                    additional_quantities["nuc_grad_kin"] = nuc_grad_kin

            if {
                "nuc_grad_kin_minus_apbe",
                "nuc_grad_kinapbe",
            }.intersection(missing_nuc_gradient_keys):
                assert classical_nuclear_gradient is not None
                if nuc_grad_kin is None:
                    nuc_grad_ks_no_xc = compute_ks_nuclear_gradient(mol_ks, gamma, "")
                    nuc_grad_kin = nuc_grad_ks_no_xc - classical_nuclear_gradient

                nuc_grad_ks_apbe = compute_ks_nuclear_gradient(mol_ks, gamma, APBE_XC_FUNCTIONAL)
                nuc_grad_kin_plus_apbe = nuc_grad_ks_apbe - classical_nuclear_gradient
                nuc_grad_kinapbe = nuc_grad_kin_plus_apbe - nuc_grad_kin

                if "nuc_grad_kinapbe" in missing_nuc_gradient_keys:
                    additional_quantities["nuc_grad_kinapbe"] = nuc_grad_kinapbe

                if "nuc_grad_kin_minus_apbe" in missing_nuc_gradient_keys:
                    additional_quantities["nuc_grad_kin_minus_apbe"] = (
                        nuc_grad_kin - nuc_grad_kinapbe
                    )

            if any(np.isnan(value).any() for value in additional_quantities.values()):
                logger.info(f"Skipped {label_path.name} due to NaN values in KS gradients.")
                return

        for transform_name in basis_transforms:
            if transform_name == "no_basis_transforms":
                continue
            gradient_generator = BASIS_TRANSFORM_GRADIENT_GENERATORS[transform_name]
            dataset_name = f"{transform_name}_grad_contracted"
            if dataset_name in existing_quantities or dataset_name in additional_quantities:
                continue
            post_transform_matrix = get_post_basis_transform_matrix(
                sample, basis_info, transform_name
            )
            grad_contracted = contract_transform_gradient(
                sample,
                gradient_generator,
                post_transform_matrix,
            )
            if np.isnan(grad_contracted).any():
                logger.info(f"Skipped {label_path.name} due to NaN in {transform_name} gradients.")
                return
            additional_quantities[dataset_name] = grad_contracted

    except Exception as e:
        logger.info(
            f"Skipped {label_path.name} ({mole_to_sum_formula(mol_ks, True)}) due to Error: {e}."
        )
        return

    if not additional_quantities:
        logger.info(f"Skipped {label_path.name} as all quantities exist.")
        return

    mode = "a" if output_file.exists() else "w"
    with zarr.ZipStore(str(output_file), mode=mode) as zipstore:
        root = zarr.open_group(store=zipstore)
        for name, data in additional_quantities.items():
            root.create_dataset(name, data=data)

    logger.info(f"Processed {label_path.name}, {ks_path.name}")


def main():
    """Script entry point to compute and save nuclear gradients."""
    parser = argparse.ArgumentParser(
        description="Compute and save nuclear gradients for the QM7X dataset."
    )
    parser.add_argument(
        "--dataset", type=str, default="QM7X_perturbed_fock", help="Dataset to process."
    )
    parser.add_argument(
        "--target_gradient_keys",
        nargs="+",
        type=str,
        default=list(DEFAULT_TARGET_GRADIENT_KEYS),
        help=(
            "Target gradient keys to generate as nuclear gradients (stored as "
            "nuc_<gradient_key>)."
        ),
    )
    parser.add_argument(
        "--basis_transforms",
        nargs="+",
        type=str,
        default=list(DEFAULT_BASIS_TRANSFORMS),
        help=("Basis transforms for which contracted transform gradients are generated."),
    )
    parser.add_argument("--ks_file_prefix", type=str, default="qm7x_", help="Path to the ks file.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use.")
    parser.add_argument(
        "--n_molecules", type=int, default=1, help="Number of molecules to process."
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index.")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads per process to use.",
    )
    parser.add_argument("--id_file", type=str, default=None, help="ID file (for OMol subsets).")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    data_dir = get_mldft_data_path() / args.dataset
    ks_dir = data_dir / "kohn_sham"
    label_dir = data_dir / "labels"
    output_path = data_dir / "nuclear_gradients"
    basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis([1, 6, 7, 8, 9])

    labels = list(sorted(label_dir.glob("*.zarr.zip")))
    start = args.start_idx
    end = min(args.start_idx + args.n_molecules, len(labels))
    labels = labels[start:end]
    output_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Processing {len(labels)} molecules from {start} to {end}.")

    allowed_ids = None
    if args.id_file is not None:
        allowed_ids = np.load(args.id_file)
        allowed_ids = set(allowed_ids)

    target_gradient_keys = set(args.target_gradient_keys)
    invalid_target_gradient_keys = target_gradient_keys.difference(SUPPORTED_TARGET_GRADIENT_KEYS)
    if invalid_target_gradient_keys:
        raise ValueError(
            f"Unknown target gradient keys: {sorted(invalid_target_gradient_keys)}. "
            f"Supported keys are {SUPPORTED_TARGET_GRADIENT_KEYS}."
        )

    basis_transforms = list(dict.fromkeys(args.basis_transforms))
    invalid_basis_transforms = set(basis_transforms).difference(SUPPORTED_BASIS_TRANSFORMS)
    if invalid_basis_transforms:
        raise ValueError(
            f"Unknown basis transforms: {sorted(invalid_basis_transforms)}. "
            f"Supported transforms are {SUPPORTED_BASIS_TRANSFORMS}."
        )

    func = partial(
        save_loss_components,
        ks_dir=ks_dir,
        ks_file_prefix=args.ks_file_prefix,
        output_path=output_path,
        basis_info=basis_info,
        num_threads=args.num_threads,
        allowed_ids=allowed_ids,
        target_gradient_keys=target_gradient_keys,
        basis_transforms=basis_transforms,
    )

    set_num_threads(args.num_threads)
    if args.num_processes <= 1:
        for l in labels:
            func(l)
    else:
        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            for _ in pool.imap_unordered(func, labels):
                pass


if __name__ == "__main__":
    main()
