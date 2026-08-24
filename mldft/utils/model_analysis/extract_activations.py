"""Copied from Corinnas repo + Modifications Includes functions to extract activations from a
model, both for single evaluation as well as during density optimization."""

import pickle
from pathlib import Path
from typing import Dict, List

import hydra
import rootutils
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import Subset
from tqdm import tqdm

from mldft.ml.data.components.dataset import OFDataset
from mldft.ml.data.components.loader import OFLoader
from mldft.ml.data.components.of_batch import OFBatch
from mldft.ml.data.components.of_data import Representation
from mldft.ml.data.datamodule import OFDataModule
from mldft.ml.models.components.equiformerv2_components.equiformerv2_net import (
    EquiformerV2Net,
)
from mldft.ml.models.components.graphformer import Graphformer, MLPStack
from mldft.ml.models.components.mlp import MLP
from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.ml.preprocess.dataset_statistics import DatasetStatistics
from mldft.ofdft.callbacks import ConvergenceCallback
from mldft.ofdft.density_optimization import density_optimization
from mldft.ofdft.functional_factory import FunctionalFactory, requires_grid
from mldft.ofdft.run_density_optimization import (
    PrepareForDensityOptimization,
    configure_dataset_indices,
    get_mldft_data_path,
)
from mldft.utils import RankedLogger
from mldft.utils.sad_guesser import SADNormalizationMode

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def setup_activation_hooks_graphormer(model: MLDFTLitModule) -> Dict[str, List[torch.Tensor]]:
    """Setup forward hooks on each layer of the model and return a dictionary that will hold the
    activations.

    The keys of this dictionary will correspond to layer names (e.g. 'layer_0', 'layer_1', etc.).
    """

    # Dictionary to store activations from each layer
    # Keys will be layer names, and values will be lists of recorded tensors
    activations: Dict[str, List[torch.Tensor]] = {}
    network = model.net

    # Layer Activations
    assert isinstance(network, Graphformer), "Model must be a Graphformer"
    layers = network.gnn_module.g3d_layers

    # Register hooks
    for idx, layer in enumerate(layers):
        layer_name = f"{layer._get_name()}_{idx}"
        activations[layer_name] = []

        def output_hook_g3d_layer(name):
            def hook_fn(module, input, kwargs, output):
                if "batch" in kwargs.keys() and kwargs["batch"] is not None:
                    batch = kwargs["batch"]
                    counts = torch.bincount(batch)
                else:
                    counts = torch.tensor([kwargs["x"].shape[0]])

                batched_activations = output.detach()
                split_activations = torch.split(batched_activations, counts.tolist(), dim=0)
                activations[name].append(split_activations)

            return hook_fn

        layer.register_forward_hook(output_hook_g3d_layer(layer_name), with_kwargs=True)

        activations[layer_name + "_attention"] = []
        original_attention = layer.compute_attention  # Save original

        def get_attention_hook(original_layer, name):
            def attention_hook(*args, **kwargs):
                # (Batch (# Number of edges), # Heads)
                output = original_layer(*args, **kwargs)

                edge_index = args[3]
                batch = activations["batch"][0]

                if batch is None:
                    counts = torch.tensor([args[0].shape[0]])
                else:
                    edge_batch = batch[edge_index]
                    counts = torch.bincount(edge_batch)

                batched_activations = output.detach()
                split_activations = torch.split(batched_activations, counts.tolist(), dim=0)
                activations[name + "_attention"].append(split_activations)
                return output

            return attention_hook

        layer.compute_attention = get_attention_hook(original_attention, layer_name)

        if idx == 0:
            activations["length"] = []

            def message_hook(name):
                def hook_fn(module, input):
                    input = input[0]
                    batch = input["index"]
                    if batch is None:
                        counts = torch.tensor([input["query_i"].shape[0]])
                    else:
                        counts = torch.bincount(batch)
                    length = input["length"]
                    batched_lengths = length.detach()
                    split_activations = torch.split(batched_lengths, counts.tolist(), dim=0)
                    activations[name].append(split_activations)

                return hook_fn

            layer.register_message_forward_pre_hook(message_hook("length"))

    def output_hook_mlp(name):
        def hook_fn(module, input, output):
            batch = input[1]
            if batch is None:
                counts = torch.tensor([input[0].shape[0]])
            else:
                counts = torch.bincount(batch)
            batched_activations = output.detach()
            split_activations = torch.split(batched_activations, counts.tolist(), dim=0)
            activations[name].append(split_activations)

        return hook_fn

    mlp_p = network.node_embedding_module.mlp_p
    activations["mlp_p"] = []
    mlp_p.register_forward_hook(output_hook_mlp("mlp_p"))

    mlp_dst = network.node_embedding_module.mlp_dst
    activations["mlp_dst"] = []
    mlp_dst.register_forward_hook(output_hook_mlp("mlp_dst"))

    first_g3d_layer = network.gnn_module.g3d_layers[0]

    def input_hook_g3d_layer(name):
        def hook_fn(module, input, kwargs):
            if "batch" in kwargs.keys() and kwargs["batch"] is not None:
                batch = kwargs["batch"]
                counts = torch.bincount(batch)
                # save most recent batch info, to split activations correctly
                activations["batch"] = [batch]
            else:
                counts = torch.tensor([kwargs["x"].shape[0]])
                activations["batch"] = [None]
            batched_inputs = kwargs["x"].detach()
            split_inputs = torch.split(batched_inputs, counts.tolist(), dim=0)
            activations[name].append(split_inputs)
            # activations["batch"].append(batch)

        return hook_fn

    activations["g3d_layer_input"] = []
    first_g3d_layer.register_forward_pre_hook(
        input_hook_g3d_layer("g3d_layer_input"), with_kwargs=True
    )

    def edge_attribute_hook(name):
        # extract edge attributes after the MLP
        def hook_fn(module, input, kwargs):
            if "batch" in kwargs.keys() and kwargs["batch"] is not None:
                batch = kwargs["batch"]
                edge_index = kwargs["edge_index"]
                edge_batch = batch[edge_index[0]]
                counts = torch.bincount(edge_batch)
            else:
                counts = torch.tensor([kwargs["x"].shape[0]])
            batched_inputs = kwargs["edge_attr"].detach()
            split_inputs = torch.split(batched_inputs, counts.tolist(), dim=0)
            activations[name].append(split_inputs)
            # activations["batch"].append(batch)

        return hook_fn

    activations["edge_attr"] = []
    network.gnn_module.register_forward_pre_hook(
        edge_attribute_hook("edge_attr"), with_kwargs=True
    )

    if isinstance(network.energy_mlp, MLPStack):
        energy_mlps = network.energy_mlp.mlps
        energy_weights = network.energy_mlp.weights

        for idx, mlp in enumerate(energy_mlps):
            layer_name = f"energy_mlp_{idx}"
            activations[layer_name] = []

            def output_hook_mlp(name):
                def hook_fn(module, input, output):
                    batch = input[1]
                    if batch is None:
                        counts = torch.tensor([input[0].shape[0]])
                    else:
                        counts = torch.bincount(batch)
                    # Multiply mlp output by weight. This corresponds to the per atom energy contribution of one readout layer
                    batched_activations = (
                        output.detach() * energy_weights[idx].detach() / len(energy_mlps)
                    )
                    split_activations = torch.split(batched_activations, counts.tolist(), dim=0)
                    activations[name].append(split_activations)

                return hook_fn

            mlp.register_forward_hook(output_hook_mlp(layer_name))

    elif isinstance(network.energy_mlp, MLP):
        activations["energy_mlp"] = []

        def output_hook_mlp(name):
            def hook_fn(module, input, output):
                batch = input[1]
                if batch is None:
                    counts = torch.tensor([input[0].shape[0]])
                else:
                    counts = torch.bincount(batch)
                batched_activations = output.detach().squeeze_(0)
                split_activations = torch.split(batched_activations, counts.tolist(), dim=0)
                activations[name].append(split_activations)

            return hook_fn

        network.energy_mlp.register_forward_hook(output_hook_mlp("energy_mlp"))

    activations["atom_ind"] = []

    def atom_ind_hook(name):
        def hook_fn(model, input, kwargs):
            if "batch" in kwargs.keys() and kwargs["batch"] is not None:
                batch = kwargs["batch"]
                counts = torch.bincount(batch)
            else:
                raise Exception("hoped for batch here")

            if "atom_ind" in kwargs.keys() and kwargs["atom_ind"] is not None:
                batched_ind = kwargs["atom_ind"].detach()
                split_ind = torch.split(batched_ind, counts.tolist(), dim=0)
                activations[name].append(split_ind)
            else:
                raise Exception("No Atom Index :(")

        return hook_fn

    network.node_embedding_module.register_forward_pre_hook(
        atom_ind_hook("atom_ind"), with_kwargs=True
    )

    return activations


def setup_activation_hooks_equiformer(model: MLDFTLitModule) -> Dict[str, List[torch.Tensor]]:
    """Setup forward hooks on each layer of the model and return a dictionary that will hold the
    activations."""
    # Dictionary to store activations from each layer
    # Keys will be layer names, and values will be lists of recorded tensors
    activations: Dict[str, List[torch.Tensor]] = {}
    network = model.net

    assert isinstance(network, EquiformerV2Net), "Model must be an EquiformerV2Net"
    blocks = network.blocks

    # Register hooks for each block
    for idx, block in enumerate(blocks):
        block_name = f"block_{idx}"
        activations[block_name] = []

        def output_hook_equiformer(name):
            def hook_fn(module, input, kwargs, output):
                if "batch" in kwargs and kwargs["batch"] is not None:
                    batch = kwargs["batch"]
                    counts = torch.bincount(batch)
                else:
                    counts = torch.tensor([input[0].shape[0]])

                batched_activations = output.embedding.detach()
                split_activations = torch.split(batched_activations, counts.tolist(), dim=0)
                activations[name].append(split_activations)

            return hook_fn

        block.register_forward_hook(output_hook_equiformer(block_name), with_kwargs=True)
    return activations


def record_batch(
    model: MLDFTLitModule, sample: torch.Tensor, activations: Dict[str, List[torch.Tensor]]
):
    """Run a sample through the model and record the activations."""
    # Clear previous activations
    for key in activations.keys():
        activations[key].clear()
    # Run the forward pass on the sample
    energy, gradient, _ = model(sample)
    return energy


def run_and_save_batch(
    model: MLDFTLitModule,
    batch: OFBatch,
    activations: Dict[str, List[torch.Tensor]],
    path: Path = None,
):
    """Run a batch (batch) through the model and save each batch's activations to a separate Zarr
    file.

    Each file will contain multiple datasets: one dataset per layer of activations and one for the error.
    """
    # Run model and record activations
    energy = record_batch(model, batch, activations)

    error = (energy - batch.energy_label).detach().cpu().numpy()

    # activations now contains entries like:
    # activations["layer_name"] = [ (activations_for_batch_0, activations_for_batch_1, ...) ]
    # Extract the tuple of activations from the list for each layer
    layer_outputs = {
        key: val[0] for key, val in activations.items() if key != "error" and key != "batch"
    }

    # Determine how many samples are in this batch
    # We can use the length of the tuple from any layer (assuming consistent batch size)
    batch_size = len(next(iter(layer_outputs.values())))

    # Save each sample in the batch to its own file
    # `n_molecule` is presumably the starting index for this batch
    if path is not None:
        for i in range(batch_size):
            filename = f"{batch[i].mol_id.split('_')[1]}.zarr.zip"
            with zarr.ZipStore(path / filename, mode="w") as zipstore:
                root = zarr.group(zipstore)
                # Save the error for this sample
                root.create_dataset("error", data=error[i])
                # Save each layer's activation for this sample
                for layer_name, acts_tuple in layer_outputs.items():
                    # acts_tuple[i] is a torch.Tensor for this sample's activation in the given layer
                    act = acts_tuple[i].cpu().numpy()
                    root.create_dataset(layer_name, data=act)


def record_activations(model: MLDFTLitModule, loader: OFLoader, path: Path, device: str):
    """Record activations for each batch in the loader."""
    model.to(device)
    model.eval()
    # Setup hooks for activation recording
    if isinstance(model.net, Graphformer):
        activations = setup_activation_hooks_graphormer(model=model)
    elif isinstance(model.net, EquiformerV2Net):
        activations = setup_activation_hooks_equiformer(model=model)
    else:
        raise ValueError(
            f"Unsupported model type: {type(model.net)}. Only Graphformer and EquiformerV2Net are supported."
        )

    cat_activations = {}

    # Run through the loader
    for index, batch in enumerate(tqdm(loader, total=len(loader.dataset) // loader.batch_size)):
        batch.to(device)
        run_and_save_batch(
            model=model,
            batch=batch,
            activations=activations,
            path=path,
        )


log = RankedLogger(__name__, rank_zero_only=True)


def extract_activations(
    ckpt_path: str,
    num_samples: int,
    out_path: str,
    random: bool = False,
    train_set: bool = False,
    full_set: bool = False,
):
    """Extracts activations from the model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_cfg_path = Path(ckpt_path).parent.parent / "hparams.yaml"
    ckpt_cfg = OmegaConf.load(ckpt_cfg_path)

    output_dir = Path(out_path)

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("high")

    # Limit the SCF iterations such that only one sample per molecule is used
    ckpt_cfg.data.datamodule.dataset_kwargs.limit_scf_iterations = [-1]

    # Instantiate the datamodule with test setup
    log.info(f"Instantiating datamodule <{ckpt_cfg.data.datamodule._target_}>")
    datamodule: OFDataModule = hydra.utils.instantiate(ckpt_cfg.data.datamodule)

    if train_set:
        datamodule.setup("fit")
        test_set = datamodule.train_set
    else:
        datamodule.setup("test")
        test_set = datamodule.test_set

    # Choose subset of the dataset
    if random:
        subset = Subset(test_set, torch.randperm(len(test_set))[:num_samples])
    else:
        subset = Subset(test_set, range(0, num_samples))

    if not full_set:
        datamodule.test_set = subset
    else:
        datamodule.test_set = test_set

    model = MLDFTLitModule.load_from_checkpoint(ckpt_path)
    loader = datamodule.test_dataloader()

    log.info("Starting inference")
    record_activations(model=model, loader=loader, path=output_dir, device=device)
    log.info("Finished inference")


class ActivationCallback(ConvergenceCallback):
    """Callback class for collecting model activations during density optimization."""

    def __init__(self, activations, out_path, save_every: int = 10):
        """
        Args:
            activations (Dict[str, List[torch.Tensor]]): Dictionary containing the activations.
        """
        super().__init__()
        self.activations = activations
        self.save_every = save_every
        self.out_path = Path(out_path)
        self.iteration = 0

    def __call__(self, minimization_locals: dict) -> None:
        """Callback function for the minimization.

        Args:
            minimization_locals (dict): Dictionary containing the local variables of the
                minimization function.
        """

        super().__call__(minimization_locals)

        if self.iteration % self.save_every == 0:
            layer_outputs = {
                key: val[0] for key, val in self.activations.items() if key != "error"
            }

            filename = f"iteration_{self.iteration}.zarr.zip"
            with zarr.ZipStore(self.out_path / filename, mode="w") as zipstore:
                root = zarr.group(zipstore)
                # Save each layer's activation for this sample
                for layer_name, acts_tuple in layer_outputs.items():
                    # acts_tuple[i] is a torch.Tensor for this sample's activation in the given layer
                    act = acts_tuple[0].cpu().numpy()
                    root.create_dataset(layer_name, data=act)

        self.iteration += 1
        for key in self.activations.keys():
            self.activations[key].clear()


def trajectory_activations(ofdft_cfg, out_path, save_every=10):
    """Extracts activations from the model during density optimization and saves them to
    out_path."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(device)

    torch.multiprocessing.set_sharing_strategy("file_system")

    model_config_path = Path(ofdft_cfg.run_path).parent.parent / "hparams.yaml"
    model_config = OmegaConf.load(model_config_path)
    transforms = instantiate(model_config.data.transforms)
    basis_info = instantiate(model_config.data.basis_info)
    add_grid = requires_grid(
        model_config.data.target_key,
        ofdft_cfg.denop_settings.negative_integrated_density_penalty_weight,
    )
    transforms.pre_transforms.insert(
        0, PrepareForDensityOptimization(basis_info, add_grid=add_grid)
    )
    transforms.add_transformation_matrix = True
    transforms.use_cached_data = False
    if ofdft_cfg.split_file_path is None:
        split_file_path = Path(model_config.data.datamodule.split_file)
    dataset_kwargs = instantiate(model_config.data.datamodule.dataset_kwargs)
    dataset_kwargs.pop("hdf5_manifest_path", None)
    dataset_kwargs.pop("denop_hdf5_manifest_path", None)
    dataset_kwargs.pop("hdf5_max_open_shards", None)
    dataset_kwargs.pop("label_path", None)
    dataset_kwargs.pop("kohn_sham_path", None)
    dataset_kwargs.pop("ground_state_coeffs", None)
    dataset_statistics_path = model_config.data.dataset_statistics.path
    if ofdft_cfg.denop_settings.initialization == "sad":
        dataset_statistics_path = dataset_statistics_path.replace(
            model_config.data.transforms.name, "no_basis_transforms"
        )

    # Prepare model / functional
    ckpt_path = ofdft_cfg.run_path
    model = MLDFTLitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(getattr(torch, ofdft_cfg.model_dtype.split(".")[1]))
    assert isinstance(model.net, Graphformer)  # Currently only Graphformer is supported
    activations = setup_activation_hooks_graphormer(model=model)

    if ofdft_cfg.denop_settings.initialization in ["sad", "sad_transformed"]:
        sad_guess_kwargs = dict(
            dataset_statistics=DatasetStatistics(dataset_statistics_path),
            normalization_mode=SADNormalizationMode.PER_ATOM_WEIGHTED,
            basis_info=basis_info,
            weigher_key="ground_state_only",
            spherical_average=True,
        )
    else:
        sad_guess_kwargs = {}

    func_factory = FunctionalFactory.from_module(
        model,
        ofdft_cfg.denop_settings.xc_functional,
        negative_integrated_density_penalty_weight=ofdft_cfg.denop_settings.negative_integrated_density_penalty_weight,
    )

    # Prepare dataset
    data_dir = get_mldft_data_path()
    with open(split_file_path, "rb") as f:
        split_dict = pickle.load(f)
    label_subdir = "labels"
    val_paths = [
        data_dir / dataset / label_subdir / label_path
        for dataset, label_path, _ in split_dict[ofdft_cfg.split]
    ]
    val_iterations = [scf_iterations for _, _, scf_iterations in split_dict[ofdft_cfg.split]]
    dataset_kwargs.update(
        {
            "limit_scf_iterations": -1,
            "additional_keys_at_ground_state": {
                "of_labels/energies/e_electron": Representation.SCALAR,
                "of_labels/energies/e_ext": Representation.SCALAR,
                "of_labels/energies/e_hartree": Representation.SCALAR,
                "of_labels/energies/e_kin": Representation.SCALAR,
                "of_labels/energies/e_kin_plus_xc": Representation.SCALAR,
                "of_labels/energies/e_kin_minus_apbe": Representation.SCALAR,
                "of_labels/energies/e_kinapbe": Representation.SCALAR,
                "of_labels/energies/e_xc": Representation.SCALAR,
                "of_labels/energies/e_tot": Representation.SCALAR,
            },
        }
    )
    dataset_indices = configure_dataset_indices(
        len(val_paths), ofdft_cfg.n_molecules, ofdft_cfg.molecule_choice, ofdft_cfg.seed
    )
    dataset = OFDataset(
        paths=[val_paths[j] for j in dataset_indices],
        num_scf_iterations_per_path=[val_iterations[j] for j in dataset_indices],
        basis_info=basis_info,
        transforms=transforms,
        **dataset_kwargs,
    )
    dataloader = OFLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
    )

    # Run density optimization
    callback = ActivationCallback(activations, out_path, save_every=save_every)

    sample = next(iter(dataloader))
    sample = sample.to_data_list()[0]
    sample.to(device)

    density_optimization(
        sample=sample,
        mol=sample.mol,
        optimizer=instantiate(ofdft_cfg.density_optimizer),
        func_factory=func_factory,
        callback=callback,
        initial_guess_str=ofdft_cfg.denop_settings.initialization,
        normalize_initial_guess=ofdft_cfg.denop_settings.normalize_initial_guess,
        proj_minao_module=model,
        sad_guess_kwargs=sad_guess_kwargs,
        convergence_criterion=ofdft_cfg.denop_settings.convergence_criterion,
        max_xc_memory=ofdft_cfg.denop_settings.max_xc_memory,
        disable_printing=True,
    )
