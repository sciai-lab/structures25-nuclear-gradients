import os
import traceback

import hydra
import hydra.experimental
import omegaconf
from hydra.core.global_hydra import GlobalHydra

from mldft.ml.models.mldft_module import MLDFTLitModule
from mldft.utils import RankedLogger
from mldft.utils.model_analysis.extract_activations import (
    extract_activations,
    trajectory_activations,
)


def parse_model_name(model_path):
    """Parse the model name from the model path."""
    try:
        name = model_path.split("/")[-3]
    except Exception:
        name = model_path
    return name


log = RankedLogger(__name__, rank_zero_only=True)


class PlotGenerator:
    """Base class for plot generators.

    Plot generators fetch the necessary data (e.g. activations, parameters) from given models and
    create figures based on given templates.
    """

    def __init__(self):
        """Initialize the plot generator."""

        self.figures = []
        self.finished = False
        self.templates = []

    def create_figures(self, data: list, labels: list):
        """Create figures based on the given templates."""

        for template in self.templates:
            try:
                self.figures.append(template.create(data, labels))
            except Exception as e:
                log.info(f"Failed to create figure {type(template)}: {e}")
                traceback.print_exc(limit=5)
                continue

    def run(self, *args, **kwargs):
        """Fetch the data and create the plots."""

    def info(self):
        """Return infostring which can be used as heading by the PlotCollection class."""

        return "PlotGenerator"


class Activations(PlotGenerator):
    """Fetches activations from a list of models for a given set of data samples."""

    def __init__(self, model_paths, num_samples, templates, random_samples=False):
        """
        Args:
            model_paths (list): List of paths to the models.
            num_samples (int): Number of data samples to fetch the activations from.
            templates (list): List of templates for the figures.
            random_samples (bool): If True, fetch random samples. Otherwise, fetch the first num_samples samples.
        """
        super().__init__()
        if not (
            isinstance(model_paths, list)
            or isinstance(model_paths, omegaconf.listconfig.ListConfig)
        ):
            model_paths = [model_paths]
        self.model_paths = model_paths
        self.num_samples = num_samples
        self.templates = templates
        self.act_paths = []
        self.random_samples = random_samples

    def run(self, out_path, *args, **kwargs):
        """Fetch the activations and create the figures."""

        log.info("Fetching activations")
        for i, model_path in enumerate(self.model_paths):
            self.act_paths.append(os.path.join(out_path, f"activations_model_{i}"))
            if not os.path.exists(self.act_paths[-1]):
                os.makedirs(self.act_paths[-1])
            else:
                # Clean directory
                for file in os.listdir(self.act_paths[-1]):
                    os.remove(os.path.join(self.act_paths[-1], file))

            extract_activations(
                model_path,
                num_samples=self.num_samples,
                out_path=self.act_paths[-1],
                random=self.random_samples,
            )

        log.info("Creating activation figures")

        act_paths = [
            [os.path.join(act_path, file) for file in os.listdir(act_path)]
            for act_path in self.act_paths
        ]
        labels = [parse_model_name(m) for m in self.model_paths]
        self.create_figures(act_paths, labels)

        self.finished = True

    def info(self):
        """Return infostring which can be used as heading by the PlotCollection class."""

        return "Activations"


class Trajectories(PlotGenerator):
    """Fetches activations from a set of models, extracted during density optimization for a given
    set of molecules."""

    def __init__(
        self,
        model_paths,
        templates=None,
        ofdft_cfg_path="density_optimization.yaml",
        mol_idc=[0],
        save_every=10,
    ):
        """
        Args:
            model_paths (list): List of paths to the models.
            templates (list): List of templates for the figures.
            ofdft_cfg_path (str): Path to the OFDFT configuration file used for denop.
            mol_idc (list): List of molecule indices.
            save_every (int): Save activations every n steps during denop.
        """
        super().__init__()
        if templates is None:
            templates = []

        if not (
            isinstance(model_paths, list)
            or isinstance(model_paths, omegaconf.listconfig.ListConfig)
        ):
            model_paths = [model_paths]
        self.model_paths = model_paths
        self.ofdft_cfg_path = ofdft_cfg_path
        if not (isinstance(mol_idc, list) or isinstance(mol_idc, omegaconf.listconfig.ListConfig)):
            mol_idc = [mol_idc]
        self.mol_idc = mol_idc
        self.templates = templates
        self.act_paths = []
        self.save_every = save_every

    def run(self, out_path, *args, **kwargs):
        """Fetch the trajectory activations and create the figures."""

        log.info("Fetching trajectory activations")

        for i, model_path in enumerate(self.model_paths):
            for j, mol_id in enumerate(self.mol_idc):
                act_path = os.path.join(out_path, f"trajectories_model_{i}/mol_{mol_id}")
                self.act_paths.append(act_path)
                if not os.path.exists(act_path):
                    os.makedirs(act_path)
                else:
                    # Clean directory
                    for file in os.listdir(act_path):
                        os.remove(os.path.join(act_path, file))

                # TODO: Not ideal. Try adding the right searchpath in hydra config
                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()

                # Initialize the Hydra context with the config path
                with hydra.initialize(config_path="../../../configs/ofdft"):
                    ofdft_cfg = hydra.compose(config_name=self.ofdft_cfg_path)

                omegaconf.OmegaConf.set_struct(ofdft_cfg, False)
                ofdft_cfg.run_path = model_path
                ofdft_cfg.molecule_choice = [mol_id]
                omegaconf.OmegaConf.set_struct(ofdft_cfg, True)
                trajectory_activations(ofdft_cfg, act_path, self.save_every)

        log.info("Creating trajectory figures")

        act_paths = [
            [os.path.join(act_path, file) for file in os.listdir(act_path)]
            for act_path in self.act_paths
        ]
        labels = [parse_model_name(m) for m in self.model_paths]
        self.create_figures(act_paths, labels)

        self.finished = True

    def info(self):
        """Return infostring which can be used as heading by the PlotCollection class."""

        return "Trajectories"


class Parameters(PlotGenerator):
    """Fetches the parameters of a set of models."""

    def __init__(self, model_paths, templates):
        """
        Args:
            model_paths (list): List of paths to the models.
            templates (list): List of templates for the figures.
        """
        super().__init__()

        self.model_paths = model_paths
        self.templates = templates
        self.params = []

    def run(self, *args, **kwargs):
        """Fetch the parameters and create the figures."""
        log.info("Fetching model parameters")

        for model_path in self.model_paths:
            model = MLDFTLitModule.load_from_checkpoint(model_path)
            self.params.append({})
            for name, param in model.named_parameters():
                # Detach to avoid graph references, then move to CPU if on GPU
                self.params[-1][name] = param.detach().cpu().numpy()

        log.info("Creating parameter figures")

        labels = [parse_model_name(m) for m in self.model_paths]
        self.create_figures(self.params, labels)

        self.finished = True

    def info(self):
        """Return infostring which can be used as heading by the PlotCollection class."""

        return "Parameters"
