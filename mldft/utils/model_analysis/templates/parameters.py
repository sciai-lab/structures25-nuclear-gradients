import numpy as np
import pandas as pd

from mldft.utils.model_analysis.templates.template import Template, adaptive_fontsize


def get_params(params: dict, param_key: str | list[str]) -> dict:
    """Extract parameters from a dictionary of parameters. If param_key is 'all', all parameters
    are returned. If param_key starts with '*', all parameters containing the string are returned.

    Args:
        params (dict): Dictionary of parameters.
        param_key (str | list[str]): Key(s) of the parameters to extract. If param_key is 'all', all parameters are returned.
    """
    if param_key == "all":
        return params

    selected_params = {}

    if isinstance(param_key, str):
        param_key = [param_key]

    for key in param_key:
        if key[0] == "*":
            key_split = key[1:].split("**")
            for k in params.keys():
                # if every element of key_split is in k
                if all([s in k for s in key_split]):
                    selected_params[k] = params[k]
        else:
            selected_params[key] = params[key]

    return selected_params


class Histogram(Template):
    """Template for creating histograms of parameters.

    Parameters for all keys provided in the param_key list are concatenated and plotted in a single
    histogram.
    """

    def __init__(self, param_key: str | list[str], num_bins=10, overlay: bool = False):
        """Initialize the template.

        Args:
            param_key (str): Key of the parameters to plot.
            num_bins (int): Number of bins for the histogram.
            overlay (bool): If True, plot all histograms in the same figure.
        """
        super().__init__(overlay)
        self.param_key = param_key
        self.num_bins = num_bins
        self.title = f"Histogram of parameters in layer {self.param_key}"

    def plot(self, data, ax, label: str = None):
        """Plot the histogram on the given axis."""
        params = get_params(data, self.param_key)
        params = np.concatenate([val.flatten() for val in params.values()])

        if label is None:
            ax.hist(params, bins=self.num_bins)
        else:
            ax.hist(params, bins=self.num_bins, label=label, alpha=0.5)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")


class SortedWeights(Template):
    """Template for plotting the sorted weight vectors of given layers."""

    def __init__(self, param_key: str | list[str]):
        """Initialize the template."""
        super().__init__()
        self.param_key = param_key
        self.title = f"Sorted weights of parameters in layer {self.param_key}"

    def plot(self, data, ax, label: str = None):
        """Plot the sorted weights on the given axis."""
        params = get_params(data, self.param_key)
        for key, val in params.items():
            weight_vector = np.sort(val.flatten())
            ax.plot(weight_vector, label=key)
        ax.set_ylabel("weight")
        ax.set_xlabel("index")
        ax.set_yscale("symlog", linthresh=1e-5)
        ax.legend(fontsize=adaptive_fontsize(ax, "small"))
        ax.grid()


class NonZeroWeights(Template):
    """Template for a table listing the number of non-zero elements in the weight matrices of the
    given layers."""

    def __init__(self, param_key: str | list[str]):
        super().__init__()
        """Initialize the template."""
        self.param_key = param_key
        self.title = f"Number of non-zero elements in weight matrices of layer {self.param_key}"

    def plot(self, data, ax, label: str = None):
        """Plot the table on the given axis."""
        params = get_params(data, self.param_key)
        num_non_zero = []
        total = []
        for key, val in params.items():
            num_non_zero.append(
                str(np.round(np.sum(np.abs(val) > 1e-4) / val.flatten().shape[0] * 100, 2)) + " %"
            )
        df = pd.DataFrame(
            {"Layer": list(params.keys()), "Number of non-zero elements": num_non_zero}
        )
        ax.axis("off")
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(adaptive_fontsize(ax, "medium"))
        ax.set_title(self.title)
        ax.grid(False)
