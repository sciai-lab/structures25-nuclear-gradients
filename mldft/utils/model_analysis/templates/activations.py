import matplotlib.pyplot as plt
import numpy as np
import zarr

from mldft.utils.model_analysis.templates.template import Template


def get_activations(zarr_paths: str | list[str], key: str | list[str]) -> dict:
    r"""Read activations from zarr.zip file and return them as a dictionary. Assumes activations of
    the form (batch, \*shape).

    Args:
        zarr_path (str): Paths to zarr.zip files containing the activations.
        key (str | list[str]): Key(s) of the activations in the zarr file. If key starts with "*", all keys containing the string are returned.
        return_keys (bool): If True, the keys of the activations are returned as well.

    Returns:
        dict: Dictionary containing lists of length equal to the number of  zarr files. Each list contains tensors of shape (batch, \*shape).
    """
    activations = {}

    if isinstance(zarr_paths, str):
        zarr_paths = [zarr_paths]

    for zarr_path in zarr_paths:
        with zarr.ZipStore(zarr_path, mode="r") as store:
            zarr_group = zarr.group(store=store, overwrite=False)

            if isinstance(key, str):
                key = [key]

            for k in key:
                if k[0] == "*" or k == "all":
                    for zarr_key in zarr_group.keys():
                        if (k[1:] in zarr_key or k == "all") and not zarr_key == "error":
                            if zarr_key not in activations.keys():
                                activations[zarr_key] = []
                            activations[zarr_key].append(zarr_group[zarr_key][:])
                else:
                    if k not in activations.keys():
                        activations[k] = []
                    activations[k].append(zarr_group[k][:])

    return activations


class Histogram(Template):
    """Template for creating histograms of activations.

    Activations for all keys provided in the act_key list are concatenated and plotted in a single
    histogram.
    """

    def __init__(self, act_key: str, num_bins=10, overlay: bool = False):
        """Initialize the template.

        Args:
            act_key (str): Key of the activations to plot.
            num_bins (int): Number of bins for the histogram.
            overlay (bool): If True, plot all histograms in the same figure.
        """
        super().__init__(overlay=overlay)
        self.act_key = act_key
        self.num_bins = num_bins
        self.title = f"Activations for {self.act_key}"

    def plot(self, paths: list, ax: plt.Axes, label: str = None):
        """Plot the histogram of activations."""
        activations = get_activations(paths, self.act_key)
        activations = np.concatenate(
            [np.concatenate(val, axis=0).flatten() for val in activations.values()]
        )
        if label is None:
            ax.hist(activations, bins=self.num_bins, alpha=0.5)
        else:
            ax.hist(activations, bins=self.num_bins, alpha=0.5, label=label)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")


class LayerDistribution(Template):
    def __init__(self, act_key: str | list[str], agg: str = "none", style: str = "line"):
        """Template for plotting the distribution of activations across different layers.

        Args:
            act_key (str | list[str]): Key(s) of the activations to plot.
            agg: Aggregation function to apply to the activations. Can be 'sum', 'mean' or 'none'.
            style (str): Style of the plot. Can be 'line' or 'box'.
        """
        super().__init__()

        self.agg = agg
        self.act_key = act_key
        self.style = style
        if self.agg == "none":
            self.title = f"Per atom activations in layers: {self.act_key}"
        else:
            self.title = f"Aggregated activations in layers: {self.act_key}"

    def plot(self, paths: list, ax: plt.Axes, label: str = None):
        """Plot onto the given axis."""
        activations = get_activations(paths, self.act_key)

        vals = list(activations.values())
        keys = list(activations.keys())

        activations = []
        for i, d in enumerate(vals):
            if self.agg == "none":
                activations.append(np.concatenate(d, axis=0).flatten())
            elif self.agg == "sum":
                activations.append(np.stack([np.sum(e, axis=0) for e in d], axis=0))
            elif self.agg == "mean":
                activations.append(np.stack([np.mean(e, axis=0) for e in d], axis=0))
            else:
                raise ValueError(f"Unknown aggregation function: {self.agg}")

        if self.style == "box":
            for i, a in enumerate(activations):
                activations[i] = a.flatten()

            ax.boxplot(activations, whis=[5, 95], showfliers=False)

        elif self.style == "line":
            activations = np.stack(activations, axis=0)
            activations = activations.reshape(activations.shape[0], -1)
            ax.plot(np.arange(1, activations.shape[0] + 1), activations)
            ax.hlines(0, 0.75, activations.shape[0] + 0.25, colors="r", linestyles="dashed")
        else:
            raise ValueError(f"Unknown plot style: {self.style}")

        ax.set_xlabel("Layer")
        ax.set_xticks(np.arange(1, len(keys) + 1))
        ax.set_xticklabels(keys, rotation=50)


class AttentionDist(Template):
    """Template for plotting the distribution of attention weights as a function of distance from
    the diagonal.

    The act_key list should first contain the keys of the attention weights and the key containing
    the length of the sequence as last element.
    """

    def __init__(self, act_key: str, num_bins: int = 10, overlay: bool = False):
        super().__init__(overlay=overlay)
        self.act_key = act_key
        self.num_bins = num_bins
        self.title = "Attention vs. Distance"

    def plot(self, paths: list, ax: plt.Axes, label: str = None):
        """Plot onto the given axis."""
        activations = get_activations(paths, self.act_key)
        # TODO: Check if this concat makes things actually easier
        for k, val in activations.items():
            activations[k] = np.concatenate(val, axis=0)
        attentions = []
        length = None
        for key, val in activations.items():
            if "attention" in key:
                attentions.append(val)
            # act_key[-1] should be the key containing the length
            elif self.act_key[-1] in key:
                length = val
        attentions = np.concatenate(attentions, axis=1)

        # Flatten first two dimensions (merge sample and batch dimensions)
        attentions = attentions.reshape(-1, attentions.shape[-1])
        length = length.flatten()

        ax.set_xlabel("Distance")
        ax.set_ylabel("Attention")
        ax.set_yscale("log")
        ax.grid(True)

        # Bin edges and bin centers
        bin_edges = np.linspace(0, np.max(length), self.num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_indices = np.digitize(length, bin_edges, right=True)

        # Calculate statistics for attention scores in each bin
        means = []
        q25 = []
        q75 = []

        for k in range(1, self.num_bins + 1):
            bin_list = attentions[bin_indices.flatten() == k].flatten()
            if len(bin_list) > 0:
                means.append(np.median(bin_list))
                q25.append(np.percentile(bin_list, 25))
                q75.append(np.percentile(bin_list, 75))
            else:
                means.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)

        # Make scatter plot
        if label is None:
            ax.plot(bin_centers, means, marker="o", linestyle="-")
        else:
            ax.plot(bin_centers, means, marker="o", linestyle="-", label=label)
        ax.fill_between(bin_centers, q25, q75, alpha=0.3)
