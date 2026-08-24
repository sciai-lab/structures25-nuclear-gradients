import matplotlib.pyplot as plt
import numpy as np

from mldft.utils.model_analysis.templates.activations import get_activations
from mldft.utils.model_analysis.templates.template import Template


class TimeHistogram(Template):
    """Template for creating histograms of activations over density optimization steps."""

    def __init__(self, act_key: str, num_bins: int = 200):
        """Initialize the template.

        Args:
            act_key (str): Key of the activations to plot.
            num_bins (int): Number of bins for the histogram.
        """
        super().__init__()

        self.act_key = act_key
        self.num_bins = num_bins
        self.title = f"Histogram of {self.act_key} over time"

    def plot(self, paths: list, ax: plt.Axes, label: str = None):
        """Plot the histogram on the given axis."""
        activations = get_activations(paths, self.act_key)
        activations = np.concatenate(
            [np.stack(val, axis=0) for val in activations.values()], axis=-1
        )
        activations = activations.reshape(activations.shape[0], -1)
        # make bins more stable
        bins = np.linspace(
            -10 * np.std(np.abs(activations)), 10 * np.std(np.abs(activations)), self.num_bins
        )
        hist = np.zeros((activations.shape[0], bins.shape[0] - 1))
        for i, a in enumerate(activations):
            hist[i], _ = np.histogram(a, bins=bins)

        # log
        hist = np.log(hist + 1)

        # plot the histogram as an image
        ax.imshow(hist.T, aspect="auto", cmap="viridis", interpolation="none")
        ax.set_xlabel("iteration")
        ax.set_ylabel("activation")
        # set y-ticks
        yticks = np.linspace(0, hist.shape[1], 5)
        yticklabels = np.linspace(bins[0], bins[-1], 5)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{l:.2f}" for l in yticklabels])
        ax.set_title("Activation histogram over time")
        # add colorbar
        sm = plt.cm.ScalarMappable(
            cmap="viridis", norm=plt.Normalize(vmin=np.min(hist), vmax=np.max(hist))
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="count")
