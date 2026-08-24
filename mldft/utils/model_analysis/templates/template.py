import math
from abc import abstractmethod

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def adaptive_fontsize(fig: Figure | Axes, size: str = "small"):
    """Adapt the fontsize used for parts of the figure based on the number of subplots."""

    if isinstance(fig, Axes):
        fig = fig.get_figure()

    # Calculate saling factor based on number of axes in the figure
    n_axes = len(fig.get_axes())
    # Corresponds to number of columns
    scaling_factor = math.ceil(math.sqrt(n_axes))

    sizes = {"small": 16, "medium": 20, "large": 24}

    return sizes[size] / scaling_factor


def format_title(title: str, max_char: int = 50):
    """Format title such that it does not exceed max_char characters."""
    return title if len(title) <= max_char else "..." + title[-max_char:]


class Template:
    """Abstract class for creating plots of activations or parameters."""

    def __init__(self, overlay: bool = False):
        """
        Args:
            overlay (bool): If True, plots for different datasets are overlaid in the same figure.
        """
        self.overlay = overlay
        self.title = "Template"

    @abstractmethod
    def plot(self, data: list | dict, ax: plt.Axes):
        """Plot activations or parameters in ax.

        Args:
            data (dict | list): Dictionary (List) containing data (paths)
            ax (plt.Axes): Axis to plot in.
        """
        raise NotImplementedError

    def create(self, data: list[dict], labels: list[str] = None) -> Figure:
        """Create a figure with plots of activations or parameters.

        Args:
            data (list[dict]): List of dictionaries containing the activations or parameters.
            labels (list[str]): List of labels for the datasets.
        """

        if self.overlay or len(data) == 1:
            fig, ax = plt.subplots(1, 1)

            for i, d in enumerate(data):
                if labels is not None:
                    self.plot(d, ax, labels[i])
                    ax.legend()
                else:
                    self.plot(d, ax)

        else:
            # Create a grid of subplots such that one obtains a grid with approximately equal number of rows and columns
            cols = math.ceil(math.sqrt(len(data)))
            rows = math.ceil(len(data) / cols)

            fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)

            # Flatten axes array for easy iteration
            axes = axes.flatten()

            for i, d in enumerate(data):
                self.plot(d, axes[i])
                if labels is not None:
                    axes[i].set_title(
                        format_title(labels[i]), fontsize=adaptive_fontsize(axes[i], "medium")
                    )

            # Hide any unused subplots
            for j in range(len(data), len(axes)):
                fig.delaxes(axes[j])

        fig.suptitle(self.title, fontsize=20)
        plt.tight_layout()

        return fig
