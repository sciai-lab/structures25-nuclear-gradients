import numpy as np
from matplotlib import pyplot as plt


def plot_matrix(m, ax=None, title=None):
    """Plot a matrix."""
    ax = ax if ax is not None else plt.gca()
    vrange = np.max([-m.min(), m.max()])
    if title is not None:
        ax.set_title(title)
    return ax.imshow(m, cmap="seismic", vmin=-vrange, vmax=vrange)
