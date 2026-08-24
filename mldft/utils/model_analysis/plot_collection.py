import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def heading(text):
    """Creates a figure with a centered headline.

    Args:
        text (str): Headline text.
    """
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=40)
    ax.axis("off")
    return fig


class PlotCollection:
    """Class which contains a collection of PlotGenerator objects and creates a summary pdf
    containing all figures from the generators."""

    def __init__(self, path, generators, name, clear_files=True):
        """
        Args:
            path (str): Path to save the summary.
            generators (list): List of PlotGenerator objects.
        """
        self.path = os.path.join(path, name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.generators = generators
        self.clear_files = clear_files
        self.size = (11.69, 8.27)

    def run(self):
        """Runs all generators."""
        for generator in self.generators:
            if not generator.finished:
                generator.run(self.path)

    def is_ready(self) -> bool:
        """
        Returns:
            bool: True if all figures from the generators have been created.
        """
        return all(generator.finished for generator in self.generators)

    def create_pdf(self):
        """Creates a summary pdf containing all figures from the generators."""
        if not self.is_ready():
            self.run()

        with PdfPages(os.path.join(self.path, "summary.pdf")) as pdf:
            for generator in self.generators:
                # Create headline for each generator
                head = heading(generator.info())
                head.set_size_inches(*self.size)
                pdf.savefig(head)
                plt.close(head)

                # Save all figures in the current generator
                for figure in generator.figures:
                    figure.set_size_inches(*self.size)
                    pdf.savefig(figure)
                    plt.close(figure)

        if self.clear_files:
            # Clear all files and dirs except the summary.pdf
            for file in os.listdir(self.path):
                if file != "summary.pdf":
                    path = os.path.join(self.path, file)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
