# %%
# class cell
import os
import pickle
from typing import Dict, List

import numpy as np
import zarr
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from mldft.utils.cluster_train.precalced_kmeanspp import kmeans_plusplus_dm


class Activation_Kmeans:
    """Class to analyse the activations of a neural network."""

    def __init__(self, folder: str, aggregation_mode: str = "mean", n_activations: int = None):
        """Initializes the activation analyser with the folder containing the activations.

        :param folder: Path to the folder containing the activations
        :param aggregation_mode: Mode of aggregation, can be 'mean', 'sum', 'max', 'min', 'minmax'
        :param n_activations: Number of activations to load, if None all activations are loaded
        """
        self.folder = folder
        # check if folder exists
        if not os.path.exists(folder):
            raise ValueError(f"Folder {folder} does not exist")
        self.aggregation_mode = aggregation_mode
        self.activations = self.load_activations(n_activations)  # load the activations
        self.aggregated_activations = self._aggregate(self.activations, aggregation_mode)
        self.embeddings = None  # embeddings for the activations
        self.n_activations = self.get_n_activations()  # number of activations

    def load_activations(self, n_activations: int = None):
        """Function to load the activations from the folder."""
        print("loading activations...")
        activations = {}  # dict for activations
        self.keys = None  # list of keys
        i = 0  # counter for number of activations
        for root, dirs, files in os.walk(self.folder, topdown=True):  # loop over every file
            for file in tqdm(files):
                with zarr.ZipStore(root + "/" + file, mode="r") as zipstore:
                    dataroot = zarr.group(zipstore, overwrite=False)
                    if n_activations is not None and i >= n_activations:
                        print(f"loaded {i} activations, stopping loading")
                        return activations
                    if self.keys is None:  # grab keys if we don't have them
                        self.keys = dataroot.keys()

                    for key in dataroot.keys():
                        if key == "error":
                            continue

                        arr = dataroot[key][:]
                        if key not in activations:  # Check if the key exists
                            activations[key] = []  # Initialize as an empty list
                        activations[key].append(arr)  # append the array to the activations dict
                i += 1  # increment the counter
        print("loaded activations")
        return activations

    def _aggregate(self, activations, aggregation_mode=None):
        """Function to aggregate the activations of single edges/node(atoms).

        Args:
            activations: dict of activations
            aggregation_mode: mode of aggregation, can be mean
        Returns:
            activations: dict of aggregated activations(where possible)
        """
        self.filtered_keys = [
            key for key in self.keys if key not in ("error", "length", "energy_mlp", "atom_ind")
        ]
        # Filter the keys to exclude one dimensional arrays

        aggregated_activations = {}  # dict for aggregated activations

        if aggregation_mode is not None:
            self.aggregation_mode = aggregation_mode

        if self.aggregation_mode == "mean":
            for key in self.filtered_keys:
                aggregated_activations[key] = []
                for i in range(len(activations[key])):
                    aggregated_activations[key].append(
                        np.mean(activations[key][i], axis=0).flatten()
                    )
                aggregated_activations[key] = np.array(aggregated_activations[key])
        elif self.aggregation_mode == "sum":
            for key in self.filtered_keys:
                aggregated_activations[key] = []
                for i in range(len(activations[key])):
                    aggregated_activations[key].append(
                        np.sum(activations[key][i], axis=0).flatten()
                    )
                aggregated_activations[key] = np.array(aggregated_activations[key])
        elif self.aggregation_mode == "max":
            # function to aggregate the activations by taking the maximumm L2 norm
            for key in self.filtered_keys:
                aggregated_activations[key] = []
                for i in range(len(activations[key])):
                    index = np.argmax(np.linalg.norm(activations[key][i], axis=1), axis=0)
                    aggregated_activations[key].append(activations[key][i][index].flatten())
                aggregated_activations[key] = np.array(aggregated_activations[key])
        elif self.aggregation_mode == "min":
            # function to aggregate the activations by taking the minimum L2 norm
            for key in self.filtered_keys:
                aggregated_activations[key] = []
                for i in range(len(activations[key])):
                    index = np.argmin(np.linalg.norm(activations[key][i], axis=1), axis=0)
                    aggregated_activations[key].append(activations[key][i][index].flatten())
                aggregated_activations[key] = np.array(aggregated_activations[key])
        elif self.aggregation_mode == "minmax":
            # function to aggregate the activations by taking the minimum and maximum L2 norm
            for key in self.filtered_keys:
                aggregated_activations[key] = []
                for i in range(len(activations[key])):
                    index_min = np.argmin(np.linalg.norm(activations[key][i], axis=1), axis=0)
                    index_max = np.argmax(np.linalg.norm(activations[key][i], axis=1), axis=0)
                    aggregated_activations[key].append(
                        np.concatenate(
                            (activations[key][i][index_min], activations[key][i][index_max])
                        ).flatten()
                    )
                aggregated_activations[key] = np.array(aggregated_activations[key])
        else:
            raise ValueError(f"Aggregation mode {self.aggregation_mode} not recognized")
        return aggregated_activations

    def get_kmeanspp_init(
        self, k=0.5, keys: List[str] = None, random_state=42
    ) -> Dict[str, np.ndarray]:
        """
        function to get kmeans++ initialization
        Args:
            n_clusters: number of clusters
            k: fraction of points to use for initialization
        Returns:
            clusters: dict of clusters
        """
        if keys is None:
            keys = self.filtered_keys

        seed_ids = {}
        for key in keys:
            print(f"computing kmeans++ init for {key}")
            data = self.aggregated_activations[key]
            n_samples, _ = data.shape
            _, ids = kmeans_plusplus(
                data, n_clusters=int(n_samples * k), random_state=random_state
            )
            seed_ids[key] = ids
        print("computed kmeans++ init")
        return seed_ids

    def get_kmeanspp_init_dm(
        self, distance_matrix, k=0.5, keys: List[str] = None, random_state=42
    ) -> Dict[str, np.ndarray]:
        """
        function to get kmeans++ initialization
        Args:
            n_clusters: number of clusters
            k: fraction of points to use for initialization
        Returns:
            clusters: dict of clusters
        """
        if keys is None:
            keys = self.filtered_keys

        seed_ids = {}
        for key in keys:
            print(f"computing kmeans++ init for {key}")
            data = self.aggregated_activations[key]
            n_samples, _ = data.shape
            ids = kmeans_plusplus_dm(
                distance_matrix, n_clusters=int(n_samples * k), random_state=random_state
            )
            seed_ids[key] = ids
        print("computed kmeans++ init")
        return seed_ids

    def save_distance_matrix(
        self, path: str, distance_matrix: np.ndarray = None, key=None
    ) -> None:
        """
        function to save the distance matrix
        Args:
            path: path to save the distance matrix
            distance_matrix: distance matrix to save if None it will be calculated
        """

        if key is None:
            raise ValueError("key must be provided to calculate distance matrix")

        if distance_matrix is None:
            distance_matrix = self.calc_distance_matrix(key=key)

        # make directory if it does not exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path, "wb") as f:
            pickle.dump(distance_matrix, f)

        print(f"saved distance matrix to {path}")

    def get_moleculeID(self, ind: List[int]) -> List[str]:
        """
        function to get the molecule id for a given index
        Args:
            ind: index of the molecule
        Returns:
            molecule: list of molecules
        """

        molecule = []
        i = 0

        for root, dirs, files in os.walk(self.folder, topdown=True):  # loop over every file
            for file in files:
                if i in ind:
                    molecule.append(file)
                i += 1

        return molecule

    def calc_distance_matrix(self, key: str) -> np.ndarray:
        """
        function to calculate the distance matrix for a given key
        Args:
            key: key of the activations
        Returns:
            distance_matrix: distance matrix
        """
        print(f"calculating distance matrix for {key}")
        X_squared_norms = np.sum(self.aggregated_activations[key] ** 2, axis=1)
        distance_matrix = euclidean_distances(
            self.aggregated_activations[key],
            self.aggregated_activations[key],
            Y_norm_squared=X_squared_norms,
            squared=True,
        )
        print("calculated distance matrix")
        return distance_matrix

    def get_n_activations(self) -> int:
        """
        function to get the number of activations
        Args:
            None
        Returns:
            n_activations: number of activations
        """
        key = self.filtered_keys[0]
        n_activations = len(self.activations[key])
        return n_activations
