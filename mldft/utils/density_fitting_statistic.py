import multiprocessing
import os
from functools import reduce
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

import hydra
import numpy as np
import regex
import torch
import zarr
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pyscf import dft
from tqdm import tqdm

from mldft.datagen.methods.density_fitting import ksdft_density_matrix
from mldft.ml.data.components.basis_info import BasisInfo
from mldft.ml.data.components.of_data import OFData
from mldft.ofdft.basis_integrals import (
    get_coulomb_matrix,
    get_coulomb_tensor,
    get_normalization_vector,
    get_overlap_matrix,
    get_overlap_tensor,
)
from mldft.utils.grids import grid_setup
from mldft.utils.molecules import (
    build_molecule_np,
    build_molecule_ofdata,
    chem_formula_from_atomic_numbers,
    load_scf,
)
from mldft.utils.multiprocess import configure_processes_and_threads


def load_mol_data(args):
    """Helper function to load molecule data in parallel."""
    file_name, mol_number, basis_info = args
    return (
        mol_number,
        OFData.from_file(
            file_name,
            -1,
            basis_info,
        ),
    )


class DensityFittingStatistic:
    """Class to compute and store statistics for the accuracy of different density fitting methods
    and OF-basis-functions.

    And to approximate the error originating from density fitting in OFDFT
    """

    def __init__(
        self,
        root_of_dataset: str,
        data_set_type: str = "QM9",
        dataset_postfix="",
        names_of_metrics=None,
        save_folder_name="statistic",
        n_mols=None,
        num_processes=1,
    ):
        """Creates a DensityFittingStatistic object which handes the sorting and loading of the
        statistic files for multiple datasets.

        Args:
            root_of_dataset (str): Path to the root directory containing the dataset directories.
            data_set_type (str): Type of dataset to process (e.g., "MD17", "QM9").
            dataset_postfix (str): Postfix for the dataset names.
            names_of_metrics (List[str]): Names of metrics to load statistics for. If None, default metrics are used.
            save_folder_name (str): Name of the folders where to look for the statistics files.
            n_mols (int): Number of molecules to process, if None all molecules are processed.
            num_processes (int): Number of parallel processes to use for loading.
        """
        self.root_path = Path(root_of_dataset)
        self.data_set_type = data_set_type
        self.postfix = dataset_postfix
        self.n_mols = n_mols
        self.num_processes = num_processes
        if names_of_metrics is None:
            self.names_of_metrics: List[str] = [
                "AE_hartree_energy",
                "AE_xc_energy",
                "AE_ext_energy",
                "AE_kin_energy",
                "AE_hartree_ext_energy",
                "AE_density",
                "L2_residual_densities",
                "L1_residual_densities",
                "L1_negative_densities",
                "L2_residual_hartree",
                "L2_residual_hartree_ext",
                "max_abs_gradient",
                "N_basis_functions",
            ]
        else:
            self.names_of_metrics = names_of_metrics
        self.search_pattern = "labels"
        self.search_pattern_kohn_sham = "kohn_sham"
        self.save_folder_name = save_folder_name

    def init_from_root_dir(self):
        """Initializes from directory which includes dataset directories It assumes that this
        directory includes a "kohn_sham" file with the .chkfiles and labels directories with the
        of-labels as well as statistic directories with the result of the statistic
        calculations."""
        if not os.path.isdir(self.root_path):
            raise FileNotFoundError(f"{self.root_path} was not found.")

        data_set_names = [
            name
            for name in os.listdir(self.root_path)
            if regex.search(f"^{self.data_set_type}{self.postfix}", name)
        ]
        self.data_set_paths = [
            join(self.root_path, data_set_name)
            for data_set_name in data_set_names
            if os.path.isdir(join(self.root_path, data_set_name))
        ]
        directories = [
            [name for name in os.listdir(data_set_path)] for data_set_path in self.data_set_paths
        ]
        ks_paths = [
            join(data_set_path, self.search_pattern_kohn_sham)
            for data_set_path in self.data_set_paths
        ]
        if self.data_set_type == "MD17":
            self.kohn_sham_dir_paths = ks_paths
            self.label_dir_paths = [
                [
                    join(path_name, sub_dir)
                    for sub_dir in directory
                    if self.search_pattern in sub_dir
                ]
                for directory, path_name in zip(directories, self.data_set_paths)
            ]
            self.statistic_dir_paths = [
                [
                    join(path_name, sub_dir)
                    for sub_dir in directory
                    if self.save_folder_name in sub_dir
                ]
                for directory, path_name in zip(directories, self.data_set_paths)
            ]
            self.atomic_numbers = [1, 6, 8]
        elif self.data_set_type == "Cumulene":
            self.kohn_sham_dir_paths = ks_paths
            self.label_dir_paths = [
                [
                    join(path_name, sub_dir)
                    for sub_dir in directory
                    if self.search_pattern in sub_dir
                ]
                for directory, path_name in zip(directories, self.data_set_paths)
            ]
            self.statistic_dir_paths = [
                [
                    join(path_name, sub_dir)
                    for sub_dir in directory
                    if self.save_folder_name in sub_dir
                ]
                for directory, path_name in zip(directories, self.data_set_paths)
            ]
            self.atomic_numbers = [1, 6]
        elif self.data_set_type == "QM9" or self.data_set_type == "QMUGS":
            # TODO change this
            # sub_folders = os.listdir(os.path.join(self.root_path, "kohn_sham"))
            sub_sub_dir_name = "dsgdb9nsd.xyz"
            self.kohn_sham_dir_paths = [
                (
                    join(ks_path, sub_sub_dir_name)
                    if os.path.isdir(join(ks_path, sub_sub_dir_name))
                    else ks_path
                )
                for ks_path in ks_paths
            ]
            self.label_dir_paths = [
                [
                    (
                        join(path_name, self.search_pattern, sub_sub_dir_name)
                        if os.path.isdir(join(path_name, self.search_pattern, sub_sub_dir_name))
                        else join(path_name, self.search_pattern)
                    )
                    # join(path_name, sub_dir, sub_sub_dir_name) if os.path.isdir(join(path_name, sub_dir, sub_sub_dir_name)) else join(path_name, sub_dir)
                    # for sub_dir in directory
                    # if self.search_pattern in sub_dir
                ]
                for directory, path_name in zip(directories, self.data_set_paths)
            ]
            self.statistic_dir_paths = [
                (
                    join(path_name, self.save_folder_name, sub_sub_dir_name)
                    if os.path.isdir(join(path_name, self.save_folder_name, sub_sub_dir_name))
                    else join(path_name, self.save_folder_name)
                )
                for path_name in self.data_set_paths
            ]
            self.atomic_numbers = [1, 6, 7, 8, 9]
        else:
            raise ValueError(f"Data set type {self.data_set_type} not supported.")

        # sort so the order of elements in both lists is the same
        self.label_dirs = []
        self.statistic_dirs = []
        self.kohn_sham_dirs = []
        self.data_set_paths_final = []
        self.config_file_paths = []
        for data_set_path, label_dir, statistic_dir, kohn_sham_dir in zip(
            self.data_set_paths,
            self.label_dir_paths,
            self.statistic_dir_paths,
            self.kohn_sham_dir_paths,
        ):
            if not os.path.isdir(kohn_sham_dir):
                logger.warning(f"Kohn Sham Directory {kohn_sham_dir} was not found.")
                continue
            if not os.path.isdir(statistic_dir):
                logger.warning(f"Statistics Directory {statistic_dir} was not found.")
                continue
            for label_sub_dir in label_dir:
                dataset_info_file = (
                    join(data_set_path, "dataset_info.yaml")
                    if os.path.isfile(join(data_set_path, "dataset_info.yaml"))
                    else join(data_set_path, label_sub_dir, "dataset_info.yaml")
                )
                if not os.path.isfile(dataset_info_file):
                    logger.warning(f"Dataset info file {label_sub_dir} was not found.")
                    continue
                self.config_file_paths.append(dataset_info_file)
                self.label_dirs.append(Path(label_sub_dir))
                self.kohn_sham_dirs.append(Path(kohn_sham_dir))

                self.statistic_dirs.append(Path(statistic_dir))
                self.data_set_paths_final.append(data_set_path)
        self.setup_data_structure()

    def from_label_dirs(
        cls,
        kohn_sham_dir,
        label_dirs: Path | str | List[str] | List[Path],
        names_of_metrics: List[str] = None,
    ):
        """Initializes the class just using the directories that are specified here.

        Args:
            kohn_sham_dir (Path): Directory containing the kohn_sham file
            label_dirs (Path): Directory containing the labels file
            names_of_metrics (List[str]): Names of metrics to calculate statistics for

        Returns:
            cls (object): Initialized class
        """
        if type(label_dirs) is not list:
            label_dirs = [label_dirs]
        dfs = cls(Path(""), names_of_metrics)
        dfs.kohn_sham_dir = kohn_sham_dir
        dfs.label_dirs = label_dirs
        return dfs

    def setup_data_structure(self):
        """Loads config files and sets up the data structure."""
        self.config_files = [
            OmegaConf.load(config_file_path) for config_file_path in self.config_file_paths
        ]

        self.basis_info_list = [
            BasisInfo.from_dataset_info_yaml(config_file_path, self.atomic_numbers)
            for config_file_path in self.config_file_paths
        ]
        self.density_fitting_methods = [
            conf["density_fitting_method"] for conf in self.config_files
        ]
        self.basis_sets = [conf["of_basis_set"] for conf in self.config_files]

        self.molecules_statistic_names = [
            os.listdir(stat_dir) if stat_dir is not None else []
            for stat_dir in self.statistic_dirs
        ]
        for lst in self.molecules_statistic_names:
            lst.sort()
        self.mol_numbers = [
            np.asarray(
                [
                    int(name.split("_")[-1].split(".")[0])
                    for name in self.molecules_statistic_names[i]
                ],
                dtype=np.int32,
            )
            for i in range(len(self.molecules_statistic_names))
        ]
        self.unique_mol_numbers = reduce(np.intersect1d, self.mol_numbers)

        self.mol_number_to_names_dicts = [
            {num: name for num, name in zip(numbers, names)}
            for numbers, names in zip(self.mol_numbers, self.molecules_statistic_names)
        ]
        self.sorted_molecules_statistic_names = [
            [mol_number_to_names_dict[num] for num in self.unique_mol_numbers]
            for mol_number_to_names_dict in self.mol_number_to_names_dicts
        ]
        if self.n_mols is not None:
            self.unique_mol_numbers = self.unique_mol_numbers[: self.n_mols]
        self.unique_density_fitting_methods = list(set(self.density_fitting_methods))
        self.unique_density_fitting_methods.sort()
        self.unique_basis_sets = list(set(self.basis_sets))
        self.unique_basis_sets.sort()

        self.of_data_files_names = [
            [
                file
                for file in os.listdir(dir)
                if isfile(join(dir, file)) and not regex.search("yaml", file)
            ]
            for dir in self.label_dirs
        ]
        self.of_data_files_numbers = [
            np.asarray([int(name.split("_")[-1].split(".")[0]) for name in lst])
            for lst in self.of_data_files_names
        ]
        self.mol_data = []

        args_list = []
        for i, number in tqdm(enumerate(self.unique_mol_numbers)):
            for j, dir in enumerate(self.label_dirs):
                if number in self.of_data_files_numbers[j]:
                    index = np.where(np.asarray(self.of_data_files_numbers[j]) == number)[0][0]
                    args_list.append(
                        (
                            join(dir, self.of_data_files_names[j][index]),
                            self.of_data_files_numbers[j][index],
                            self.basis_info_list[j],
                        )
                    )
                    break

        logger.info(f"Starting loading molecule data with {self.num_processes} processes")
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # Using imap_unordered instead of starmap_async
            mol_list = list(
                tqdm(
                    pool.imap_unordered(load_mol_data, args_list),
                    total=len(self.unique_mol_numbers),
                    position=0,
                    smoothing=0,
                    dynamic_ncols=True,
                )
            )
        mol_dict = {pair[0]: pair[1] for pair in mol_list}
        self.mol_data = [mol_dict[num] for num in self.unique_mol_numbers]

        self.unique_mol_numbers_to_index = {
            unique_mol_number: i for i, unique_mol_number in enumerate(self.unique_mol_numbers)
        }
        self.mol_atomic_numbers = [data.atomic_numbers for data in self.mol_data]
        self.mol_n_atoms = np.asarray(
            [atomic_numbers.shape[0] for atomic_numbers in self.mol_atomic_numbers]
        )
        self.mol_n_basis_functios = np.asarray(
            [data.ground_state_coeffs.shape[0] for data in self.mol_data]
        )
        self.mol_elektrons = np.asarray(
            [np.sum(atomic_numbers) for atomic_numbers in self.mol_atomic_numbers]
        )
        self.mol_n_heavy_atoms = np.asarray(
            [np.sum(atomic_numbers > 1) for atomic_numbers in self.mol_atomic_numbers]
        )
        self.mol_chem_formula = [
            chem_formula_from_atomic_numbers(atomic_numbers)
            for atomic_numbers in self.mol_atomic_numbers
        ]
        logger.info(
            f"Found {len(self.label_dirs)} directories with {len(self.unique_mol_numbers)} molecules"
        )

    def get_mol_dictionary(self, path: Path | str):
        """Get a dictionary which maps the mol number of every molecule-data-file found in the
        given directory to the name of its file.

        Args:
            path (Path): Directory containing files that contain the mol number in their file name

        Returns:
            dict: Dictionary {mol_number:int -> file_name:str}
        """
        if path == "":
            return None
        return {
            int(file.split("_")[-1].split(".")[0]): file
            for file in os.listdir(path)
            if isfile(join(path, file)) and not regex.search("yaml", file)
        }

    def load_statistics(self):
        """Load the statistic files and store them in self.metrics_of_dataset."""
        self.metrics_of_dataset = []
        for i, dir in enumerate(self.statistic_dirs):
            metric_data = []
            if len(self.molecules_statistic_names[i]) < 1:
                self.metrics_of_dataset.append([])
                logger.warning("No Datasets found")
                continue
            with zarr.open(dir / self.molecules_statistic_names[i][0], mode="r") as root:
                found_all_metrics = True
                for name in self.names_of_metrics:
                    if name not in root:
                        logger.warning(
                            f"Metric {name} was not found in dataset {dir}. Skipping this directory for now"
                        )
                        found_all_metrics = False
                        break
                if not found_all_metrics:
                    continue
                for file_name in tqdm(self.molecules_statistic_names[i]):
                    with zarr.open(dir / file_name, mode="r") as root:
                        metric_dict = {}
                        for name in self.names_of_metrics:
                            metric_dict[name] = np.asarray(root[name])
                        metric_data.append(metric_dict)
                self.metrics_of_dataset.append(metric_data)

    def get_metrics_at_index(self, metric: str, index=-1) -> np.ndarray:
        """Returns an array of all the metric values.

        Args:
            metric: The name of the metric to return
            index: the index of iteration to return

        Returns:
            Array of shape (n_datasets,n_unique_mol_numbers)
        Note:
            index = 0 is initial guess
            index = -1 is the ground state
        """

        if not hasattr(self, "metrics_of_dataset"):
            self.load_statistics()

        data = np.zeros((len(self.statistic_dirs), len(self.unique_mol_numbers)))
        data[:] = np.nan
        for i, dir in enumerate(self.statistic_dirs):
            k = 0  # current_index
            for j, num in tqdm(enumerate(self.mol_numbers[i])):
                if num in self.unique_mol_numbers:
                    data[i, np.argwhere(self.unique_mol_numbers == num)] = self.metrics_of_dataset[
                        i
                    ][j][metric][index]
        return data

    def get_all_metrics_at_index(self, index=-1):
        """Returns an array of all the metric values.

        Args:
            metric: The name of the metric to return
            index: the index of iteration to return

        Returns:
            Array of shape (n_datasets,n_unique_mol_numbers,n_metrics)
        Note:
            index = 0 is initial guess
            index = -1 is the ground state
        """
        if not hasattr(self, "metrics_of_dataset"):
            self.load_statistics()

        data = np.zeros(
            (len(self.statistic_dirs), len(self.unique_mol_numbers), len(self.names_of_metrics))
        )
        data[:] = np.nan
        for i, dir in enumerate(self.statistic_dirs):
            # k = 0  # current_index
            for j, num in tqdm(enumerate(self.mol_numbers[i])):
                if num in self.unique_mol_numbers:
                    for m, metric in enumerate(self.names_of_metrics):
                        data[i, np.argwhere(self.unique_mol_numbers == num), m] = (
                            self.metrics_of_dataset[i][j][metric][index]
                        )
        return data

    def get_all_metrics_at_index_unsorted(self, index=-1):
        """Returns an array of all the metric values.

        Args:
            metric: The name of the metric to return
            index: the index of iteration to return

        Returns:
            Array of shape (n_datasets,n_unique_mol_numbers,n_metrics)
        Note:
            index = 0 is initial guess
            index = -1 is the ground state
        """
        if not hasattr(self, "metrics_of_dataset"):
            self.load_statistics()

        data = (
            np.zeros(
                (
                    len(self.statistic_dirs),
                    max([len(dir) for dir in self.mol_numbers]),
                    len(self.names_of_metrics),
                )
            )
            * np.nan
        )
        data[:] = np.nan
        for i, dir in enumerate(self.statistic_dirs):
            k = 0  # current_index
            for j, num in tqdm(enumerate(self.mol_numbers[i])):
                for m, metric in enumerate(self.names_of_metrics):
                    data[i, j, m] = self.metrics_of_dataset[i][j][metric][index]
        return data

    def get_all_metrics_at_every_step(self):
        """Returns an array of all the metric values.

        Args:
            metric: The name of the metric to return
            index: the index of iteration to return

        Returns:
            Array of shape (n_datasets,n_unique_mol_numbers,n_metrics)
        Note:
            index = 0 is initial guess
            index = -1 is the ground state
        """
        if not hasattr(self, "metrics_of_dataset"):
            self.load_statistics()
        self.cycles = []
        for i, dir in enumerate(self.statistic_dirs):
            cycles_dir = []
            for j, num in tqdm(enumerate(self.mol_numbers[i])):
                cycle = self.metrics_of_dataset[i][j][self.names_of_metrics[0]].shape[0]
                cycles_dir.append(cycle)
            self.cycles.append(np.asarray(cycles_dir, dtype=np.int32))

        max_cycle = np.max(np.concatenate(self.cycles))

        data = np.zeros(
            (
                len(self.statistic_dirs),
                len(self.unique_mol_numbers),
                len(self.names_of_metrics),
                max_cycle,
            )
        )
        data[:] = np.nan
        for i, dir in enumerate(self.statistic_dirs):
            k = 0  # current_index
            for j, num in enumerate(self.mol_numbers[i]):
                while k < len(self.unique_mol_numbers):
                    if self.unique_mol_numbers[k] == num:
                        for m, metric in enumerate(self.names_of_metrics):
                            data[i, k, m, : self.cycles[i][j]] = self.metrics_of_dataset[i][j][
                                metric
                            ][:]
                        k += 1
                        break
                    k += 1
        return data

    def sort_by_dataset_and_density_fitting_method(
        self, arr: np.ndarray, return_dicts: bool = False
    ):
        """Reshapes an array of the form (n_directories,:) into one of the form
        (n_unique_basis_sets,n_unique_df_methods,:)

        Args:
            arr: the array to be sorted
            return_dicts: Whether to return dictionaries mapping df_names and basis_set_names to indices

        Returns:
            The reshaped and sorted array
        """
        new_array = np.zeros(
            [len(self.unique_basis_sets), len(self.unique_density_fitting_methods)]
            + list(np.asarray(arr.shape)[1:]),
            dtype=arr.dtype,
        )
        if arr.dtype in (np.float64, np.float32, np.double, np.floating):
            new_array[:] = np.nan
        basis_set_name_to_num = {
            basis_set_name: i for i, basis_set_name in enumerate(self.unique_basis_sets)
        }
        df_method_name_to_num = {
            df_method_name: i
            for i, df_method_name in enumerate(self.unique_density_fitting_methods)
        }
        for i in range(len(self.statistic_dirs)):
            new_array[
                basis_set_name_to_num[self.basis_sets[i]],
                df_method_name_to_num[self.density_fitting_methods[i]],
            ] = arr[i]
        if return_dicts:
            return new_array, basis_set_name_to_num, df_method_name_to_num
        return new_array

    def get_of_data_file_by_dataset_and_density_fitting_method(self, return_dicts=False):
        """Reshapes the array of labels directories from the form (n_directories,:) into one of the
        form (n_unique_basis_sets,n_unique_df_methods,:)

        Args:
            arr: the array to be sorted
            return_dicts: Whether to return dictionaries mapping df_names and basis_set_names to indices

        Returns:
            The reshaped and sorted array
        """
        new_array = np.empty(
            [len(self.unique_basis_sets), len(self.unique_density_fitting_methods)], dtype="S200"
        )
        new_array[:] = ""
        basis_set_name_to_num = {
            basis_set_name: i for i, basis_set_name in enumerate(self.unique_basis_sets)
        }
        df_method_name_to_num = {
            df_method_name: i
            for i, df_method_name in enumerate(self.unique_density_fitting_methods)
        }
        for i in range(len(self.statistic_dirs)):
            new_array[
                basis_set_name_to_num[self.basis_sets[i]],
                df_method_name_to_num[self.density_fitting_methods[i]],
            ] = self.label_dirs[i]
        if return_dicts:
            return new_array, basis_set_name_to_num, df_method_name_to_num
        return new_array

    def create_statistics_for_all_datasets(
        self,
        root_path: Path | str,
        dataset_type="MD17",
        parallel=True,
        num_processes=None,
        override=False,
        search_pattern="^labels$",
        kohn_sham_folder="kohn_sham",
        postfix="",
        save_folder_name="statistic",
        n_molecules=None,
        start_index=None,
        iterations=None,
    ):
        """Calculates statistics for all datasets that are in subfolders of the given root path.

        Args:
            root_path: the root directory containing the datasets
            dataset_type: the type of dataset to calculate statistics for(MD17 and QM9 supported)
            parallel: Whether to parallelize the calculation over multiple cores
            num_processes: Number of parallel processes to use if parallel = True
            override: Whether to overwrite existing statistic files
            search_pattern: the pattern used to search the labels directory
            kohn_sham_folder: the name of the folder containing the kohn_sham files
            postfix: a postfix for the dataset names,
            save_folder_name: name of the folder where the statistics are saved
            n_molecules: Number of molecules to process, if None all molecules are processed.
            start_index: Index to start processing from, if None starts from 0.
            iterations: List of iterations to process, if None processes all iterations.
        """
        data_set_names = [
            name
            for name in os.listdir(self.root_path)
            if regex.search(f"^{dataset_type}{postfix}", name)
        ]

        for data_set_name in data_set_names:
            logger.info(f"Creating statistics for Dataset: {data_set_name}")
            self.create_statistic_dataset(
                join(root_path, data_set_name),
                dataset_type=dataset_type,
                parallel=parallel,
                num_processes=num_processes,
                override=override,
                search_pattern=search_pattern,
                kohn_sham_folder=kohn_sham_folder,
                save_folder_name=save_folder_name,
                n_molecules=n_molecules,
                start_index=start_index,
                iterations=iterations,
            )

    def create_statistic_dataset(
        self,
        root_path: Path | str,
        dataset_type="MD17",
        parallel=True,
        num_processes=None,
        override=False,
        search_pattern="^labels$",
        kohn_sham_folder="kohn_sham",
        save_folder_name="statistic",
        n_molecules=None,
        start_index=None,
        iterations=None,
    ):
        """Calculates statistics files for all labels that are found inside the given director.

        Args:
            root_path: the root directory containing the labels directories
            dataset_type: the type of dataset to calculate statistics for(MD17 and QM9 supported)
            parallel: Whether to parallelize the calculation over multiple cores
            num_processes: Number of parallel processes to use if parallel = True
            search_pattern: the pattern used to search the labels directory
        """
        list_of_dirnames = listdir(root_path)
        if kohn_sham_folder not in list_of_dirnames:
            logger.warning(
                f"There is no kohn_sham directory at {root_path} to compare against. Skipping this Dataset"
            )
            return

        ks_dir = join(root_path, kohn_sham_folder)
        label_directories = [
            dir_name for dir_name in list_of_dirnames if regex.search(search_pattern, dir_name)
        ]
        for label_directory in label_directories:
            save_dir_name = regex.sub("labels", save_folder_name, label_directory)
            logger.info(f"Preceding with {save_dir_name} of dataset type {dataset_type}")

            if not os.path.exists(join(root_path, save_dir_name)):
                os.mkdir(join(root_path, save_dir_name))
            sub_dir = "dsgdb9nsd.xyz"  # For backward compatibility with QM9
            if os.path.isdir(join(ks_dir, sub_dir)):
                ks_dir = join(ks_dir, sub_dir)
            if os.path.isdir(join(root_path, label_directory, sub_dir)):
                label_directory = join(label_directory, sub_dir)
            logger.info(f"KS-files at {ks_dir} and Labels at{join(root_path, label_directory)}")
            if not os.path.exists(join(root_path, save_dir_name)):
                os.mkdir(join(root_path, save_dir_name))
            if parallel:
                self.create_statistic_parallel(
                    join(root_path, save_dir_name),
                    join(ks_dir),
                    join(root_path, label_directory),
                    dataset_type=dataset_type,
                    num_processes=num_processes,
                    override=override,
                    n_molecules=n_molecules,
                    start_index=start_index,
                    iterations=iterations,
                )
            else:
                raise NotImplementedError()

    @staticmethod
    def create_statistic(args: tuple):
        """Creates statistic files for one given ks_file, and of_data_file.

        Args:
            tuple of:
            root_path: path to the directory containing where the statistic is saved
            ks_file_path: path to the kohn_sham-file
            of_data_file_path: path to the of_data_file
            number: number of the molecule
            basis_info: basis_info-object
            names_of_metrics: names of all the metrics are to be computed
        """
        (
            root_path,
            ks_file_path,
            of_data_file_path,
            number,
            basis_info,
            names_of_metrics,
            iterations,
        ) = args
        root_path, ks_file_path, of_data_file_path = (
            Path(root_path),
            Path(ks_file_path),
            Path(of_data_file_path),
        )
        dataset_kwargs = {}
        logger.info(f"Starting calculation of statistic for molecule {number}.")
        """Compute the statistic for the density_fitting performance of the dataset and save it to
        a zarr.zip file."""
        if not isfile(of_data_file_path):
            logger.warning(f"File : {of_data_file_path} does not exist")
            return

        compare_with_kohn_sham = bool(
            {
                "AE_density",
                "L2_norm_residual_densities",
                "L2_residual_densities",
                "L1_residual_densities",
                "L1_negative_densities",
            }
            & set(names_of_metrics)
        )

        if compare_with_kohn_sham:
            results, initialization, data_of_iterations = load_scf(ks_file_path)
            n_iterations = len(data_of_iterations)
            grid_level = results["grid_level"]
        else:
            root = zarr.open(of_data_file_path, mode="r")
            of_labels = root["of_labels"]
            n_iterations = of_labels["n_scf_steps"][()] - 1
            grid_level = 3

        if iterations is None:
            iterations = list(range(n_iterations + 1))
        if isinstance(iterations, int):
            if iterations > n_iterations:
                iterations = n_iterations
            iterations = [iterations]
        iterations = [i for i in iterations if i <= n_iterations]
        if len(iterations) == 0:
            return

        energies = ["e_hartree", "e_xc", "e_ext", "e_kin", "e_tot"]
        add_keys = {"ks_labels/energies/" + e: "scalar" for e in energies}
        add_keys.update({"of_labels/energies/" + e: "scalar" for e in energies})
        grads = ["grad_kin", "grad_hartree", "grad_xc", "grad_ext"]
        add_keys.update({"of_labels/spatial/" + grad: "gradient" for grad in grads})
        try:
            of_data_list = [
                OFData.from_file(
                    of_data_file_path,
                    iteration,
                    basis_info=basis_info,
                    additional_keys_at_scf_iteration=add_keys,
                )
                for iteration in range(n_iterations + 1)
                if iteration in iterations
            ]
        except Exception as e:
            logger.warning(f"Could not open file at {of_data_file_path} because of {e}")
            return

        # filter zeros
        if np.any(of_data_list[0].atomic_numbers == 0):
            real_atom_mask = of_data_list[0].atomic_numbers != 0
            real_atoms = of_data_list[0].atomic_numbers[real_atom_mask]
            real_positions = np.asarray(of_data_list[0].pos)[real_atom_mask]
            mol_with_orbital_basis = build_molecule_np(
                real_atoms,
                real_positions,
                "6-31G(2df,p)",
                unit="Bohr",
            )
        else:
            mol_with_orbital_basis = build_molecule_np(
                of_data_list[0].atomic_numbers,
                np.asarray(of_data_list[0].pos),
                "6-31G(2df,p)",
                unit="Bohr",
            )

        mol_with_density_basis = build_molecule_ofdata(of_data_list[0], basis_info.basis_dict)

        if of_data_list[0]["coeffs"].shape[0] != mol_with_density_basis.nao:
            logger.warning("This directory is corrupted!!")
            return

        # precompute some tensors that are commonly used
        _tensors = {}
        if bool(
            {"AE_density", "L2_norm_residual_densities", "L2_residual_densities"}
            & set(names_of_metrics)
        ):
            _tensors.update(
                {
                    "of_basis_integrals": np.asarray(
                        get_normalization_vector(mol_with_density_basis)
                    ),
                    "orbitals_overlap_matrix": np.asarray(
                        get_overlap_matrix(mol_with_orbital_basis)
                    ),
                    "W_overlap": np.asarray(get_overlap_matrix(mol_with_density_basis)),
                    "L_overlap": np.asarray(
                        get_overlap_tensor(mol_with_density_basis, mol_with_orbital_basis)
                    ),
                }
            )
        elif "L2_gradient_overlap" in names_of_metrics:
            _tensors.update(
                {
                    "W_overlap": np.asarray(get_overlap_matrix(mol_with_density_basis)),
                }
            )

        if bool(
            {"L1_residual_densities", "L1_negative_densities", "L2_residual_densities"}
            & set(names_of_metrics)
        ):
            grid_orbitals = grid_setup(mol_with_orbital_basis, grid_level=grid_level)
            _tensors.update(
                {
                    "grid_weights": grid_orbitals.weights,
                    "grid_coords": grid_orbitals.coords,
                    "ao_orbitals": dft.numint.eval_ao(
                        mol_with_orbital_basis, grid_orbitals.coords, deriv=0
                    ),
                    "ao_densities": dft.numint.eval_ao(
                        mol_with_density_basis, grid_orbitals.coords, deriv=0
                    ),
                }
            )
        if "L2_residual_hartree" in names_of_metrics:
            _tensors.update(
                {
                    "W_coulomb": np.asarray(get_coulomb_matrix(mol_with_density_basis)),
                    "L_coulomb": np.asarray(
                        get_coulomb_tensor(mol_with_density_basis, mol_with_orbital_basis)
                    ),
                }
            )
            # "D_coulomb" : np.asarray(get_L2_coulomb_matrix(mol_with_orbital_basis)),
            # "external_potential_of" : np.asarray(get_nuclear_attraction_vector(mol_with_density_basis)),
            # "external_potential_orbitals" : np.asarray(get_nuclear_attraction_matrix(mol_with_orbital_basis)),
        if compare_with_kohn_sham:
            temp_gamma_list = [
                ksdft_density_matrix(
                    data["molecular_coeffs_orbitals"], data["occupation_numbers_orbitals"]
                )
                for i, data in enumerate(data_of_iterations)
                if i + 1 in iterations
            ]
            if 0 in iterations:
                temp_gamma_list = [initialization["first_density_matrix"]]
                +temp_gamma_list

            gammas = np.stack(temp_gamma_list)
            _tensors.update({"gammas": gammas})
        of_coeffs = np.stack([_of_data["coeffs"] for _of_data in of_data_list])
        _tensors.update({"of_coeffs": of_coeffs})

        metric_values = {}
        for metric_name in names_of_metrics:
            if hasattr(DensityFittingStatistic, metric_name):
                metric_values.update(
                    {
                        metric_name: getattr(DensityFittingStatistic, metric_name)(
                            of_data_list, _tensors
                        )
                    }
                )
        dataset_kwargs.setdefault("compressor", None)
        with zarr.ZipStore(root_path.joinpath(f"{number:06}.zarr.zip"), mode="w") as zipstore:
            root = zarr.open(zipstore, mode="w")
            for metric_name, metric_value in metric_values.items():
                root.create_dataset(metric_name, data=metric_value, **dataset_kwargs)

    def create_statistic_parallel(
        self,
        root_path: Path,
        ks_files_path: Path,
        of_data_files_path: Path,
        basis_info: BasisInfo = None,
        dataset_type="MD17",
        num_processes=None,
        override=False,
        n_molecules=None,
        start_index=None,
        iterations=None,
        **dataset_kwargs,
    ):
        """Creates the statistics files for a single directory in single process mode.

        Args:
            root_path (Path): Path to the directory where the statistics files are stored.
            ks_files_path (Path): Path to the directory where the ks files are stored.
            of_data_files_path (Path): Path to the directory where the ofdata files are stored.
            basis_info (BasisInfo, optional): Basis_info class. Defaults to None.
            dataset_type (str, optional): Dataset type to be computed. Defaults to "MD17".
            num_processes (int, optional): Number of parallel processes, if None use all available cores. Defaults to None.
            **dataset_kwargs (dict, optional): Keyword arguments to be passed to the Dataset constructor.
        """
        num_processes, num_threads = configure_processes_and_threads(num_processes, None)

        if dataset_type == "MD17" or dataset_type == "Cumulene":
            padding_digits = 6
            ks_files = [
                f
                for f in listdir(ks_files_path)
                if isfile(join(ks_files_path, f)) and not regex.search("yaml", f)
            ]
            ks_files.sort()
            if n_molecules is not None:
                ks_files = ks_files[start_index : start_index + n_molecules]
            else:
                ks_files = ks_files[start_index:]
            numbers = [int(f.split("_")[-1].split(".")[0]) for f in ks_files]
            of_data_files_names = [f"{number:0{padding_digits}}.zarr.zip" for number in numbers]
            atomic_numbers = [1, 6, 8]
        elif dataset_type == "QM9" or dataset_type == "QMUGS":
            padding_digits = 7
            ks_files = [
                f
                for f in listdir(ks_files_path)
                if isfile(join(ks_files_path, f)) and regex.search("chk", f)
            ]
            ks_files.sort()
            if n_molecules is not None:
                ks_files = ks_files[start_index : start_index + n_molecules]
            else:
                ks_files = ks_files[start_index:]
            numbers = [int(f.split("_")[-1].split(".")[0]) for f in ks_files]
            of_data_files_names = [f"{number:0{padding_digits}}.zarr.zip" for number in numbers]
            atomic_numbers = [1, 6, 7, 8, 9]
        else:
            raise NotImplementedError()
        logger.info(f"Found {len(ks_files)} kohn_sham files")
        if len(ks_files) == 0:
            logger.info(f"Found no ks-files at {ks_files_path}")

        if override:
            already_processed = np.asarray(
                [
                    not os.path.isfile(join(of_data_files_path, of_data_files_names[i]))
                    for i, ks_file in enumerate(ks_files)
                ]
            )
        else:
            already_processed = np.asarray(
                [
                    os.path.isfile(join(root_path, f"{numbers[i]:06}.zarr.zip"))
                    or not os.path.isfile(join(of_data_files_path, of_data_files_names[i]))
                    for i, ks_file in enumerate(ks_files)
                ]
            )
        logger.info(
            f"Skipped {already_processed.sum()} calculations. {len(ks_files)-already_processed.sum()} left."
        )
        num_processes = min(len(ks_files) - already_processed.sum(), num_processes)
        logger.info(
            f"Starting calculation of statistic in multi process mode with {num_processes} threads."
        )
        if num_processes == 0:
            return

        if basis_info is None:
            if os.path.isfile(join(of_data_files_path, "dataset_info.yaml")):
                basis_info = BasisInfo.from_dataset_info_yaml(
                    join(of_data_files_path, "dataset_info.yaml"), atomic_numbers
                )
            elif os.path.isfile(join(of_data_files_path, "..", "dataset_info.yaml")):
                basis_info = BasisInfo.from_dataset_info_yaml(
                    join(of_data_files_path, "..", "dataset_info.yaml"), atomic_numbers
                )
            else:
                dataset_info_path = join(of_data_files_path, "dataset_info.yaml")
                dataset_info_path2 = join(of_data_files_path, "..", "dataset_info.yaml")
                logger.warning(
                    f"No basis info found for dataset type {dataset_type} at {dataset_info_path} or {dataset_info_path2}"
                )
                basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis(atomic_numbers)

        args_list = [
            (
                root_path,
                ks_files_path + "/" + ks_file,
                of_data_files_path + "/" + of_data_files_names[i],
                numbers[i],
                basis_info,
                self.names_of_metrics,
                iterations,
            )
            for i, ks_file in enumerate(ks_files)
            if not already_processed[i]
        ]
        with multiprocessing.Pool(processes=num_processes) as pool:
            # Using imap_unordered instead of starmap_async
            list(
                tqdm(
                    pool.imap_unordered(DensityFittingStatistic.create_statistic, args_list),
                    total=len(ks_files),
                    position=0,
                    smoothing=0,
                    dynamic_ncols=True,
                )
            )

    @staticmethod
    def AE_hartree_energy(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the Hartree energy :math:`|E_{H}(C)-E_{H}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        """
        ks_energies = np.asarray(
            [_of_data["ks_labels/energies/e_hartree"] for _of_data in of_data_list]
        )
        of_energies = np.asarray(
            [_of_data["of_labels/energies/e_hartree"] for _of_data in of_data_list]
        )
        return np.abs(ks_energies - of_energies)

    @staticmethod
    def AE_total_energy(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the total energy :math:`|E_{tot}(C)-E_{tot}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Note:
            0 per usual, as :math:`E_{kin}(p)` is defined such as :math:`E_{tot}(C)=E_{tot}(p)`
        """
        ks_energies = np.asarray(
            [_of_data["ks_labels/energies/e_tot"] for _of_data in of_data_list]
        )
        of_energies = np.asarray(
            [_of_data["of_labels/energies/e_tot"] for _of_data in of_data_list]
        )
        return np.abs(ks_energies - of_energies)

    @staticmethod
    def AE_xc_energy(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the exchange correlation energy :math:`|E_{xc}(C)-E_{xc}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        """
        ks_energies = np.asarray(
            [_of_data["ks_labels/energies/e_xc"] for _of_data in of_data_list]
        )
        of_energies = np.asarray(
            [_of_data["of_labels/energies/e_xc"] for _of_data in of_data_list]
        )
        return np.abs(ks_energies - of_energies)

    @staticmethod
    def AE_ext_energy(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the external energy :math:`|E_{ext}(C)-E_{ext}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        """
        ks_energies = np.asarray(
            [_of_data["ks_labels/energies/e_ext"] for _of_data in of_data_list]
        )
        of_energies = np.asarray(
            [_of_data["of_labels/energies/e_ext"] for _of_data in of_data_list]
        )
        return np.abs(ks_energies - of_energies)

    @staticmethod
    def AE_kin_energy(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the not interacting kinetic energy :math:`|T_{S}(C)-T_{S}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        """
        ks_energies = np.asarray(
            [_of_data["ks_labels/energies/e_kin"] for _of_data in of_data_list]
        )
        of_energies = np.asarray(
            [_of_data["of_labels/energies/e_kin"] for _of_data in of_data_list]
        )
        return np.abs(ks_energies - of_energies)

    @staticmethod
    def AE_hartree_ext_energy(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the sum of external and hartree energy :math:`|E_{ext}(C)+E_{H}(C)-E_{ext}(p)-E_{H}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        """
        ks_energies = np.asarray(
            [
                _of_data["ks_labels/energies/e_hartree"] + _of_data["ks_labels/energies/e_ext"]
                for _of_data in of_data_list
            ]
        )
        of_energies = np.asarray(
            [
                _of_data["of_labels/energies/e_hartree"] + _of_data["of_labels/energies/e_ext"]
                for _of_data in of_data_list
            ]
        )
        return np.abs(ks_energies - of_energies)

    @staticmethod
    def AE_density(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The absolute error of the integrated density (electron number) :math:`|N_{elec}(C)-N_{elec}(p)|`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        """
        ks_densities_orbitals = np.einsum(
            "ij,kij->k", tensors["orbitals_overlap_matrix"], tensors["gammas"], optimize=True
        )
        of_densities = np.einsum(
            "i,ji->j", tensors["of_basis_integrals"], tensors["of_coeffs"], optimize=True
        )
        # TODO find out why this throws an error
        return np.abs(ks_densities_orbitals - of_densities)

    @staticmethod
    def L2_residual_densities(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        r"""The L2-norm of the residual density :math:`\int dr\left(\rho(C)-\rho(p)\right)^2` entirely evaluated on the grid
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            (because pyscf doesn`t support int4c1e)
            Same as L2_norm_residual_density, but faster
        """
        rho_ks_on_grid = np.einsum(
            "ij,ljk,ik->li",
            tensors["ao_orbitals"],
            tensors["gammas"],
            tensors["ao_orbitals"],
            optimize=True,
        )
        rho_of_on_grid = np.einsum(
            "ij,kj->ki", tensors["ao_densities"], tensors["of_coeffs"], optimize=True
        )
        rho2_on_grid = (rho_ks_on_grid - rho_of_on_grid) ** 2
        return np.sqrt(np.einsum("ki,i->k", rho2_on_grid, tensors["grid_weights"], optimize=True))

    @staticmethod
    def L1_residual_densities(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        r"""The L1-norm of the residual density :math:`\int dr\|left(\rho(C)-\rho(p)\right)|` entirely evaluated on the grid
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            (because pyscf doesn`t support int4c1e)
            Same as L2_norm_residual_density, but faster
        """
        rho_ks_on_grid = np.einsum(
            "ij,ljk,ik->li",
            tensors["ao_orbitals"],
            tensors["gammas"],
            tensors["ao_orbitals"],
            optimize=True,
        )
        rho_of_on_grid = np.einsum(
            "ij,kj->ki", tensors["ao_densities"], tensors["of_coeffs"], optimize=True
        )
        rho2_on_grid = np.abs(rho_ks_on_grid - rho_of_on_grid)
        return np.einsum("ki,i->k", rho2_on_grid, tensors["grid_weights"], optimize=True)

    @staticmethod
    def L1_negative_densities(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        r"""The L2-norm of the residual density :math:`\int dr\left(\rho(C)-\rho(p)\right)^2` entirely evaluated on the grid
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            (because pyscf doesn`t support int4c1e)
            Same as L2_norm_residual_density, but faster
        """
        rho_of_on_grid = np.einsum(
            "ij,kj->ki", tensors["ao_densities"], tensors["of_coeffs"], optimize=True
        )
        rho2_on_grid = -rho_of_on_grid * (rho_of_on_grid <= 0)
        return np.einsum("ki,i->k", rho2_on_grid, tensors["grid_weights"], optimize=True)

    @staticmethod
    def L2_norm_residual_density(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        r"""The L2-norm of the residual density :math:`\int dr\left(\rho(C)-\rho(p)\right)^2`
        The of- and overlap parts are calculated using pyscf matrices the L2 norm of the pure orbital density
        has to be evaluated on the grid
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            (because pyscf doesn`t support int4c1e)
            Same as L2_residual_densities, but slower
        """
        rho_on_grid = np.einsum(
            "ij,ljk,ik->li",
            tensors["ao_orbitals"],
            tensors["gammas"],
            tensors["ao_orbitals"],
            optimize=True,
        )
        rho2_on_grid = rho_on_grid**2
        L2_norms_orbital_density = np.einsum(
            "ij,j->i", rho2_on_grid, tensors["grid_weights"], optimize=True
        )
        pWp = np.einsum(
            "li,ij,lj->l",
            tensors["of_coeffs"],
            tensors["W_overlap"],
            tensors["of_coeffs"],
            optimize=True,
        )
        pLgamma = np.einsum(
            "ijk,ljk,li->l",
            tensors["L_overlap"],
            tensors["gammas"],
            tensors["of_coeffs"],
            optimize=True,
        )
        return np.sqrt(pWp - 2 * pLgamma + L2_norms_orbital_density)

    @staticmethod
    def L2_residual_hartree_num(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        r"""The residual hartree energy :math:`E_{H}(\rho(C)-\rho(p))` evaluated on the grid
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        rho_ks_on_grid = np.einsum(
            "ij,ljk,ik->li",
            tensors["ao_orbitals"],
            tensors["gammas"],
            tensors["ao_orbitals"],
            optimize=True,
        )
        rho_of_on_grid = np.einsum(
            "ij,kj->ki", tensors["ao_densities"], tensors["of_coeffs"], optimize=True
        )

        coords = torch.from_numpy(tensors["grid_coords"])
        potential = torch.zeros((rho_ks_on_grid.shape[1], rho_ks_on_grid.shape[1]))
        n = 88
        for i in range(rho_ks_on_grid.shape[1] // n):
            potential[:, i * n : (i + 1) * n] = torch.linalg.vector_norm(
                coords[:, None, :] - coords[None, i * n : (i + 1) * n, :], dim=2
            )
        # divide safely
        nonzero_mask = potential != 0
        torch.div(1, potential[nonzero_mask], out=potential[nonzero_mask])
        potential = np.asarray(potential)
        diff = rho_ks_on_grid - rho_of_on_grid
        return np.einsum(
            "li,ij,lj,i,j->l",
            diff,
            potential,
            diff,
            tensors["grid_weights"],
            tensors["grid_weights"],
            optimize=True,
        )

    @staticmethod
    def L2_residual_hartree(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        r"""The residual hartree energy :math:`E_{H}(\rho(C)-\rho(p))`
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        overlap_hartree = np.einsum(
            "li,ijk,ljk->l",
            tensors["of_coeffs"],
            tensors["L_coulomb"],
            tensors["gammas"],
            optimize=True,
        )
        ks_hartree = np.asarray(
            [_of_data["ks_labels/energies/e_hartree"] for _of_data in of_data_list]
        )
        of_hartree = np.asarray(
            [_of_data["of_labels/energies/e_hartree"] for _of_data in of_data_list]
        )
        return 2 * ks_hartree - 2 * overlap_hartree + 2 * of_hartree

    @staticmethod
    def N_basis_functions(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The number of basis functions per molecule
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        return tensors["of_coeffs"].shape[1] * np.ones(tensors["of_coeffs"].shape[0])

    @staticmethod
    def max_abs_gradient(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The maximum absolute gradient, the gradient at the ground state should be 0
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        abs_ground_state_gradient = np.stack(
            [
                np.max(
                    np.abs(
                        _of_data["of_labels/spatial/grad_kin"]
                        + _of_data["of_labels/spatial/grad_hartree"]
                        + _of_data["of_labels/spatial/grad_xc"]
                        + _of_data["of_labels/spatial/grad_ext"]
                    )
                )
                for _of_data in of_data_list
            ]
        )
        return abs_ground_state_gradient

    @staticmethod
    def L2_gradient(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The L2 norm of the gradient at the ground state, the gradient at the ground state should be 0
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        total_ground_state_gradient = np.stack(
            [
                _of_data["of_labels/spatial/grad_kin"]
                + _of_data["of_labels/spatial/grad_hartree"]
                + _of_data["of_labels/spatial/grad_xc"]
                + _of_data["of_labels/spatial/grad_ext"]
                for _of_data in of_data_list
            ]
        )
        l2_norm_gradient = np.sqrt(
            np.einsum("ij,ij->i", total_ground_state_gradient, total_ground_state_gradient)
        )
        return l2_norm_gradient

    @staticmethod
    def L2_gradient_overlap(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The L2 norm of the gradient at the ground state, the gradient at the ground state should be 0
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        total_ground_state_gradient = np.stack(
            [
                _of_data["of_labels/spatial/grad_kin"]
                + _of_data["of_labels/spatial/grad_hartree"]
                + _of_data["of_labels/spatial/grad_xc"]
                + _of_data["of_labels/spatial/grad_ext"]
                for _of_data in of_data_list
            ]
        )
        l2_norm_gradient = np.sqrt(
            np.einsum(
                "ij,jk,ik->i",
                total_ground_state_gradient,
                tensors["W_overlap"],
                total_ground_state_gradient,
            )
        )
        return l2_norm_gradient

    @staticmethod
    def L2_residual_hartree_ext(of_data_list: List[OFData], tensors: dict[str, np.ndarray]):
        """The sum of residual hartree energy and residual external energy squared
        Args:
            of_data_list (List[OFData]): List of OFData objects
            tensors (dict[str,np.ndarray]): Dictionary of tensors used for some metrics
        Notes:
            Used as a sanity check for L2_residual_hartree, but is very slow and uses lots of ram
        """
        return (
            DensityFittingStatistic.L2_residual_hartree(of_data_list, tensors)
            + DensityFittingStatistic.AE_ext_energy(of_data_list, tensors) ** 2
        )


@hydra.main(
    version_base="1.3",
    config_path="../../configs/datagen",
    config_name="density_fitting_statistic",
)
def main(cfg: DictConfig) -> None:
    dfs = DensityFittingStatistic(
        cfg.get("root_directory"), names_of_metrics=cfg.get("names_of_metrics")
    )

    parallel = (cfg.get("num_processes") is None) or cfg.get("num_processes") > 1
    dfs.create_statistics_for_all_datasets(
        cfg.get("root_directory"),
        cfg.get("data_set_type"),
        parallel=parallel,
        num_processes=cfg.get("num_processes"),
        override=cfg.get("override"),
        postfix=cfg.get("postfix"),
        search_pattern=cfg.get("search_pattern"),
        kohn_sham_folder=cfg.get("kohn_sham_folder"),
        save_folder_name=cfg.get("save_folder_name"),
        n_molecules=cfg.get("n_molecules"),
        start_index=cfg.get("start_index"),
        iterations=cfg.get("iterations"),
    )


if __name__ == "__main__":
    main()
