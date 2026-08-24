import os
import pickle

import yaml
from tqdm import tqdm


def sub_split_from_IDs(complete_path, save_path, molecule_ids):
    """Create a sub-split from a complete split file based on molecule IDs.

    Args:
        complete_path (str): Path to the complete split file.
        save_path (str): Path where the new split file will be saved.
        molecule_ids (list): List of molecule IDs to filter the train set.
    Returns:
        dict: The modified split dictionary containing the new train set.
    """
    # load split file
    with open(complete_path, "rb") as f:
        split = pickle.load(f)

    original_train = split["train"]
    subset_indices = []
    train_size = 0
    for i, item in tqdm(enumerate(original_train)):
        # print(item[1])
        if item[1] in molecule_ids:
            subset_indices.append(i)
            train_size += item[2]

    split["train"] = [original_train[i] for i in subset_indices]
    split["sizes"]["train"] = train_size

    print(save_path)
    # make directory if it does not exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    # save split file
    with open(save_path, "wb") as f:
        pickle.dump(split, f)
    # also save as yaml

    with open(save_path.replace(".pkl", ".yaml"), "w") as f:
        yaml.dump(split, f, sort_keys=False, default_flow_style=None)
    print(f"Saved split to {save_path}")
    return split
