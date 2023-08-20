"""
This script regroups a few functions that are used to work with a custom Dataset, which will be the input of our convolutional neural
network. The class inherits from Pytorch's abstract class Dataset, and implements the methods __len__ and __getitem__.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pymatgen.symmetry.groups import SpaceGroup


def get_dictionary(filepath):
    """Return a dictionary with integers as values and space groups as values, from the Parquet file.

        Args:
            - filepath (string) : path of the file to be parsed

        Returns:
            - int2group (dict) : dictionary with labels (space groups) as values
    """
    int2group = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            label = line.strip()
            int2group[i] = label
    return int2group


def get_mapping(filepath):
    """Returns a dictionary with the index (incremented of 1) corresponding to a space group.
        Args:
            - filename (str) : path of the file to be parsed (here space_groups.txt)

        Returns:
            - space_group_mapping (dict) : dictionary which does the mapping between space groups and indexes
    """
    int2group = get_dictionary(filepath)
    space_group_mapping = {name: i+1 for i, name in int2group.items()}
    return space_group_mapping


class XRDPatternDataset(Dataset):
    """Class that inherits from abstract class Dataset, generates the materials' formulas, the XRD pattern (angles and intensity),
       and the corresponding space group to be predicted.
    """
    def __init__(self, xrd_file):
        """Constructor of the class XRDPatternDataset.
            Args:
                - xrd_file (str) : Parquet file containing all the data
        """
        self.dataframe = pd.read_parquet(xrd_file, engine="pyarrow")
        space_groups = self.dataframe["Space Group"].unique().tolist()
        crystal_systems = self.dataframe["Crystal System"].unique().tolist()
        space_group_mapping = {}
        for i, group in enumerate(space_groups):
            space_group_mapping[group] = i+1
        self.space_group_mapping = space_group_mapping
        self.nb_space_group = len(space_group_mapping)
        self.nb_crystal_systems = len(crystal_systems)

    def __len__(self):
        """Returns the size of the dataframe (number of rows).
            Returns:
                - the size (i.e. number of rows) of the dataset (int)
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Access and returns the data at row index idx in the dataframe.
            Args:
                - idx (int) : index where to get information

            Returns:
                - angles (torch.Tensor), intensities (torch.Tensor) and space group / crystal system (torch.Tensor)
        """
        xrd_pattern = self.dataframe.iloc[idx]
        intensities = np.array(xrd_pattern[2], dtype=float)
        angles = np.array(xrd_pattern[1], dtype=float)
        intensities = torch.tensor(intensities)
        angles = torch.tensor(angles)
        space_group = torch.tensor(self.space_group_mapping[self.dataframe.iloc[idx, 3]], dtype=torch.long)
        crystal_systems = np.array(xrd_pattern[4], dtype=int)
        crystal_systems = torch.tensor(crystal_systems)
        return angles, intensities, crystal_systems


if __name__ == "__main__":
    print("obtaining space groups")
    with open('./data/space_groups.txt', 'w') as f:
        for i in range(1, 231):
            space_group = SpaceGroup.from_int_number(i)
            group_name = str(space_group).split(" ")[1]
            f.write(f"{group_name}\n")
