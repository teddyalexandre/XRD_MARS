import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pymatgen.symmetry.groups import SpaceGroup


def get_dictionary(filepath):
    """Return a dictionary with integers as values and space groups as values,
    from the Parquet file
        Args:
            - filepath (string) : path of the file to be parsed
        Returns:
            - int2group : dictionary with labels (space groups) as values"""
    int2group = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            label = line.strip()
            int2group[i] = label
    return int2group


def get_mapping(filepath):
    """Returns the values from the dictionary
        Args:
            - filename (string) : path of the file to be parsed
        Returns:
            - space_group_mapping : dictionary which does the mapping between space groups and indexes"""
    int2group = get_dictionary(filepath)
    space_group_mapping = {name: i+1 for i, name in int2group.items()}
    return space_group_mapping


class XRDPatternDataset(Dataset):
    """Class that generates the XRD pattern, the angles, the intensity and the space group to be predicted"""
    def __init__(self, xrd_file):
        """Constructor of the class
            Args:
                - xrd_file : Parquet file containing all the data
                - space_group_mapping : dictionary with the space groups as values
        """
        self.dataframe = pd.read_parquet(xrd_file, engine="pyarrow")
        space_groups = self.dataframe["Space Group"].unique().tolist()
        space_group_mapping = {}
        for i, group in enumerate(space_groups):
            space_group_mapping[group] = i+1
        self.space_group_mapping = space_group_mapping
        self.nb_space_group = len(space_group_mapping)

    def __len__(self):
        """Returns the size of the dataframe (number of rows)"""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Returns the information at row idx
            Args:
                - idx (int) : index where to get information
            Returns:
                - angles, intensities and space_group tensors
        """
        xrd_pattern = self.dataframe.iloc[idx, 0]
        intensities = np.array(xrd_pattern[0], dtype=float)
        angles = np.array(xrd_pattern[1], dtype=float)
        intensities = torch.tensor(intensities)
        angles = torch.tensor(angles)
        space_group = torch.tensor(self.space_group_mapping[self.dataframe.iloc[idx, 1]], dtype=torch.long)
        return angles, intensities, space_group


if __name__ == "__main__":
    print("obtaining space groups")
    with open('../tests/space_groups.txt', 'w') as f:
        for i in range(1, 231):
            space_group = SpaceGroup.from_int_number(i)
            group_name = str(space_group).split(" ")[1]
            f.write(f"{group_name}\n")
