import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pymatgen.symmetry.groups import SpaceGroup


def get_dictionary(filepath):
    int2group = {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            label = line.strip()
            int2group[i] = label
    return int2group


def get_mapping(filepath):
    int2group = get_dictionary(filepath)
    space_group_mapping = {name: i for i, name in int2group.items()}
    return space_group_mapping


class XRDPatternDataset(Dataset):
    def __init__(self, xrd_file, space_group_mapping):
        self.dataframe = pd.read_parquet(xrd_file)
        self.space_group_mapping = space_group_mapping

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        xrd_pattern = self.dataframe.iloc[idx, 0]
        xrd_pattern = np.array(xrd_pattern, dtype=float)
        intensities = torch.tensor(xrd_pattern[0])
        angles = torch.tensor(xrd_pattern[1])
        space_group = torch.tensor(self.space_group_mapping[self.dataframe.iloc[idx, 1]])
        return angles, intensities, space_group


if __name__ == "__main__":
    print("obtaining space group")
    with open('./space_groups.txt', 'w') as f:
        for i in range(1, 231):
            space_group = SpaceGroup.from_int_number(i)
            group_name = str(space_group).split(" ")[1]
            f.write(f"{group_name}\n")
