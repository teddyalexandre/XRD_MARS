import pytest
import numpy as np
import torch
from data import XRDPatternDataset, get_dictionary, get_mapping
from pathlib import Path


@pytest.fixture
def space_group_mapping():
    filepath = Path(__file__).parent / 'space_groups.txt'
    return get_mapping(filepath)


def test_get_dictionary_returns_dict():
    filepath = Path(__file__).parent / 'space_groups.txt'
    result = get_dictionary(filepath)
    assert isinstance(result, dict)


def test_get_mapping_returns_dict():
    filepath = Path(__file__).parent / 'space_groups.txt'
    result = get_mapping(filepath)
    assert isinstance(result, dict)


def test_get_mapping_swaps_keys_and_values():
    filepath = Path(__file__).parent / 'space_groups.txt'
    dict1 = get_dictionary(filepath)
    dict2 = get_mapping(filepath)
    for key in dict1:
        assert dict2[dict1[key]] == key


def test_dataset_length(space_group_mapping):
    dataset = XRDPatternDataset('./pow_xrd.parquet', space_group_mapping)
    assert len(dataset) > 0


def test_dataset_item_type(space_group_mapping):
    dataset = XRDPatternDataset('./pow_xrd.parquet', space_group_mapping)
    angles, intensities, space_group = dataset[0]
    assert isinstance(angles, torch.Tensor)
    assert isinstance(intensities, torch.Tensor)
    assert isinstance(space_group, torch.Tensor)


def test_dataset_item_shape(space_group_mapping):
    dataset = XRDPatternDataset('./pow_xrd.parquet', space_group_mapping)
    angles, intensities, space_group = dataset[0]
    assert angles.dim() == 1
    assert intensities.dim() == 1
    assert space_group.dim() == 0


def test_dataset_item_dtype(space_group_mapping):
    dataset = XRDPatternDataset('./pow_xrd.parquet', space_group_mapping)
    angles, intensities, space_group = dataset[0]
    assert angles.dtype == torch.float32
    assert intensities.dtype == torch.float32
    assert space_group.dtype == torch.int64
