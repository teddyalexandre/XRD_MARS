import pytest
from models import vector_size, conv_output_size


def test_vector_size():
    parms = {
        'kernels': [100, 50, 25],
        'strides': [5, 5, 2],
        'input_size': 10001,
        'conv_channels': 80
    }
    s = vector_size(parms)
    assert s == 880


def test_conv_output_size1():
    s = conv_output_size(10001, 5, 100, 50)
    assert s == 2001


def test_conv_output_size2():
    s = conv_output_size(2001, 2, 3)
    assert s == 1000


def test_conv_output_size3():
    s = conv_output_size(1000, 5, 50, 24)
    assert s == 200


def test_conv_output_size3():
    s = conv_output_size(200, 3, 3)
    assert s == 66


def test_conv_output_size4():
    s = conv_output_size(66, 2, 25, 12)
    assert s == 33


def test_conv_output_size4():
    s = conv_output_size(33, 3, 3)
    assert s == 11