import pytest
import numpy as np
from numpy_tutorial.tensors import (
    create_3d_tensor,
    tensor_addition,
    tensor_dot_product
)

def test_create_3d_tensor():
    tensor = create_3d_tensor((2, 3, 4))
    assert tensor.shape == (2, 3, 4)

def test_tensor_addition():
    t1 = np.ones((2, 2, 2))
    t2 = np.ones((2, 2, 2)) * 2
    res = tensor_addition(t1, t2)
    assert np.array_equal(res, np.ones((2, 2, 2)) * 3)

def test_tensor_dot_product():
    t1 = np.ones((2, 3))
    t2 = np.ones((3, 2))
    # Produto de matriz é um subconjunto de tensordot
    res = tensor_dot_product(t1, t2, axes=1)
    expected = np.full((2, 2), 3.0)
    assert np.array_equal(res, expected)
