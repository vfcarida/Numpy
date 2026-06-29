import pytest
import numpy as np
from numpy_tutorial.arrays import (
    create_array,
    create_2d_array,
    replace_elements,
    find_common_elements,
    reverse_array,
    sort_array,
    get_n_largest
)

def test_create_array():
    arr = create_array([1, 2, 3])
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.array([1, 2, 3]))

def test_create_2d_array():
    arr = create_2d_array(2, 3, 5)
    assert arr.shape == (2, 3)
    assert np.all(arr == 5)

def test_replace_elements():
    arr = np.array([1, 2, 2, 3])
    new_arr = replace_elements(arr, 2, 99)
    assert np.array_equal(new_arr, np.array([1, 99, 99, 3]))
    assert np.array_equal(arr, np.array([1, 2, 2, 3]))  # Garantir que não mutou o original

def test_find_common_elements():
    a1 = np.array([1, 2, 3, 4])
    a2 = np.array([3, 4, 5, 6])
    common = find_common_elements(a1, a2)
    assert np.array_equal(common, np.array([3, 4]))

def test_reverse_array():
    arr = np.array([1, 2, 3])
    rev = reverse_array(arr)
    assert np.array_equal(rev, np.array([3, 2, 1]))

def test_sort_array():
    arr = np.array([3, 1, 2])
    assert np.array_equal(sort_array(arr), np.array([1, 2, 3]))
    assert np.array_equal(sort_array(arr, descending=True), np.array([3, 2, 1]))

def test_get_n_largest():
    arr = np.array([10, 5, 20, 1, 8])
    largest = get_n_largest(arr, 2)
    assert np.array_equal(largest, np.array([20, 10]))
