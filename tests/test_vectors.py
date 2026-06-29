import pytest
import numpy as np
from numpy_tutorial.vectors import (
    add_vectors,
    multiply_vectors,
    dot_product,
    vector_length,
    normalize_vector,
    vector_angle,
    cross_product
)

def test_add_vectors():
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    assert np.array_equal(add_vectors(v1, v2), np.array([5, 7, 9]))

def test_multiply_vectors():
    v1 = np.array([1, 2, 3])
    v2 = np.array([2, 3, 4])
    assert np.array_equal(multiply_vectors(v1, v2), np.array([2, 6, 12]))

def test_dot_product():
    v1 = np.array([1, 2])
    v2 = np.array([3, 4])
    assert dot_product(v1, v2) == 11

def test_vector_length():
    v = np.array([3, 4])
    assert np.isclose(vector_length(v), 5.0)

def test_normalize_vector():
    v = np.array([3, 4])
    norm_v = normalize_vector(v)
    assert np.isclose(vector_length(norm_v), 1.0)

def test_vector_angle():
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    angle = vector_angle(v1, v2)
    assert np.isclose(angle, np.pi / 2)

def test_cross_product():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    cross = cross_product(v1, v2)
    assert np.array_equal(cross, np.array([0, 0, 1]))
