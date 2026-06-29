import pytest
import numpy as np
from numpy_tutorial.matrices import (
    create_identity_matrix,
    transpose_matrix,
    matrix_determinant,
    matrix_inverse,
    matrix_rank,
    kronecker_product,
    matrix_dot_product,
    solve_linear_equations
)

def test_create_identity_matrix():
    i_mat = create_identity_matrix(3)
    assert np.array_equal(i_mat, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

def test_transpose_matrix():
    mat = np.array([[1, 2], [3, 4]])
    trans = transpose_matrix(mat)
    assert np.array_equal(trans, np.array([[1, 3], [2, 4]]))

def test_matrix_determinant():
    mat = np.array([[1, 2], [3, 4]])
    det = matrix_determinant(mat)
    assert np.isclose(det, -2.0)

def test_matrix_inverse():
    mat = np.array([[1, 2], [3, 4]])
    inv = matrix_inverse(mat)
    identity = np.matmul(mat, inv)
    assert np.allclose(identity, np.eye(2))

def test_matrix_rank():
    mat = np.array([[1, 2], [2, 4]]) # Linhas dependentes
    assert matrix_rank(mat) == 1

def test_kronecker_product():
    m1 = np.array([[1, 2], [3, 4]])
    m2 = np.array([[0, 5], [6, 7]])
    kron = kronecker_product(m1, m2)
    assert kron.shape == (4, 4)

def test_matrix_dot_product():
    m1 = np.array([[1, 2], [3, 4]])
    m2 = np.array([[2, 0], [1, 2]])
    res = matrix_dot_product(m1, m2)
    expected = np.array([[4, 4], [10, 8]])
    assert np.array_equal(res, expected)

def test_solve_linear_equations():
    # 3x + y = 9
    # x + 2y = 8
    A = np.array([[3, 1], [1, 2]])
    B = np.array([9, 8])
    X = solve_linear_equations(A, B)
    assert np.allclose(X, np.array([2, 3]))
