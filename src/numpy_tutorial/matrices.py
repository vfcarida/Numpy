import numpy as np

def create_identity_matrix(size: int) -> np.ndarray:
    """
    Cria uma matriz Identidade quadrada (diagonal com 1, restante 0).

    Args:
        size (int): Ordem da matriz (N x N).

    Returns:
        np.ndarray: A matriz Identidade.
    """
    return np.eye(size)

def transpose_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Obtém a transposta de uma matriz, invertendo linhas e colunas.

    Args:
        matrix (np.ndarray): A matriz original.

    Returns:
        np.ndarray: A matriz transposta.
    """
    return np.transpose(matrix)

def matrix_determinant(matrix: np.ndarray) -> float:
    """
    Calcula o determinante de uma matriz quadrada.

    Args:
        matrix (np.ndarray): Matriz quadrada.

    Returns:
        float: Valor do determinante.
    """
    return np.linalg.det(matrix)

def matrix_inverse(matrix: np.ndarray) -> np.ndarray:
    """
    Calcula a matriz inversa. Somente aplicável se a matriz for quadrada e o determinante não nulo.

    Args:
        matrix (np.ndarray): A matriz original.

    Returns:
        np.ndarray: Matriz inversa.
    """
    return np.linalg.inv(matrix)

def matrix_rank(matrix: np.ndarray) -> int:
    """
    Retorna o posto (rank) da matriz (número de linhas ou colunas linearmente independentes).

    Args:
        matrix (np.ndarray): Matriz de entrada.

    Returns:
        int: Posto da matriz.
    """
    return np.linalg.matrix_rank(matrix)

def kronecker_product(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Realiza o produto de Kronecker entre duas matrizes (produto tensorial).

    Args:
        m1 (np.ndarray): Matriz 1.
        m2 (np.ndarray): Matriz 2.

    Returns:
        np.ndarray: Matriz de blocos resultante.
    """
    return np.kron(m1, m2)

def matrix_dot_product(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """
    Calcula a multiplicação tradicional de matrizes (Dot Product).
    O número de colunas de m1 deve ser igual ao de linhas de m2.

    Args:
        m1 (np.ndarray): Matriz 1 (N x K).
        m2 (np.ndarray): Matriz 2 (K x M).

    Returns:
        np.ndarray: Matriz resultante (N x M).
    """
    return np.matmul(m1, m2)

def solve_linear_equations(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Resolve um sistema de equações lineares da forma AX = B.

    Args:
        A (np.ndarray): Matriz dos coeficientes (quadrada N x N).
        B (np.ndarray): Vetor ou matriz dos termos independentes (N x M).

    Returns:
        np.ndarray: Solução X.
    """
    return np.linalg.solve(A, B)
