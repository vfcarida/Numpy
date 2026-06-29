import numpy as np

def add_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Realiza a soma vetorial entre dois vetores.

    Args:
        v1 (np.ndarray): Vetor 1.
        v2 (np.ndarray): Vetor 2.

    Returns:
        np.ndarray: Vetor resultante da soma.
    """
    return np.add(v1, v2)

def multiply_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Realiza a multiplicação elemento a elemento (Hadamard product) entre vetores.

    Args:
        v1 (np.ndarray): Vetor 1.
        v2 (np.ndarray): Vetor 2.

    Returns:
        np.ndarray: Vetor resultante da multiplicação escalar elemento a elemento.
    """
    return np.multiply(v1, v2)

def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcula o Produto Escalar (Dot Product) de dois vetores.

    Args:
        v1 (np.ndarray): Vetor 1.
        v2 (np.ndarray): Vetor 2.

    Returns:
        float: Resultado escalar.
    """
    return np.dot(v1, v2)

def vector_length(v: np.ndarray) -> float:
    """
    Calcula o comprimento (norma L2) de um vetor.

    Args:
        v (np.ndarray): O vetor.

    Returns:
        float: Comprimento do vetor.
    """
    return np.linalg.norm(v)

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normaliza um vetor para que seu comprimento seja igual a 1 (vetor unitário).

    Args:
        v (np.ndarray): O vetor original.

    Returns:
        np.ndarray: O vetor normalizado.
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calcula o ângulo (em radianos) entre dois vetores usando o arco-cosseno do dot product normalizado.

    Args:
        v1 (np.ndarray): Vetor 1.
        v2 (np.ndarray): Vetor 2.

    Returns:
        float: Ângulo em radianos.
    """
    unit_v1 = normalize_vector(v1)
    unit_v2 = normalize_vector(v2)
    # np.clip garante que o valor de dot product fique dentro do domínio de arccos [-1.0, 1.0]
    dot_prod = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    return np.arccos(dot_prod)

def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calcula o produto vetorial (Cross Product) para vetores no espaço 3D.

    Args:
        v1 (np.ndarray): Vetor 1 (tamanho 3).
        v2 (np.ndarray): Vetor 2 (tamanho 3).

    Returns:
        np.ndarray: O vetor resultante ortogonal a v1 e v2.
    """
    return np.cross(v1, v2)
