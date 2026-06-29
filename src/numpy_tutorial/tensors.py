import numpy as np

def create_3d_tensor(shape: tuple) -> np.ndarray:
    """
    Cria um tensor tridimensional (3D) preenchido com números aleatórios (distribuição normal padrão).

    Args:
        shape (tuple): Tupla (profundidade, linhas, colunas).

    Returns:
        np.ndarray: Tensor 3D randômico.
    """
    return np.random.randn(*shape)

def tensor_addition(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Adiciona dois tensores elemento a elemento.

    Args:
        t1 (np.ndarray): Tensor 1.
        t2 (np.ndarray): Tensor 2.

    Returns:
        np.ndarray: Tensor resultante da soma.
    """
    return np.add(t1, t2)

def tensor_dot_product(t1: np.ndarray, t2: np.ndarray, axes: int = 1) -> np.ndarray:
    """
    Calcula o produto escalar sobre tensores (contração tensorial generalizada).

    Args:
        t1 (np.ndarray): Tensor 1.
        t2 (np.ndarray): Tensor 2.
        axes (int): Número de eixos sobre os quais contrair (padrão é 1).

    Returns:
        np.ndarray: Tensor resultante do produto escalar.
    """
    return np.tensordot(t1, t2, axes=axes)
