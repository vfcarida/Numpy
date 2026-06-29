import numpy as np
from typing import List, Tuple, Any

def create_array(elements: List[Any]) -> np.ndarray:
    """
    Cria um array Numpy a partir de uma lista Python.

    Args:
        elements (List[Any]): Lista de elementos para conversão.

    Returns:
        np.ndarray: O array instanciado.
    """
    return np.array(elements)

def create_2d_array(rows: int, cols: int, fill_value: int = 0) -> np.ndarray:
    """
    Cria um array 2D (matriz) preenchido com um valor específico.

    Args:
        rows (int): Número de linhas.
        cols (int): Número de colunas.
        fill_value (int): Valor para preencher as posições. Padrão é 0.

    Returns:
        np.ndarray: Array 2D gerado.
    """
    return np.full((rows, cols), fill_value)

def replace_elements(array: np.ndarray, condition_val: Any, new_val: Any) -> np.ndarray:
    """
    Substitui todos os elementos iguais a 'condition_val' por 'new_val'.

    Args:
        array (np.ndarray): O array original.
        condition_val (Any): Valor a ser procurado.
        new_val (Any): Valor substituto.

    Returns:
        np.ndarray: Um novo array com os elementos substituídos.
    """
    # Copia o array para evitar efeitos colaterais mutáveis (boas práticas)
    new_array = array.copy()
    new_array[new_array == condition_val] = new_val
    return new_array

def find_common_elements(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Encontra a interseção (elementos comuns) entre dois arrays.

    Args:
        array1 (np.ndarray): Primeiro array.
        array2 (np.ndarray): Segundo array.

    Returns:
        np.ndarray: Elementos comuns ordenados e únicos.
    """
    return np.intersect1d(array1, array2)

def reverse_array(array: np.ndarray) -> np.ndarray:
    """
    Inverte a ordem dos elementos de um array 1D.

    Args:
        array (np.ndarray): Array a ser invertido.

    Returns:
        np.ndarray: O array invertido.
    """
    return array[::-1]

def sort_array(array: np.ndarray, descending: bool = False) -> np.ndarray:
    """
    Ordena os elementos de um array.

    Args:
        array (np.ndarray): Array para ordenação.
        descending (bool): Define se a ordem será decrescente.

    Returns:
        np.ndarray: O array ordenado.
    """
    sorted_arr = np.sort(array)
    if descending:
        return sorted_arr[::-1]
    return sorted_arr

def get_n_largest(array: np.ndarray, n: int) -> np.ndarray:
    """
    Retorna os N maiores valores do array 1D.

    Args:
        array (np.ndarray): O array de entrada.
        n (int): Quantidade de elementos.

    Returns:
        np.ndarray: Array contendo os N maiores valores.
    """
    # Retorna usando particionamento rápido (O(N)) e então ordenando o resultado (O(K log K))
    if n >= len(array):
        return sort_array(array, descending=True)
    partitioned = np.partition(array, -n)[-n:]
    return np.sort(partitioned)[::-1]
