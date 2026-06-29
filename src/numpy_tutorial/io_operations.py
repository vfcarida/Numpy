import numpy as np

def save_array_to_npy(array: np.ndarray, filepath: str) -> None:
    """
    Salva um array Numpy em um arquivo binário formato .npy.

    Args:
        array (np.ndarray): O array para salvar.
        filepath (str): Caminho do arquivo de destino.
    """
    np.save(filepath, array)

def load_array_from_npy(filepath: str) -> np.ndarray:
    """
    Carrega um array Numpy salvo no formato binário .npy.

    Args:
        filepath (str): Caminho do arquivo a ser lido.

    Returns:
        np.ndarray: O array carregado.
    """
    return np.load(filepath)

def save_to_txt(array: np.ndarray, filepath: str, delimiter: str = ',') -> None:
    """
    Salva um array Numpy em um arquivo de texto.

    Args:
        array (np.ndarray): O array a ser salvo.
        filepath (str): Caminho do arquivo (ex: .txt ou .csv).
        delimiter (str): Separador de colunas (padrão é vírgula).
    """
    np.savetxt(filepath, array, delimiter=delimiter)

def load_from_txt(filepath: str, delimiter: str = ',') -> np.ndarray:
    """
    Lê um array Numpy de um arquivo de texto.

    Args:
        filepath (str): Caminho do arquivo.
        delimiter (str): Delimitador usado no arquivo.

    Returns:
        np.ndarray: Array resultante da leitura.
    """
    return np.loadtxt(filepath, delimiter=delimiter)
