"""Funciones para Unir los datos recolectados
"""
import os
from tqdm import tqdm
import pandas as pd

DEFAULT_EXTENSION = ".json"
DEFAULT_ENCODING = "utf-8"

def merge_data(dir_path: str):
    """Unir los datos recolectados

    :param dir_path: Ruta de los datos recolectados
    :type dir_path: str
    :return: DataFrame con los datos unidos
    :rtype: DataFrame
    """

    print("merge data from ", dir_path)
    dfs = []
    for filename in tqdm(os.listdir(dir_path)):
        if filename.endswith(DEFAULT_EXTENSION):
            full_path = os.path.join(dir_path, filename) # Recuperar la ruta completa
            df = pd.read_json(full_path, encoding=DEFAULT_ENCODING) # Cargar los datos
            term = filename.replace(DEFAULT_EXTENSION, "")
            df["termino"] = "_".join(term.lower().split()) # Agregar columna con el témino de búsqueda
            dfs.append(df)
    merged_df =  pd.concat(dfs, ignore_index=True)
    return merged_df
