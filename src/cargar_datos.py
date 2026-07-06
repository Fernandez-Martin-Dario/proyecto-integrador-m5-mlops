import json
from pathlib import Path

import pandas as pd

def cargar_config(ruta_config: str = "src/config.json") -> dict:
    """
    Carga el archivo config.json del proyecto.

    Parameters
    ----------
    ruta_config : str
        Ruta al archivo de configuración.

    Returns
    -------
    dict
        Diccionario con la configuración del proyecto.
    """

    ruta_config = Path(ruta_config)

    if not ruta_config.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta_config}")

    with open(ruta_config, "r", encoding="utf-8") as archivo:
        config = json.load(archivo)

    return config

def cargar_base(ruta_config: str = "src/config.json") -> pd.DataFrame:
    """
    Carga la base de datos definida en config.json.

    Parameters
    ----------
    ruta_config : str
        Ruta al archivo de configuración.

    Returns
    -------
    pd.DataFrame
        DataFrame con la base cargada.
    """

    config = cargar_config(ruta_config)

    ruta_config = Path(ruta_config)
    carpeta_config = ruta_config.parent

    ruta_base = carpeta_config / config["base_datos"]

    if not ruta_base.exists():
        raise FileNotFoundError(f"No se encontró la base de datos: {ruta_base}")

    df = pd.read_excel(ruta_base)

    return df

if __name__ == "__main__":

    config = cargar_config()
    df = cargar_base()

    target = config["target"]

    if target not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo: {target}")

    print("Base cargada correctamente.")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    print(f"Target encontrado: {target}")