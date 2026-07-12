"""
Entrenamiento y evaluación de modelos supervisados.

Este módulo reutiliza la carga de datos y el feature engineering
desarrollados en los pasos anteriores del proyecto.
"""

from pathlib import Path

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from cargar_datos import cargar_base, cargar_config
from ft_engineering import (
    aplicar_feature_engineering,
    crear_preprocesador,
    separar_features_target,
)

def preparar_datos():
    """
    Carga la configuración y la base, aplica feature engineering
    y divide los datos en entrenamiento y prueba.

    La división conserva la proporción de clases del target mediante
    stratify, debido al fuerte desbalance de Pago_atiempo.
    """

    ruta_config = Path(__file__).resolve().parent / "config.json"

    config = cargar_config(ruta_config)
    df = cargar_base(ruta_config)

    df_transformado = aplicar_feature_engineering(df)

    X, y = separar_features_target(
        df_transformado,
        target=config["target"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preparar_datos()

    print("Datos preparados correctamente.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    print("\nDistribución y_train:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).round(4))

    print("\nDistribución y_test:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True).round(4))
