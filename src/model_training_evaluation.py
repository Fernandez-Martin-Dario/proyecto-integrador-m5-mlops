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

from sklearn.dummy import DummyClassifier

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
def build_model(modelo: BaseEstimator) -> Pipeline:
    """
    Construye un pipeline completo de preprocesamiento y modelado.

    Parameters
    ----------
    modelo : BaseEstimator
        Modelo supervisado compatible con la API de scikit-learn.

    Returns
    -------
    Pipeline
        Pipeline compuesto por el preprocesador y el modelo.
    """

    pipeline_modelo = Pipeline(
        steps=[
            ("preprocesador", crear_preprocesador()),
            ("modelo", modelo),
        ]
    )

    return pipeline_modelo
def summarize_classification(
    nombre_modelo: str,
    modelo: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    """
    Resume el desempeño de un modelo de clasificación.

    Se priorizan las métricas de la clase 0, que representa a los clientes
    que no pagaron a tiempo y constituye la clase minoritaria.
    """

    y_pred = modelo.predict(X_test)

    resumen = {
        "modelo": nombre_modelo,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_clase_0": precision_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "recall_clase_0": recall_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
        "f1_clase_0": f1_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0,
        ),
    }

    roc_auc_clase_0 = float("nan")

    if hasattr(modelo, "predict_proba"):
        probabilidades = modelo.predict_proba(X_test)

        clases_modelo = modelo.named_steps["modelo"].classes_
        indice_clase_0 = list(clases_modelo).index(0)

        y_test_clase_0 = (y_test == 0).astype(int)

        roc_auc_clase_0 = roc_auc_score(
            y_test_clase_0,
            probabilidades[:, indice_clase_0],
        )

    resumen["roc_auc_clase_0"] = roc_auc_clase_0

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    reporte = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No paga a tiempo", "Paga a tiempo"],
        output_dict=True,
        zero_division=0,
    )

    return resumen, matriz, reporte
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

    modelo_base = build_model(
        DummyClassifier(strategy="most_frequent")
    )

    modelo_base.fit(X_train, y_train)

    resumen_base, matriz_base, _ = summarize_classification(
        nombre_modelo="DummyClassifier",
        modelo=modelo_base,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nResultados del modelo base:")
    print(pd.Series(resumen_base))

    print("\nMatriz de confusión del modelo base:")
    print(matriz_base)