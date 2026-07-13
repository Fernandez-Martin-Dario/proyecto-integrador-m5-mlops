"""
Entrenamiento y evaluación de modelos supervisados.

Este módulo reutiliza la carga de datos y el feature engineering
desarrollados en los pasos anteriores del proyecto.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

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
        target=config["target"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def build_model(
    modelo: BaseEstimator,
    columnas_excluir: list[str] | None = None,
) -> Pipeline:
    """
    Construye un pipeline completo de preprocesamiento y modelado.

    Parameters
    ----------
    modelo : BaseEstimator
        Modelo supervisado compatible con la API de scikit-learn.

    columnas_excluir : list[str] | None
        Variables que no deben utilizarse durante el preprocesamiento
        ni el entrenamiento del modelo.

    Returns
    -------
    Pipeline
        Pipeline compuesto por el preprocesador y el modelo.
    """

    pipeline_modelo = Pipeline(
        steps=[
            (
                "preprocesador",
                crear_preprocesador(
                    columnas_excluir=columnas_excluir,
                ),
            ),
            (
                "modelo",
                modelo,
            ),
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
        "accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
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
        target_names=[
            "No paga a tiempo",
            "Paga a tiempo",
        ],
        output_dict=True,
        zero_division=0,
    )

    return resumen, matriz, reporte


def graficar_comparacion_modelos(
    tabla_modelos: pd.DataFrame,
) -> None:
    """
    Grafica las métricas principales de los modelos sin puntaje.

    Parameters
    ----------
    tabla_modelos : pd.DataFrame
        Tabla con las métricas de los modelos comparados.
    """

    metricas = [
        "precision_clase_0",
        "recall_clase_0",
        "f1_clase_0",
        "roc_auc_clase_0",
    ]

    datos_grafico = (
        tabla_modelos
        .set_index("modelo")[metricas]
    )

    eje = datos_grafico.plot(
        kind="bar",
        figsize=(12, 6),
    )

    eje.set_title(
        "Comparación de modelos sin la variable puntaje"
    )
    eje.set_xlabel("Modelo")
    eje.set_ylabel("Valor de la métrica")
    eje.set_ylim(0, 1)

    plt.xticks(
        rotation=15,
        ha="right",
    )
    plt.legend(title="Métrica")
    plt.tight_layout()
    plt.show()


def optimizar_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """
    Optimiza Gradient Boosting mediante validación cruzada estratificada.

    La búsqueda utiliza únicamente los datos de entrenamiento y prioriza
    el F1 de la clase 0, correspondiente a quienes no pagan a tiempo.
    """

    modelo_pipeline = build_model(
        modelo=GradientBoostingClassifier(
            random_state=42,
        ),
        columnas_excluir=["puntaje"],
    )

    parametros = {
        "modelo__n_estimators": [
            100,
            200,
        ],
        "modelo__learning_rate": [
            0.03,
            0.05,
        ],
        "modelo__max_depth": [
            2,
            3,
        ],
        "modelo__min_samples_leaf": [
            1,
            5,
        ],
    }

    validacion_cruzada = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    metrica_f1_clase_0 = make_scorer(
        f1_score,
        pos_label=0,
        zero_division=0,
    )

    busqueda = GridSearchCV(
        estimator=modelo_pipeline,
        param_grid=parametros,
        scoring=metrica_f1_clase_0,
        cv=validacion_cruzada,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    pesos_entrenamiento = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    busqueda.fit(
        X_train,
        y_train,
        modelo__sample_weight=pesos_entrenamiento,
    )

    return busqueda


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

    # ========================================================
    # Modelo de referencia
    # ========================================================

    modelo_base = build_model(
        modelo=DummyClassifier(
            strategy="most_frequent",
        )
    )

    modelo_base.fit(
        X_train,
        y_train,
    )

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

    # ========================================================
    # Regresión Logística con puntaje
    # ========================================================

    regresion_logistica = build_model(
        modelo=LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        )
    )

    regresion_logistica.fit(
        X_train,
        y_train,
    )

    resumen_logistica, matriz_logistica, _ = summarize_classification(
        nombre_modelo="Regresión Logística",
        modelo=regresion_logistica,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nResultados de Regresión Logística:")
    print(pd.Series(resumen_logistica))

    print("\nMatriz de confusión de Regresión Logística:")
    print(matriz_logistica)

    # ========================================================
    # Regresión Logística sin puntaje
    # ========================================================

    regresion_logistica_sin_puntaje = build_model(
        modelo=LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000,
        ),
        columnas_excluir=["puntaje"],
    )

    regresion_logistica_sin_puntaje.fit(
        X_train,
        y_train,
    )

    (
        resumen_logistica_sin_puntaje,
        matriz_logistica_sin_puntaje,
        _,
    ) = summarize_classification(
        nombre_modelo="Regresión Logística sin puntaje",
        modelo=regresion_logistica_sin_puntaje,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nResultados de Regresión Logística sin puntaje:")
    print(pd.Series(resumen_logistica_sin_puntaje))

    print("\nMatriz de confusión sin puntaje:")
    print(matriz_logistica_sin_puntaje)

    # ========================================================
    # Random Forest sin puntaje
    # ========================================================

    random_forest_sin_puntaje = build_model(
        modelo=RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        columnas_excluir=["puntaje"],
    )

    random_forest_sin_puntaje.fit(
        X_train,
        y_train,
    )

    (
        resumen_random_forest,
        matriz_random_forest,
        _,
    ) = summarize_classification(
        nombre_modelo="Random Forest sin puntaje",
        modelo=random_forest_sin_puntaje,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nResultados de Random Forest sin puntaje:")
    print(pd.Series(resumen_random_forest))

    print("\nMatriz de confusión de Random Forest sin puntaje:")
    print(matriz_random_forest)

    # ========================================================
    # Gradient Boosting sin puntaje
    # ========================================================

    pesos_entrenamiento = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    gradient_boosting_sin_puntaje = build_model(
        modelo=GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        columnas_excluir=["puntaje"],
    )

    gradient_boosting_sin_puntaje.fit(
        X_train,
        y_train,
        modelo__sample_weight=pesos_entrenamiento,
    )

    (
        resumen_gradient_boosting,
        matriz_gradient_boosting,
        _,
    ) = summarize_classification(
        nombre_modelo="Gradient Boosting sin puntaje",
        modelo=gradient_boosting_sin_puntaje,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nResultados de Gradient Boosting sin puntaje:")
    print(pd.Series(resumen_gradient_boosting))

    print("\nMatriz de confusión de Gradient Boosting sin puntaje:")
    print(matriz_gradient_boosting)

    # ========================================================
    # Optimización de Gradient Boosting
    # ========================================================

    busqueda_gradient_boosting = optimizar_gradient_boosting(
        X_train=X_train,
        y_train=y_train,
    )

    print("\nMejores hiperparámetros de Gradient Boosting:")
    print(busqueda_gradient_boosting.best_params_)

    print("\nMejor F1 promedio de clase 0 en validación cruzada:")
    print(round(busqueda_gradient_boosting.best_score_, 4))

    mejor_gradient_boosting = (
        busqueda_gradient_boosting.best_estimator_
    )

    (
        resumen_gradient_boosting_optimizado,
        matriz_gradient_boosting_optimizado,
        _,
    ) = summarize_classification(
        nombre_modelo="Gradient Boosting optimizado sin puntaje",
        modelo=mejor_gradient_boosting,
        X_test=X_test,
        y_test=y_test,
    )

    print("\nResultados de Gradient Boosting optimizado:")
    print(pd.Series(resumen_gradient_boosting_optimizado))

    print("\nMatriz de confusión de Gradient Boosting optimizado:")
    print(matriz_gradient_boosting_optimizado)

    # ========================================================
    # Tabla comparativa
    # ========================================================

    tabla_resultados = pd.DataFrame(
        [
            {
                **resumen_base,
                "escenario": "Referencia",
            },
            {
                **resumen_logistica,
                "escenario": "Con puntaje sospechoso",
            },
            {
                **resumen_logistica_sin_puntaje,
                "escenario": "Sin puntaje",
            },
            {
                **resumen_random_forest,
                "escenario": "Sin puntaje",
            },
            {
                **resumen_gradient_boosting,
                "escenario": "Sin puntaje",
            },
              {
                **resumen_gradient_boosting_optimizado,
                "escenario": "Sin puntaje",
            },
        ]
    )

    columnas_tabla = [
        "modelo",
        "escenario",
        "accuracy",
        "precision_clase_0",
        "recall_clase_0",
        "f1_clase_0",
        "roc_auc_clase_0",
    ]

    tabla_resultados = tabla_resultados[columnas_tabla]

    print("\nTabla comparativa completa:")
    print(
        tabla_resultados
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # Ranking conservador sin puntaje
    # ========================================================

    tabla_modelos_sin_puntaje = (
        tabla_resultados[
            tabla_resultados["escenario"] == "Sin puntaje"
        ]
        .sort_values(
            by=[
                "f1_clase_0",
                "roc_auc_clase_0",
                "recall_clase_0",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nRanking de modelos sin puntaje:")
    print(
        tabla_modelos_sin_puntaje
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # Gráfico comparativo
    # ========================================================

    graficar_comparacion_modelos(
        tabla_modelos_sin_puntaje
    )

    # ========================================================
    # Selección del modelo final
    # ========================================================

    modelos_entrenados = {
        "Regresión Logística sin puntaje": (
            regresion_logistica_sin_puntaje
        ),
        "Random Forest sin puntaje": (
            random_forest_sin_puntaje
        ),
        "Gradient Boosting sin puntaje": (
            gradient_boosting_sin_puntaje
        ),
        "Gradient Boosting optimizado sin puntaje": (
            mejor_gradient_boosting
        ),
    }

    nombre_modelo_final = (
        tabla_modelos_sin_puntaje.iloc[0]["modelo"]
    )

    modelo_final = modelos_entrenados[nombre_modelo_final]

    print("\nModelo final seleccionado:")
    print(nombre_modelo_final)

    print(
        "Criterio de selección: mayor F1 de la clase 0 "
        "entre los modelos sin puntaje."
    )