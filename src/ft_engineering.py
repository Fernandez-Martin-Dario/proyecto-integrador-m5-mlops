# ============================================================
# Ingeniería de características - Proyecto Integrador M5
# ============================================================

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Categorías válidas detectadas en el EDA para tendencia_ingresos
CATEGORIAS_VALIDAS_TENDENCIA = [
    "Creciente",
    "Decreciente",
    "Estable"
]
def normalizar_tendencia_ingresos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza la variable tendencia_ingresos y conserva información
    sobre valores inconsistentes detectados durante el EDA.

    Durante el EDA se observó que tendencia_ingresos contiene:
    - categorías válidas: Creciente, Decreciente, Estable
    - valores nulos
    - valores numéricos o inconsistentes

    Estrategia:
    1. Se conserva la columna tendencia_ingresos solo con categorías válidas.
    2. Los valores no válidos se transforman en nulos.
    3. Se crea una nueva variable binaria:
       tendencia_ingresos_inconsistente
       que indica si el valor original era inconsistente.

    Esto permite limpiar la variable categórica sin perder la señal
    de que existía una inconsistencia en el dato original.
    """

    df_transformado = df.copy()

    columna = "tendencia_ingresos"
    columna_inconsistente = "tendencia_ingresos_inconsistente"

    if columna not in df_transformado.columns:
        return df_transformado

    tendencia_original = df_transformado[columna]

    tendencia_limpia = tendencia_original.astype("object")

    tendencia_limpia = tendencia_limpia.where(
        tendencia_limpia.notna(),
        np.nan
    )

    tendencia_limpia = tendencia_limpia.apply(
        lambda valor: valor.strip() if isinstance(valor, str) else valor
    )

    es_nulo_original = tendencia_original.isna()

    es_categoria_valida = tendencia_limpia.isin(
        CATEGORIAS_VALIDAS_TENDENCIA
    )

    df_transformado[columna_inconsistente] = np.where(
        (~es_nulo_original) & (~es_categoria_valida),
        1,
        0
    )

    df_transformado[columna] = np.where(
        es_categoria_valida,
        tendencia_limpia,
        np.nan
    )

    return df_transformado
def crear_variables_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables temporales a partir de fecha_prestamo.

    La fecha original no se utiliza directamente para el modelado, ya que
    los modelos supervisados no trabajan correctamente con valores datetime
    crudos. En su lugar, se generan variables numéricas interpretables.
    """

    df_transformado = df.copy()

    columna_fecha = "fecha_prestamo"

    if columna_fecha not in df_transformado.columns:
        return df_transformado

    df_transformado[columna_fecha] = pd.to_datetime(
        df_transformado[columna_fecha],
        errors="coerce"
    )

    df_transformado["anio_prestamo"] = df_transformado[columna_fecha].dt.year
    df_transformado["mes_prestamo"] = df_transformado[columna_fecha].dt.month
    df_transformado["dia_semana_prestamo"] = df_transformado[columna_fecha].dt.dayofweek

    df_transformado = df_transformado.drop(columns=[columna_fecha])

    return df_transformado
def aplicar_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las transformaciones de ingeniería de características definidas
    para el proyecto.

    Esta función centraliza el proceso para asegurar que las mismas
    transformaciones se apliquen de forma consistente durante entrenamiento,
    evaluación y futuras predicciones.
    """

    df_transformado = df.copy()

    df_transformado = normalizar_tendencia_ingresos(df_transformado)
    df_transformado = crear_variables_fecha(df_transformado)

    return df_transformado
def separar_features_target(
    df: pd.DataFrame,
    target: str = "Pago_atiempo"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa el dataset en variables predictoras X y variable objetivo y.

    La variable objetivo no debe ingresar al pipeline de preprocesamiento
    como predictor, ya que eso produciría fuga de información.
    """

    if target not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    return X, y
def crear_preprocesador() -> ColumnTransformer:
    """
    Crea el preprocesador para las variables predictoras.

    Estrategia:
    - Variables numéricas:
        imputación de nulos con mediana
        escalado con StandardScaler

    - Variables categóricas:
        imputación de nulos con "Sin dato"
        codificación con OneHotEncoder

    La variable tipo_credito se trata como categórica porque, aunque está
    almacenada como número, representa un código de tipo de crédito y no
    una magnitud numérica continua.
    """

    columnas_categoricas = [
        "tipo_credito",
        "tipo_laboral",
        "tendencia_ingresos"
    ]

    columnas_numericas = [
        "capital_prestado",
        "plazo_meses",
        "edad_cliente",
        "salario_cliente",
        "total_otros_prestamos",
        "cuota_pactada",
        "puntaje",
        "puntaje_datacredito",
        "cant_creditosvigentes",
        "huella_consulta",
        "saldo_mora",
        "saldo_total",
        "saldo_principal",
        "saldo_mora_codeudor",
        "creditos_sectorFinanciero",
        "creditos_sectorCooperativo",
        "creditos_sectorReal",
        "promedio_ingresos_datacredito",
        "tendencia_ingresos_inconsistente",
        "anio_prestamo",
        "mes_prestamo",
        "dia_semana_prestamo"
    ]

    pipeline_numerico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler())
        ]
    )

    pipeline_categorico = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="constant", fill_value="Sin dato")),
            ("codificador", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerico, columnas_numericas),
            ("categoricas", pipeline_categorico, columnas_categoricas)
        ]
    )

    return preprocesador
