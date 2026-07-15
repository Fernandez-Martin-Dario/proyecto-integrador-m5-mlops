from __future__ import annotations
import pandas as pd
import numpy as np
from src.cargar_datos import cargar_base, cargar_config


def separar_periodos_monitoreo(
    df: pd.DataFrame,
    fecha_corte: str,
    columna_fecha: str = "fecha_prestamo",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el dataset en un período histórico de referencia
    y un período actual para monitoreo.
    """

    datos = df.copy()

    datos[columna_fecha] = pd.to_datetime(
        datos[columna_fecha],
        errors="coerce",
    )

    fecha_corte = pd.Timestamp(fecha_corte)

    referencia = datos.loc[
        datos[columna_fecha] < fecha_corte
    ].copy()

    actual = datos.loc[
        datos[columna_fecha] >= fecha_corte
    ].copy()

    if referencia.empty:
        raise ValueError(
            "El período de referencia quedó vacío."
        )

    if actual.empty:
        raise ValueError(
            "El período actual quedó vacío."
        )

    return referencia, actual

def calcular_psi_numerico(
    referencia: pd.Series,
    actual: pd.Series,
    cantidad_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calcula el Population Stability Index (PSI)
    entre una variable numérica histórica y su período actual.
    """

    serie_referencia = pd.to_numeric(
        referencia,
        errors="coerce",
    )

    serie_actual = pd.to_numeric(
        actual,
        errors="coerce",
    )

    if serie_referencia.empty or serie_actual.empty:
        raise ValueError(
            "Las series de referencia y actual no pueden estar vacías."
        )

    if serie_referencia.dropna().empty:
        raise ValueError(
            "La serie de referencia no contiene valores numéricos válidos."
        )

    if serie_actual.dropna().empty:
        raise ValueError(
            "La serie actual no contiene valores numéricos válidos."
        )

    cuantiles = np.linspace(
        0,
        1,
        cantidad_bins + 1,
    )

    limites = serie_referencia.dropna().quantile(
        cuantiles
    ).to_numpy(dtype=float)

    limites = np.unique(limites)

    if len(limites) < 2:
        valor_constante = float(
            serie_referencia.dropna().iloc[0]
        )

        limites = np.array(
            [-np.inf, valor_constante, np.inf]
        )

    else:
        limites[0] = -np.inf
        limites[-1] = np.inf

    bins_referencia = pd.cut(
        serie_referencia,
        bins=limites,
        include_lowest=True,
        duplicates="drop",
    )

    bins_actual = pd.cut(
        serie_actual,
        bins=limites,
        include_lowest=True,
        duplicates="drop",
    )

    proporcion_referencia = (
        bins_referencia.value_counts(sort=False)
        / len(serie_referencia)
    )

    proporcion_actual = (
        bins_actual.value_counts(sort=False)
        / len(serie_actual)
    )

    distribucion_referencia = np.append(
        proporcion_referencia.to_numpy(dtype=float),
        serie_referencia.isna().mean(),
    )

    distribucion_actual = np.append(
        proporcion_actual.to_numpy(dtype=float),
        serie_actual.isna().mean(),
    )

    distribucion_referencia = np.clip(
        distribucion_referencia,
        epsilon,
        None,
    )

    distribucion_actual = np.clip(
        distribucion_actual,
        epsilon,
        None,
    )

    psi = np.sum(
        (distribucion_actual - distribucion_referencia)
        * np.log(
            distribucion_actual
            / distribucion_referencia
        )
    )

    return float(psi)
def clasificar_psi(valor_psi: float) -> str:
    """
    Clasifica la magnitud del cambio según el valor de PSI.
    """

    if valor_psi < 0.10:
        return "Estable"

    if valor_psi < 0.25:
        return "Cambio moderado"

    return "Cambio importante"


def analizar_drift_numerico(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    columnas_excluir: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcula el PSI para todas las variables numéricas
    presentes en los períodos de referencia y actual.
    """

    columnas_excluir = set(columnas_excluir or [])

    columnas_referencia = set(
        referencia.select_dtypes(include=np.number).columns
    )

    columnas_actual = set(
        actual.select_dtypes(include=np.number).columns
    )

    columnas_numericas = sorted(
        (columnas_referencia & columnas_actual)
        - columnas_excluir
    )

    resultados = []

    for columna in columnas_numericas:
        valor_psi = calcular_psi_numerico(
            referencia[columna],
            actual[columna],
        )

        resultados.append(
            {
                "variable": columna,
                "psi": valor_psi,
                "clasificacion": clasificar_psi(valor_psi),
            }
        )

    return (
        pd.DataFrame(resultados)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )
def calcular_psi_categorico(
    referencia: pd.Series,
    actual: pd.Series,
    epsilon: float = 1e-6,
) -> float:
    """
    Calcula el Population Stability Index (PSI)
    entre dos distribuciones categóricas.
    """

    serie_referencia = (
        referencia
        .astype("string")
        .fillna("Sin dato")
    )

    serie_actual = (
        actual
        .astype("string")
        .fillna("Sin dato")
    )

    if serie_referencia.empty or serie_actual.empty:
        raise ValueError(
            "Las series de referencia y actual no pueden estar vacías."
        )

    categorias = sorted(
        set(serie_referencia.unique())
        | set(serie_actual.unique())
    )

    proporcion_referencia = (
        serie_referencia
        .value_counts(normalize=True)
        .reindex(categorias, fill_value=0)
        .to_numpy(dtype=float)
    )

    proporcion_actual = (
        serie_actual
        .value_counts(normalize=True)
        .reindex(categorias, fill_value=0)
        .to_numpy(dtype=float)
    )

    proporcion_referencia = np.clip(
        proporcion_referencia,
        epsilon,
        None,
    )

    proporcion_actual = np.clip(
        proporcion_actual,
        epsilon,
        None,
    )

    psi = np.sum(
        (proporcion_actual - proporcion_referencia)
        * np.log(
            proporcion_actual
            / proporcion_referencia
        )
    )

    return float(psi)