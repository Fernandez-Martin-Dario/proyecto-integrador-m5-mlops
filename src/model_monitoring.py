from __future__ import annotations

import pandas as pd

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