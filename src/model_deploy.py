"""
API para disponibilizar el modelo final del Proyecto Integrador M5.

Este módulo carga el pipeline entrenado, recibe datos de nuevos créditos,
aplica la ingeniería de características y devuelve una predicción.
"""

from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.cargar_datos import cargar_config
from src.ft_engineering import aplicar_feature_engineering


RUTA_CONFIG = Path(__file__).resolve().parent / "config.json"

config = cargar_config(RUTA_CONFIG)

RUTA_MODELO = (
    RUTA_CONFIG.parent / config["ruta_modelo"]
).resolve()


app = FastAPI(
    title="API de predicción de pagos",
    description=(
        "API para estimar si un cliente pagará su crédito a tiempo."
    ),
    version="1.0.0",
)


class DatosCredito(BaseModel):
    """
    Datos originales de un crédito utilizados para generar una predicción.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    tipo_credito: Literal[4, 6, 7, 9, 10, 68]
    fecha_prestamo: date
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: Literal[
        "Empleado",
        "Independiente",
    ]
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: float | None = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float | None = None
    saldo_total: float | None = None
    saldo_principal: float | None = None
    saldo_mora_codeudor: float | None = None
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float | None = None
    tendencia_ingresos: str | None = None


def cargar_modelo():
    """
    Carga desde disco el pipeline entrenado utilizado por la API.
    """

    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo entrenado: {RUTA_MODELO}"
        )

    return joblib.load(RUTA_MODELO)


modelo = cargar_modelo()


@app.get("/")
def estado_api() -> dict:
    """
    Informa si la API y el modelo están disponibles.
    """

    return {
        "estado": "API activa",
        "modelo_cargado": True,
        "version": app.version,
    }


@app.post("/predict")
def predecir_pago(datos: DatosCredito) -> dict:
    """
    Genera una predicción para un nuevo crédito.
    """

    try:
        datos_entrada = pd.DataFrame(
            [
                datos.model_dump(
                    mode="json",
                )
            ]
        )

        datos_transformados = aplicar_feature_engineering(
            datos_entrada
        )

        prediccion = int(
            modelo.predict(datos_transformados)[0]
        )

        probabilidades = modelo.predict_proba(
            datos_transformados
        )[0]

        probabilidades_por_clase = {
            int(clase): float(probabilidad)
            for clase, probabilidad in zip(
                modelo.classes_,
                probabilidades,
            )
        }

        significado = (
            "Pagará a tiempo"
            if prediccion == 1
            else "No pagará a tiempo"
        )

        return {
            "prediccion": prediccion,
            "significado": significado,
            "probabilidad_clase_0": probabilidades_por_clase.get(0),
            "probabilidad_clase_1": probabilidades_por_clase.get(1),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudo generar la predicción: "
                f"{error}"
            ),
        ) from error


@app.post("/predict-batch")
def predecir_pagos_lote(
    registros: list[DatosCredito],
) -> dict:
    """
    Genera predicciones para múltiples créditos en una sola solicitud.
    """

    if not registros:
        raise HTTPException(
            status_code=400,
            detail="La lista de registros no puede estar vacía.",
        )

    try:
        datos_entrada = pd.DataFrame(
            [
                registro.model_dump(mode="json")
                for registro in registros
            ]
        )

        datos_transformados = aplicar_feature_engineering(
            datos_entrada
        )

        predicciones = modelo.predict(
            datos_transformados
        )

        probabilidades = modelo.predict_proba(
            datos_transformados
        )

        clases = [
            int(clase)
            for clase in modelo.classes_
        ]

        indice_clase_0 = clases.index(0)
        indice_clase_1 = clases.index(1)

        resultados = []

        for indice, prediccion in enumerate(predicciones):
            prediccion = int(prediccion)

            significado = (
                "Pagará a tiempo"
                if prediccion == 1
                else "No pagará a tiempo"
            )

            resultados.append(
                {
                    "registro": indice + 1,
                    "prediccion": prediccion,
                    "significado": significado,
                    "probabilidad_clase_0": float(
                        probabilidades[indice][indice_clase_0]
                    ),
                    "probabilidad_clase_1": float(
                        probabilidades[indice][indice_clase_1]
                    ),
                }
            )

        return {
            "cantidad_registros": len(resultados),
            "resultados": resultados,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "No se pudieron generar las predicciones por lote: "
                f"{error}"
            ),
        ) from error
