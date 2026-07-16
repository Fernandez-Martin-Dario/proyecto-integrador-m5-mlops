import altair as alt
import pandas as pd
import streamlit as st

from src.cargar_datos import cargar_base, cargar_config
from src.model_monitoring import generar_reporte_monitoreo


st.set_page_config(
    page_title="Proyecto Integrador M5",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def cargar_reporte_monitoreo():
    """
    Carga los datos y genera el reporte de monitoreo.

    Streamlit guarda el resultado en caché para evitar
    repetir el cálculo en cada actualización de la página.
    """

    config = cargar_config()
    df = cargar_base()

    reporte = generar_reporte_monitoreo(
        df=df,
        fecha_corte=config["fecha_corte_monitoreo"],
        columna_target=config["target"],
        columnas_categoricas=[
            "tipo_credito",
            "tipo_laboral",
            "tendencia_ingresos",
        ],
    )

    return reporte


st.title("Proyecto Integrador M5")
st.subheader("Predicción y monitoreo de pagos de créditos")

st.write(
    "Aplicación desarrollada para visualizar el modelo "
    "y monitorear posibles cambios en los datos."
)

reporte = cargar_reporte_monitoreo()

periodos = reporte["periodos"]

cantidad_referencia = int(
    periodos.loc[
        periodos["periodo"] == "Referencia",
        "cantidad_registros",
    ].iloc[0]
)

cantidad_actual = int(
    periodos.loc[
        periodos["periodo"] == "Actual",
        "cantidad_registros",
    ].iloc[0]
)

drift_predictoras = pd.concat(
    [
        reporte["drift_numerico"],
        reporte["drift_categorico"],
    ],
    ignore_index=True,
)

cantidad_drift_importante = int(
    (
        drift_predictoras["clasificacion"]
        == "Cambio importante"
    ).sum()
)

cantidad_drift_moderado = int(
    (
        drift_predictoras["clasificacion"]
        == "Cambio moderado"
    ).sum()
)

st.header("Resumen ejecutivo")

columna_1, columna_2, columna_3, columna_4 = st.columns(4)

columna_1.metric(
    "Registros de referencia",
    f"{cantidad_referencia:,}".replace(",", "."),
)

columna_2.metric(
    "Registros actuales",
    f"{cantidad_actual:,}".replace(",", "."),
)

columna_3.metric(
    "Cambios importantes",
    cantidad_drift_importante,
)

columna_4.metric(
    "Cambios moderados",
    cantidad_drift_moderado,
)

st.header("Períodos de monitoreo")

st.dataframe(
    reporte["periodos"],
    use_container_width=True,
    hide_index=True,
)
st.header("Drift de variables numéricas")

st.caption(
    "El PSI compara la distribución histórica con la distribución actual. "
    "Valores inferiores a 0.10 se consideran estables, entre 0.10 y 0.25 "
    "indican un cambio moderado y superiores a 0.25 un cambio importante."
)

drift_numerico = reporte["drift_numerico"].copy()

drift_numerico["psi"] = drift_numerico["psi"].round(4)

st.dataframe(
    drift_numerico,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Comparación visual del PSI numérico")

grafico_psi_numerico = (
    drift_numerico
    .nlargest(10, "psi")
    .sort_values("psi", ascending=False)
)

grafico_barras_psi = (
    alt.Chart(grafico_psi_numerico)
    .mark_bar()
    .encode(
        x=alt.X(
            "psi:Q",
            title="PSI",
        ),
        y=alt.Y(
            "variable:N",
            title=None,
            sort="-x",
            axis=alt.Axis(
                labelLimit=320,
            ),
        ),
        tooltip=[
            alt.Tooltip(
                "variable:N",
                title="Variable",
            ),
            alt.Tooltip(
                "psi:Q",
                title="PSI",
                format=".4f",
            ),
            alt.Tooltip(
                "clasificacion:N",
                title="Clasificación",
            ),
        ],
    )
    .properties(
        height=420,
    )
)

st.altair_chart(
    grafico_barras_psi,
    use_container_width=True,
)
st.header("Drift de variables categóricas")

drift_categorico = reporte["drift_categorico"].copy()

drift_categorico["psi"] = drift_categorico["psi"].round(4)

st.dataframe(
    drift_categorico,
    use_container_width=True,
    hide_index=True,
)
st.header("Drift del target")

drift_target = reporte["drift_target"].copy()

drift_target["proporcion_referencia"] = (
    drift_target["proporcion_referencia"] * 100
).round(2)

drift_target["proporcion_actual"] = (
    drift_target["proporcion_actual"] * 100
).round(2)

drift_target["diferencia_puntos_porcentuales"] = (
    drift_target["diferencia_puntos_porcentuales"]
    .round(2)
)

drift_target["psi"] = drift_target["psi"].round(4)

st.dataframe(
    drift_target,
    use_container_width=True,
    hide_index=True,
)