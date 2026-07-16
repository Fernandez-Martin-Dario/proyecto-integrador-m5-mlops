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
st.header("Drift de variables categóricas")

drift_categorico = reporte["drift_categorico"].copy()

drift_categorico["psi"] = drift_categorico["psi"].round(4)

st.dataframe(
    drift_categorico,
    use_container_width=True,
    hide_index=True,
)