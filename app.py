import altair as alt
import pandas as pd
import streamlit as st

from src.cargar_datos import cargar_base, cargar_config
from src.model_monitoring import generar_reporte_monitoreo
from src.model_deploy import predecir_dataframe

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

def leer_archivo_prediccion(archivo) -> pd.DataFrame:
    """
    Lee archivos Excel o CSV cargados desde Streamlit.
    """

    nombre_archivo = archivo.name.lower()

    if nombre_archivo.endswith(".xlsx"):
        return pd.read_excel(archivo)

    if nombre_archivo.endswith(".csv"):
        ultimo_error = None

        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                archivo.seek(0)

                datos = pd.read_csv(
                    archivo,
                    encoding=encoding,
                )

                # Algunos CSV utilizan punto y coma como separador.
                if datos.shape[1] == 1:
                    archivo.seek(0)

                    datos_punto_coma = pd.read_csv(
                        archivo,
                        encoding=encoding,
                        sep=";",
                    )

                    if datos_punto_coma.shape[1] > 1:
                        datos = datos_punto_coma

                return datos

            except Exception as error:
                ultimo_error = error

        raise ValueError(
            f"No se pudo leer el archivo CSV: {ultimo_error}"
        )

    raise ValueError(
        "Formato no permitido. Utilice CSV o Excel XLSX."
    )


st.title("Proyecto Integrador M5")
st.subheader("Predicción y monitoreo de pagos de créditos")

st.write(
    "Aplicación desarrollada para visualizar el modelo "
    "y monitorear posibles cambios en los datos."
)

st.header("Predicción por lotes")

st.write(
    "Cargue un archivo CSV o Excel con uno o varios créditos. "
    "La aplicación aplicará el mismo pipeline utilizado por la API."
)

archivo_prediccion = st.file_uploader(
    "Seleccione un archivo",
    type=["csv", "xlsx"],
    key="archivo_prediccion",
)

if archivo_prediccion is not None:
    try:
        datos_cargados = leer_archivo_prediccion(
            archivo_prediccion
        )

        st.success(
            f"Archivo cargado correctamente: "
            f"{len(datos_cargados)} registros."
        )

        st.subheader("Vista previa del archivo")

        st.dataframe(
            datos_cargados.head(20),
            use_container_width=True,
            hide_index=True,
        )

        if st.button(
            "Generar predicciones",
            type="primary",
        ):
            with st.spinner(
                "Procesando registros..."
            ):
                resultados_lote = predecir_dataframe(
                    datos_cargados
                )

            st.session_state["resultados_lote"] = (
                resultados_lote
            )

            st.success(
                "Predicciones generadas correctamente."
            )

    except Exception as error:
        st.error(
            f"No se pudo leer el archivo: {error}"
        )

if "resultados_lote" in st.session_state:
    resultados_lote = st.session_state[
        "resultados_lote"
    ]

    st.subheader("Resultados de la predicción")

    st.dataframe(
        resultados_lote,
        use_container_width=True,
        hide_index=True,
    )

    archivo_resultados = resultados_lote.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Descargar resultados en CSV",
        data=archivo_resultados,
        file_name="predicciones_creditos.csv",
        mime="text/csv",
    )

st.divider()

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