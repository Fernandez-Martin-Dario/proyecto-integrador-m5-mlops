# Proyecto Integrador M5

## Predicción, monitoreo y despliegue de pagos de créditos

Proyecto desarrollado como parte del Módulo 5 de la carrera de Data Science de Henry.

El objetivo es construir un flujo reproducible de Machine Learning y MLOps que permita analizar información histórica de créditos, entrenar modelos de clasificación, monitorear posibles cambios en la distribución de los datos y exponer predicciones mediante una API contenerizada.

## Índice

- [Caso de negocio](#caso-de-negocio)
- [Objetivo del proyecto](#objetivo-del-proyecto)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Flujo de trabajo](#flujo-de-trabajo)
- [Análisis exploratorio de datos](#análisis-exploratorio-de-datos)
- [Ingeniería de características](#ingeniería-de-características)
- [Entrenamiento y evaluación de modelos](#entrenamiento-y-evaluación-de-modelos)
- [Monitoreo y detección de data drift](#monitoreo-y-detección-de-data-drift)
- [Aplicación Streamlit](#aplicación-streamlit)
- [API de predicción con FastAPI](#api-de-predicción-con-fastapi)
- [Contenerización con Docker](#contenerización-con-docker)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Resultados principales](#resultados-principales)
- [Versionamiento](#versionamiento)
- [Autor](#autor)

## Caso de negocio

Una empresa financiera necesita anticipar el comportamiento de pago de sus clientes para identificar aquellos créditos con mayor riesgo de no ser abonados en tiempo y forma.

El dataset utilizado contiene 10.763 registros y 23 variables relacionadas con las características del crédito, la situación económica del cliente, su historial financiero y su comportamiento de pago.

La variable objetivo es `Pago_atiempo`:

- `1`: el cliente pagó a tiempo.
- `0`: el cliente no pagó a tiempo.

La distribución del target presenta un fuerte desbalance:

- Clase `1`: 10.252 registros, aproximadamente el 95,25 %.
- Clase `0`: 511 registros, aproximadamente el 4,75 %.

Debido a este desbalance, la evaluación no se basa únicamente en `accuracy`. Se priorizan métricas como `precision`, `recall`, `F1-score`, matriz de confusión y `ROC-AUC`, prestando especial atención a la clase `0`.

## Objetivo del proyecto

Desarrollar un pipeline de datos y Machine Learning que permita:

- Cargar y validar la base de datos del proyecto.
- Realizar un análisis exploratorio de los datos.
- Aplicar ingeniería de características y preprocesamiento.
- Entrenar y comparar distintos modelos supervisados.
- Seleccionar el modelo con mejor desempeño sobre la clase minoritaria.
- Detectar posibles cambios o `data drift` entre períodos.
- Visualizar los resultados de monitoreo mediante una aplicación desarrollada con Streamlit.
- Serializar el pipeline final para reutilizar el mismo preprocesamiento y modelo en inferencia.
- Exponer predicciones individuales y por lotes mediante una API desarrollada con FastAPI.
- Empaquetar la API y sus dependencias en una imagen reproducible de Docker.
- Mantener trazabilidad y versionamiento del proyecto mediante Git y GitHub.

## Estructura del repositorio

```text
Proyecto_Integrador_M5_Fernandez_Martin_Dario/
│
├── .dockerignore
├── app.py
├── Base_de_datos.xlsx
├── Dockerfile
├── README.md
├── requirements-api.txt
├── requirements.txt
├── set_up.bat
│
└── src/
    ├── cargar_datos.py
    ├── comprension_eda.ipynb
    ├── config.json
    ├── ft_engineering.py
    ├── model_deploy.py
    ├── model_monitoring.py
    └── model_training_evaluation.py
```

### Descripción de los archivos principales

- `.dockerignore`: exclusiones utilizadas para reducir el contexto de construcción y evitar copiar archivos locales innecesarios a la imagen.
- `app.py`: aplicación desarrollada con Streamlit para visualizar los resultados del monitoreo y del data drift.
- `Base_de_datos.xlsx`: dataset utilizado en el proyecto.
- `Dockerfile`: definición reproducible de la imagen que entrena el modelo y ejecuta la API con Uvicorn.
- `README.md`: documentación general del caso de negocio, metodología, resultados y ejecución.
- `requirements-api.txt`: dependencias mínimas necesarias para entrenar el pipeline dentro de la imagen y ejecutar la API.
- `requirements.txt`: dependencias del entorno completo de desarrollo.
- `set_up.bat`: script de configuración inicial del entorno en Windows.
- `src/config.json`: parámetros generales del proyecto, como target, semilla aleatoria, tamaño del conjunto de prueba y fecha de corte para monitoreo.
- `src/cargar_datos.py`: funciones destinadas a cargar la configuración y la base de datos.
- `src/comprension_eda.ipynb`: notebook del análisis exploratorio de datos.
- `src/ft_engineering.py`: funciones de limpieza, transformación, ingeniería de características y preprocesamiento.
- `src/model_training_evaluation.py`: preparación de datos, entrenamiento, evaluación, optimización, selección y serialización del pipeline final.
- `src/model_monitoring.py`: separación temporal de los datos, cálculo del PSI y generación del reporte de monitoreo.
- `src/model_deploy.py`: API FastAPI, validación de entradas, carga del modelo y endpoints de predicción individual y por lotes.

## Flujo de trabajo

El proyecto fue desarrollado de manera incremental siguiendo un flujo de trabajo reproducible:

1. **Carga y configuración**
   - Lectura de parámetros desde `src/config.json`.
   - Carga de la base mediante `src/cargar_datos.py`.
   - Validación de dimensiones, columnas y variable objetivo.

2. **Análisis exploratorio**
   - Revisión de tipos de datos.
   - Identificación de valores nulos y categorías inconsistentes.
   - Análisis de la distribución del target.
   - Estudio de variables numéricas, categóricas y temporales.
   - Evaluación de correlaciones y posibles fugas de información.

3. **Ingeniería de características**
   - Normalización de `tendencia_ingresos`.
   - Creación de una variable indicadora de inconsistencias.
   - Extracción de año, mes y día de la semana desde `fecha_prestamo`.
   - Imputación de valores faltantes.
   - Escalado de variables numéricas.
   - Codificación One-Hot de variables categóricas.

4. **Entrenamiento y evaluación**
   - Separación estratificada en conjuntos de entrenamiento y prueba.
   - Construcción de pipelines reproducibles.
   - Entrenamiento de modelos base y modelos supervisados.
   - Comparación mediante métricas enfocadas en la clase minoritaria.
   - Optimización de hiperparámetros.
   - Selección del modelo final.

5. **Monitoreo**
   - Separación temporal entre datos históricos y actuales.
   - Cálculo del Population Stability Index (PSI).
   - Análisis de drift numérico, categórico y del target.
   - Generación de un reporte integrado de monitoreo.

6. **Visualización**
   - Desarrollo de una aplicación con Streamlit.
   - Visualización de períodos, métricas de drift y resumen ejecutivo.
   - Gráfico interactivo de PSI mediante Altair.

7. **API e inferencia**
   - Entrenamiento y serialización del pipeline final con `joblib`.
   - Carga única del modelo al iniciar la aplicación.
   - Validación de entradas mediante Pydantic.
   - Predicciones individuales y por lotes con FastAPI.

8. **Contenerización**
   - Definición de una imagen basada en Python 3.11.
   - Instalación de dependencias específicas para la API.
   - Generación del modelo durante la construcción de la imagen.
   - Ejecución de la API mediante Uvicorn dentro de Docker.

9. **Control de versiones**
   - Uso de ramas `feature`, `developer`, `certification` y `main`.
   - Commits incrementales por funcionalidad.
   - Versionamiento mediante tags semánticos.

## Análisis exploratorio de datos

El dataset contiene 10.763 registros y 23 columnas.

Las variables se distribuyen en:

- Variables numéricas relacionadas con montos, saldos, ingresos, puntajes e historial crediticio.
- Variables categóricas como `tipo_laboral` y `tendencia_ingresos`.
- Una variable temporal: `fecha_prestamo`.
- Una variable objetivo: `Pago_atiempo`.

### Distribución del target

La variable `Pago_atiempo` presenta un fuerte desbalance:

| Clase | Significado      | Cantidad | Porcentaje aproximado |
|-------|------------------|----------|-----------------------|
| 0     | No pagó a tiempo |    511   |        4,75 %         |
| 1     | Pagó a tiempo    | 10.252   |       95,25 %         |

Este desbalance implica que un modelo que prediga siempre la clase mayoritaria podría alcanzar una `accuracy` cercana al 95 %, pero no sería útil para detectar clientes con riesgo de incumplimiento.

### Valores faltantes e inconsistencias

Durante el análisis se detectaron:

- Valores faltantes en distintas variables.
- Aproximadamente un 27 % de valores nulos en `promedio_ingresos_datacredito`.
- Valores inconsistentes en `tendencia_ingresos`, donde se encontraron números mezclados con las categorías esperadas.
- Categorías válidas principales: `Creciente`, `Decreciente` y `Estable`.

Los valores inconsistentes de `tendencia_ingresos` fueron transformados en valores faltantes y se creó una variable adicional para conservar la información de que el registro original presentaba una inconsistencia.

### Análisis temporal

La variable `fecha_prestamo` contiene registros desde noviembre de 2024 hasta abril de 2026.

La proporción de pagos a tiempo se mantiene elevada durante todos los meses analizados, generalmente entre el 93 % y el 98 %.

### Posible fuga de información

Durante el análisis se observó que la variable `puntaje` presenta una separación completa entre las clases:

- Clase `0`: valores máximos cercanos a 62,67.
- Clase `1`: valores mínimos cercanos a 63,81.

Esta separación podría indicar que la variable contiene información directa o indirectamente derivada del target.

Debido a que no se dispone de un diccionario de datos que confirme su origen, `puntaje` fue considerada una variable sospechosa de fuga de información.

Por este motivo se entrenaron modelos con y sin esta variable, priorizando para la selección final los modelos que no utilizan `puntaje`.

El análisis de `saldo_mora` mostró que esta variable no determina automáticamente el target, ya que existen registros de ambas clases cuando su valor es cero, mayor que cero o faltante.

## Ingeniería de características

La lógica de transformación y preprocesamiento se encuentra en `src/ft_engineering.py`.

### Normalización de `tendencia_ingresos`

La variable `tendencia_ingresos` contenía valores válidos y registros inconsistentes.

Se conservaron como categorías válidas:

- `Creciente`
- `Decreciente`
- `Estable`

Los valores que no pertenecían a estas categorías fueron reemplazados por valores faltantes.

Además, se creó la variable binaria `tendencia_ingresos_inconsistente`.

Esta variable permite conservar la información de que el valor original presentaba una anomalía:

- `1`: el valor original era inconsistente.
- `0`: el valor original era válido o faltante.

### Variables temporales

La columna `fecha_prestamo` fue transformada en las siguientes variables:

- `anio_prestamo`
- `mes_prestamo`
- `dia_semana_prestamo`

Luego de extraer esta información, la fecha original deja de ser utilizada directamente por el modelo.

### Separación entre variables predictoras y target

La función de separación genera:

- `X`: conjunto de variables predictoras.
- `y`: variable objetivo `Pago_atiempo`.

De esta manera se evita que el target forme parte accidentalmente del preprocesamiento o del entrenamiento.

### Preprocesamiento de variables numéricas

Las variables numéricas se procesan mediante un pipeline compuesto por:

1. Imputación de valores faltantes utilizando la mediana.
2. Estandarización mediante `StandardScaler`.

La mediana fue elegida porque es menos sensible a valores extremos que la media.

### Preprocesamiento de variables categóricas

Las variables categóricas se procesan mediante:

1. Imputación de valores faltantes con la categoría `Sin dato`.
2. Codificación mediante `OneHotEncoder`.
3. Configuración `handle_unknown="ignore"` para evitar errores ante categorías nuevas.

Las principales variables categóricas son:

- `tipo_credito`
- `tipo_laboral`
- `tendencia_ingresos`

Aunque `tipo_credito` está almacenada numéricamente, se trata como variable categórica porque sus valores representan tipos de crédito y no una magnitud continua.

### Exclusión de variables

El preprocesador permite excluir columnas específicas antes del entrenamiento.

Esta funcionalidad fue utilizada para comparar modelos:

- Con `puntaje`.
- Sin `puntaje`.

La exclusión se realiza dentro del pipeline, permitiendo mantener una estructura reproducible y evitando modificaciones manuales del dataset.

## Entrenamiento y evaluación de modelos

La lógica de preparación de datos, entrenamiento, evaluación, optimización y selección se encuentra en `src/model_training_evaluation.py`.

### División de los datos

El dataset fue dividido de manera estratificada para conservar la proporción original de las clases:

- Conjunto de entrenamiento: 8.610 registros.
- Conjunto de prueba: 2.153 registros.
- Tamaño del conjunto de prueba: 20 %.
- Semilla aleatoria: 42.

La distribución aproximada de la clase minoritaria `0` se mantuvo en ambos conjuntos:

- Entrenamiento: 4,75 %.
- Prueba: 4,74 %.

La estratificación es importante porque evita que una clase poco frecuente quede subrepresentada en alguno de los conjuntos.

### Pipeline de entrenamiento

Cada modelo fue integrado dentro de un pipeline compuesto por:

1. Preprocesamiento de variables numéricas y categóricas.
2. Imputación de valores faltantes.
3. Escalado de variables numéricas cuando corresponde.
4. Codificación de variables categóricas.
5. Entrenamiento del modelo.

Este enfoque evita aplicar transformaciones antes de la separación entre entrenamiento y prueba, reduciendo el riesgo de fuga de información.

### Modelos evaluados

Se entrenaron y compararon los siguientes modelos:

- `DummyClassifier`.
- Regresión Logística con `puntaje`.
- Regresión Logística sin `puntaje`.
- Random Forest sin `puntaje`.
- Gradient Boosting sin `puntaje`.
- Gradient Boosting optimizado sin `puntaje`.

El `DummyClassifier` fue utilizado como línea base para demostrar que una `accuracy` elevada no implica necesariamente un modelo útil cuando el target está desbalanceado.

### Métricas utilizadas

La evaluación se centró especialmente en la clase `0`, correspondiente a los clientes que no pagaron a tiempo.

Las principales métricas utilizadas fueron:

- `Accuracy`: proporción total de predicciones correctas.
- `Precision` de la clase `0`: proporción de alertas de incumplimiento que fueron correctas.
- `Recall` de la clase `0`: proporción de clientes incumplidores que fueron detectados.
- `F1-score` de la clase `0`: equilibrio entre precision y recall.
- `ROC-AUC`: capacidad general del modelo para separar ambas clases.
- Matriz de confusión: distribución de aciertos y errores por clase.

### Resultados obtenidos

| Modelo                                     | Accuracy | Precision clase 0 | Recall clase 0 | F1 clase 0 | ROC-AUC |
|--------------------------------------------|----------|-------------------|----------------|------------|---------|
| Dummy Classifier                           | 0,9526   |       0,0000      |     0,0000     |   0,0000   | 0,5000  |
| Regresión Logística con `puntaje`          | 0,9986   |       0,9714      |     1,0000     |   0,9855   | 1,0000  |
| Regresión Logística sin `puntaje`          | 0,6438   |       0,0754      |     0,5784     |   0,1333   | 0,6660  |
| Random Forest sin `puntaje`                | 0,9540   |       0,8000      |     0,0392     |   0,0748   | 0,6774  |
| Gradient Boosting sin `puntaje`            | 0,7348   |       0,0850      |     0,4706     |   0,1439   | 0,6801  |
| Gradient Boosting optimizado sin `puntaje` | 0,7733   |       0,0928      |     0,4314     |   0,1528   | 0,6657  |

### Interpretación de los resultados

El `DummyClassifier` alcanzó una `accuracy` superior al 95 %, pero no detectó ningún caso de la clase `0`.

Esto demuestra que la `accuracy` no es suficiente para evaluar un problema fuertemente desbalanceado.

La Regresión Logística con `puntaje` obtuvo resultados casi perfectos. Sin embargo, debido a la separación completa observada entre las clases, este desempeño fue considerado sospechoso y compatible con una posible fuga de información.

Al excluir `puntaje`, el rendimiento disminuyó considerablemente, mostrando una situación más realista y exigente.

Random Forest obtuvo una `accuracy` elevada, pero solo detectó aproximadamente el 3,92 % de los incumplidores. Por este motivo no fue seleccionado, aunque su `precision` para la clase `0` fuera alta.

Gradient Boosting consiguió un mejor equilibrio entre la detección de la clase minoritaria y el nivel general de errores.

### Optimización de Gradient Boosting

La optimización se realizó mediante:

- `GridSearchCV`.
- Validación cruzada estratificada de 5 particiones.
- Métrica de optimización: F1-score de la clase `0`.
- Pesos de muestra balanceados.

La búsqueda evaluó 16 combinaciones de hiperparámetros, completando 80 entrenamientos.

Los mejores hiperparámetros encontrados fueron:

```python
{
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 1,
    "n_estimators": 200,
}
```

El mejor F1 promedio obtenido durante la validación cruzada fue aproximadamente 0,1644.

### Selección del modelo final

El modelo seleccionado fue:

```text
Gradient Boosting optimizado sin puntaje
```

La selección se realizó utilizando un criterio conservador:

1. Excluir modelos que utilizan la variable sospechosa `puntaje`.
2. Priorizar el F1-score de la clase `0`.
3. Utilizar ROC-AUC y recall como métricas complementarias.

Aunque el desempeño obtenido todavía es limitado, este modelo representa la alternativa más consistente entre los modelos evaluados sin utilizar la variable sospechosa de fuga de información.

Los resultados también indican que será necesario continuar trabajando en nuevas variables, técnicas de balanceo y calibración del umbral de decisión para mejorar la detección de clientes con riesgo de incumplimiento.

## Monitoreo y detección de data drift

La lógica de monitoreo se encuentra en `src/model_monitoring.py`.

El objetivo es comparar la distribución de los datos históricos con la distribución de un período más reciente para detectar cambios que podrían afectar el comportamiento del modelo.

### División temporal

La fecha de corte se encuentra configurada en `src/config.json`:

```text
2025-07-01
```

A partir de esta fecha se definieron dos períodos:

| Período    | Desde      | Hasta      | Registros |
|------------|------------|------------|-----------|
| Referencia | 2024-11-26 | 2025-06-30 | 8.378     |
| Actual     | 2025-07-01 | 2026-04-26 | 2.385     |

El período de referencia representa el comportamiento histórico utilizado como base de comparación.

El período actual representa la población más reciente que se desea monitorear.

### Métrica utilizada

Para medir el cambio entre las distribuciones se utilizó el `Population Stability Index` o PSI.

Los umbrales de interpretación fueron:

| Valor de PSI         | Clasificación     |
|----------------------|-------------------|
| Menor a 0,10         | Estable           |
| Entre 0,10 y 0,25    | Cambio moderado   |
| Mayor o igual a 0,25 | Cambio importante |

El PSI fue calculado para:

- Variables numéricas.
- Variables categóricas.
- Variable objetivo.

### Drift en variables numéricas

Los principales resultados fueron:

| Variable                        | PSI      | Clasificación     |
|---------------------------------|----------|-------------------|
| `plazo_meses`                   | 0,299388 | Cambio importante |
| `promedio_ingresos_datacredito` | 0,252737 | Cambio importante |
| `total_otros_prestamos`         | 0,206363 | Cambio moderado   |
| `cuota_pactada`                 | 0,078254 | Estable           |
| `capital_prestado`              | 0,053165 | Estable           |
| `salario_cliente`               | 0,038609 | Estable           |

Las demás variables numéricas evaluadas se mantuvieron dentro del rango considerado estable.

Los resultados indican que `plazo_meses` y `promedio_ingresos_datacredito` presentan cambios importantes entre los períodos y deberían ser monitoreadas con especial atención.

La variable `total_otros_prestamos` presenta un cambio moderado que todavía no alcanza el umbral crítico, pero requiere seguimiento.

### Drift en variables categóricas

Los resultados obtenidos fueron:

| Variable             | PSI      | Clasificación |
|----------------------|----------|---------------|
| `tendencia_ingresos` | 0,065527 | Estable       |
| `tipo_credito`       | 0,005562 | Estable       |
| `tipo_laboral`       | 0,000802 | Estable       |

Las variables categóricas no presentan cambios relevantes entre el período de referencia y el período actual.

### Drift del target

El análisis del target se concentró en la proporción de la clase `0`, correspondiente a los clientes que no pagaron a tiempo.

| Métrica                                   | Resultado                 |
|-------------------------------------------|---------------------------|
| Proporción de clase `0` en referencia     | 5,19 %                    |
| Proporción de clase `0` en período actual | 3,19 %                    |
| Diferencia                                | -2,01 puntos porcentuales |
| PSI                                       | 0,010211                  |
| Clasificación                             | Estable                   |

Aunque la proporción de incumplimientos disminuyó en el período actual, el valor del PSI indica que el cambio todavía se encuentra dentro del rango considerado estable.

### Reporte integrado

La función principal de monitoreo genera un reporte con cuatro componentes:

- `periodos`
- `drift_numerico`
- `drift_categorico`
- `drift_target`

El monitoreo puede ejecutarse desde la raíz del proyecto mediante:

```bash
python -m src.model_monitoring
```

## Aplicación Streamlit

La aplicación de visualización se encuentra en `app.py`.

Su objetivo es presentar los resultados del monitoreo de manera clara e interactiva, facilitando la identificación de variables con cambios relevantes.

La aplicación muestra:

- Título y descripción del proyecto.
- Resumen ejecutivo.
- Cantidad de registros del período de referencia.
- Cantidad de registros del período actual.
- Cantidad de variables con cambio importante.
- Cantidad de variables con cambio moderado.
- Tabla de períodos.
- Tabla de drift numérico.
- Tabla de drift categórico.
- Tabla de drift del target.
- Gráfico horizontal de las 10 variables con mayor PSI.
- Tooltips interactivos con el nombre completo de cada variable y su valor de PSI.

La aplicación utiliza:

- `streamlit`
- `pandas`
- `altair`
- `st.cache_data`
- `cargar_base`
- `cargar_config`
- `generar_reporte_monitoreo`

Para ejecutarla desde la raíz del proyecto:

```bash
python -m streamlit run app.py
```

La aplicación queda disponible localmente en:

```text
http://localhost:8501
```

## API de predicción con FastAPI

La API de inferencia se encuentra en `src/model_deploy.py` y utiliza FastAPI para exponer el pipeline final como un servicio HTTP.

El modelo desplegado es el Gradient Boosting optimizado sin la variable `puntaje`. El pipeline incluye el preprocesamiento y el estimador entrenado, por lo que una predicción reutiliza exactamente las mismas transformaciones aplicadas durante el entrenamiento.

### Serialización y carga del modelo

La función `entrenar_y_guardar_modelo_final` de `src/model_training_evaluation.py`:

1. Construye el pipeline final sin `puntaje`.
2. Entrena Gradient Boosting con los hiperparámetros seleccionados.
3. Aplica pesos de muestra balanceados.
4. Guarda el pipeline mediante `joblib`.

La ruta se define en `src/config.json` y genera localmente:

```text
models/modelo_final.joblib
```

La carpeta `models` no se versiona porque el archivo es un artefacto generado. La API carga el modelo una sola vez al iniciar el módulo.

### Validación de entradas

El cuerpo de las solicitudes se valida mediante un modelo de Pydantic llamado `DatosCredito`.

El esquema contiene las variables necesarias para inferencia, pero excluye deliberadamente:

- `Pago_atiempo`, porque es la variable que se desea predecir.
- `puntaje`, porque fue identificada como sospechosa de fuga de información.

Pydantic controla tipos, campos obligatorios, valores nulos permitidos y categorías restringidas antes de enviar los datos al pipeline.

### Endpoints disponibles

| Método | Endpoint         | Función                                                              |
|--------|------------------|----------------------------------------------------------------------|
| `GET`  | `/`              | Verifica que la API esté activa y que el modelo haya sido cargado.   |
| `POST` | `/predict`       | Recibe un crédito en formato JSON y devuelve una predicción.         |
| `POST` | `/predict-batch` | Recibe una lista JSON de créditos y devuelve predicciones por lotes. |

La respuesta de una predicción incluye:

- Clase predicha: `0` o `1`.
- Interpretación de la clase.
- Probabilidad estimada para la clase `0`.
- Probabilidad estimada para la clase `1`.

Interpretación:

- Clase `0`: no pagará a tiempo.
- Clase `1`: pagará a tiempo.

Una solicitud vacía a `/predict-batch` devuelve un error HTTP `400`. Los datos que no cumplen el esquema de entrada son rechazados automáticamente con un error HTTP `422`.

### Ejecutar la API localmente

Primero debe existir el artefacto `models/modelo_final.joblib`. Luego, desde la raíz del proyecto:

```bash
python -m uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000
```

La API queda disponible en:

```text
http://127.0.0.1:8000
```

La documentación interactiva generada por FastAPI puede consultarse en:

```text
http://127.0.0.1:8000/docs
```

El esquema alternativo ReDoc queda disponible en:

```text
http://127.0.0.1:8000/redoc
```

## Contenerización con Docker

La API fue contenerizada para disponer de un entorno reproducible e independiente de la configuración local.

### Archivos utilizados

- `Dockerfile`: define la imagen, instala las dependencias, copia el código, entrena el pipeline final y ejecuta Uvicorn.
- `.dockerignore`: evita copiar el entorno virtual, cachés, artefactos locales, reportes y otros archivos innecesarios.
- `requirements-api.txt`: contiene las dependencias necesarias para construir el modelo y servir la API.

La imagen utiliza como base:

```text
python:3.11-slim
```

Durante la construcción se copia el dataset y se ejecuta el entrenamiento del modelo. De esta manera, el artefacto queda generado dentro de la imagen sin depender del archivo local `models/modelo_final.joblib`.

### Construir la imagen

Desde la raíz del proyecto:

```bash
docker build -t proyecto-m5-api:1.0.0 .
```

### Crear y ejecutar el contenedor

```bash
docker run --name proyecto-m5-api-container -p 8000:8000 proyecto-m5-api:1.0.0
```

La documentación interactiva queda disponible en:

```text
http://127.0.0.1:8000/docs
```

### Detener y volver a iniciar el contenedor

Para detenerlo desde otra terminal:

```bash
docker stop proyecto-m5-api-container
```

Para volver a iniciarlo mostrando los logs:

```bash
docker start -a proyecto-m5-api-container
```

La construcción de la imagen, el inicio del contenedor, el endpoint de estado y la predicción individual fueron validados correctamente.

## Instalación y ejecución

### Requisitos previos

Para la ejecución local:

- Windows 11.
- Python 3.11.9.
- Git.
- Visual Studio Code.
- Terminal integrada de Visual Studio Code.

Para la ejecución contenerizada:

- Docker Desktop con el motor de Docker activo.

### Clonar el repositorio

```bash
git clone https://github.com/Fernandez-Martin-Dario/proyecto-integrador-m5-mlops.git
```

Ingresar a la carpeta del proyecto:

```bash
cd proyecto-integrador-m5-mlops
```

### Crear el entorno virtual

```bash
python -m venv .venv
```

### Activar el entorno virtual

En Windows:

```bash
.venv\Scripts\activate
```

### Instalar las dependencias

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### Generar el modelo final para ejecución local

Desde la raíz del proyecto:

```bash
python -c "from src.model_training_evaluation import preparar_datos, entrenar_y_guardar_modelo_final; X_train, _, y_train, _ = preparar_datos(); entrenar_y_guardar_modelo_final(X_train, y_train)"
```

Este comando crea:

```text
models/modelo_final.joblib
```

### Verificar la carga de datos

Desde la raíz del proyecto:

```bash
python -c "from src.cargar_datos import cargar_base; df = cargar_base(); print(df.shape)"
```

La salida esperada para el dataset utilizado es:

```text
(10763, 23)
```

### Ejecutar el monitoreo

```bash
python -m src.model_monitoring
```

### Ejecutar la aplicación Streamlit

```bash
python -m streamlit run app.py
```

### Ejecutar la API local

```bash
python -m uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000
```

Documentación interactiva:

```text
http://127.0.0.1:8000/docs
```

### Ejecutar la API con Docker

Construir la imagen:

```bash
docker build -t proyecto-m5-api:1.0.0 .
```

Crear y ejecutar el contenedor:

```bash
docker run --name proyecto-m5-api-container -p 8000:8000 proyecto-m5-api:1.0.0
```

## Resultados principales

Los principales hallazgos del proyecto fueron:

1. **Desbalance del target**
   La clase `0` representa aproximadamente el 4,75 % del dataset. Por este motivo, la `accuracy` no es suficiente para evaluar correctamente los modelos.

2. **Posible fuga de información**
   La variable `puntaje` presenta una separación completa entre las clases. Como no existe un diccionario de datos que permita confirmar su origen, fue tratada como una variable sospechosa y se priorizaron modelos que no la utilizan.

3. **Modelo seleccionado**
   El modelo final fue Gradient Boosting optimizado sin `puntaje`, con los siguientes resultados sobre el conjunto de prueba:

   - `Accuracy`: 0,7733.
   - `Precision` de clase `0`: 0,0928.
   - `Recall` de clase `0`: 0,4314.
   - `F1-score` de clase `0`: 0,1528.
   - `ROC-AUC`: 0,6657.

4. **Variables con mayor drift**
   Las variables `plazo_meses` y `promedio_ingresos_datacredito` presentaron cambios importantes.

5. **Cambio moderado**
   La variable `total_otros_prestamos` presentó un cambio moderado.

6. **Estabilidad del target**
   El PSI del target fue 0,010211, por lo que su distribución se clasificó como estable.

7. **API de inferencia**
   El pipeline final fue serializado y expuesto mediante FastAPI con validación de entradas, predicción individual y predicción por lotes.

8. **Contenerización**
   La imagen `proyecto-m5-api:1.0.0` fue construida correctamente y el servicio fue validado dentro de un contenedor Docker mediante el endpoint de estado y una predicción individual.

9. **Limitaciones actuales**
   El desempeño sobre la clase minoritaria todavía es limitado. Será necesario continuar evaluando nuevas variables, técnicas de balanceo, calibración del umbral y estrategias de reentrenamiento. La API permite consumir el modelo actual, pero no elimina estas limitaciones estadísticas.

## Versionamiento

El proyecto utiliza versionamiento semántico y un flujo de ramas controlado:

```text
feature → developer → certification → main → tag
```

La rama `main` recibe únicamente versiones completas y estables.

Versiones definidas:

| Versión  | Alcance                                                                                                              |
|----------|----------------------------------------------------------------------------------------------------------------------|
| `v1.0.0` | Avance 1: carga de datos y análisis exploratorio inicial.                                                            |
| `v1.0.1` | Cambio administrativo de la rama principal de `master` a `main`.                                                     |
| `v1.1.0` | Ingeniería de características y preprocesamiento.                                                                    |
| `v1.2.0` | Entrenamiento, evaluación, optimización y selección del modelo final.                                                |
| `v1.3.0` | Cierre del Avance 3: monitoreo, detección de data drift, aplicación Streamlit y documentación del proyecto.          |
| `v1.4.0` | Cierre del Avance 4: serialización del modelo, API FastAPI, predicción individual y por lotes, Docker y documentación final. |

Cada tag se crea después de integrar y validar completamente su alcance en `main`.

El formato utilizado para los tags anotados es:

```text
Version 1.4.0 - API de prediccion y contenerizacion con Docker
```

## Autor

**Martín Darío Fernández**

- GitHub: [Fernandez-Martin-Dario](https://github.com/Fernandez-Martin-Dario)
- Repositorio: [proyecto-integrador-m5-mlops](https://github.com/Fernandez-Martin-Dario/proyecto-integrador-m5-mlops)
