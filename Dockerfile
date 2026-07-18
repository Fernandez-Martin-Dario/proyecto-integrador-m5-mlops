FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .

RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY Base_de_datos.xlsx .
COPY src ./src

RUN python -c "from src.model_training_evaluation import preparar_datos, entrenar_y_guardar_modelo_final; X_train, _, y_train, _ = preparar_datos(); entrenar_y_guardar_modelo_final(X_train, y_train)"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]