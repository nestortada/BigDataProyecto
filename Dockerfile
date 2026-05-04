FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY Code ./Code

RUN mkdir -p \
    Data/Raw \
    Data/Processed \
    Models/trained/all_models \
    Reports/Model/feature_importance

COPY Models/trained/all_models/RandomForest_early_pipeline.joblib Models/trained/all_models/
COPY Reports/Model/feature_importance/RandomForest_early_fi.csv Reports/Model/feature_importance/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"

CMD ["uvicorn", "Code.Operationalization.secop_fastapi.main:app", "--host", "0.0.0.0", "--port", "8000"]
