# SECOP Transparencia Temprana - FastAPI

Aplicacion FastAPI para servir el formulario de prediccion temprana de riesgo de baja transparencia y consumir el modelo `RandomForest_early_pipeline.joblib`.

## Ejecucion

Desde la raiz del proyecto:

### Recomendado en Windows/OneDrive

```bash
uvicorn Code.Operationalization.secop_fastapi.main:app --host 127.0.0.1 --port 8000
```

### Desarrollo con recarga automatica

Si usas `--reload`, limita la carpeta vigilada a esta app. No ejecutes `uvicorn --reload` vigilando toda la raiz del proyecto, porque puede intentar escanear `.venv/lib64` y fallar con `WinError 1920`.

```bash
uvicorn Code.Operationalization.secop_fastapi.main:app --reload --reload-dir Code/Operationalization/secop_fastapi --host 127.0.0.1 --port 8000
```

Luego abrir:

```text
http://127.0.0.1:8000
```

## Ejecucion con Docker

Desde la raiz del proyecto:

```bash
docker compose up --build
```

Luego abrir:

```text
http://127.0.0.1:8000
```

Para desarrollo con recarga automatica:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Para cargar el parquet raw en PostgreSQL desde el contenedor:

```bash
docker compose --profile jobs run --rm secop-db-load
```

## Endpoints

- `GET /`: formulario web.
- `GET /api/health`: estado del servicio y carga del modelo.
- `POST /api/predict`: prediccion JSON usando las variables tempranas del proceso SECOP II.

El backend calcula automaticamente `precio_base_log = log1p(precio_base)` y clasifica riesgo alto cuando la probabilidad de clase 1 es mayor o igual a `0.645`.
