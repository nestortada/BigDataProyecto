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

## Endpoints

- `GET /`: formulario web.
- `GET /api/health`: estado del servicio y carga del modelo.
- `POST /api/predict`: prediccion JSON usando las variables tempranas del proceso SECOP II.

El backend calcula automaticamente `precio_base_log = log1p(precio_base)` y clasifica riesgo alto cuando la probabilidad de clase 1 es mayor o igual a `0.645`.
