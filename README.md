# SECOP Transparencia Temprana

Proyecto academico de Big Data para analizar procesos de contratacion publica de SECOP II y estimar, en etapas tempranas, el riesgo de baja transparencia de un proceso contractual.

El proyecto esta alineado con el ODS 16.6, que busca instituciones eficaces, responsables y transparentes. La idea central no es predecir adjudicaciones, sino transformar datos abiertos de contratacion en senales medibles de completitud, trazabilidad, coherencia temporal y competencia.

## Objetivo

Construir un flujo completo de datos y analitica para:

- Descargar procesos SECOP II desde datos.gov.co.
- Preparar una base analitica sin columnas posadjudicacion que puedan contaminar el analisis temprano.
- Limpiar, normalizar y enriquecer los datos con variables de calidad institucional.
- Entrenar modelos de machine learning que clasifiquen riesgo de baja transparencia.
- Operacionalizar el mejor flujo disponible mediante una API FastAPI y un formulario web.

El resultado operativo es una aplicacion llamada **SECOP Transparencia Temprana**, que recibe informacion basica de un proceso y devuelve una probabilidad de riesgo, una etiqueta y factores clave del modelo.

## Como esta implementado

La solucion esta organizada por etapas reproducibles:

| Etapa | Implementacion | Salidas principales |
| --- | --- | --- |
| Descarga | `Code/DataPrep/descarga_datos_secop.py` usa Socrata/datos.gov.co. | `Data/Raw/secop_procesos.parquet` |
| Vista analitica | `Code/Operationalization/crear_datos_analisis_secop.py` elimina campos posadjudicacion. | `Data/Raw/datos_analisis.parquet` |
| Limpieza | `Code/DataPrep/limpieza_datos_analisis_secop.py` normaliza textos, fechas, numericos, nulos y duplicados. | `Data/Processed/Limpieza/datos_analisis_limpio.parquet` |
| Feature engineering | `Code/DataPrep/feature_engineering_secop.py` crea scores, flags y targets de transparencia. | `Data/Processed/Limpieza/datos_feature_engineering.parquet` |
| Analisis | Scripts descriptivos, EDA e inferenciales en `Code/Operationalization/`. | `Data/Processed/`, `Docs/`, `Reports/` |
| Modelado | `Code/Operationalization/pipeline_modelado_secop.py` compara modelos sklearn y opcionales. | `Models/trained/`, `Reports/Model/` |
| API | `Code/Operationalization/secop_fastapi/main.py` sirve el modelo y el formulario web. | `GET /`, `GET /api/health`, `POST /api/predict` |

La API carga el artefacto:

```text
Models/trained/all_models/RandomForest_early_pipeline.joblib
```

y usa la importancia de variables de:

```text
Reports/Model/feature_importance/RandomForest_early_fi.csv
```

El modelo operativo trabaja con variables tempranas como entidad, NIT, ubicacion, fase, modalidad, tipo de contrato, categoria, descripcion y precio base. El backend calcula automaticamente `precio_base_log = log1p(precio_base)` y clasifica:

- `Riesgo alto` si la probabilidad es mayor o igual a `0.645`.
- `Revisar` si la probabilidad esta entre `0.35` y `0.645`.
- `Riesgo bajo` si la probabilidad es menor a `0.35`.

## Estructura del proyecto

```text
Code/
  DataPrep/                 # Descarga, limpieza y feature engineering
  Operationalization/       # Analisis, modelado, carga a Postgres y FastAPI
Data/
  Raw/                      # Datos fuente y vista analitica
  Processed/                # Datos limpios, EDA, descriptivo, inferencial y modelado
Models/
  trained/                  # Pipelines entrenados en formato joblib
Reports/
  Model/                    # Metricas, curvas, matrices, umbrales e importancias
Docs/
  Project/                  # Informes academicos del proyecto
Dockerfile
docker-compose.yml
docker-compose.dev.yml
requirements.txt
```

## Requisitos

Para ejecutar con Docker:

- Docker Desktop o Docker Engine.
- Docker Compose.

Para ejecutar localmente:

- Python 3.11 recomendado.
- Dependencias de `requirements.txt`.
- Los artefactos de modelo en `Models/trained/all_models/`.

## Configuracion

1. Crear el archivo `.env` desde el ejemplo:

```bash
cp .env.example .env
```

En PowerShell:

```powershell
Copy-Item .env.example .env
```

2. Revisar las variables principales:

```text
POSTGRES_USER=secop
POSTGRES_PASSWORD=secop_password
POSTGRES_DB=secop
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
API_PORT=8000
```

Las variables de Socrata son opcionales y solo se necesitan si se quiere volver a descargar datos desde datos.gov.co.

## Ejecucion recomendada con Docker

Desde la raiz del proyecto:

```bash
docker compose up --build
```

Esto levanta:

- `postgres`: base de datos PostgreSQL 16.
- `api`: aplicacion FastAPI en el puerto configurado, por defecto `8000`.

Abrir en el navegador:

```text
http://127.0.0.1:8000
```

Tambien se puede revisar la documentacion interactiva de la API:

```text
http://127.0.0.1:8000/docs
```

Verificacion rapida:

```bash
curl http://127.0.0.1:8000/api/health
```

## Cargar datos en PostgreSQL

El servicio de prediccion no necesita consultar Postgres para responder, pero el proyecto incluye un job para cargar el parquet crudo en la base de datos.

Con los contenedores disponibles, ejecutar:

```bash
docker compose --profile jobs run --rm secop-db-load
```

Este job ejecuta:

```text
Code/Operationalization/postgres_carga_datos.py
```

y carga `Data/Raw/secop_procesos.parquet` en la tabla `secop_procesos`.

## Ejecucion en desarrollo

Para levantar la API con recarga automatica dentro de Docker:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Para ejecutar localmente sin Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn Code.Operationalization.secop_fastapi.main:app --host 127.0.0.1 --port 8000
```

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn Code.Operationalization.secop_fastapi.main:app --host 127.0.0.1 --port 8000
```

Si se usa recarga automatica local, limitar la carpeta vigilada a la app:

```bash
uvicorn Code.Operationalization.secop_fastapi.main:app --reload --reload-dir Code/Operationalization/secop_fastapi --host 127.0.0.1 --port 8000
```

## Ejemplo de prediccion por API

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "entidad": "Alcaldia Mayor de Bogota",
    "nit_entidad": "899999061",
    "departamento_entidad": "Bogota D.C.",
    "ciudad_entidad": "Bogota",
    "ordenentidad": "Territorial",
    "nombre_del_procedimiento": "Adquisicion de equipos tecnologicos",
    "descripci_n_del_procedimiento": "Compra de equipos para fortalecimiento institucional",
    "fase": "Presentacion de oferta",
    "precio_base": 250000000,
    "modalidad_de_contratacion": "Licitacion publica",
    "justificaci_n_modalidad_de": "Contratacion publica competitiva",
    "codigo_principal_de_categoria": "43211500",
    "tipo_de_contrato": "Compraventa",
    "categorias_adicionales": "Equipos de computo"
  }'
```

Ejemplo de respuesta abreviada:

```json
{
  "probabilidad_riesgo": 0.123456,
  "umbral": 0.645,
  "clase_predicha": 0,
  "etiqueta": "Riesgo bajo",
  "modelo": "RandomForest_early_pipeline.joblib",
  "factores_clave": [
    {
      "feature": "modalidad_de_contratacion",
      "label": "Modalidad",
      "importance": 0.123456
    }
  ]
}
```

## Reproducir el pipeline completo

Si se quiere regenerar artefactos desde cero, el orden sugerido es:

```bash
python Code/DataPrep/descarga_datos_secop.py
python Code/Operationalization/crear_datos_analisis_secop.py
python Code/DataPrep/limpieza_datos_analisis_secop.py
python Code/DataPrep/feature_engineering_secop.py
python Code/Operationalization/pipeline_modelado_secop.py
```

Notas:

- La descarga desde Socrata puede tardar y depende de la conexion y limites de datos.gov.co.
- El archivo `.env` debe existir para que los scripts encuentren la raiz del proyecto y la configuracion de base de datos.
- Los modelos opcionales como XGBoost, LightGBM o CatBoost solo se ejecutan si sus librerias estan instaladas.

## Problemas frecuentes

- Si `docker compose up --build` falla porque el puerto `8000` esta ocupado, cambiar `API_PORT` en `.env`.
- Si Postgres no inicia por puerto ocupado, cambiar `POSTGRES_PORT` en `.env`.
- Si la API reporta que no encuentra el modelo, validar que exista `Models/trained/all_models/RandomForest_early_pipeline.joblib`.
- En Windows con OneDrive, evitar ejecutar `uvicorn --reload` sobre toda la raiz del proyecto; usar `--reload-dir Code/Operationalization/secop_fastapi`.
