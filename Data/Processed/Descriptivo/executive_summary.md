# Executive Summary - SECOP Descriptive Analysis

## Context
This descriptive profile was generated over `processed/datos_feature_engineering.parquet` to support evidence on traceability, completeness, coherence, competition, and information quality in public procurement processes aligned with ODS 16.6.

## Dataset Overview
- Rows: 429,541
- Columns: 100
- Memory usage: 520.06 MB
- Numeric columns: 32
- Binary numeric columns: 23
- Categorical columns: 19
- Datetime columns: 3

## Data Quality
- Exact duplicate rows across all columns: 0.
- `id_del_proceso`: 0 duplicate rows across 0 repeated keys.
- `referencia_del_proceso`: 52,346 duplicate rows across 13,692 repeated keys.
- `id_del_portafolio`: 34,599 duplicate rows across 14,324 repeated keys.
- `urlproceso`: 50 duplicate rows across 2 repeated keys.
- `id_del_proceso, referencia_del_proceso`: 0 duplicate rows across 0 repeated keys.
- `ratio_proveedores_vs_invitados`: 401,321 nulls (93.43%).
- `categorias_adicionales`: 360,572 nulls (83.94%).
- `fecha_de_apertura_de_respuesta`: 356,703 nulls (83.04%).
- `dias_recepcion_a_apertura`: 356,703 nulls (83.04%).
- `dias_publicacion_a_apertura`: 356,703 nulls (83.04%).
- `dias_hasta_recepcion`: 335,358 nulls (78.07%).
- `fecha_de_recepcion_de`: 335,314 nulls (78.06%).
- `dias_publicacion_a_recepcion`: 335,314 nulls (78.06%).
- `ciudad_entidad`: 90,675 nulls (21.11%).
- `departamento_entidad`: 7,735 nulls (1.80%).

## Descriptive Highlights
- `precio_base`: mean 403548398.1969, median 14487500.0000, p95 496002248.4000, IQR outliers 65,775 (15.31%).
- `total_respuestas`: mean 0.5269, median 0.0000, p95 3.0000, IQR outliers 62,103 (14.46%).
- `transparency_score`: mean 0.6409, median 0.5900, p95 0.9550, IQR outliers 173,720 (40.44%).
- `nombre_del_procedimiento`: 251,183 categories; top category `PRESTACION DE SERVICIOS PROFESIONALES` represents 2.91%.
- `categorias_adicionales`: 33,745 categories; top category `None` represents 83.94%.
- `codigo_principal_de_categoria`: 8,594 categories; top category `V1.80111600` represents 27.27%.
- `nombre_de_la_unidad_de`: 3,454 categories; top category `SUBDIRECCION DE CONTRATACION` represents 1.86%.
- `entidad`: 1,985 categories; top category `(Secretaría Distrital de Integración Social)` represents 1.78%.

## Temporal Coverage
- `fecha_de_publicacion` mapped to `fecha_de_publicacion_del` spans from 2021-04-12T00:00:00+00:00 to 2022-01-07T00:00:00+00:00 with 429,541 non-null records.
- `fecha_de_recepcion` mapped to `fecha_de_recepcion_de` spans from 2021-03-15T00:00:00+00:00 to 2023-07-21T00:00:00+00:00 with 94,227 non-null records.
- `fecha_de_apertura` mapped to `fecha_de_apertura_de_respuesta` spans from 2021-04-13T00:00:00+00:00 to 2023-07-22T00:00:00+00:00 with 72,838 non-null records.

## Transparency and Institutional Risk Signals
- `score_completitud`: mean 0.9552, median 1.0000, min 0.2000, max 1.0000.
- `score_trazabilidad`: mean 0.9966, median 1.0000, min 0.4000, max 1.0000.
- `score_temporal`: mean 0.1918, median 0.0000, min 0.0000, max 1.0000.
- `score_competencia`: mean 0.2863, median 0.2000, min 0.2000, max 1.0000.
- `transparency_score`: mean 0.6409, median 0.5900, min 0.3475, max 1.0000.
- `nivel_riesgo_transparencia`: most common category `medio` with 78.62% share.
- `confianza_label`: most common category `baja` with 61.73% share.

## Risks for Next Stages
- High-null columns may weaken downstream modeling and can reduce reliability in institutional transparency indicators.
- Repeated process identifiers should be reviewed before using the data for entity-level benchmarking or process-level aggregation.
- Strong skewness and outliers in monetary and participation variables suggest using robust statistics and capped transformations in later stages.
- Temporal gaps in publication, reception, and opening dates can affect traceability analyses and timeline-based controls.