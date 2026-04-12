# Operationalization

Carpeta destinada al codigo de despliegue, automatizacion y consumo en produccion.

## Flujos disponibles

- `postgres_carga_datos.py`: carga el parquet de SECOP hacia PostgreSQL usando Prefect.
- `analisis_ods16_secop_prefect.py`: ejecuta un pipeline de analisis descriptivo y calidad de datos orientado al ODS 16.6.

## Ejecucion del analisis ODS 16.6

Desde PowerShell:

```powershell
Set-Location "C:\Users\nesto\OneDrive - Universidad de la Sabana\Universidad\2026-1\Big data\Proyecto"
.\.envProyecto\Scripts\Activate.ps1
python .\Code\Operationalization\analisis_ods16_secop_prefect.py
```

## Salidas esperadas

El flujo guarda artefactos en `Data/Processed/ods16_secop/`:

- `variable_types.csv`
- `variable_type_counts.csv`
- `null_profile.csv`
- `duplicates_summary.csv`
- `key_cardinality.csv`
- `semantic_null_tokens.csv`
- `frequency_index.csv`
- tablas de frecuencia por columna en `Data/Processed/ods16_secop/frequency_tables/`
- `univariate_statistics.csv`
- `correlation_matrix.csv`
- `covariance_matrix.csv`
- `top_correlations.csv`
- `entity_completeness.csv`
- `modalidad_vs_precio_base.csv`
- `summary.json`
- Graficos PNG en `Data/Processed/ods16_secop/plots/`
- Graficos categoricos en `plots/categorical/`
- Stacked bar plots en `plots/stacked/`
- Scatter plots en `plots/scatter/`
- Dot-Line plots en `plots/line_dot/`
