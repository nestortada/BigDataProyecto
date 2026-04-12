from __future__ import annotations

import inspect
import json
import re
import time
import warnings
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
    PLOTTING_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)

try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
except ImportError:
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def flow(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def get_run_logger():
        raise RuntimeError("Prefect no esta instalado en el entorno actual.")


SOURCE_PARQUET = Path("Data/Raw/datos_analisis.parquet")
OUTPUT_DIR = Path("Data/Processed/ods16_secop")

DATE_COLUMNS = [
    "fecha_de_publicacion",
    "fecha_de_publicacion_del",
    "fecha_de_ultima_publicaci",
    "fecha_de_recepcion_de",
    "fecha_de_apertura_de_respuesta",
    "fecha_de_apertura_efectiva",
]

NUMERIC_COLUMNS = [
    "precio_base",
    "duracion",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "proveedores_que_manifestaron",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_unicos_con",
    "numero_de_lotes",
    "visualizaciones_del",
]

TEXT_COLUMNS = [
    "nombre_del_procedimiento",
    "descripci_n_del_procedimiento",
]

KEY_CATEGORICAL_COLUMNS = [
    "entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "ordenentidad",
    "codigo_entidad",
    "nombre_del_procedimiento",
    "fase",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "subtipo_de_contrato",
    "estado_del_procedimiento",
    "id_estado_del_procedimiento",
    "estado_de_apertura_del_proceso",
    "estado_resumen",
    "nombre_de_la_unidad_de",
]

POTENTIAL_LEAKAGE_COLUMNS = {
    "estado_del_procedimiento",
    "id_estado_del_procedimiento",
    "estado_de_apertura_del_proceso",
    "estado_resumen",
    "fecha_de_ultima_publicaci",
    "fecha_de_apertura_efectiva",
}

HIGH_NULL_THRESHOLD = 0.30
NEAR_CONSTANT_THRESHOLD = 0.95
RARE_CATEGORY_THRESHOLD = 0.01
SHORT_TEXT_THRESHOLD = 10

warnings.filterwarnings("ignore", category=FutureWarning)


def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Code").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path.cwd())
SOURCE_PARQUET = PROJECT_ROOT / SOURCE_PARQUET
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR


def log(message: str) -> None:
    print(f"[EDA SECOP] {message}")


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger = None

        try:
            logger = get_run_logger()
        except Exception:
            logger = None

        try:
            return func(*args, **kwargs)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            if logger is not None:
                logger.info("La tarea %s tardo %.2f segundos.", func.__name__, elapsed_seconds)
            else:
                print(f"La tarea {func.__name__} tardo {elapsed_seconds:.2f} segundos.")

    return wrapper


def validate_inputs(func):
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_arguments = signature.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()

        for arg_name, arg_value in bound_arguments.arguments.items():
            if isinstance(arg_value, str) and not arg_value.strip():
                raise ValueError(f"El parametro '{arg_name}' no puede ser una cadena vacia.")

            if isinstance(arg_value, Path) and arg_name in {"path", "base_dir", "output_path", "source_path"}:
                if arg_name in {"path", "source_path"} and not arg_value.exists():
                    raise FileNotFoundError(f"No se encontro la ruta requerida: {arg_value}")

            if arg_name == "df" and not isinstance(arg_value, pd.DataFrame):
                raise TypeError("'df' debe ser un pandas.DataFrame.")

            if arg_name in {"null_profile", "low_variability_profile", "duplicates_summary", "conversion_summary", "temporal_anomalies"} and not isinstance(arg_value, pd.DataFrame):
                raise TypeError(f"'{arg_name}' debe ser un pandas.DataFrame.")

            if arg_name == "output_dirs":
                if not isinstance(arg_value, dict):
                    raise TypeError("'output_dirs' debe ser un diccionario.")
                required_keys = {"base", "frequency_tables", "cross_tabs", "diagnostics"}
                if not required_keys.issubset(arg_value.keys()):
                    raise ValueError("'output_dirs' no contiene las carpetas requeridas.")

        return func(*args, **kwargs)

    return wrapper


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


@task(name="configurar_entorno_eda_secop")
@timing_decorator
def configure_environment() -> None:
    try:
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.max_colwidth", 160)
        pd.set_option("display.width", 220)
        if PLOTTING_AVAILABLE:
            sns.set_theme(style="whitegrid", palette="deep")
            plt.rcParams["figure.figsize"] = (12, 6)
            plt.rcParams["axes.titlesize"] = 12
            plt.rcParams["axes.labelsize"] = 10
    except Exception as e:
        log(f"Error al configurar el entorno: {e}")
        raise
    finally:
        log("Configuracion de entorno finalizada.")


@task(name="crear_directorios_eda_secop")
@timing_decorator
@validate_inputs
def ensure_directories(base_dir: Path) -> dict[str, Path]:
    try:
        directories = {
            "base": base_dir,
            "plots": base_dir / "plots",
            "numeric_plots": base_dir / "plots" / "numeric",
            "categorical_plots": base_dir / "plots" / "categorical",
            "text_plots": base_dir / "plots" / "text",
            "temporal_plots": base_dir / "plots" / "temporal",
            "bivariate_plots": base_dir / "plots" / "bivariate",
            "correlation_plots": base_dir / "plots" / "correlation",
            "frequency_tables": base_dir / "frequency_tables",
            "cross_tabs": base_dir / "cross_tabs",
            "diagnostics": base_dir / "diagnostics",
        }

        for directory in directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        return directories
    except Exception as e:
        log(f"Error al crear directorios de salida: {e}")
        raise
    finally:
        log("Provision de directorios finalizada.")


@timing_decorator
def save_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        log(f"Error al guardar CSV '{path}': {e}")
        raise
    finally:
        log(f"Finalizo el guardado de CSV: {path.name}")


@timing_decorator
def save_json(payload: dict[str, Any], path: Path) -> None:
    try:
        serializable = json.loads(json.dumps(payload, default=json_safe))
        path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log(f"Error al guardar JSON '{path}': {e}")
        raise
    finally:
        log(f"Finalizo el guardado de JSON: {path.name}")


@timing_decorator
def save_plot(fig: Any, path: Path) -> None:
    try:
        if not PLOTTING_AVAILABLE or fig is None:
            return
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as e:
        log(f"Error al guardar grafico '{path}': {e}")
        raise
    finally:
        if PLOTTING_AVAILABLE and fig is not None:
            plt.close(fig)


def normalize_text_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    if text == "":
        return pd.NA
    return text


def flatten_url(value: Any) -> Any:
    if isinstance(value, dict):
        url = value.get("url")
        return normalize_text_value(url)
    return normalize_text_value(value)


def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if pd.isna(value):
        return pd.NA
    return value


def hashable_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    subset = df[columns].copy() if columns else df.copy()
    for column in subset.columns:
        subset[column] = subset[column].map(make_hashable)
    return subset


@task(name="cargar_dataset_eda_secop")
@timing_decorator
@validate_inputs
def load_dataset(path: Path) -> pd.DataFrame:
    try:
        log(f"Leyendo parquet: {path}")
        df = pd.read_parquet(path)
        log(f"Dataset cargado con shape {df.shape}")
        return df
    except Exception as e:
        log(f"Error al leer el parquet: {e}")
        raise
    finally:
        log("Carga de dataset finalizada.")


@task(name="preparar_dataframe_eda_secop")
@timing_decorator
@validate_inputs
def prepare_dataframe(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = raw_df.copy()
        conversion_records: list[dict[str, Any]] = []

        for column in df.columns:
            if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                if column == "urlproceso":
                    continue
                df[column] = df[column].map(normalize_text_value)

        df["url"] = df["urlproceso"].map(flatten_url) if "urlproceso" in df.columns else pd.NA

        for column in DATE_COLUMNS:
            if column not in df.columns:
                conversion_records.append(
                    {
                        "column": column,
                        "conversion_type": "datetime",
                        "status": "missing_column",
                        "non_null_before": 0,
                        "non_null_after": 0,
                        "coerced_to_null": 0,
                        "parse_success_pct": 0.0,
                    }
                )
                continue

            non_null_before = int(df[column].notna().sum())
            parsed = pd.to_datetime(df[column], errors="coerce", utc=True)
            non_null_after = int(parsed.notna().sum())
            coerced_to_null = max(non_null_before - non_null_after, 0)
            parse_success_pct = round((non_null_after / non_null_before) * 100, 2) if non_null_before else 0.0
            df[column] = parsed
            conversion_records.append(
                {
                    "column": column,
                    "conversion_type": "datetime",
                    "status": "converted",
                    "non_null_before": non_null_before,
                    "non_null_after": non_null_after,
                    "coerced_to_null": coerced_to_null,
                    "parse_success_pct": parse_success_pct,
                }
            )

        for column in NUMERIC_COLUMNS:
            if column not in df.columns:
                conversion_records.append(
                    {
                        "column": column,
                        "conversion_type": "numeric",
                        "status": "missing_column",
                        "non_null_before": 0,
                        "non_null_after": 0,
                        "coerced_to_null": 0,
                        "parse_success_pct": 0.0,
                    }
                )
                continue

            non_null_before = int(df[column].notna().sum())
            numeric = pd.to_numeric(df[column], errors="coerce")
            non_null_after = int(numeric.notna().sum())
            coerced_to_null = max(non_null_before - non_null_after, 0)
            parse_success_pct = round((non_null_after / non_null_before) * 100, 2) if non_null_before else 0.0
            df[column] = numeric
            conversion_records.append(
                {
                    "column": column,
                    "conversion_type": "numeric",
                    "status": "converted",
                    "non_null_before": non_null_before,
                    "non_null_after": non_null_after,
                    "coerced_to_null": coerced_to_null,
                    "parse_success_pct": parse_success_pct,
                }
            )

        conversion_summary = pd.DataFrame(conversion_records)
        return df, conversion_summary
    except Exception as e:
        log(f"Error durante la preparacion del dataframe: {e}")
        raise
    finally:
        log("Preparacion del dataframe finalizada.")


@task(name="inferir_grupos_variables_eda_secop")
@timing_decorator
@validate_inputs
def infer_variable_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    try:
        datetime_columns = [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
        numeric_columns = [
            column for column in df.columns if pd.api.types.is_numeric_dtype(df[column]) and column not in datetime_columns
        ]
        text_columns = [column for column in TEXT_COLUMNS if column in df.columns]
        technical_columns = [column for column in ["id_del_proceso", "referencia_del_proceso", "urlproceso", "url"] if column in df.columns]
        categorical_columns = [
            column
            for column in df.columns
            if column not in datetime_columns + numeric_columns + text_columns + technical_columns
        ]

        return {
            "datetime": sorted(datetime_columns),
            "numeric": sorted(numeric_columns),
            "text": sorted(text_columns),
            "categorical": sorted(categorical_columns),
            "technical": sorted(technical_columns),
        }
    except Exception as e:
        log(f"Error al inferir grupos de variables: {e}")
        raise
    finally:
        log("Inferencia de grupos de variables finalizada.")


@task(name="perfilar_dataset_eda_secop")
@timing_decorator
@validate_inputs
def build_overview(df: pd.DataFrame) -> pd.DataFrame:
    try:
        hashable_df = hashable_frame(df)
        return pd.DataFrame(
            [
                {
                    "n_rows": len(df),
                    "n_columns": len(df.columns),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2),
                    "exact_duplicate_rows": int(hashable_df.duplicated().sum()),
                    "id_del_proceso_unique": int(df["id_del_proceso"].nunique(dropna=True)) if "id_del_proceso" in df.columns else None,
                    "referencia_del_proceso_unique": int(df["referencia_del_proceso"].nunique(dropna=True)) if "referencia_del_proceso" in df.columns else None,
                }
            ]
        )
    except Exception as e:
        log(f"Error al construir el overview del dataset: {e}")
        raise
    finally:
        log("Overview del dataset finalizado.")


def build_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )


@task(name="perfilar_nulos_eda_secop")
@timing_decorator
@validate_inputs
def build_null_profile(df: pd.DataFrame) -> pd.DataFrame:
    try:
        profile = pd.DataFrame(
            {
                "column": df.columns,
                "null_count": df.isna().sum().values,
                "null_pct": (df.isna().mean().values * 100).round(4),
                "non_null_count": df.notna().sum().values,
            }
        )
        return profile.sort_values(["null_pct", "null_count"], ascending=[False, False]).reset_index(drop=True)
    except Exception as e:
        log(f"Error al construir el perfil de nulos: {e}")
        raise
    finally:
        log("Perfil de nulos finalizado.")


@task(name="perfilar_duplicados_eda_secop")
@timing_decorator
@validate_inputs
def build_duplicates_summary(df: pd.DataFrame) -> pd.DataFrame:
    try:
        records = []
        hashable_df = hashable_frame(df)
        for label, subset in {
            "exact_all_columns": None,
            "id_del_proceso": ["id_del_proceso"] if "id_del_proceso" in df.columns else None,
            "referencia_del_proceso": ["referencia_del_proceso"] if "referencia_del_proceso" in df.columns else None,
            "id_y_referencia": [column for column in ["id_del_proceso", "referencia_del_proceso"] if column in df.columns],
        }.items():
            if label != "exact_all_columns" and (subset is None or len(subset) == 0):
                continue

            if label == "exact_all_columns":
                duplicate_rows = int(hashable_df.duplicated().sum())
                duplicate_keys = None
            else:
                mask = hashable_df.duplicated(subset=subset, keep=False)
                duplicate_rows = int(mask.sum())
                duplicate_keys = int(hashable_df.loc[mask, subset].drop_duplicates().shape[0]) if duplicate_rows else 0

            records.append(
                {
                    "duplicate_type": label,
                    "subset": ", ".join(subset) if subset else "all_columns",
                    "duplicate_rows": duplicate_rows,
                    "duplicate_groups_or_keys": duplicate_keys,
                }
            )

        return pd.DataFrame(records)
    except Exception as e:
        log(f"Error al construir el resumen de duplicados: {e}")
        raise
    finally:
        log("Resumen de duplicados finalizado.")


@task(name="perfilar_baja_variabilidad_eda_secop")
@timing_decorator
@validate_inputs
def build_low_variability_profile(df: pd.DataFrame) -> pd.DataFrame:
    try:
        records = []
        for column in df.columns:
            hashable_series = df[column].map(make_hashable)
            non_null = hashable_series.dropna()
            value_counts = hashable_series.value_counts(dropna=False, normalize=True)
            dominant_share = float(value_counts.iloc[0]) if not value_counts.empty else np.nan
            dominant_value = value_counts.index[0] if not value_counts.empty else None
            records.append(
                {
                    "column": column,
                    "nunique_including_null": int(hashable_series.nunique(dropna=False)),
                    "nunique_excluding_null": int(hashable_series.nunique(dropna=True)),
                    "dominant_value": json_safe(dominant_value),
                    "dominant_share_pct": round(dominant_share * 100, 4) if pd.notna(dominant_share) else np.nan,
                    "is_near_constant": bool(pd.notna(dominant_share) and dominant_share >= NEAR_CONSTANT_THRESHOLD),
                    "is_single_value_non_null": bool(non_null.nunique(dropna=True) <= 1) if len(non_null) else False,
                }
            )

        return pd.DataFrame(records).sort_values(
            ["is_near_constant", "dominant_share_pct"], ascending=[False, False]
        ).reset_index(drop=True)
    except Exception as e:
        log(f"Error al perfilar la baja variabilidad: {e}")
        raise
    finally:
        log("Perfil de baja variabilidad finalizado.")


@task(name="construir_recomendaciones_columnas_eda_secop")
@timing_decorator
@validate_inputs
def build_column_recommendations(
    null_profile: pd.DataFrame,
    low_variability_profile: pd.DataFrame,
) -> pd.DataFrame:
    try:
        null_map = null_profile.set_index("column")
        low_var_map = low_variability_profile.set_index("column")
        columns = sorted(set(null_profile["column"]).union(low_variability_profile["column"]).union(POTENTIAL_LEAKAGE_COLUMNS))

        records = []
        for column in columns:
            null_pct = float(null_map.loc[column, "null_pct"]) if column in null_map.index else 0.0
            dominant_share_pct = (
                float(low_var_map.loc[column, "dominant_share_pct"]) if column in low_var_map.index else np.nan
            )

            recommendations = []
            if null_pct > HIGH_NULL_THRESHOLD * 100:
                recommendations.append("high_nulls")
            if pd.notna(dominant_share_pct) and dominant_share_pct >= NEAR_CONSTANT_THRESHOLD * 100:
                recommendations.append("near_constant")
            if column in POTENTIAL_LEAKAGE_COLUMNS:
                recommendations.append("potential_leakage")
            if column == "id_del_proceso":
                recommendations.append("primary_identifier")
            if column == "referencia_del_proceso":
                recommendations.append("non_unique_reference_identifier")

            records.append(
                {
                    "column": column,
                    "null_pct": round(null_pct, 4),
                    "dominant_share_pct": round(dominant_share_pct, 4) if pd.notna(dominant_share_pct) else np.nan,
                    "recommendation_count": len(recommendations),
                    "recommendations": " | ".join(recommendations) if recommendations else "ok",
                }
            )

        return pd.DataFrame(records).sort_values(
            ["recommendation_count", "null_pct", "dominant_share_pct"], ascending=[False, False, False]
        ).reset_index(drop=True)
    except Exception as e:
        log(f"Error al construir recomendaciones de columnas: {e}")
        raise
    finally:
        log("Construccion de recomendaciones de columnas finalizada.")


def safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column] if column in df.columns else pd.Series(dtype="object")


@task(name="analisis_univariado_numerico_eda_secop")
@timing_decorator
@validate_inputs
def run_numeric_univariate_analysis(df: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    try:
        records = []
        numeric_columns = [column for column in NUMERIC_COLUMNS if column in df.columns]

        for column in numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                records.append(
                    {
                        "column": column,
                        "non_null_count": 0,
                        "null_count": int(series.isna().sum()),
                        "mean": np.nan,
                        "median": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "p01": np.nan,
                        "p05": np.nan,
                        "p25": np.nan,
                        "p50": np.nan,
                        "p75": np.nan,
                        "p95": np.nan,
                        "p99": np.nan,
                        "max": np.nan,
                        "iqr": np.nan,
                        "lower_bound_iqr": np.nan,
                        "upper_bound_iqr": np.nan,
                        "outlier_count_iqr": 0,
                        "outlier_pct_iqr": 0.0,
                    }
                )
                continue

            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)

            records.append(
                {
                    "column": column,
                    "non_null_count": int(non_null.shape[0]),
                    "null_count": int(series.isna().sum()),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if non_null.shape[0] > 1 else 0.0,
                    "min": float(non_null.min()),
                    "p01": float(non_null.quantile(0.01)),
                    "p05": float(non_null.quantile(0.05)),
                    "p25": float(q1),
                    "p50": float(non_null.quantile(0.50)),
                    "p75": float(q3),
                    "p95": float(non_null.quantile(0.95)),
                    "p99": float(non_null.quantile(0.99)),
                    "max": float(non_null.max()),
                    "iqr": float(iqr),
                    "lower_bound_iqr": float(lower_bound),
                    "upper_bound_iqr": float(upper_bound),
                    "outlier_count_iqr": int(outlier_mask.sum()),
                    "outlier_pct_iqr": round((outlier_mask.mean() * 100), 4),
                }
            )

            if PLOTTING_AVAILABLE:
                fig, ax = plt.subplots()
                sns.histplot(non_null, bins=40, kde=True, ax=ax, color="#2a9d8f")
                ax.set_title(f"Histograma - {column}")
                ax.set_xlabel(column)
                save_plot(fig, output_dirs["numeric_plots"] / f"{column}_hist.png")

                fig, ax = plt.subplots()
                sns.boxplot(x=non_null, ax=ax, color="#e76f51")
                ax.set_title(f"Boxplot - {column}")
                ax.set_xlabel(column)
                save_plot(fig, output_dirs["numeric_plots"] / f"{column}_box.png")

        return pd.DataFrame(records)
    except Exception as e:
        log(f"Error en el analisis univariado numerico: {e}")
        raise
    finally:
        log("Analisis univariado numerico finalizado.")


@task(name="analisis_univariado_categorico_eda_secop")
@timing_decorator
@validate_inputs
def run_categorical_univariate_analysis(df: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    try:
        records = []
        technical_columns = {"id_del_proceso", "referencia_del_proceso", "urlproceso", "url"}
        candidate_columns = [
            column
            for column in df.columns
            if column not in NUMERIC_COLUMNS + DATE_COLUMNS
            and column not in TEXT_COLUMNS
            and column not in technical_columns
        ]

        for column in candidate_columns:
            if column == "urlproceso":
                continue

            series = safe_series(df, column)
            non_null = series.dropna()
            frequencies = series.value_counts(dropna=False)
            frequencies_rel = series.value_counts(dropna=False, normalize=True)

            if frequencies.empty:
                continue

            top_value = frequencies.index[0]
            top_count = int(frequencies.iloc[0])
            top_share_pct = float(frequencies_rel.iloc[0] * 100)
            rare_categories_count = int((frequencies_rel[frequencies_rel < RARE_CATEGORY_THRESHOLD]).shape[0])

            records.append(
                {
                    "column": column,
                    "non_null_count": int(non_null.shape[0]),
                    "null_count": int(series.isna().sum()),
                    "nunique_excluding_null": int(series.nunique(dropna=True)),
                    "top_category": json_safe(top_value),
                    "top_count": top_count,
                    "top_share_pct": round(top_share_pct, 4),
                    "rare_categories_count": rare_categories_count,
                }
            )

            frequency_table = pd.DataFrame(
                {
                    "category": [json_safe(value) for value in frequencies.index],
                    "count": frequencies.values,
                    "relative_pct": (frequencies_rel.values * 100).round(4),
                }
            )
            save_csv(frequency_table, output_dirs["frequency_tables"] / f"{column}_frequency.csv")

            plot_table = frequency_table.head(15).copy()
            if plot_table.empty:
                continue

            if PLOTTING_AVAILABLE:
                fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(plot_table))))
                sns.barplot(data=plot_table, y="category", x="count", ax=ax, color="#457b9d")
                ax.set_title(f"Top categorias - {column}")
                ax.set_xlabel("Conteo")
                ax.set_ylabel(column)
                save_plot(fig, output_dirs["categorical_plots"] / f"{column}_top_categories.png")

        return pd.DataFrame(records).sort_values(
            ["nunique_excluding_null", "top_share_pct"], ascending=[False, False]
        ).reset_index(drop=True)
    except Exception as e:
        log(f"Error en el analisis univariado categorico: {e}")
        raise
    finally:
        log("Analisis univariado categorico finalizado.")


@task(name="analisis_texto_eda_secop")
@timing_decorator
@validate_inputs
def run_text_analysis(df: pd.DataFrame, output_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        summary_records = []
        enriched_df = df.copy()

        for column in TEXT_COLUMNS:
            if column not in enriched_df.columns:
                continue

            text_series = enriched_df[column].fillna("").astype(str)
            lengths = text_series.str.len()
            enriched_df[f"{column}_longitud"] = lengths

            summary_records.append(
                {
                    "column": column,
                    "non_null_count": int(enriched_df[column].notna().sum()),
                    "null_count": int(enriched_df[column].isna().sum()),
                    "empty_or_blank_pct": round((((text_series.str.strip() == "").mean()) * 100), 4),
                    "short_text_pct": round(((lengths < SHORT_TEXT_THRESHOLD).mean() * 100), 4),
                    "mean_length": round(float(lengths.mean()), 4),
                    "median_length": round(float(lengths.median()), 4),
                    "p95_length": round(float(lengths.quantile(0.95)), 4),
                    "max_length": int(lengths.max()),
                }
            )

            if PLOTTING_AVAILABLE:
                fig, ax = plt.subplots()
                sns.histplot(lengths, bins=40, kde=True, ax=ax, color="#8d99ae")
                ax.set_title(f"Distribucion de longitud - {column}")
                ax.set_xlabel("Longitud")
                save_plot(fig, output_dirs["text_plots"] / f"{column}_length_hist.png")

        return pd.DataFrame(summary_records), enriched_df
    except Exception as e:
        log(f"Error en el analisis de texto: {e}")
        raise
    finally:
        log("Analisis de texto finalizado.")


@task(name="derivar_variables_temporales_eda_secop")
@timing_decorator
@validate_inputs
def derive_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        enriched_df = df.copy()

        for column in DATE_COLUMNS:
            if column not in enriched_df.columns:
                enriched_df[column] = pd.NaT

        enriched_df["dias_hasta_recepcion"] = (
            enriched_df["fecha_de_recepcion_de"] - enriched_df["fecha_de_publicacion"]
        ).dt.days
        enriched_df["dias_hasta_apertura"] = (
            enriched_df["fecha_de_apertura_de_respuesta"] - enriched_df["fecha_de_publicacion"]
        ).dt.days

        apertura_final = enriched_df["fecha_de_apertura_efectiva"].combine_first(enriched_df["fecha_de_apertura_de_respuesta"])
        apertura_final = apertura_final.combine_first(enriched_df["fecha_de_recepcion_de"])
        enriched_df["duracion_proceso_si_aplica"] = (apertura_final - enriched_df["fecha_de_publicacion"]).dt.days

        anomaly_definitions = {
            "flag_recepcion_antes_publicacion": enriched_df["fecha_de_recepcion_de"] < enriched_df["fecha_de_publicacion"],
            "flag_apertura_antes_recepcion": enriched_df["fecha_de_apertura_de_respuesta"] < enriched_df["fecha_de_recepcion_de"],
            "flag_apertura_efectiva_antes_recepcion": enriched_df["fecha_de_apertura_efectiva"] < enriched_df["fecha_de_recepcion_de"],
            "flag_publicacion_posterior_apertura": enriched_df["fecha_de_publicacion"] > enriched_df["fecha_de_apertura_de_respuesta"],
            "flag_publicacion_posterior_apertura_efectiva": enriched_df["fecha_de_publicacion"] > enriched_df["fecha_de_apertura_efectiva"],
            "flag_ultima_publicacion_antes_publicacion": enriched_df["fecha_de_ultima_publicaci"] < enriched_df["fecha_de_publicacion"],
        }

        for column, condition in anomaly_definitions.items():
            enriched_df[column] = condition.fillna(False)

        anomaly_columns = list(anomaly_definitions)
        enriched_df["anomalia_temporal"] = enriched_df[anomaly_columns].any(axis=1)

        temporal_summary_records = []
        for column in DATE_COLUMNS + ["dias_hasta_recepcion", "dias_hasta_apertura", "duracion_proceso_si_aplica"]:
            if column not in enriched_df.columns:
                continue

            series = enriched_df[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                temporal_summary_records.append(
                    {
                        "column": column,
                        "dtype": "datetime",
                        "non_null_count": int(series.notna().sum()),
                        "null_count": int(series.isna().sum()),
                        "min": json_safe(series.min()),
                        "max": json_safe(series.max()),
                        "mean": None,
                        "median": None,
                    }
                )
            else:
                numeric_series = pd.to_numeric(series, errors="coerce")
                temporal_summary_records.append(
                    {
                        "column": column,
                        "dtype": str(series.dtype),
                        "non_null_count": int(numeric_series.notna().sum()),
                        "null_count": int(numeric_series.isna().sum()),
                        "min": json_safe(numeric_series.min()),
                        "max": json_safe(numeric_series.max()),
                        "mean": json_safe(numeric_series.mean()),
                        "median": json_safe(numeric_series.median()),
                    }
                )

        temporal_summary = pd.DataFrame(temporal_summary_records)

        anomalies = enriched_df.loc[
            enriched_df["anomalia_temporal"],
            [
                "id_del_proceso",
                "referencia_del_proceso",
                "entidad",
                "estado_del_procedimiento",
                "estado_resumen",
                "fecha_de_publicacion",
                "fecha_de_recepcion_de",
                "fecha_de_apertura_de_respuesta",
                "fecha_de_apertura_efectiva",
                *anomaly_columns,
            ],
        ].copy()

        return enriched_df, temporal_summary, anomalies
    except Exception as e:
        log(f"Error al derivar variables temporales: {e}")
        raise
    finally:
        log("Derivacion de variables temporales finalizada.")


@task(name="construir_indicadores_transparencia_eda_secop")
@timing_decorator
@validate_inputs
def build_transparency_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        indicators = df.copy()

        key_identifier_columns = [column for column in ["id_del_proceso", "referencia_del_proceso", "codigo_entidad"] if column in indicators.columns]
        competencia_columns = [
            column
            for column in [
                "proveedores_con_invitacion",
                "proveedores_que_manifestaron",
                "respuestas_al_procedimiento",
                "respuestas_externas",
                "conteo_de_respuestas_a_ofertas",
                "proveedores_unicos_con",
            ]
            if column in indicators.columns
        ]

        indicators["completitud_registro"] = indicators.notna().mean(axis=1).round(4)
        indicators["tiene_url"] = indicators["url"].notna() if "url" in indicators.columns else False
        indicators["tiene_descripcion"] = indicators["descripci_n_del_procedimiento"].notna() if "descripci_n_del_procedimiento" in indicators.columns else False
        indicators["tiene_precio_base"] = indicators["precio_base"].fillna(0).gt(0) if "precio_base" in indicators.columns else False
        indicators["tiene_ids_clave"] = indicators[key_identifier_columns].notna().all(axis=1) if key_identifier_columns else False

        estado_resumen_ok = indicators["estado_resumen"].fillna("No Definido").ne("No Definido") if "estado_resumen" in indicators.columns else False
        estado_procedimiento_ok = indicators["estado_del_procedimiento"].notna() if "estado_del_procedimiento" in indicators.columns else False
        estado_apertura_ok = indicators["estado_de_apertura_del_proceso"].notna() if "estado_de_apertura_del_proceso" in indicators.columns else False
        indicators["consistencia_estados"] = estado_resumen_ok & estado_procedimiento_ok & estado_apertura_ok

        if competencia_columns:
            indicators["indicador_competencia_reportada"] = indicators[competencia_columns].fillna(0).sum(axis=1).gt(0)
        else:
            indicators["indicador_competencia_reportada"] = False

        temporal_ok_columns = [
            column
            for column in [
                "fecha_de_publicacion",
                "fecha_de_recepcion_de",
                "fecha_de_apertura_de_respuesta",
            ]
            if column in indicators.columns
        ]
        if temporal_ok_columns:
            indicators["indicador_trazabilidad_temporal"] = indicators[temporal_ok_columns].notna().all(axis=1) & ~indicators["anomalia_temporal"]
        else:
            indicators["indicador_trazabilidad_temporal"] = False

        output_columns = [
            column
            for column in [
                "id_del_proceso",
                "referencia_del_proceso",
                "entidad",
                "departamento_entidad",
                "ciudad_entidad",
                "estado_del_procedimiento",
                "estado_resumen",
                "completitud_registro",
                "tiene_url",
                "tiene_descripcion",
                "tiene_precio_base",
                "tiene_ids_clave",
                "consistencia_estados",
                "indicador_competencia_reportada",
                "indicador_trazabilidad_temporal",
                "anomalia_temporal",
            ]
            if column in indicators.columns
        ]
        row_level_indicators = indicators[output_columns].copy()

        entity_group_columns = [
            column
            for column in ["entidad", "nit_entidad", "codigo_entidad", "departamento_entidad", "ciudad_entidad"]
            if column in indicators.columns
        ]

        entity_summary = (
            indicators.groupby(entity_group_columns, dropna=False)
            .agg(
                procesos=("id_del_proceso", "count"),
                completitud_promedio=("completitud_registro", "mean"),
                pct_con_url=("tiene_url", "mean"),
                pct_con_descripcion=("tiene_descripcion", "mean"),
                pct_con_precio_base=("tiene_precio_base", "mean"),
                pct_ids_clave=("tiene_ids_clave", "mean"),
                pct_consistencia_estados=("consistencia_estados", "mean"),
                pct_competencia_reportada=("indicador_competencia_reportada", "mean"),
                pct_trazabilidad_temporal=("indicador_trazabilidad_temporal", "mean"),
                pct_anomalia_temporal=("anomalia_temporal", "mean"),
            )
            .reset_index()
        )

        pct_columns = [column for column in entity_summary.columns if column.startswith("pct_")]
        entity_summary[pct_columns] = entity_summary[pct_columns].mul(100).round(4)
        entity_summary["completitud_promedio"] = entity_summary["completitud_promedio"].mul(100).round(4)
        entity_summary = entity_summary.sort_values(["pct_anomalia_temporal", "procesos"], ascending=[False, False])

        return row_level_indicators, entity_summary
    except Exception as e:
        log(f"Error al construir indicadores de transparencia: {e}")
        raise
    finally:
        log("Construccion de indicadores de transparencia finalizada.")


def save_single_value_diagnostic(column: str, reason: str, output_dirs: dict[str, Path]) -> None:
    diagnostic = pd.DataFrame([{"column": column, "status": "skipped", "reason": reason}])
    save_csv(diagnostic, output_dirs["diagnostics"] / f"{column}_diagnostic.csv")


def plot_price_vs_category(df: pd.DataFrame, category_column: str, output_dirs: dict[str, Path]) -> pd.DataFrame:
    if category_column not in df.columns or "precio_base" not in df.columns:
        return pd.DataFrame()

    analysis_df = df[[category_column, "precio_base"]].dropna().copy()
    if analysis_df.empty:
        save_single_value_diagnostic(category_column, "sin datos no nulos para el cruce con precio_base", output_dirs)
        return pd.DataFrame()

    if analysis_df[category_column].nunique(dropna=True) <= 1:
        save_single_value_diagnostic(category_column, "variable categorica constante", output_dirs)
        grouped = analysis_df.groupby(category_column, dropna=False)["precio_base"].agg(["count", "mean", "median", "min", "max"]).reset_index()
        save_csv(grouped, output_dirs["diagnostics"] / f"precio_base_vs_{category_column}_diagnostic.csv")
        return grouped

    grouped = (
        analysis_df.groupby(category_column, dropna=False)["precio_base"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )

    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.4 * analysis_df[category_column].nunique())))
        order = grouped[category_column].tolist()
        sns.boxplot(data=analysis_df, x="precio_base", y=category_column, order=order, ax=ax, color="#90be6d")
        ax.set_title(f"Precio base vs {category_column}")
        ax.set_xlabel("precio_base")
        ax.set_ylabel(category_column)
        save_plot(fig, output_dirs["bivariate_plots"] / f"precio_base_vs_{category_column}_box.png")

    save_csv(grouped, output_dirs["cross_tabs"] / f"precio_base_vs_{category_column}.csv")
    return grouped


def build_crosstab(
    df: pd.DataFrame,
    index_col: str,
    column_col: str,
    output_dirs: dict[str, Path],
    top_n_index: int | None = None,
) -> pd.DataFrame:
    if index_col not in df.columns or column_col not in df.columns:
        return pd.DataFrame()

    subset = df[[index_col, column_col]].dropna().copy()
    if subset.empty:
        return pd.DataFrame()

    if top_n_index:
        top_categories = subset[index_col].value_counts().head(top_n_index).index
        subset = subset[subset[index_col].isin(top_categories)]

    crosstab = pd.crosstab(subset[index_col], subset[column_col]).reset_index()
    save_csv(crosstab, output_dirs["cross_tabs"] / f"{index_col}_vs_{column_col}.csv")
    return crosstab


def plot_stacked_crosstab(crosstab_df: pd.DataFrame, index_col: str, output_path: Path, title: str) -> None:
    if not PLOTTING_AVAILABLE or crosstab_df.empty or crosstab_df.shape[1] <= 1:
        return

    plot_df = crosstab_df.set_index(index_col)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.45 * len(plot_df))))
    plot_df.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(title)
    ax.set_xlabel("Conteo")
    ax.set_ylabel(index_col)
    save_plot(fig, output_path)


def analyze_modalidad_vs_competencia(df: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    required_columns = [column for column in ["modalidad_de_contratacion", "indicador_competencia_reportada"] if column in df.columns]
    if len(required_columns) < 2:
        return pd.DataFrame()

    grouped = (
        df.groupby("modalidad_de_contratacion", dropna=False)["indicador_competencia_reportada"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"count": "procesos", "mean": "pct_competencia_reportada"})
    )
    grouped["pct_competencia_reportada"] = grouped["pct_competencia_reportada"].mul(100).round(4)
    save_csv(grouped, output_dirs["cross_tabs"] / "modalidad_de_contratacion_vs_competencia.csv")

    if grouped["modalidad_de_contratacion"].nunique(dropna=True) <= 1:
        save_single_value_diagnostic("modalidad_de_contratacion_vs_competencia", "modalidad constante", output_dirs)
        return grouped

    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots()
        sns.barplot(data=grouped, x="modalidad_de_contratacion", y="pct_competencia_reportada", ax=ax, color="#f4a261")
        ax.set_title("Modalidad de contratacion vs competencia reportada")
        ax.set_xlabel("modalidad_de_contratacion")
        ax.set_ylabel("% competencia reportada")
        ax.tick_params(axis="x", rotation=30)
        save_plot(fig, output_dirs["bivariate_plots"] / "modalidad_de_contratacion_vs_competencia.png")
    return grouped


def analyze_tipo_vs_numero_lotes(df: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    required_columns = [column for column in ["tipo_de_contrato", "numero_de_lotes"] if column in df.columns]
    if len(required_columns) < 2:
        return pd.DataFrame()

    subset = df[required_columns].dropna().copy()
    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset.groupby("tipo_de_contrato")["numero_de_lotes"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    save_csv(grouped, output_dirs["cross_tabs"] / "tipo_de_contrato_vs_numero_de_lotes.csv")

    if subset["tipo_de_contrato"].nunique(dropna=True) <= 1 or subset["numero_de_lotes"].nunique(dropna=True) <= 1:
        save_single_value_diagnostic("tipo_de_contrato_vs_numero_de_lotes", "alguna variable sin variacion suficiente", output_dirs)
        return grouped

    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, max(6, 0.35 * subset["tipo_de_contrato"].nunique())))
        sns.boxplot(data=subset, x="numero_de_lotes", y="tipo_de_contrato", ax=ax, color="#6d597a")
        ax.set_title("Numero de lotes vs tipo de contrato")
        ax.set_xlabel("numero_de_lotes")
        ax.set_ylabel("tipo_de_contrato")
        save_plot(fig, output_dirs["bivariate_plots"] / "tipo_de_contrato_vs_numero_de_lotes_box.png")
    return grouped


def analyze_temporal_vs_estado(df: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    required_columns = [
        column for column in ["estado_del_procedimiento", "dias_hasta_recepcion", "dias_hasta_apertura", "duracion_proceso_si_aplica"] if column in df.columns
    ]
    if len(required_columns) < 2:
        return pd.DataFrame()

    grouped = (
        df.groupby("estado_del_procedimiento", dropna=False)[["dias_hasta_recepcion", "dias_hasta_apertura", "duracion_proceso_si_aplica"]]
        .median(numeric_only=True)
        .reset_index()
    )
    save_csv(grouped, output_dirs["cross_tabs"] / "temporal_vs_estado_del_procedimiento.csv")

    melted = df[["estado_del_procedimiento", "dias_hasta_recepcion", "dias_hasta_apertura", "duracion_proceso_si_aplica"]].melt(
        id_vars="estado_del_procedimiento",
        var_name="metrica_temporal",
        value_name="dias",
    ).dropna()

    if melted.empty:
        return grouped

    if PLOTTING_AVAILABLE:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(data=melted, x="metrica_temporal", y="dias", hue="estado_del_procedimiento", ax=ax)
        ax.set_title("Variables temporales vs estado del proceso")
        ax.set_xlabel("Metrica temporal")
        ax.set_ylabel("Dias")
        ax.legend(loc="best", fontsize=8)
        save_plot(fig, output_dirs["temporal_plots"] / "variables_temporales_vs_estado.png")
    return grouped


@task(name="correlacion_eda_secop")
@timing_decorator
@validate_inputs
def build_correlation_outputs(df: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    try:
        numeric_candidates = [column for column in NUMERIC_COLUMNS + ["dias_hasta_recepcion", "dias_hasta_apertura", "duracion_proceso_si_aplica"] if column in df.columns]
        usable_columns = [column for column in numeric_candidates if df[column].nunique(dropna=True) > 1]

        if len(usable_columns) < 2:
            save_single_value_diagnostic("correlation_matrix", "menos de dos variables numericas con variacion", output_dirs)
            return pd.DataFrame()

        corr_df = df[usable_columns].corr(numeric_only=True)

        if PLOTTING_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Heatmap de correlacion")
            save_plot(fig, output_dirs["correlation_plots"] / "correlation_heatmap.png")

        return corr_df.reset_index().rename(columns={"index": "variable"})
    except Exception as e:
        log(f"Error al construir la correlacion: {e}")
        raise
    finally:
        log("Construccion de correlacion finalizada.")


def create_summary_payload(
    df: pd.DataFrame,
    variable_groups: dict[str, list[str]],
    null_profile: pd.DataFrame,
    duplicates_summary: pd.DataFrame,
    low_variability_profile: pd.DataFrame,
    temporal_anomalies: pd.DataFrame,
    conversion_summary: pd.DataFrame,
) -> dict[str, Any]:
    top_nulls = null_profile.head(10).to_dict(orient="records")
    constant_columns = low_variability_profile.loc[low_variability_profile["is_near_constant"], "column"].tolist()
    conversion_issues = conversion_summary.loc[conversion_summary["coerced_to_null"] > 0].to_dict(orient="records")

    findings = [
        f"El dataset tiene {len(df):,} filas y {len(df.columns):,} columnas despues de la preparacion analitica.",
        f"Se detectaron {int(duplicates_summary.loc[duplicates_summary['duplicate_type'] == 'exact_all_columns', 'duplicate_rows'].fillna(0).iloc[0])} duplicados exactos tras normalizar la URL." if not duplicates_summary.empty else "No se pudo calcular duplicados exactos.",
        "referencia_del_proceso no es una llave unica y debe tratarse como identificador de referencia.",
        "modalidad_de_contratacion es constante en este corte y no debe interpretarse como driver explicativo.",
        "Las variables de competencia muestran baja o nula variabilidad en este corte y requieren contexto antes de modelar.",
        f"Se encontraron {len(temporal_anomalies):,} registros con al menos una anomalia temporal.",
    ]

    return {
        "source_parquet": str(SOURCE_PARQUET),
        "output_dir": str(OUTPUT_DIR),
        "plotting_available": PLOTTING_AVAILABLE,
        "plotting_import_error": PLOTTING_IMPORT_ERROR,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "variable_groups": variable_groups,
        "top_null_columns": top_nulls,
        "near_constant_columns": constant_columns,
        "potential_conversion_issues": conversion_issues,
        "temporal_anomalies_count": int(len(temporal_anomalies)),
        "findings": findings,
    }


@flow(name="analisis_eda_secop_flow")
@timing_decorator
def main() -> None:
    try:
        configure_environment()
        output_dirs = ensure_directories(OUTPUT_DIR)

        raw_df = load_dataset(SOURCE_PARQUET)
        cleaned_df, conversion_summary = prepare_dataframe(raw_df)
        variable_groups = infer_variable_groups(cleaned_df)

        log("Guardando artefactos de perfilado inicial")
        dataset_overview = build_overview(cleaned_df)
        dtypes_df = build_dtypes(cleaned_df)
        null_profile = build_null_profile(cleaned_df)
        duplicates_summary = build_duplicates_summary(cleaned_df)
        low_variability_profile = build_low_variability_profile(cleaned_df)
        column_recommendations = build_column_recommendations(null_profile, low_variability_profile)

        save_csv(dataset_overview, OUTPUT_DIR / "dataset_overview.csv")
        save_csv(dtypes_df, OUTPUT_DIR / "dtypes.csv")
        save_csv(null_profile, OUTPUT_DIR / "null_profile.csv")
        save_csv(duplicates_summary, OUTPUT_DIR / "duplicates_summary.csv")
        save_csv(low_variability_profile, OUTPUT_DIR / "low_variability_columns.csv")
        save_csv(column_recommendations, OUTPUT_DIR / "column_recommendations.csv")
        save_csv(conversion_summary, OUTPUT_DIR / "conversion_summary.csv")
        save_json(variable_groups, OUTPUT_DIR / "variable_groups.json")

        log("Ejecutando analisis univariado")
        numeric_summary = run_numeric_univariate_analysis(cleaned_df, output_dirs)
        categorical_summary = run_categorical_univariate_analysis(cleaned_df, output_dirs)
        text_summary, with_text_df = run_text_analysis(cleaned_df, output_dirs)

        save_csv(numeric_summary, OUTPUT_DIR / "univariate_numeric_summary.csv")
        save_csv(categorical_summary, OUTPUT_DIR / "univariate_categorical_summary.csv")
        save_csv(text_summary, OUTPUT_DIR / "text_quality_summary.csv")

        log("Construyendo variables temporales e indicadores de transparencia")
        temporal_df, temporal_summary, temporal_anomalies = derive_temporal_features(with_text_df)
        row_level_indicators, entity_quality_summary = build_transparency_indicators(temporal_df)

        save_csv(temporal_summary, OUTPUT_DIR / "temporal_summary.csv")
        save_csv(temporal_anomalies, OUTPUT_DIR / "temporal_anomalies.csv")
        save_csv(row_level_indicators, OUTPUT_DIR / "transparency_quality_indicators.csv")
        save_csv(entity_quality_summary, OUTPUT_DIR / "entity_quality_summary.csv")

        log("Generando analisis bivariado y multivariado")
        plot_price_vs_category(temporal_df, "modalidad_de_contratacion", output_dirs)
        plot_price_vs_category(temporal_df, "tipo_de_contrato", output_dirs)

        depto_estado = build_crosstab(temporal_df, "departamento_entidad", "estado_resumen", output_dirs, top_n_index=15)
        plot_stacked_crosstab(
            depto_estado,
            "departamento_entidad",
            output_dirs["bivariate_plots"] / "departamento_entidad_vs_estado_resumen.png",
            "Departamento de entidad vs estado resumen",
        )

        analyze_modalidad_vs_competencia(temporal_df, output_dirs)
        analyze_tipo_vs_numero_lotes(temporal_df, output_dirs)
        analyze_temporal_vs_estado(temporal_df, output_dirs)

        correlation_matrix = build_correlation_outputs(temporal_df, output_dirs)
        save_csv(correlation_matrix, OUTPUT_DIR / "correlation_matrix.csv")

        summary_payload = create_summary_payload(
            temporal_df,
            variable_groups,
            null_profile,
            duplicates_summary,
            low_variability_profile,
            temporal_anomalies,
            conversion_summary,
        )
        save_json(summary_payload, OUTPUT_DIR / "summary.json")

        log("Analisis completado. Artefactos guardados en Data/Processed/ods16_secop/")
    except Exception as e:
        log(f"Error durante la ejecucion del flow de EDA: {e}")
        raise
    finally:
        log("Flow de EDA finalizado.")


if __name__ == "__main__":
    main()
