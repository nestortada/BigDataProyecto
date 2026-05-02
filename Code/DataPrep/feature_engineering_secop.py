from __future__ import annotations

import argparse
import inspect
import math
import os
import re
import time
import unicodedata
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

os.environ.setdefault("PREFECT_HOME", str(Path("/tmp") / "prefect"))

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback para entornos minimos

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


try:
    from prefect import flow, task
    from prefect.logging import get_run_logger

    PREFECT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback para ejecutar sin Prefect
    PREFECT_AVAILABLE = False

    class _PrintLogger:
        def info(self, message: str, *args: Any) -> None:
            print(message % args if args else message)

        def warning(self, message: str, *args: Any) -> None:
            print(message % args if args else message)

        def error(self, message: str, *args: Any) -> None:
            print(message % args if args else message)

    def get_run_logger() -> _PrintLogger:
        return _PrintLogger()

    def task(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[..., Any]:
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def flow(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[..., Any]:
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1 and not decorator_kwargs:
            return decorator_args[0]

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


PROJECT_MARKERS = {".env", "Code", "Data"}
DEFAULT_INPUT_PARQUET = Path("Data/Processed/Limpieza/datos_analisis_limpio.parquet")
DEFAULT_OUTPUT_PARQUET = Path("Data/Processed/Limpieza/datos_feature_engineering.parquet")
DEFAULT_REPORTS_DIR = Path("Reports/FeatureEngineering")
DEFAULT_TABLE_NAME = "secop_feature_engineering"

TARGET_THRESHOLD_LEGACY = 0.65
TARGET_V2_SCORE_CUTOFF = 0.58
TARGET_V2_MIN_EVIDENCE_COVERAGE = 0.75
TARGET_V2_MIN_WEAK_DIMENSIONS = 2

THRESHOLD_SENSITIVITY_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
GROUP_AUDIT_COLUMNS = [
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "ordenentidad",
    "departamento_entidad",
]
WARNING_CATEGORICAL_COLUMNS = [
    "modalidad_de_contratacion",
    "justificaci_n_modalidad_de",
    "tipo_de_contrato",
    "ordenentidad",
    "departamento_entidad",
    "entidad",
]

MIN_REQUIRED_COLUMNS = {
    "id_del_proceso",
    "referencia_del_proceso",
    "id_del_portafolio",
    "descripci_n_del_procedimiento",
    "codigo_principal_de_categoria",
    "precio_base",
    "departamento_entidad",
    "ciudad_entidad",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "urlproceso",
}

DATETIME_COLUMNS = [
    "fecha_de_publicacion_del",
    "fecha_de_recepcion_de",
    "fecha_de_apertura_de_respuesta",
]
COMPETITION_COLUMNS = [
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_con_invitacion",
    "proveedores_unicos_con",
]
NUMERIC_COLUMNS = [
    "precio_base",
    "proveedores_invitados",
    *COMPETITION_COLUMNS,
]
BOOLEAN_COLUMNS = [
    "anomalia_temporal",
    "fue_duplicado_exacto",
    "fue_duplicado_logico",
]

COMPLETENESS_WEIGHTS = {
    "flag_tiene_descripcion_util": 0.30,
    "flag_tiene_categoria": 0.20,
    "flag_tiene_precio_base": 0.20,
    "flag_tiene_ubicacion_entidad": 0.15,
    "flag_tiene_tipo_modalidad": 0.15,
}
TRACEABILITY_WEIGHTS = {
    "flag_id_proceso_valido": 0.35,
    "flag_referencia_valida": 0.25,
    "flag_tiene_url_publica": 0.25,
    "flag_portafolio_disponible": 0.15,
}
TEMPORAL_WEIGHTS = {
    "flag_fechas_temporales_disponibles": 0.25,
    "flag_publicacion_antes_recepcion": 0.30,
    "flag_recepcion_antes_apertura": 0.20,
    "flag_ventana_recepcion_razonable": 0.15,
    "flag_ventana_apertura_razonable": 0.10,
}
COMPETITION_WEIGHTS = {
    "flag_datos_competencia_disponibles": 0.20,
    "flag_hubo_participacion": 0.30,
    "flag_hubo_competencia_minima": 0.35,
    "intensidad_competencia_normalizada": 0.15,
}
TRANSPARENCY_SCORE_WEIGHTS = {
    "score_completitud": 0.30,
    "score_trazabilidad": 0.25,
    "score_temporal": 0.25,
    "score_competencia": 0.20,
}

TEXT_PLACEHOLDER_PATTERN = re.compile(
    r"^(?:|n/?a|na|nan|null|none|sin\s+informacion|"
    r"sin\s+definir|no\s+aplica|no\s+definid[oa]|no\s+registrado)$",
    re.IGNORECASE,
)
PROCESS_ID_PATTERN = re.compile(r"^CO\d+\.[A-Z0-9_]+\.[A-Z0-9._-]+$", re.IGNORECASE)
PUBLIC_URL_PATTERN = re.compile(r"^https?://.*secop.*", re.IGNORECASE)
VALID_TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_logger() -> Any:
    try:
        return get_run_logger()
    except Exception:
        return None


def log_message(message: str, level: str = "info") -> None:
    logger = get_logger()
    if logger is None:
        print(message)
        return
    log_method = getattr(logger, level, logger.info)
    log_method(message)


def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            log_message(f"La tarea {func.__name__} tardo {elapsed_seconds:.2f} segundos.")

    return wrapper


def validate_inputs(func: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_arguments = signature.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()
        for arg_name, arg_value in bound_arguments.arguments.items():
            if arg_name == "df" and not isinstance(arg_value, pd.DataFrame):
                raise TypeError("'df' debe ser un pandas.DataFrame.")
            if arg_name in {"parquet_path", "output_path", "reports_dir"}:
                if not str(arg_value).strip():
                    raise ValueError(f"El parametro '{arg_name}' no puede estar vacio.")
            if arg_name == "parquet_path" and not resolve_project_path(arg_value).exists():
                raise FileNotFoundError(f"No se encontro el parquet: {resolve_project_path(arg_value)}")
            if arg_name == "table_name" and not VALID_TABLE_NAME_PATTERN.fullmatch(str(arg_value)):
                raise ValueError("'table_name' debe ser un identificador SQL simple y valido.")
        return func(*args, **kwargs)

    return wrapper


def build_db_config() -> dict[str, Any]:
    port_value = os.getenv("POSTGRES_PORT")
    try:
        port = int(port_value) if port_value else None
    except ValueError:
        port = None
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "port": port,
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }


def db_config_is_complete(db_config: dict[str, Any]) -> bool:
    return all(db_config.get(key) not in {None, ""} for key in ["host", "port", "database", "user", "password"])


def extract_scalar_value(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ["url", "href", "value"]:
            if key in value:
                return value[key]
    return value


def get_series(df: pd.DataFrame, column_name: str, default: Any = pd.NA) -> pd.Series:
    if column_name in df.columns:
        return df[column_name]
    return pd.Series(default, index=df.index)


def normalize_text_series(series: pd.Series) -> pd.Series:
    normalized = series.map(extract_scalar_value).astype("string").fillna("").str.strip().str.lower()
    return normalized.map(
        lambda value: unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    )


def non_placeholder_mask(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    # Usar re de Python evita diferencias del backend string[pyarrow] en Windows.
    is_placeholder = normalized.map(lambda value: bool(TEXT_PLACEHOLDER_PATTERN.fullmatch(str(value))))
    return normalized.ne("") & ~is_placeholder


def regex_match_mask(series: pd.Series, pattern: re.Pattern[str]) -> pd.Series:
    normalized = normalize_text_series(series)
    return normalized.map(lambda value: bool(pattern.match(str(value))))


def bool_to_int8(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype("int8")


def clip_series(series: pd.Series | np.ndarray, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    if isinstance(series, pd.Series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(lower)
        return pd.Series(np.clip(numeric, lower, upper), index=series.index, dtype="float64")
    numeric = pd.Series(series, dtype="float64").fillna(lower)
    return pd.Series(np.clip(numeric, lower, upper), dtype="float64")


def coerce_datetime_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[column_name], errors="coerce", utc=True)


def coerce_numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column_name], errors="coerce")


def coerce_bool_series(series: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(int(default)).ne(0)
    normalized = normalize_text_series(series)
    true_values = {"true", "1", "si", "s", "yes", "y", "verdadero"}
    false_values = {"false", "0", "no", "n", "falso", ""}
    result = normalized.isin(true_values)
    placeholder = normalized.map(lambda value: bool(TEXT_PLACEHOLDER_PATTERN.fullmatch(str(value))))
    known_false = normalized.isin(false_values) | placeholder
    return result.where(result | known_false, default).astype(bool)


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype="float64")
    for column_name, weight in weights.items():
        if column_name not in df.columns:
            raise KeyError(f"No existe la columna requerida para score: {column_name}")
        score = score + pd.to_numeric(df[column_name], errors="coerce").fillna(0.0) * weight
    return clip_series(score)


def postgres_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMPTZ"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    return "TEXT"


def create_postgres_table(df: pd.DataFrame, db_config: dict[str, Any], table_name: str) -> None:
    import psycopg2

    connection = None
    cursor = None
    try:
        columns_sql = [f'"{column}" {postgres_type_for_series(df[column])}' for column in df.columns]
        create_sql = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            {", ".join(columns_sql)}
        );
        """
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute(create_sql)
        connection.commit()
    except Exception:
        if connection is not None:
            connection.rollback()
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


@task(name="extract_data")
@timing_decorator
@validate_inputs
def extract_data(parquet_path: str | Path) -> pd.DataFrame:
    input_path = resolve_project_path(parquet_path)
    df = pd.read_parquet(input_path)
    log_message(f"Archivo leido correctamente: {input_path}")
    log_message(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
    return df


@task(name="validate_data")
@timing_decorator
@validate_inputs
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(MIN_REQUIRED_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Faltan columnas minimas requeridas: {missing_columns}")

    validated_df = df.copy()
    for column_name in DATETIME_COLUMNS:
        if column_name in validated_df.columns:
            validated_df[column_name] = coerce_datetime_series(validated_df, column_name)
    for column_name in NUMERIC_COLUMNS:
        if column_name in validated_df.columns:
            validated_df[column_name] = coerce_numeric_series(validated_df, column_name)
    for column_name in BOOLEAN_COLUMNS:
        if column_name in validated_df.columns:
            validated_df[column_name] = coerce_bool_series(validated_df[column_name])

    log_message("Validacion de columnas y tipos completada.")
    return validated_df


@task(name="feature_engineering")
@timing_decorator
@validate_inputs
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    engineered_df = df.copy()

    descripcion = get_series(engineered_df, "descripci_n_del_procedimiento")
    descripcion_text = descripcion.astype("string").fillna("").str.strip()
    descripcion_len = descripcion_text.str.len().fillna(0).astype("int64")
    engineered_df["longitud_descripcion"] = descripcion_len
    engineered_df["flag_tiene_descripcion_util"] = bool_to_int8(non_placeholder_mask(descripcion) & descripcion_len.ge(20))
    engineered_df["flag_tiene_categoria"] = bool_to_int8(
        non_placeholder_mask(get_series(engineered_df, "codigo_principal_de_categoria"))
    )
    engineered_df["flag_tiene_precio_base"] = bool_to_int8(coerce_numeric_series(engineered_df, "precio_base").gt(0))
    engineered_df["flag_tiene_ubicacion_entidad"] = bool_to_int8(
        non_placeholder_mask(get_series(engineered_df, "departamento_entidad"))
        & non_placeholder_mask(get_series(engineered_df, "ciudad_entidad"))
    )
    engineered_df["flag_tiene_tipo_modalidad"] = bool_to_int8(
        non_placeholder_mask(get_series(engineered_df, "modalidad_de_contratacion"))
        & non_placeholder_mask(get_series(engineered_df, "tipo_de_contrato"))
    )
    completeness_flags = list(COMPLETENESS_WEIGHTS.keys())
    engineered_df["missing_required_fields_count"] = (
        len(completeness_flags) - engineered_df[completeness_flags].sum(axis=1)
    ).astype("int8")

    process_id = get_series(engineered_df, "id_del_proceso")
    engineered_df["flag_id_proceso_valido"] = bool_to_int8(regex_match_mask(process_id, PROCESS_ID_PATTERN))
    referencia = get_series(engineered_df, "referencia_del_proceso")
    engineered_df["flag_referencia_valida"] = bool_to_int8(
        non_placeholder_mask(referencia) & referencia.astype("string").fillna("").str.strip().str.len().ge(5)
    )
    url = get_series(engineered_df, "urlproceso")
    engineered_df["flag_tiene_url_publica"] = bool_to_int8(regex_match_mask(url, PUBLIC_URL_PATTERN))
    duplicate_mask = pd.Series(False, index=engineered_df.index)
    for duplicate_column in ["fue_duplicado_exacto", "fue_duplicado_logico"]:
        duplicate_mask = duplicate_mask | coerce_bool_series(get_series(engineered_df, duplicate_column, False))
    engineered_df["penalizacion_duplicado"] = bool_to_int8(duplicate_mask)
    engineered_df["flag_portafolio_disponible"] = bool_to_int8(
        non_placeholder_mask(get_series(engineered_df, "id_del_portafolio"))
    )

    publicacion = coerce_datetime_series(engineered_df, "fecha_de_publicacion_del")
    recepcion = coerce_datetime_series(engineered_df, "fecha_de_recepcion_de")
    apertura = coerce_datetime_series(engineered_df, "fecha_de_apertura_de_respuesta")
    engineered_df["dias_publicacion_a_recepcion"] = (recepcion - publicacion).dt.total_seconds().div(86400.0)
    engineered_df["dias_recepcion_a_apertura"] = (apertura - recepcion).dt.total_seconds().div(86400.0)
    engineered_df["dias_publicacion_a_apertura"] = (apertura - publicacion).dt.total_seconds().div(86400.0)
    engineered_df["flag_tiene_fecha_publicacion"] = bool_to_int8(publicacion.notna())
    engineered_df["flag_tiene_fecha_recepcion"] = bool_to_int8(recepcion.notna())
    engineered_df["flag_tiene_fecha_apertura_respuesta"] = bool_to_int8(apertura.notna())
    engineered_df["flag_fechas_temporales_disponibles"] = bool_to_int8(
        publicacion.notna() & recepcion.notna() & apertura.notna()
    )
    engineered_df["flag_publicacion_antes_recepcion"] = bool_to_int8(
        engineered_df["dias_publicacion_a_recepcion"].ge(0).fillna(False)
    )
    engineered_df["flag_recepcion_antes_apertura"] = bool_to_int8(
        engineered_df["dias_recepcion_a_apertura"].ge(0).fillna(False)
    )
    engineered_df["flag_ventana_recepcion_razonable"] = bool_to_int8(
        engineered_df["dias_publicacion_a_recepcion"].between(0, 90, inclusive="both").fillna(False)
    )
    engineered_df["flag_ventana_apertura_razonable"] = bool_to_int8(
        engineered_df["dias_recepcion_a_apertura"].between(0, 30, inclusive="both").fillna(False)
    )
    temporal_anomaly_source = coerce_bool_series(get_series(engineered_df, "anomalia_temporal", False))
    temporal_incoherence = (
        (publicacion.notna() & recepcion.notna() & engineered_df["dias_publicacion_a_recepcion"].lt(0))
        | (recepcion.notna() & apertura.notna() & engineered_df["dias_recepcion_a_apertura"].lt(0))
        | temporal_anomaly_source
    )
    engineered_df["flag_incoherencia_temporal_observada"] = bool_to_int8(temporal_incoherence)
    engineered_df["flag_coherencia_temporal_global"] = bool_to_int8(
        engineered_df["flag_fechas_temporales_disponibles"].eq(1)
        & engineered_df["flag_publicacion_antes_recepcion"].eq(1)
        & engineered_df["flag_recepcion_antes_apertura"].eq(1)
        & engineered_df["flag_ventana_recepcion_razonable"].eq(1)
        & engineered_df["flag_ventana_apertura_razonable"].eq(1)
        & engineered_df["flag_incoherencia_temporal_observada"].eq(0)
    )

    respuestas_al_procedimiento = coerce_numeric_series(engineered_df, "respuestas_al_procedimiento")
    respuestas_externas = coerce_numeric_series(engineered_df, "respuestas_externas")
    conteo_ofertas = coerce_numeric_series(engineered_df, "conteo_de_respuestas_a_ofertas")
    proveedores_con_invitacion = coerce_numeric_series(engineered_df, "proveedores_con_invitacion")
    proveedores_unicos = coerce_numeric_series(engineered_df, "proveedores_unicos_con")

    competition_observed = pd.DataFrame(
        {column_name: coerce_numeric_series(engineered_df, column_name).notna() for column_name in COMPETITION_COLUMNS},
        index=engineered_df.index,
    ).any(axis=1)
    total_respuestas = (
        respuestas_al_procedimiento.fillna(0)
        + respuestas_externas.fillna(0)
        + conteo_ofertas.fillna(0)
    )
    total_interes = proveedores_con_invitacion.fillna(0) + proveedores_unicos.fillna(0)
    engineered_df["total_respuestas"] = total_respuestas.astype("float64")
    engineered_df["total_interes_oferentes"] = total_interes.astype("float64")
    engineered_df["flag_datos_competencia_disponibles"] = bool_to_int8(competition_observed)
    engineered_df["flag_hubo_participacion"] = bool_to_int8(
        competition_observed & (engineered_df["total_respuestas"].gt(0) | engineered_df["total_interes_oferentes"].gt(0))
    )
    engineered_df["flag_hubo_competencia_minima"] = bool_to_int8(
        competition_observed
        & (
            proveedores_unicos.fillna(0).ge(2)
            | respuestas_al_procedimiento.fillna(0).ge(2)
            | conteo_ofertas.fillna(0).ge(2)
        )
    )
    engineered_df["flag_sin_competencia_observada"] = bool_to_int8(
        competition_observed & engineered_df["flag_hubo_participacion"].eq(0)
    )
    competition_signal = total_respuestas + proveedores_unicos.fillna(0)
    engineered_df["intensidad_competencia_normalizada"] = clip_series(
        np.log1p(competition_signal.clip(lower=0)) / math.log1p(10)
    )

    log_message("Feature engineering completado.")
    return engineered_df


@task(name="build_scores")
@timing_decorator
@validate_inputs
def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored_df = df.copy()
    scored_df["score_completitud"] = weighted_score(scored_df, COMPLETENESS_WEIGHTS)
    scored_df["score_trazabilidad_base"] = weighted_score(scored_df, TRACEABILITY_WEIGHTS)
    scored_df["score_trazabilidad"] = clip_series(
        scored_df["score_trazabilidad_base"] - scored_df["penalizacion_duplicado"].astype(float) * 0.35
    )
    scored_df["score_temporal"] = weighted_score(scored_df, TEMPORAL_WEIGHTS)
    scored_df["score_competencia"] = weighted_score(scored_df, COMPETITION_WEIGHTS)
    scored_df["transparency_score"] = weighted_score(scored_df, TRANSPARENCY_SCORE_WEIGHTS)
    log_message("Scores de transparencia construidos.")
    return scored_df


@task(name="build_target")
@timing_decorator
@validate_inputs
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    target_df = df.copy()
    score = pd.to_numeric(target_df["transparency_score"], errors="coerce").fillna(0.0)

    target_df["riesgo_baja_transparencia"] = score.lt(TARGET_THRESHOLD_LEGACY).astype("int8")
    target_df["nivel_riesgo_transparencia"] = np.select(
        [score.lt(0.50), score.lt(TARGET_THRESHOLD_LEGACY)],
        ["alto", "medio"],
        default="bajo",
    )

    # Las dimensiones base son observables porque sus columnas minimas son obligatorias.
    target_df["flag_evidencia_completitud_suficiente"] = np.int8(1)
    target_df["flag_evidencia_trazabilidad_suficiente"] = np.int8(1)

    date_presence_count = (
        target_df["flag_tiene_fecha_publicacion"].astype(int)
        + target_df["flag_tiene_fecha_recepcion"].astype(int)
        + target_df["flag_tiene_fecha_apertura_respuesta"].astype(int)
    )
    target_df["flag_evidencia_temporal_suficiente"] = bool_to_int8(date_presence_count.ge(2))
    target_df["flag_evidencia_competencia_suficiente"] = target_df["flag_datos_competencia_disponibles"].astype("int8")

    evidence_flags = [
        "flag_evidencia_completitud_suficiente",
        "flag_evidencia_trazabilidad_suficiente",
        "flag_evidencia_temporal_suficiente",
        "flag_evidencia_competencia_suficiente",
    ]
    target_df["target_dimension_count_v2"] = target_df[evidence_flags].sum(axis=1).astype("int8")
    target_df["evidence_coverage_v2"] = clip_series(target_df["target_dimension_count_v2"].astype(float) / 4.0)

    low_completeness = target_df["score_completitud"].lt(0.60)
    low_traceability = target_df["score_trazabilidad"].lt(0.60) | target_df["penalizacion_duplicado"].eq(1)
    low_temporal = (
        target_df["flag_evidencia_temporal_suficiente"].eq(1)
        & (target_df["score_temporal"].lt(0.55) | target_df["flag_incoherencia_temporal_observada"].eq(1))
    )
    low_competition = (
        target_df["flag_evidencia_competencia_suficiente"].eq(1)
        & target_df["score_competencia"].lt(0.50)
    )

    weak_dimension_count = (
        low_completeness.astype(int)
        + low_traceability.astype(int)
        + low_temporal.astype(int)
        + low_competition.astype(int)
    )
    non_temporal_weak_count = (
        low_completeness.astype(int) + low_traceability.astype(int) + low_competition.astype(int)
    )
    sufficient_evidence = target_df["evidence_coverage_v2"].ge(TARGET_V2_MIN_EVIDENCE_COVERAGE)
    enough_distance = score.le(TARGET_V2_SCORE_CUTOFF)
    consistent_low_signals = weak_dimension_count.ge(TARGET_V2_MIN_WEAK_DIMENSIONS)

    target_df["riesgo_baja_transparencia_v2"] = (
        sufficient_evidence
        & enough_distance
        & consistent_low_signals
        & non_temporal_weak_count.ge(1)
    ).astype("int8")

    dimensions_observed = target_df["target_dimension_count_v2"].replace(0, np.nan).astype(float)
    weak_ratio = weak_dimension_count.astype(float) / dimensions_observed
    signal_consistency = ((weak_ratio - 0.5).abs() * 2).fillna(0.0)
    distance_to_threshold = score.sub(TARGET_THRESHOLD_LEGACY).abs()

    high_confidence = (
        target_df["evidence_coverage_v2"].ge(0.75)
        & target_df["target_dimension_count_v2"].ge(3)
        & distance_to_threshold.ge(0.10)
        & signal_consistency.ge(0.50)
    )
    medium_confidence = (
        target_df["evidence_coverage_v2"].ge(0.50)
        & target_df["target_dimension_count_v2"].ge(2)
        & distance_to_threshold.ge(0.05)
        & signal_consistency.ge(0.25)
    )
    target_df["target_confidence_v2"] = np.select(
        [high_confidence, medium_confidence],
        ["alta", "media"],
        default="baja",
    )
    target_df.loc[target_df["evidence_coverage_v2"].lt(0.50), "target_confidence_v2"] = "baja"

    log_message("Targets legacy y v2 construidos.")
    return target_df


def distribution_table(series: pd.Series, name: str) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).sort_index()
    total = max(int(counts.sum()), 1)
    return pd.DataFrame(
        {
            "variable": name,
            "valor": counts.index.astype(str),
            "conteo": counts.values.astype(int),
            "porcentaje": (counts.values / total * 100).round(4),
        }
    )


def build_threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    score = pd.to_numeric(df["transparency_score"], errors="coerce")
    for threshold in THRESHOLD_SENSITIVITY_VALUES:
        positives = int(score.lt(threshold).sum())
        negatives = int(score.ge(threshold).sum())
        rows.append(
            {
                "threshold": threshold,
                "positivos": positives,
                "negativos": negatives,
                "porcentaje_positivo": round(positives / total * 100, 4),
                "porcentaje_negativo": round(negatives / total * 100, 4),
            }
        )
    return pd.DataFrame(rows)


def build_group_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit_frames: list[pd.DataFrame] = []
    target = "riesgo_baja_transparencia_v2"
    for column_name in GROUP_AUDIT_COLUMNS:
        if column_name not in df.columns:
            continue
        group_values = df[column_name].astype("string").fillna("Sin informacion").replace("", "Sin informacion")
        grouped = (
            pd.DataFrame({"group_value": group_values, target: df[target].astype(int)})
            .groupby("group_value", dropna=False)[target]
            .agg(total="count", positivos="sum")
            .reset_index()
        )
        grouped["group_variable"] = column_name
        grouped["negativos"] = grouped["total"] - grouped["positivos"]
        grouped["tasa_positiva"] = (grouped["positivos"] / grouped["total"]).round(6)
        audit_frames.append(
            grouped[
                ["group_variable", "group_value", "total", "positivos", "negativos", "tasa_positiva"]
            ]
        )
    if not audit_frames:
        return pd.DataFrame(
            columns=["group_variable", "group_value", "total", "positivos", "negativos", "tasa_positiva"]
        )
    return pd.concat(audit_frames, ignore_index=True).sort_values(
        ["group_variable", "tasa_positiva", "total"],
        ascending=[True, False, False],
    )


def build_warning_report(df: pd.DataFrame) -> str:
    lines = [
        "ADVERTENCIAS DE DOMINANCIA DEL TARGET V2",
        "=" * 70,
        "",
        "Criterios:",
        "- Diferencia extrema: spread de tasas positivas >= 0.65 entre categorias con soporte suficiente.",
        "- Dominancia positiva: una categoria concentra >= 50% de todos los positivos.",
        "",
    ]
    target = df["riesgo_baja_transparencia_v2"].astype(int)
    total_rows = max(len(df), 1)
    total_positives = int(target.sum())
    global_rate = float(total_positives / total_rows)
    min_support = max(30, int(total_rows * 0.001))
    warning_count = 0

    lines.append(f"Filas evaluadas: {len(df):,}")
    lines.append(f"Positivos v2: {total_positives:,} ({global_rate:.4%})")
    lines.append(f"Soporte minimo para spread: {min_support:,} filas")
    lines.append("")

    for column_name in WARNING_CATEGORICAL_COLUMNS:
        if column_name not in df.columns:
            lines.append(f"- {column_name}: columna no disponible; se omite.")
            continue
        values = df[column_name].astype("string").fillna("Sin informacion").replace("", "Sin informacion")
        stats = (
            pd.DataFrame({"value": values, "target": target})
            .groupby("value", dropna=False)["target"]
            .agg(total="count", positivos="sum")
            .reset_index()
        )
        stats["tasa_positiva"] = stats["positivos"] / stats["total"]
        stats["share_positivos"] = stats["positivos"] / max(total_positives, 1)
        supported = stats[stats["total"].ge(min_support)].copy()
        spread = float(supported["tasa_positiva"].max() - supported["tasa_positiva"].min()) if not supported.empty else 0.0
        dominant = stats.sort_values("share_positivos", ascending=False).head(1)

        lines.append(f"- {column_name}: categorias={len(stats):,}, spread_soportado={spread:.4f}")
        if spread >= 0.65:
            warning_count += 1
            max_row = supported.sort_values("tasa_positiva", ascending=False).iloc[0]
            min_row = supported.sort_values("tasa_positiva", ascending=True).iloc[0]
            lines.append(
                "  ADVERTENCIA spread extremo: "
                f"max='{max_row['value']}' ({max_row['tasa_positiva']:.4%}) vs "
                f"min='{min_row['value']}' ({min_row['tasa_positiva']:.4%})."
            )
        if not dominant.empty:
            dominant_row = dominant.iloc[0]
            if float(dominant_row["share_positivos"]) >= 0.50 and int(dominant_row["positivos"]) > 0:
                warning_count += 1
                lines.append(
                    "  ADVERTENCIA dominancia positiva: "
                    f"'{dominant_row['value']}' concentra {dominant_row['share_positivos']:.4%} "
                    f"de positivos con tasa {dominant_row['tasa_positiva']:.4%}."
                )
    if warning_count == 0:
        lines.append("No se detectaron advertencias automaticas bajo los umbrales definidos.")
    return "\n".join(lines)


def build_label_quality_report(df: pd.DataFrame) -> str:
    lines = [
        "REPORTE DE CALIDAD DEL TARGET V2",
        "=" * 70,
        "",
        f"Filas: {len(df):,}",
        f"Columnas: {len(df.columns):,}",
        "",
        "Distribucion target legacy:",
        distribution_table(df["riesgo_baja_transparencia"], "riesgo_baja_transparencia").to_string(index=False),
        "",
        "Distribucion target v2:",
        distribution_table(df["riesgo_baja_transparencia_v2"], "riesgo_baja_transparencia_v2").to_string(index=False),
        "",
        "Confianza target v2:",
        distribution_table(df["target_confidence_v2"], "target_confidence_v2").to_string(index=False),
        "",
        "Cobertura de evidencia v2:",
        distribution_table(df["evidence_coverage_v2"], "evidence_coverage_v2").to_string(index=False),
        "",
        "Dimensiones observadas v2:",
        distribution_table(df["target_dimension_count_v2"], "target_dimension_count_v2").to_string(index=False),
        "",
        "Cruce target legacy vs target v2:",
        pd.crosstab(
            df["riesgo_baja_transparencia"],
            df["riesgo_baja_transparencia_v2"],
            rownames=["legacy"],
            colnames=["v2"],
            dropna=False,
        ).to_string(),
        "",
    ]
    low_evidence = df["evidence_coverage_v2"].lt(0.50)
    low_evidence_positive = int((low_evidence & df["riesgo_baja_transparencia_v2"].eq(1)).sum())
    lines.append(f"Casos con evidencia baja (<0.50): {int(low_evidence.sum()):,}")
    lines.append(f"Positivos v2 con evidencia baja: {low_evidence_positive:,}")
    return "\n".join(lines)


def build_summary_report(
    df: pd.DataFrame,
    input_path: Path,
    output_path: Path,
    reports_dir: Path,
    postgres_status: dict[str, Any],
) -> str:
    generated_features = [
        "longitud_descripcion",
        "flag_tiene_descripcion_util",
        "flag_tiene_categoria",
        "flag_tiene_precio_base",
        "flag_tiene_ubicacion_entidad",
        "flag_tiene_tipo_modalidad",
        "missing_required_fields_count",
        "flag_id_proceso_valido",
        "flag_referencia_valida",
        "flag_tiene_url_publica",
        "penalizacion_duplicado",
        "flag_portafolio_disponible",
        "dias_publicacion_a_recepcion",
        "dias_recepcion_a_apertura",
        "dias_publicacion_a_apertura",
        "total_respuestas",
        "total_interes_oferentes",
        "intensidad_competencia_normalizada",
    ]
    scores = [
        "score_completitud",
        "score_trazabilidad_base",
        "score_trazabilidad",
        "score_temporal",
        "score_competencia",
        "transparency_score",
    ]
    targets = [
        "riesgo_baja_transparencia",
        "nivel_riesgo_transparencia",
        "riesgo_baja_transparencia_v2",
        "target_confidence_v2",
        "evidence_coverage_v2",
        "target_dimension_count_v2",
    ]
    lines = [
        "RESUMEN FEATURE ENGINEERING SECOP II",
        "=" * 70,
        "",
        f"Input parquet: {input_path}",
        f"Output parquet: {output_path}",
        f"Reports dir: {reports_dir}",
        f"Prefect disponible: {PREFECT_AVAILABLE}",
        "",
        f"Filas procesadas: {len(df):,}",
        f"Columnas finales: {len(df.columns):,}",
        f"Positivos legacy: {int(df['riesgo_baja_transparencia'].sum()):,}",
        f"Positivos v2: {int(df['riesgo_baja_transparencia_v2'].sum()):,}",
        "",
        "Features principales generados:",
        *[f"- {column}" for column in generated_features if column in df.columns],
        "",
        "Scores generados:",
        *[f"- {column}" for column in scores if column in df.columns],
        "",
        "Targets y metadatos de target generados:",
        *[f"- {column}" for column in targets if column in df.columns],
        "",
        "Reportes generados:",
        "- target_threshold_sensitivity.csv",
        "- target_v2_group_audit.csv",
        "- target_v2_warning_report.txt",
        "- target_v2_label_quality_report.txt",
        "- feature_engineering_summary.txt",
        "",
        "Estado PostgreSQL:",
        f"- status: {postgres_status.get('status')}",
        f"- rows_uploaded: {postgres_status.get('rows_uploaded', 0)}",
        f"- table_name: {postgres_status.get('table_name', '')}",
        f"- detail: {postgres_status.get('detail', '')}",
    ]
    return "\n".join(lines)


@task(name="write_quality_reports")
@timing_decorator
@validate_inputs
def write_quality_reports(df: pd.DataFrame, reports_dir: str | Path) -> dict[str, str]:
    reports_path = resolve_project_path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    threshold_path = reports_path / "target_threshold_sensitivity.csv"
    group_audit_path = reports_path / "target_v2_group_audit.csv"
    warning_path = reports_path / "target_v2_warning_report.txt"
    quality_path = reports_path / "target_v2_label_quality_report.txt"

    build_threshold_sensitivity(df).to_csv(threshold_path, index=False)
    build_group_audit(df).to_csv(group_audit_path, index=False)
    warning_path.write_text(build_warning_report(df), encoding="utf-8")
    quality_path.write_text(build_label_quality_report(df), encoding="utf-8")

    log_message(f"Reportes de target escritos en: {reports_path}")
    return {
        "target_threshold_sensitivity": str(threshold_path),
        "target_v2_group_audit": str(group_audit_path),
        "target_v2_warning_report": str(warning_path),
        "target_v2_label_quality_report": str(quality_path),
    }


@task(name="write_summary_report")
@timing_decorator
@validate_inputs
def write_summary_report(
    df: pd.DataFrame,
    input_path: str | Path,
    output_path: str | Path,
    reports_dir: str | Path,
    postgres_status: dict[str, Any],
) -> Path:
    reports_path = resolve_project_path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    summary_path = reports_path / "feature_engineering_summary.txt"
    summary = build_summary_report(
        df=df,
        input_path=resolve_project_path(input_path),
        output_path=resolve_project_path(output_path),
        reports_dir=reports_path,
        postgres_status=postgres_status,
    )
    summary_path.write_text(summary, encoding="utf-8")
    log_message(f"Resumen de feature engineering escrito en: {summary_path}")
    return summary_path


@task(name="save_parquet")
@timing_decorator
@validate_inputs
def save_parquet(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_file = resolve_project_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    log_message(f"Parquet final guardado en: {output_file}")
    return output_file


@task(name="upload_postgres")
@timing_decorator
@validate_inputs
def upload_postgres(df: pd.DataFrame, db_config: dict[str, Any], table_name: str) -> dict[str, Any]:
    if not db_config_is_complete(db_config):
        return {
            "status": "skipped_missing_config",
            "rows_uploaded": 0,
            "table_name": table_name,
            "detail": "Configuracion PostgreSQL incompleta en .env.",
        }
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        return {
            "status": "skipped_missing_dependency",
            "rows_uploaded": 0,
            "table_name": table_name,
            "detail": "psycopg2 no esta instalado.",
        }

    import psycopg2

    upload_df = df.copy()
    for column_name in upload_df.columns:
        if pd.api.types.is_datetime64_any_dtype(upload_df[column_name]):
            upload_df[column_name] = upload_df[column_name].astype("string")

    connection = None
    cursor = None
    try:
        create_postgres_table(upload_df, db_config, table_name)
        buffer = StringIO()
        upload_df.to_csv(buffer, index=False, na_rep="\\N")
        buffer.seek(0)

        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        quoted_columns = ", ".join(f'"{column}"' for column in upload_df.columns)
        copy_sql = (
            f"COPY {table_name} ({quoted_columns}) "
            "FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '\\N')"
        )
        cursor.copy_expert(copy_sql, buffer)
        connection.commit()
        rows_uploaded = int(len(upload_df))
        log_message(f"Se cargaron {rows_uploaded:,} filas en PostgreSQL: {table_name}")
        return {
            "status": "uploaded",
            "rows_uploaded": rows_uploaded,
            "table_name": table_name,
            "detail": "Carga completada.",
        }
    except Exception as exc:
        if connection is not None:
            connection.rollback()
        log_message(f"Error durante la carga a PostgreSQL: {exc}", level="error")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def call_pipeline_step(step: Callable[..., Any], *args: Any, use_prefect_tasks: bool, **kwargs: Any) -> Any:
    if use_prefect_tasks:
        return step(*args, **kwargs)
    raw_step = getattr(step, "fn", step)
    return raw_step(*args, **kwargs)


def run_feature_engineering_pipeline(
    parquet_path: str | Path,
    output_path: str | Path,
    table_name: str,
    skip_postgres: bool,
    reports_dir: str | Path,
    use_prefect_tasks: bool = False,
) -> dict[str, Any]:
    df = call_pipeline_step(extract_data, parquet_path, use_prefect_tasks=use_prefect_tasks)
    df = call_pipeline_step(validate_data, df, use_prefect_tasks=use_prefect_tasks)
    df = call_pipeline_step(feature_engineering, df, use_prefect_tasks=use_prefect_tasks)
    df = call_pipeline_step(build_scores, df, use_prefect_tasks=use_prefect_tasks)
    df = call_pipeline_step(build_target, df, use_prefect_tasks=use_prefect_tasks)
    call_pipeline_step(write_quality_reports, df, reports_dir, use_prefect_tasks=use_prefect_tasks)
    saved_path = call_pipeline_step(save_parquet, df, output_path, use_prefect_tasks=use_prefect_tasks)

    if skip_postgres:
        postgres_status = {
            "status": "skipped_by_cli",
            "rows_uploaded": 0,
            "table_name": table_name,
            "detail": "Carga omitida por --skip-postgres.",
        }
        log_message("Carga a PostgreSQL omitida por CLI.")
    else:
        postgres_status = call_pipeline_step(
            upload_postgres,
            df,
            build_db_config(),
            table_name,
            use_prefect_tasks=use_prefect_tasks,
        )

    summary_path = call_pipeline_step(
        write_summary_report,
        df,
        parquet_path,
        saved_path,
        reports_dir,
        postgres_status,
        use_prefect_tasks=use_prefect_tasks,
    )
    result = {
        "rows_processed": int(len(df)),
        "columns_final": int(len(df.columns)),
        "output_path": str(saved_path),
        "reports_dir": str(resolve_project_path(reports_dir)),
        "summary_report": str(summary_path),
        "postgres_status": postgres_status,
        "prefect_available": PREFECT_AVAILABLE,
    }
    log_message(f"Flujo completado correctamente: {result}")
    return result


@flow(name="secop_feature_engineering_flow", log_prints=True)
def secop_feature_engineering_flow(
    parquet_path: str | Path = PROJECT_ROOT / DEFAULT_INPUT_PARQUET,
    output_path: str | Path = PROJECT_ROOT / DEFAULT_OUTPUT_PARQUET,
    table_name: str = DEFAULT_TABLE_NAME,
    skip_postgres: bool = False,
    reports_dir: str | Path = PROJECT_ROOT / DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    return run_feature_engineering_pipeline(
        parquet_path=parquet_path,
        output_path=output_path,
        table_name=table_name,
        skip_postgres=skip_postgres,
        reports_dir=reports_dir,
        use_prefect_tasks=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de feature engineering SECOP II con target de transparencia v2."
    )
    parser.add_argument(
        "--input-path",
        default=str(PROJECT_ROOT / DEFAULT_INPUT_PARQUET),
        help="Ruta del parquet limpio de entrada.",
    )
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / DEFAULT_OUTPUT_PARQUET),
        help="Ruta del parquet final con features, scores y targets.",
    )
    parser.add_argument(
        "--table-name",
        default=DEFAULT_TABLE_NAME,
        help="Nombre de tabla destino en PostgreSQL.",
    )
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Guarda parquet y reportes sin cargar a PostgreSQL.",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(PROJECT_ROOT / DEFAULT_REPORTS_DIR),
        help="Directorio donde se guardan los reportes de feature engineering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # El CLI ejecuta los mismos pasos de forma directa para funcionar en
    # entornos locales/sandbox donde Prefect no puede iniciar su servidor efimero.
    # El flow decorado queda disponible para orquestacion Prefect real.
    run_feature_engineering_pipeline(
        parquet_path=args.input_path,
        output_path=args.output_path,
        table_name=args.table_name,
        skip_postgres=args.skip_postgres,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
