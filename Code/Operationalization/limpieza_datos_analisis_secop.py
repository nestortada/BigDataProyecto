from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import time
import unicodedata
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

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


RAW_MISSING_TOKENS = frozenset({"", "no definido", "n/a", "na", "null", "none"})
RAW_NOT_APPLICABLE_TOKENS = frozenset({"no aplica"})
RAW_NOT_YET_AVAILABLE_TOKENS = frozenset({"no adjudicado"})
RAW_UNKNOWN_TOKENS = frozenset({"no especificado"})
RAW_PLACEHOLDER_TOKENS = (
    RAW_MISSING_TOKENS | RAW_NOT_APPLICABLE_TOKENS | RAW_NOT_YET_AVAILABLE_TOKENS | RAW_UNKNOWN_TOKENS
)

DATE_COLUMNS = [
    "fecha_de_publicacion_fase",
    "fecha_de_publicacion_fase_2",
    "fecha_de_publicacion_fase_3",
    "fecha_de_publicacion",
    "fecha_de_publicacion_del",
    "fecha_de_ultima_publicaci",
    "fecha_de_recepcion_de",
    "fecha_de_apertura_de_respuesta",
    "fecha_de_apertura_efectiva",
    "fecha_adjudicacion",
]

NUMERIC_COLUMNS = [
    "precio_base",
    "duracion",
    "visualizaciones_del",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "proveedores_que_manifestaron",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_unicos_con",
    "numero_de_lotes",
]

TRACEABILITY_EXTERNAL_COLUMNS = [
    "id_del_proceso",
    "referencia_del_proceso",
    "id_del_portafolio",
    "urlproceso",
]

HIGH_CARDINALITY_TRACEABILITY_COLUMNS = [
    "codigo_entidad",
    "nit_entidad",
    "codigo_principal_de_categoria",
    "codigo_pci",
]

LEAKAGE_COLUMNS = [
    "estado_del_procedimiento",
    "id_estado_del_procedimiento",
    "estado_de_apertura_del_proceso",
    "estado_resumen",
    "fecha_adjudicacion",
    "fecha_de_ultima_publicaci",
    "fecha_de_apertura_efectiva",
]

LOW_VARIABILITY_COLUMNS = [
    "subtipo_de_contrato",
    "proveedores_que_manifestaron",
]

REDUNDANT_COLUMNS = [
    "ppi",
    "fecha_de_publicacion_fase_3",
]

ALMOST_EMPTY_COLUMNS = [
    "fecha_de_publicacion_fase",
    "fecha_de_publicacion_fase_2",
    "fecha_de_publicacion",
]

CRITICAL_COMPLETENESS_COLUMNS = [
    "entidad",
    "codigo_entidad",
    "fecha_de_publicacion_del",
    "nombre_del_procedimiento",
    "descripci_n_del_procedimiento",
    "precio_base",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "urlproceso",
]

COMPETITION_COLUMNS = [
    "proveedores_con_invitacion",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_unicos_con",
]

DURATION_UNIT_MAP = {
    "día(s)": "día(s)",
    "dia(s)": "día(s)",
    "dias": "día(s)",
    "día": "día(s)",
    "dia": "día(s)",
    "mes(es)": "Mes(es)",
    "mes": "Mes(es)",
    "meses": "Mes(es)",
    "año(s)": "Año(s)",
    "ano(s)": "Año(s)",
    "año": "Año(s)",
    "ano": "Año(s)",
    "años": "Año(s)",
    "anos": "Año(s)",
    "hora(s)": "Hora(s)",
    "hora": "Hora(s)",
    "horas": "Hora(s)",
    "semana(s)": "Semana(s)",
    "semana": "Semana(s)",
    "semanas": "Semana(s)",
}

DURATION_TO_DAYS = {
    "día(s)": 1.0,
    "Mes(es)": 30.0,
    "Año(s)": 365.0,
    "Hora(s)": 1.0 / 24.0,
    "Semana(s)": 7.0,
}


def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / ".env").exists() and (candidate / "Data").exists() and (candidate / "Code").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path.cwd())
load_dotenv(PROJECT_ROOT / ".env")

SOURCE_PARQUET = PROJECT_ROOT / "Data" / "Raw" / "datos_analisis.parquet"
OUTPUT_DIR = PROJECT_ROOT / "processed"
OUTPUT_PARQUET = OUTPUT_DIR / "datos_analisis_limpio.parquet"
OUTPUT_SUMMARY = OUTPUT_DIR / "datos_analisis_limpio_validacion.json"
TABLE_NAME = "datos_analisis_limpio"
POSTGRES_CONTAINER = os.getenv("POSTGRES_CONTAINER", "docker_postgres")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")) if os.getenv("POSTGRES_PORT") else None,
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def log(message: str) -> None:
    print(f"[LIMPIEZA SECOP] {message}")


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
                log(f"La tarea {func.__name__} tardo {elapsed_seconds:.2f} segundos.")

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

            if arg_name in {"source_path", "source_parquet"}:
                path_value = Path(arg_value)
                if not path_value.exists():
                    raise FileNotFoundError(f"No se encontro el archivo requerido: {path_value}")

            if arg_name == "df" and not isinstance(arg_value, pd.DataFrame):
                raise TypeError("'df' debe ser un pandas.DataFrame.")

            if arg_name == "db_config":
                required_keys = {"host", "port", "database", "user", "password"}
                if not isinstance(arg_value, dict):
                    raise TypeError("'db_config' debe ser un diccionario.")
                if not required_keys.issubset(arg_value.keys()):
                    raise ValueError("'db_config' debe incluir host, port, database, user y password.")
                if any(arg_value.get(key) in {None, ""} for key in required_keys):
                    raise ValueError("'db_config' contiene parametros vacios.")

            if arg_name == "table_name":
                if not isinstance(arg_value, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", arg_value):
                    raise ValueError("'table_name' debe ser un identificador SQL simple y valido.")

        return func(*args, **kwargs)

    return wrapper


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def normalize_text_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return value
    if pd.isna(value):
        return pd.NA
    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else pd.NA


def normalize_token(value: Any) -> str | None:
    cleaned = normalize_text_value(value)
    if pd.isna(cleaned):
        return None
    token = str(cleaned).lower()
    token = unicodedata.normalize("NFKD", token)
    token = token.encode("ascii", "ignore").decode("ascii")
    token = re.sub(r"\s+", " ", token).strip()
    return token or None


def normalize_multivalue_text(value: Any) -> Any:
    cleaned = normalize_text_value(value)
    if pd.isna(cleaned):
        return pd.NA
    text = str(cleaned)
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not pieces:
        return pd.NA
    stable_pieces = list(dict.fromkeys(pieces))
    return ", ".join(stable_pieces)


def flatten_url(value: Any) -> Any:
    if isinstance(value, dict):
        return normalize_text_value(value.get("url"))
    if pd.isna(value):
        return pd.NA
    return normalize_text_value(value)


def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, set):
        return json.dumps(sorted(value), ensure_ascii=False)
    return value


def postgres_type_for_series(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMPTZ"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    return "TEXT"


def docker_psql_base_command(db_config: dict, table_name: str | None = None) -> list[str]:
    _ = table_name
    return [
        "docker",
        "exec",
        "-i",
        "-e",
        f"PGPASSWORD={db_config['password']}",
        POSTGRES_CONTAINER,
        "psql",
        "-U",
        db_config["user"],
        "-d",
        db_config["database"],
        "-v",
        "ON_ERROR_STOP=1",
    ]


def run_docker_psql(db_config: dict, sql: str) -> subprocess.CompletedProcess:
    command = docker_psql_base_command(db_config) + ["-f", "-"]
    return subprocess.run(
        command,
        input=sql.encode("utf-8"),
        check=True,
        capture_output=True,
    )


def load_via_docker_psql(df: pd.DataFrame, db_config: dict, table_name: str) -> int:
    buffer = StringIO()
    df.to_csv(buffer, index=False, na_rep="\\N")
    buffer.seek(0)
    columns_sql = ", ".join(df.columns)
    copy_sql = f"\\copy {table_name} ({columns_sql}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '\\N')"
    command = docker_psql_base_command(db_config, table_name) + ["-c", copy_sql]
    subprocess.run(
        command,
        input=buffer.getvalue().encode("utf-8"),
        check=True,
        capture_output=True,
    )
    return int(len(df))


def validate_via_docker_psql(db_config: dict, table_name: str) -> int:
    command = docker_psql_base_command(db_config, table_name) + ["-t", "-A", "-c", f"SELECT COUNT(*) FROM {table_name};"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return int(completed.stdout.strip())


def canonicalize_text_series(
    series: pd.Series,
    preserve_not_applicable: bool = False,
    preserve_unknown: bool = False,
    missing_label: str | None = None,
) -> pd.Series:
    normalized = series.map(normalize_text_value).astype("string")
    tokens = normalized.map(normalize_token)
    result = normalized.copy()

    missing_mask = tokens.isin(RAW_MISSING_TOKENS | RAW_NOT_YET_AVAILABLE_TOKENS)
    result = result.mask(missing_mask, pd.NA if missing_label is None else missing_label)

    if preserve_not_applicable:
        result = result.mask(tokens.isin(RAW_NOT_APPLICABLE_TOKENS), "NO_APLICA")
    else:
        result = result.mask(tokens.isin(RAW_NOT_APPLICABLE_TOKENS), pd.NA if missing_label is None else missing_label)

    if preserve_unknown:
        result = result.mask(tokens.isin(RAW_UNKNOWN_TOKENS), "UNIDAD_DESCONOCIDA")
    else:
        result = result.mask(tokens.isin(RAW_UNKNOWN_TOKENS), pd.NA if missing_label is None else missing_label)

    return result.astype("string")


def summarize_placeholder_counts(df: pd.DataFrame) -> dict[str, int]:
    placeholder_counts: dict[str, int] = {}
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            tokens = df[column].astype("string").map(normalize_token)
            count = int(tokens.isin(RAW_PLACEHOLDER_TOKENS).sum())
            if count > 0:
                placeholder_counts[column] = count
    return dict(sorted(placeholder_counts.items(), key=lambda item: item[1], reverse=True))


def build_profile(df: pd.DataFrame) -> dict[str, Any]:
    hashable_df = df.copy()
    for column in hashable_df.columns:
        hashable_df[column] = hashable_df[column].map(make_hashable)

    real_null_pct = {column: round(float(value), 4) for column, value in (df.isna().mean() * 100).items()}
    expanded_null_pct: dict[str, float] = {}
    cardinality: dict[str, int] = {}
    constant_columns: list[str] = []
    near_constant_columns: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            tokens = series.astype("string").map(normalize_token)
            placeholder_mask = tokens.isin(RAW_PLACEHOLDER_TOKENS)
        else:
            placeholder_mask = pd.Series(False, index=series.index)

        expanded_null_pct[column] = round(float((series.isna() | placeholder_mask).mean() * 100), 4)

        hashable_series = hashable_df[column]
        nunique = int(hashable_series.nunique(dropna=False))
        cardinality[column] = nunique
        if nunique <= 1:
            constant_columns.append(column)

        value_counts = hashable_series.value_counts(dropna=False, normalize=True)
        if not value_counts.empty and float(value_counts.iloc[0]) >= 0.95:
            near_constant_columns.append(
                {
                    "column": column,
                    "dominant_share_pct": round(float(value_counts.iloc[0] * 100), 4),
                    "nunique_including_null": nunique,
                }
            )

    duplicate_rows_exact = int(hashable_df.duplicated().sum())
    duplicate_rows_by_process = int(hashable_df.duplicated(subset=["id_del_proceso"]).sum()) if "id_del_proceso" in df.columns else 0

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "real_null_pct": real_null_pct,
        "expanded_null_pct": expanded_null_pct,
        "placeholder_counts": summarize_placeholder_counts(df),
        "duplicate_rows_exact": duplicate_rows_exact,
        "duplicate_rows_by_id_del_proceso": duplicate_rows_by_process,
        "cardinality": cardinality,
        "constant_columns": sorted(constant_columns),
        "near_constant_columns": sorted(near_constant_columns, key=lambda item: item["dominant_share_pct"], reverse=True),
    }


def normalize_duration_unit(value: Any) -> Any:
    token = normalize_token(value)
    if token is None:
        return pd.NA
    if token in RAW_UNKNOWN_TOKENS:
        return "UNIDAD_DESCONOCIDA"
    if token in RAW_NOT_APPLICABLE_TOKENS:
        return "NO_APLICA"
    canonical = DURATION_UNIT_MAP.get(token)
    if canonical is not None:
        return canonical
    return "UNIDAD_DESCONOCIDA"


def compute_groupwise_price_threshold(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    min_group_size: int = 50,
    quantile: float = 0.995,
) -> pd.Series:
    if value_column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    if group_column not in df.columns:
        valid = df[[value_column]].copy()
        valid = valid.loc[valid[value_column].notna() & valid[value_column].ge(0)]
        global_threshold = float(valid[value_column].quantile(quantile)) if not valid.empty else np.nan
        return pd.Series(global_threshold, index=df.index, dtype="float64")

    valid = df[[value_column, group_column]].copy()
    valid = valid.loc[valid[value_column].notna() & valid[value_column].ge(0)]
    global_threshold = float(valid[value_column].quantile(quantile)) if not valid.empty else np.nan
    if valid.empty:
        return pd.Series(global_threshold, index=df.index, dtype="float64")

    grouped = valid.groupby(group_column)[value_column]
    group_sizes = grouped.size()
    group_thresholds = grouped.quantile(quantile)
    usable_groups = group_sizes[group_sizes >= min_group_size].index
    group_thresholds = group_thresholds.loc[group_thresholds.index.intersection(usable_groups)]

    thresholds = df[group_column].map(group_thresholds)
    thresholds = thresholds.fillna(global_threshold)
    return thresholds.astype("float64")


def classify_columns(existing_columns: list[str], final_columns: list[str], transformed_columns: set[str]) -> dict[str, list[str]]:
    existing_set = set(existing_columns)
    final_set = set(final_columns)

    leakage_present = sorted(column for column in LEAKAGE_COLUMNS if column in existing_set)
    low_variability_present = sorted(column for column in LOW_VARIABILITY_COLUMNS if column in existing_set)
    redundant_present = sorted(column for column in REDUNDANT_COLUMNS if column in existing_set)
    almost_empty_present = sorted(column for column in ALMOST_EMPTY_COLUMNS if column in existing_set)

    traceability_external = sorted(column for column in TRACEABILITY_EXTERNAL_COLUMNS if column in final_set)
    preserve_transformed = sorted(column for column in final_set if column in transformed_columns)
    preserve_direct = sorted(
        column
        for column in final_set
        if column not in transformed_columns and column not in traceability_external
    )

    removed_total = sorted(existing_set - final_set)
    generic_removed = sorted(
        column
        for column in removed_total
        if column not in set(leakage_present + low_variability_present + redundant_present + almost_empty_present)
    )

    return {
        "eliminar": generic_removed,
        "conservar": preserve_direct,
        "conservar_con_transformacion": preserve_transformed,
        "solo_trazabilidad_externa": traceability_external,
        "excluir_por_leakage": leakage_present,
        "excluir_por_constancia_o_baja_varianza": low_variability_present,
        "excluir_por_redundancia": redundant_present + almost_empty_present,
    }


def build_column_rules(existing_columns: list[str]) -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {column: [] for column in existing_columns}

    for column in existing_columns:
        if column in ALMOST_EMPTY_COLUMNS:
            rules[column].append("eliminada por cobertura casi nula")
        if column in LEAKAGE_COLUMNS:
            rules[column].append("excluida del dataset final por leakage o desenlace tardio")
        if column in LOW_VARIABILITY_COLUMNS:
            rules[column].append("excluida por constancia o variabilidad insuficiente")
        if column in REDUNDANT_COLUMNS:
            rules[column].append("excluida por redundancia")

    if "fecha_de_publicacion_del" in rules:
        rules["fecha_de_publicacion_del"].append("convertida a datetime y usada como fecha ancla")
    if "fecha_de_recepcion_de" in rules:
        rules["fecha_de_recepcion_de"].append("convertida a datetime y usada para dias_hasta_recepcion sin imputacion")
    if "fecha_de_apertura_de_respuesta" in rules:
        rules["fecha_de_apertura_de_respuesta"].append("convertida a datetime y preservada como hito temporal temprano")
    if "fecha_de_apertura_efectiva" in rules:
        rules["fecha_de_apertura_efectiva"].append("convertida a datetime solo para auditoria y derivacion de flags")
    if "precio_base" in rules:
        rules["precio_base"].append("convertida a numerico; negativos a NaN; cero preservado; derivadas precio_base_es_cero y precio_base_log")
    if "duracion" in rules:
        rules["duracion"].append("convertida a numerico y normalizada a duracion_dias")
    if "unidad_de_duracion" in rules:
        rules["unidad_de_duracion"].append("normalizada con mapeo explicito a dias; unidades desconocidas marcadas")
    if "descripci_n_del_procedimiento" in rules:
        rules["descripci_n_del_procedimiento"].append("placeholders tratados como ausencia semantica y derivada tiene_descripcion")
    if "categorias_adicionales" in rules:
        rules["categorias_adicionales"].append("texto estabilizado y derivada tiene_categorias_adicionales")
    if "ciudad_de_la_unidad_de" in rules:
        rules["ciudad_de_la_unidad_de"].append("No Aplica preservado como categoria explicita NO_APLICA")
    if "urlproceso" in rules:
        rules["urlproceso"].append("aplanada desde dict.url y preservada para trazabilidad externa")

    return {column: actions for column, actions in rules.items() if actions}


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working_df = df.copy()
    working_df["_row_order"] = np.arange(len(working_df), dtype=np.int64)

    existing_columns = [column for column in working_df.columns if column != "_row_order"]
    transformed_columns: set[str] = set()
    warning_messages: list[str] = []

    try:
        if "urlproceso" in working_df.columns:
            working_df["urlproceso"] = working_df["urlproceso"].map(flatten_url).astype("string")
            transformed_columns.add("urlproceso")

        for column in working_df.columns:
            if column in {"urlproceso", "_row_order"}:
                continue
            if pd.api.types.is_object_dtype(working_df[column]) or pd.api.types.is_string_dtype(working_df[column]):
                working_df[column] = working_df[column].map(normalize_text_value)

        for column in DATE_COLUMNS:
            if column in working_df.columns:
                working_df[column] = pd.to_datetime(working_df[column], errors="coerce", utc=True)
                transformed_columns.add(column)

        for column in NUMERIC_COLUMNS:
            if column in working_df.columns:
                working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
                transformed_columns.add(column)

        for column in [
            "entidad",
            "departamento_entidad",
            "ciudad_entidad",
            "ordenentidad",
            "codigo_pci",
            "id_del_proceso",
            "referencia_del_proceso",
            "id_del_portafolio",
            "nombre_del_procedimiento",
            "fase",
            "modalidad_de_contratacion",
            "justificaci_n_modalidad_de",
            "nombre_de_la_unidad_de",
            "tipo_de_contrato",
            "codigo_principal_de_categoria",
            "codigo_entidad",
            "nit_entidad",
        ]:
            if column in working_df.columns:
                working_df[column] = canonicalize_text_series(working_df[column])
                transformed_columns.add(column)

        if "descripci_n_del_procedimiento" in working_df.columns:
            working_df["descripci_n_del_procedimiento"] = canonicalize_text_series(working_df["descripci_n_del_procedimiento"])
            transformed_columns.add("descripci_n_del_procedimiento")

        if "departamento_entidad" in working_df.columns:
            working_df["departamento_entidad"] = canonicalize_text_series(working_df["departamento_entidad"])

        if "ciudad_entidad" in working_df.columns:
            working_df["ciudad_entidad"] = canonicalize_text_series(working_df["ciudad_entidad"])

        if "ciudad_de_la_unidad_de" in working_df.columns:
            working_df["ciudad_de_la_unidad_de"] = canonicalize_text_series(
                working_df["ciudad_de_la_unidad_de"], preserve_not_applicable=True
            )
            transformed_columns.add("ciudad_de_la_unidad_de")

        if "subtipo_de_contrato" in working_df.columns:
            working_df["subtipo_de_contrato"] = canonicalize_text_series(working_df["subtipo_de_contrato"])
            transformed_columns.add("subtipo_de_contrato")

        if "categorias_adicionales" in working_df.columns:
            categorized = canonicalize_text_series(working_df["categorias_adicionales"])
            working_df["categorias_adicionales"] = categorized.map(normalize_multivalue_text).astype("string")
            transformed_columns.add("categorias_adicionales")

        if "unidad_de_duracion" in working_df.columns:
            working_df["unidad_de_duracion"] = working_df["unidad_de_duracion"].map(normalize_duration_unit).astype("string")
            transformed_columns.add("unidad_de_duracion")
            unknown_duration_unit_mask = working_df["unidad_de_duracion"].eq("UNIDAD_DESCONOCIDA")
        else:
            unknown_duration_unit_mask = pd.Series(False, index=working_df.index)

        price_negative_count = 0
        if "precio_base" in working_df.columns:
            price_negative_mask = working_df["precio_base"].lt(0)
            price_negative_count = int(price_negative_mask.sum())
            working_df.loc[price_negative_mask, "precio_base"] = np.nan
            working_df["precio_base_es_cero"] = working_df["precio_base"].fillna(-1).eq(0)
            working_df["precio_base_log"] = np.where(
                working_df["precio_base"].notna() & working_df["precio_base"].ge(0),
                np.log1p(working_df["precio_base"]),
                np.nan,
            )
            price_thresholds = compute_groupwise_price_threshold(working_df, "precio_base", "tipo_de_contrato")
            working_df["flag_precio_base_extremo"] = (
                working_df["precio_base"].notna()
                & price_thresholds.notna()
                & working_df["precio_base"].gt(price_thresholds)
            )
            transformed_columns.update({"precio_base_es_cero", "precio_base_log", "flag_precio_base_extremo"})

        invalid_duration_count = 0
        if "duracion" in working_df.columns:
            invalid_duration_mask = working_df["duracion"].lt(0)
            invalid_duration_count = int(invalid_duration_mask.sum())
            working_df.loc[invalid_duration_mask, "duracion"] = np.nan

            duration_multiplier = working_df["unidad_de_duracion"].map(DURATION_TO_DAYS) if "unidad_de_duracion" in working_df.columns else pd.Series(np.nan, index=working_df.index)
            valid_duration_mask = working_df["duracion"].notna() & duration_multiplier.notna()
            working_df["duracion_dias"] = np.where(
                valid_duration_mask,
                working_df["duracion"] * duration_multiplier,
                np.nan,
            )

            positive_duration_days = working_df["duracion_dias"].dropna()
            duration_threshold_days = float(max(positive_duration_days.quantile(0.999), 3650.0)) if not positive_duration_days.empty else 3650.0
            working_df["duracion_improbable"] = working_df["duracion_dias"].gt(duration_threshold_days).fillna(False)
            working_df["unidad_de_duracion_desconocida"] = unknown_duration_unit_mask.fillna(False)
            transformed_columns.update({"duracion_dias", "duracion_improbable", "unidad_de_duracion_desconocida"})
        else:
            duration_threshold_days = None

        for column in [
            "visualizaciones_del",
            "proveedores_invitados",
            "proveedores_con_invitacion",
            "respuestas_al_procedimiento",
            "respuestas_externas",
            "conteo_de_respuestas_a_ofertas",
            "proveedores_unicos_con",
            "numero_de_lotes",
        ]:
            if column in working_df.columns:
                negative_mask = working_df[column].lt(0)
                if bool(negative_mask.any()):
                    working_df.loc[negative_mask, column] = np.nan

        working_df["tiene_descripcion"] = working_df["descripci_n_del_procedimiento"].notna() if "descripci_n_del_procedimiento" in working_df.columns else False
        working_df["tiene_urlproceso"] = working_df["urlproceso"].notna() if "urlproceso" in working_df.columns else False
        working_df["tiene_categorias_adicionales"] = working_df["categorias_adicionales"].notna() if "categorias_adicionales" in working_df.columns else False
        working_df["ausencia_geografica"] = False
        if "departamento_entidad" in working_df.columns or "ciudad_entidad" in working_df.columns:
            geo_components = []
            if "departamento_entidad" in working_df.columns:
                geo_components.append(working_df["departamento_entidad"].isna())
            if "ciudad_entidad" in working_df.columns:
                geo_components.append(working_df["ciudad_entidad"].isna())
            if geo_components:
                working_df["ausencia_geografica"] = pd.concat(geo_components, axis=1).any(axis=1)

        working_df["tiene_fecha_recepcion"] = working_df["fecha_de_recepcion_de"].notna() if "fecha_de_recepcion_de" in working_df.columns else False
        working_df["tiene_fecha_apertura_respuesta"] = (
            working_df["fecha_de_apertura_de_respuesta"].notna() if "fecha_de_apertura_de_respuesta" in working_df.columns else False
        )
        working_df["tiene_fecha_apertura_efectiva"] = (
            working_df["fecha_de_apertura_efectiva"].notna() if "fecha_de_apertura_efectiva" in working_df.columns else False
        )

        if {"fecha_de_publicacion_del", "fecha_de_recepcion_de"}.issubset(working_df.columns):
            delta_recepcion = (working_df["fecha_de_recepcion_de"] - working_df["fecha_de_publicacion_del"]).dt.days
            working_df["dias_hasta_recepcion"] = delta_recepcion.where(delta_recepcion.ge(0))
            flag_recepcion_antes_publicacion = delta_recepcion.lt(0).fillna(False)
        else:
            working_df["dias_hasta_recepcion"] = np.nan
            flag_recepcion_antes_publicacion = pd.Series(False, index=working_df.index)

        if {"fecha_de_recepcion_de", "fecha_de_apertura_de_respuesta"}.issubset(working_df.columns):
            delta_apertura_respuesta = (
                working_df["fecha_de_apertura_de_respuesta"] - working_df["fecha_de_recepcion_de"]
            ).dt.days
            flag_apertura_respuesta_antes_recepcion = delta_apertura_respuesta.lt(0).fillna(False)
        else:
            flag_apertura_respuesta_antes_recepcion = pd.Series(False, index=working_df.index)

        if {"fecha_de_publicacion_del", "fecha_de_apertura_de_respuesta"}.issubset(working_df.columns):
            delta_apertura_publicacion = (
                working_df["fecha_de_apertura_de_respuesta"] - working_df["fecha_de_publicacion_del"]
            ).dt.days
            flag_apertura_respuesta_antes_publicacion = delta_apertura_publicacion.lt(0).fillna(False)
        else:
            flag_apertura_respuesta_antes_publicacion = pd.Series(False, index=working_df.index)

        if {"fecha_de_recepcion_de", "fecha_de_apertura_efectiva"}.issubset(working_df.columns):
            delta_apertura_efectiva = (working_df["fecha_de_apertura_efectiva"] - working_df["fecha_de_recepcion_de"]).dt.days
            working_df["anomalia_apertura_antes_recepcion"] = delta_apertura_efectiva.lt(0).fillna(False)
        else:
            working_df["anomalia_apertura_antes_recepcion"] = False

        temporal_columns_for_missing = [
            column
            for column in [
                "fecha_de_publicacion_del",
                "fecha_de_recepcion_de",
                "fecha_de_apertura_de_respuesta",
                "fecha_de_apertura_efectiva",
            ]
            if column in working_df.columns
        ]
        working_df["missing_temporal_count"] = working_df[temporal_columns_for_missing].isna().sum(axis=1) if temporal_columns_for_missing else 0
        working_df["anomalia_temporal"] = (
            flag_recepcion_antes_publicacion
            | flag_apertura_respuesta_antes_recepcion
            | flag_apertura_respuesta_antes_publicacion
            | working_df["anomalia_apertura_antes_recepcion"].fillna(False)
        )

        available_competition_columns = [column for column in COMPETITION_COLUMNS if column in working_df.columns]
        if available_competition_columns:
            competition_base = working_df[available_competition_columns].fillna(0).clip(lower=0)
            working_df["competencia_reportada"] = competition_base.sum(axis=1).gt(0)
            working_df["intensidad_competencia_log"] = np.log1p(competition_base.sum(axis=1))
        else:
            working_df["competencia_reportada"] = False
            working_df["intensidad_competencia_log"] = 0.0

        if {"proveedores_unicos_con", "proveedores_invitados"}.issubset(working_df.columns):
            denominator_mask = working_df["proveedores_invitados"].gt(0)
            numerator = working_df["proveedores_unicos_con"]
            denominator = working_df["proveedores_invitados"]
            ratio = pd.Series(np.nan, index=working_df.index, dtype="float64")
            valid_ratio_mask = denominator_mask & numerator.notna()
            ratio.loc[valid_ratio_mask] = numerator.loc[valid_ratio_mask] / denominator.loc[valid_ratio_mask]
            working_df["ratio_proveedores_vs_invitados"] = ratio
        else:
            working_df["ratio_proveedores_vs_invitados"] = np.nan

        available_critical_columns = [column for column in CRITICAL_COMPLETENESS_COLUMNS if column in working_df.columns]
        if available_critical_columns:
            working_df["completitud_critica"] = working_df[available_critical_columns].notna().mean(axis=1).round(4)
        else:
            working_df["completitud_critica"] = np.nan

        transformed_columns.update(
            {
                "tiene_descripcion",
                "tiene_urlproceso",
                "tiene_categorias_adicionales",
                "ausencia_geografica",
                "tiene_fecha_recepcion",
                "tiene_fecha_apertura_respuesta",
                "tiene_fecha_apertura_efectiva",
                "dias_hasta_recepcion",
                "missing_temporal_count",
                "anomalia_apertura_antes_recepcion",
                "anomalia_temporal",
                "competencia_reportada",
                "intensidad_competencia_log",
                "ratio_proveedores_vs_invitados",
                "completitud_critica",
            }
        )

        hashable_for_duplicates = working_df.drop(columns=["_row_order"]).copy()
        for column in hashable_for_duplicates.columns:
            hashable_for_duplicates[column] = hashable_for_duplicates[column].map(make_hashable)

        exact_duplicate_any = hashable_for_duplicates.duplicated(keep=False)
        exact_duplicate_drop = hashable_for_duplicates.duplicated(keep="first")
        working_df["fue_duplicado_exacto"] = exact_duplicate_any
        exact_duplicates_removed = int(exact_duplicate_drop.sum())
        working_df = working_df.loc[~exact_duplicate_drop].copy()

        if "id_del_proceso" in working_df.columns:
            logical_duplicate_any = working_df.duplicated(subset=["id_del_proceso"], keep=False)
            working_df["fue_duplicado_logico"] = logical_duplicate_any

            candidate_columns = [
                column
                for column in working_df.columns
                if column
                not in {
                    "_row_order",
                    "completeness_score",
                    "fue_duplicado_exacto",
                    "fue_duplicado_logico",
                    *LEAKAGE_COLUMNS,
                    *LOW_VARIABILITY_COLUMNS,
                    *REDUNDANT_COLUMNS,
                    *ALMOST_EMPTY_COLUMNS,
                }
            ]
            working_df["completeness_score"] = working_df[candidate_columns].notna().sum(axis=1)

            sort_columns = ["id_del_proceso", "completeness_score", "fecha_de_ultima_publicaci", "_row_order"]
            ascending = [True, False, False, True]
            available_sort_columns = [column for column in sort_columns if column in working_df.columns]
            available_ascending = [ascending[idx] for idx, column in enumerate(sort_columns) if column in working_df.columns]
            working_df = working_df.sort_values(available_sort_columns, ascending=available_ascending, na_position="last")
            logical_duplicates_removed = int(working_df.duplicated(subset=["id_del_proceso"], keep="first").sum())
            working_df = working_df.drop_duplicates(subset=["id_del_proceso"], keep="first").copy()
        else:
            working_df["fue_duplicado_logico"] = False
            working_df["completeness_score"] = np.nan
            logical_duplicates_removed = 0

        drop_columns = [
            column
            for column in (
                LEAKAGE_COLUMNS
                + LOW_VARIABILITY_COLUMNS
                + REDUNDANT_COLUMNS
                + ALMOST_EMPTY_COLUMNS
                + ["completeness_score", "_row_order"]
            )
            if column in working_df.columns
        ]
        working_df = working_df.drop(columns=drop_columns)

        final_columns = list(working_df.columns)
        classification = classify_columns(existing_columns, final_columns, transformed_columns)

        if "fecha_de_ultima_publicaci" in existing_columns and "fecha_de_publicacion_del" in existing_columns:
            warning_messages.append(
                "fecha_de_ultima_publicaci es identica a fecha_de_publicacion_del en el dataset real y se excluyo por redundancia/leakage."
            )
        if "fecha_de_publicacion_fase_3" in existing_columns:
            warning_messages.append(
                "fecha_de_publicacion_fase_3 es casi redundante con la fecha ancla y se excluyo del dataset final."
            )
        if "subtipo_de_contrato" in existing_columns:
            warning_messages.append(
                "subtipo_de_contrato aparece completamente constante en datos_analisis.parquet y no aporta variacion modelable."
            )
        if "proveedores_que_manifestaron" in existing_columns:
            warning_messages.append(
                "proveedores_que_manifestaron es constante en el dataset real y se excluyo por baja varianza."
            )

        cleaning_summary = {
            "rows_initial": int(len(df)),
            "rows_after_exact_deduplication": int(len(df) - exact_duplicates_removed),
            "rows_final": int(len(working_df)),
            "columns_initial": int(len(existing_columns)),
            "columns_final": int(len(working_df.columns)),
            "exact_duplicates_removed": exact_duplicates_removed,
            "logical_duplicates_removed": logical_duplicates_removed,
            "negative_price_base_corrected": price_negative_count,
            "negative_duration_corrected": invalid_duration_count,
            "duracion_improbable_threshold_days": duration_threshold_days,
            "duracion_dias_null_due_to_invalid_or_unknown": int(
                (
                    working_df["duracion_dias"].isna()
                    & (
                        working_df["duracion"].isna()
                        | working_df["unidad_de_duracion_desconocida"].fillna(False)
                        | working_df["unidad_de_duracion"].eq("NO_APLICA")
                    )
                ).sum()
            )
            if {"duracion_dias", "duracion", "unidad_de_duracion_desconocida", "unidad_de_duracion"}.issubset(working_df.columns)
            else 0,
            "temporal_anomalies_final": int(working_df["anomalia_temporal"].fillna(False).sum()) if "anomalia_temporal" in working_df.columns else 0,
            "classification": classification,
            "column_rules": build_column_rules(existing_columns),
            "derived_columns_created": sorted(
                column
                for column in transformed_columns
                if column not in existing_columns and column in working_df.columns
            ),
            "warnings": warning_messages,
        }

        return working_df, cleaning_summary
    except Exception as exc:
        log(f"Error durante la limpieza del dataset: {exc}")
        raise
    finally:
        log("Fase de limpieza de dataset finalizada.")


def build_final_validation(
    clean_df: pd.DataFrame,
    initial_profile: dict[str, Any],
    cleaning_summary: dict[str, Any],
) -> dict[str, Any]:
    final_profile = build_profile(clean_df)
    remaining_placeholders = summarize_placeholder_counts(clean_df)

    validation_summary = {
        "initial_profile": initial_profile,
        "cleaning_summary": cleaning_summary,
        "final_profile": final_profile,
        "final_quality_checks": {
            "rows_initial_vs_final": {
                "initial": cleaning_summary["rows_initial"],
                "final": cleaning_summary["rows_final"],
            },
            "columns_initial_vs_final": {
                "initial": cleaning_summary["columns_initial"],
                "final": cleaning_summary["columns_final"],
            },
            "duplicate_rows_by_id_del_proceso_final": int(
                clean_df.duplicated(subset=["id_del_proceso"]).sum()
            )
            if "id_del_proceso" in clean_df.columns
            else 0,
            "columns_with_placeholder_semantic_tokens_remaining": remaining_placeholders,
            "constant_columns_remaining": final_profile["constant_columns"],
            "null_rate_final_by_column": final_profile["real_null_pct"],
            "temporal_anomalies_final": cleaning_summary["temporal_anomalies_final"],
            "negative_price_base_corrected": cleaning_summary["negative_price_base_corrected"],
            "duracion_dias_null_due_to_invalid_or_unknown": cleaning_summary["duracion_dias_null_due_to_invalid_or_unknown"],
        },
        "output_artifacts": {
            "output_parquet": str(OUTPUT_PARQUET),
            "output_summary": str(OUTPUT_SUMMARY),
            "postgres_table": f"public.{TABLE_NAME}",
        },
        "discrepancies_vs_documents": [
            "datos_analisis.parquet no contiene las columnas clasicas de adjudicacion del parquet crudo original (adjudicado, id_adjudicacion, valor_total_adjudicacion y proveedor adjudicado).",
            "fecha_de_ultima_publicaci es 100% igual a fecha_de_publicacion_del en el dataset real.",
            "fecha_de_publicacion_fase_3 coincide con la fecha ancla en aproximadamente 99.11% de las filas comparables.",
            "subtipo_de_contrato es totalmente constante y proveedores_que_manifestaron tambien resulta constante en el dataset real.",
        ],
    }
    return validation_summary


@task(name="crear_directorio_salida_datos_analisis_limpio")
@timing_decorator
def ensure_output_directory(output_dir: str = str(OUTPUT_DIR)) -> str:
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return str(output_path)
    except Exception as exc:
        log(f"Error al crear el directorio de salida: {exc}")
        raise
    finally:
        log("Provision del directorio de salida finalizada.")


@task(name="leer_datos_analisis_parquet")
@timing_decorator
@validate_inputs
def load_dataset(source_path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(source_path)
        log(f"Parquet cargado correctamente con shape {df.shape}")
        return df
    except Exception as exc:
        log(f"Error al leer el parquet de entrada: {exc}")
        raise
    finally:
        log("Lectura del parquet de entrada finalizada.")


@task(name="perfilar_datos_analisis_inicial")
@timing_decorator
@validate_inputs
def profile_initial_dataset(df: pd.DataFrame) -> dict[str, Any]:
    try:
        profile = build_profile(df)
        log(
            "Perfil inicial generado: "
            f"{profile['rows']:,} filas, {profile['columns']} columnas, "
            f"{profile['duplicate_rows_exact']:,} duplicados exactos."
        )
        return profile
    except Exception as exc:
        log(f"Error al perfilar el dataset inicial: {exc}")
        raise
    finally:
        log("Perfilado inicial finalizado.")


@task(name="limpiar_datos_analisis_secop")
@timing_decorator
@validate_inputs
def clean_analysis_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    return clean_dataset(df)


@task(name="validar_datos_analisis_limpio")
@timing_decorator
@validate_inputs
def validate_clean_dataset(
    df: pd.DataFrame,
    initial_profile: dict[str, Any],
    cleaning_summary: dict[str, Any],
) -> dict[str, Any]:
    try:
        summary = build_final_validation(df, initial_profile, cleaning_summary)
        log(
            "Validacion final generada: "
            f"{summary['final_profile']['rows']:,} filas finales y "
            f"{summary['final_profile']['columns']} columnas."
        )
        return summary
    except Exception as exc:
        log(f"Error al validar el dataset limpio: {exc}")
        raise
    finally:
        log("Validacion final de calidad terminada.")


@task(name="guardar_parquet_datos_analisis_limpio")
@timing_decorator
@validate_inputs
def save_clean_parquet(df: pd.DataFrame, output_path: str = str(OUTPUT_PARQUET)) -> str:
    try:
        parquet_path = Path(output_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        log(f"Parquet limpio guardado en {parquet_path}")
        return str(parquet_path)
    except Exception as exc:
        log(f"Error al guardar el parquet limpio: {exc}")
        raise
    finally:
        log("Escritura de parquet limpio finalizada.")


@task(name="guardar_resumen_validacion_datos_analisis_limpio")
@timing_decorator
def save_validation_summary(summary: dict[str, Any], output_path: str = str(OUTPUT_SUMMARY)) -> str:
    try:
        summary_path = Path(output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(json_safe(summary), file, ensure_ascii=False, indent=2)
        log(f"Resumen de validacion guardado en {summary_path}")
        return str(summary_path)
    except Exception as exc:
        log(f"Error al guardar el resumen de validacion: {exc}")
        raise
    finally:
        log("Escritura del resumen de validacion finalizada.")


@task(name="crear_tabla_postgres_datos_analisis_limpio")
@timing_decorator
@validate_inputs
def create_table(df: pd.DataFrame, db_config: dict, table_name: str) -> None:
    conn = None
    cur = None
    try:
        columns_sql = [f'"{column}" {postgres_type_for_series(df[column])}' for column in df.columns]
        create_sql = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            {", ".join(columns_sql)}
        );
        """
        try:
            import psycopg2

            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute(create_sql)
            conn.commit()
        except ModuleNotFoundError:
            run_docker_psql(db_config, create_sql)
        log(f"Tabla '{table_name}' creada correctamente en PostgreSQL.")
    except Exception as exc:
        log(f"Error al crear la tabla '{table_name}': {exc}")
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        log(f"Finalizo la creacion de la tabla '{table_name}'.")


@task(name="cargar_postgres_datos_analisis_limpio")
@timing_decorator
@validate_inputs
def load_data_to_postgres(df: pd.DataFrame, db_config: dict, table_name: str) -> int:
    conn = None
    cur = None
    try:
        try:
            import psycopg2

            buffer = StringIO()
            df.to_csv(buffer, index=False, na_rep="\\N")
            buffer.seek(0)

            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            copy_sql = (
                f"COPY {table_name} ({', '.join(df.columns)}) "
                "FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '\\N')"
            )
            cur.copy_expert(copy_sql, buffer)
            conn.commit()
            loaded_rows = int(len(df))
        except ModuleNotFoundError:
            loaded_rows = load_via_docker_psql(df, db_config, table_name)
        log(f"Se cargaron {loaded_rows:,} filas en '{table_name}'.")
        return loaded_rows
    except Exception as exc:
        log(f"Error al cargar datos en PostgreSQL: {exc}")
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        log(f"Finalizo la carga de datos en la tabla '{table_name}'.")


@task(name="validar_carga_postgres_datos_analisis_limpio")
@timing_decorator
@validate_inputs
def validate_load(db_config: dict, table_name: str) -> int:
    conn = None
    cur = None
    try:
        try:
            import psycopg2

            conn = psycopg2.connect(**db_config)
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name};")
            total_rows = int(cur.fetchone()[0])
        except ModuleNotFoundError:
            total_rows = validate_via_docker_psql(db_config, table_name)
        log(f"Validacion exitosa. Total de filas en '{table_name}': {total_rows:,}")
        return total_rows
    except Exception as exc:
        log(f"Error al validar la tabla '{table_name}': {exc}")
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
        log(f"Finalizo la validacion de la tabla '{table_name}'.")


@flow(name="limpieza_datos_analisis_secop_flow")
@timing_decorator
@validate_inputs
def clean_analysis_data_flow(
    source_parquet: str = str(SOURCE_PARQUET),
    output_parquet: str = str(OUTPUT_PARQUET),
    output_summary: str = str(OUTPUT_SUMMARY),
    table_name: str = TABLE_NAME,
) -> dict[str, Any]:
    try:
        ensure_output_directory(str(Path(output_parquet).parent))
        raw_df = load_dataset(source_parquet)
        initial_profile = profile_initial_dataset(raw_df)
        clean_df, cleaning_summary = clean_analysis_dataset(raw_df)
        validation_summary = validate_clean_dataset(clean_df, initial_profile, cleaning_summary)
        save_clean_parquet(clean_df, output_parquet)
        save_validation_summary(validation_summary, output_summary)
        create_table(clean_df, DB_CONFIG, table_name)
        loaded_rows = load_data_to_postgres(clean_df, DB_CONFIG, table_name)
        validated_rows = validate_load(DB_CONFIG, table_name)
        validation_summary["postgres"] = {
            "table_name": f"public.{table_name}",
            "loaded_rows": loaded_rows,
            "validated_rows": validated_rows,
        }
        save_validation_summary(validation_summary, output_summary)
        return validation_summary
    except Exception as exc:
        log(f"El flujo de limpieza de datos de analisis fallo: {exc}")
        raise
    finally:
        log("Ejecucion del flujo de limpieza finalizada.")


@timing_decorator
@validate_inputs
def run_without_prefect(
    source_parquet: str = str(SOURCE_PARQUET),
    output_parquet: str = str(OUTPUT_PARQUET),
    output_summary: str = str(OUTPUT_SUMMARY),
    table_name: str = TABLE_NAME,
) -> dict[str, Any]:
    try:
        ensure_output_directory.fn(str(Path(output_parquet).parent)) if hasattr(ensure_output_directory, "fn") else ensure_output_directory(str(Path(output_parquet).parent))
        raw_df = load_dataset.fn(source_parquet) if hasattr(load_dataset, "fn") else load_dataset(source_parquet)
        initial_profile = profile_initial_dataset.fn(raw_df) if hasattr(profile_initial_dataset, "fn") else profile_initial_dataset(raw_df)
        clean_df, cleaning_summary = clean_analysis_dataset.fn(raw_df) if hasattr(clean_analysis_dataset, "fn") else clean_analysis_dataset(raw_df)
        validation_summary = (
            validate_clean_dataset.fn(clean_df, initial_profile, cleaning_summary)
            if hasattr(validate_clean_dataset, "fn")
            else validate_clean_dataset(clean_df, initial_profile, cleaning_summary)
        )
        if hasattr(save_clean_parquet, "fn"):
            save_clean_parquet.fn(clean_df, output_parquet)
        else:
            save_clean_parquet(clean_df, output_parquet)
        if hasattr(save_validation_summary, "fn"):
            save_validation_summary.fn(validation_summary, output_summary)
        else:
            save_validation_summary(validation_summary, output_summary)
        if hasattr(create_table, "fn"):
            create_table.fn(clean_df, DB_CONFIG, table_name)
        else:
            create_table(clean_df, DB_CONFIG, table_name)
        if hasattr(load_data_to_postgres, "fn"):
            loaded_rows = load_data_to_postgres.fn(clean_df, DB_CONFIG, table_name)
        else:
            loaded_rows = load_data_to_postgres(clean_df, DB_CONFIG, table_name)
        if hasattr(validate_load, "fn"):
            validated_rows = validate_load.fn(DB_CONFIG, table_name)
        else:
            validated_rows = validate_load(DB_CONFIG, table_name)
        validation_summary["postgres"] = {
            "table_name": f"public.{table_name}",
            "loaded_rows": loaded_rows,
            "validated_rows": validated_rows,
        }
        if hasattr(save_validation_summary, "fn"):
            save_validation_summary.fn(validation_summary, output_summary)
        else:
            save_validation_summary(validation_summary, output_summary)
        return validation_summary
    except Exception as exc:
        log(f"La ejecucion directa del flujo fallo: {exc}")
        raise
    finally:
        log("Ejecucion directa del flujo de limpieza finalizada.")


def main() -> None:
    try:
        summary = run_without_prefect()
        final_rows = summary.get("final_profile", {}).get("rows")
        log(f"Proceso finalizado con {final_rows:,} filas limpias." if final_rows is not None else "Proceso finalizado.")
    except Exception as exc:
        log(f"No fue posible completar la limpieza y carga: {exc}")
        raise
    finally:
        log("Script finalizado.")


if __name__ == "__main__":
    main()
