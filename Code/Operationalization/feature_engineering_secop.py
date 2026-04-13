from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import re
import subprocess
import sys
import time
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_DIR = BOOTSTRAP_ROOT / ".python_packages"

if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

DEPENDENCY_MAP = {
    "prefect": "prefect",
    "psycopg2": "psycopg2-binary",
}

PROJECT_MARKERS = {".env", "Code", "processed"}
INPUT_PARQUET = Path("processed/datos_analisis_limpio.parquet")
OUTPUT_PARQUET = Path("processed/datos_feature_engineering.parquet")
OUTPUT_TABLE = "secop_feature_engineering"

REQUIRED_COLUMNS = {
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
    "fecha_de_publicacion_del",
    "fecha_de_recepcion_de",
    "fecha_de_apertura_de_respuesta",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_con_invitacion",
    "proveedores_unicos_con",
}

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
    r"^(?:|n/?a|na|null|none|sin\s+informacion|sin\s+definir|no\s+aplica|no\s+definid[oa]|no\s+registrado)$"
)
PROCESS_ID_PATTERN = re.compile(r"^CO\d+\.[A-Z0-9_]+\.[A-Z0-9._-]+$", re.IGNORECASE)
PUBLIC_URL_PATTERN = re.compile(
    r"^https?://community\.secop\.gov\.co/Public/Tendering/OpportunityDetail/Index\?noticeUID=.+$",
    re.IGNORECASE,
)


def install_missing_dependency(module_name: str, package_name: str) -> bool:
    try:
        LOCAL_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--target", str(LOCAL_PACKAGE_DIR), package_name],
            check=True,
            capture_output=True,
            text=True,
        )
        if str(LOCAL_PACKAGE_DIR) not in sys.path:
            sys.path.insert(0, str(LOCAL_PACKAGE_DIR))
        importlib.invalidate_caches()
        return True
    except Exception:
        print(
            f"No fue posible instalar automaticamente la dependencia '{package_name}' "
            f"para el modulo '{module_name}' en {LOCAL_PACKAGE_DIR}."
        )
        return False


def ensure_runtime_dependencies() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for module_name, package_name in DEPENDENCY_MAP.items():
        try:
            __import__(module_name)
            status[module_name] = True
        except ImportError:
            status[module_name] = install_missing_dependency(module_name, package_name)
    return status


DEPENDENCY_STATUS = ensure_runtime_dependencies()

try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

    def task(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def flow(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def get_run_logger() -> Any:
        raise RuntimeError("Prefect no esta disponible en el entorno actual.")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path.cwd())
load_dotenv(PROJECT_ROOT / ".env")


def get_logger() -> Any | None:
    try:
        return get_run_logger()
    except Exception:
        return None


def log_message(message: str, level: str = "info") -> None:
    logger = get_logger()
    if logger is not None:
        log_method = getattr(logger, level, logger.info)
        log_method(message)
    else:
        print(message)


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


def validate(func: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_arguments = signature.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()

        for arg_name, arg_value in bound_arguments.arguments.items():
            if arg_name in {"parquet_path", "output_path"}:
                path_value = Path(arg_value)
                if arg_name == "parquet_path" and not path_value.exists():
                    raise FileNotFoundError(f"No se encontro el archivo parquet: {path_value}")

            if arg_name == "df" and not isinstance(arg_value, pd.DataFrame):
                raise TypeError("'df' debe ser un pandas.DataFrame.")

            if arg_name == "table_name":
                if not isinstance(arg_value, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", arg_value):
                    raise ValueError("'table_name' debe ser un identificador SQL simple y valido.")

            if arg_name == "db_config" and arg_value is not None:
                required_keys = {"host", "port", "database", "user", "password"}
                if not isinstance(arg_value, dict):
                    raise TypeError("'db_config' debe ser un diccionario.")
                missing_keys = required_keys.difference(arg_value.keys())
                if missing_keys:
                    raise ValueError(
                        f"'db_config' debe incluir host, port, database, user y password. Faltan: {sorted(missing_keys)}"
                    )

            if isinstance(arg_value, str) and arg_name not in {"table_name"} and not arg_value.strip():
                raise ValueError(f"El parametro '{arg_name}' no puede ser una cadena vacia.")

        return func(*args, **kwargs)

    return wrapper


def build_db_config() -> dict[str, Any]:
    port_value = os.getenv("POSTGRES_PORT")
    return {
        "host": os.getenv("POSTGRES_HOST"),
        "port": int(port_value) if port_value else None,
        "database": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }


def db_config_is_complete(db_config: dict[str, Any]) -> bool:
    return all(db_config.get(key) not in {None, ""} for key in ["host", "port", "database", "user", "password"])


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip().str.lower()


def non_placeholder_mask(series: pd.Series) -> pd.Series:
    normalized = normalize_text_series(series)
    return normalized.ne("") & ~normalized.str.fullmatch(TEXT_PLACEHOLDER_PATTERN.pattern, na=False)


def clip_series(series: pd.Series, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    return pd.Series(np.clip(series.astype(float), lower, upper), index=series.index, dtype="float64")


def bool_to_int8(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype("int8")


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=df.index, dtype="float64")
    for column_name, weight in weights.items():
        score = score.add(df[column_name].astype(float) * weight, fill_value=0.0)
    return clip_series(score)


def coerce_datetime_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[column_name], errors="coerce", utc=True)


def coerce_numeric_column(df: pd.DataFrame, column_name: str, fill_value: float = 0.0) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(fill_value, index=df.index, dtype="float64")
    numeric_series = pd.to_numeric(df[column_name], errors="coerce")
    return numeric_series.fillna(fill_value)


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


def create_postgres_table(df: pd.DataFrame, db_config: dict[str, Any], table_name: str) -> None:
    psycopg2 = __import__("psycopg2")
    connection = None
    cursor = None
    try:
        columns_sql = [f'"{column}" {postgres_type_for_series(df[column])}' for column in df.columns]
        ddl_sql = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            {", ".join(columns_sql)}
        );
        """
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute(ddl_sql)
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
@validate
def extract_data(parquet_path: str) -> pd.DataFrame:
    try:
        parquet_file = Path(parquet_path)
        df = pd.read_parquet(parquet_file, engine="pyarrow")
        log_message(f"Archivo leido correctamente: {parquet_file}")
        log_message(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
        return df
    except Exception as e:
        log_message(f"Error al leer el archivo parquet: {e}", level="error")
        raise
    finally:
        log_message("Finalizo la lectura del archivo parquet.")


@task(name="validate_data")
@timing_decorator
@validate
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        missing_columns = sorted(REQUIRED_COLUMNS.difference(df.columns))
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas en el dataset: {missing_columns}")

        validated_df = df.copy()
        datetime_columns = [
            "fecha_de_publicacion_del",
            "fecha_de_recepcion_de",
            "fecha_de_apertura_de_respuesta",
        ]
        numeric_columns = [
            "precio_base",
            "respuestas_al_procedimiento",
            "respuestas_externas",
            "conteo_de_respuestas_a_ofertas",
            "proveedores_con_invitacion",
            "proveedores_unicos_con",
        ]
        bool_columns = ["anomalia_temporal", "fue_duplicado_exacto", "fue_duplicado_logico"]

        for column_name in datetime_columns:
            validated_df[column_name] = coerce_datetime_column(validated_df, column_name)

        for column_name in numeric_columns:
            validated_df[column_name] = pd.to_numeric(validated_df[column_name], errors="coerce")

        for column_name in bool_columns:
            if column_name in validated_df.columns:
                validated_df[column_name] = validated_df[column_name].fillna(False).astype(bool)

        log_message("Validacion del dataset completada correctamente.")
        return validated_df
    except Exception as e:
        log_message(f"Error durante la validacion del dataset: {e}", level="error")
        raise
    finally:
        log_message("Finalizo la validacion del dataset.")


@task(name="feature_engineering")
@timing_decorator
@validate
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    try:
        engineered_df = df.copy()

        descripcion_mask = non_placeholder_mask(engineered_df["descripci_n_del_procedimiento"])
        descripcion_length = (
            engineered_df["descripci_n_del_procedimiento"].astype("string").fillna("").str.strip().str.len().astype("Int64")
        )
        engineered_df["longitud_descripcion"] = descripcion_length
        engineered_df["flag_tiene_descripcion_util"] = bool_to_int8(descripcion_mask & descripcion_length.fillna(0).ge(20))
        engineered_df["flag_tiene_categoria"] = bool_to_int8(non_placeholder_mask(engineered_df["codigo_principal_de_categoria"]))
        engineered_df["flag_tiene_precio_base"] = bool_to_int8(pd.to_numeric(engineered_df["precio_base"], errors="coerce").gt(0))
        engineered_df["flag_tiene_ubicacion_entidad"] = bool_to_int8(
            non_placeholder_mask(engineered_df["departamento_entidad"]) & non_placeholder_mask(engineered_df["ciudad_entidad"])
        )
        engineered_df["flag_tiene_tipo_modalidad"] = bool_to_int8(
            non_placeholder_mask(engineered_df["modalidad_de_contratacion"]) & non_placeholder_mask(engineered_df["tipo_de_contrato"])
        )
        required_flag_columns = list(COMPLETENESS_WEIGHTS.keys())
        engineered_df["missing_required_fields_count"] = (
            len(required_flag_columns) - engineered_df[required_flag_columns].sum(axis=1)
        ).astype("int8")

        process_id_series = normalize_text_series(engineered_df["id_del_proceso"]).str.upper()
        engineered_df["flag_id_proceso_valido"] = bool_to_int8(process_id_series.str.match(PROCESS_ID_PATTERN, na=False))
        engineered_df["flag_referencia_valida"] = bool_to_int8(
            non_placeholder_mask(engineered_df["referencia_del_proceso"])
            & engineered_df["referencia_del_proceso"].astype("string").fillna("").str.strip().str.len().ge(5)
        )
        engineered_df["flag_tiene_url_publica"] = bool_to_int8(
            normalize_text_series(engineered_df["urlproceso"]).str.match(PUBLIC_URL_PATTERN, na=False)
        )
        duplicate_mask = pd.Series(False, index=engineered_df.index)
        for duplicate_column in ["fue_duplicado_exacto", "fue_duplicado_logico"]:
            if duplicate_column in engineered_df.columns:
                duplicate_mask = duplicate_mask | engineered_df[duplicate_column].fillna(False).astype(bool)
        engineered_df["penalizacion_duplicado"] = bool_to_int8(duplicate_mask)
        engineered_df["flag_portafolio_disponible"] = bool_to_int8(non_placeholder_mask(engineered_df["id_del_portafolio"]))

        publicacion = engineered_df["fecha_de_publicacion_del"]
        recepcion = engineered_df["fecha_de_recepcion_de"]
        apertura = engineered_df["fecha_de_apertura_de_respuesta"]
        engineered_df["dias_publicacion_a_recepcion"] = (recepcion - publicacion).dt.total_seconds().div(86400.0)
        engineered_df["dias_recepcion_a_apertura"] = (apertura - recepcion).dt.total_seconds().div(86400.0)
        engineered_df["dias_publicacion_a_apertura"] = (apertura - publicacion).dt.total_seconds().div(86400.0)
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
        temporal_anomaly = engineered_df["anomalia_temporal"].fillna(False).astype(bool) if "anomalia_temporal" in engineered_df.columns else False
        engineered_df["flag_coherencia_temporal_global"] = bool_to_int8(
            engineered_df["flag_fechas_temporales_disponibles"].eq(1)
            & engineered_df["flag_publicacion_antes_recepcion"].eq(1)
            & engineered_df["flag_recepcion_antes_apertura"].eq(1)
            & engineered_df["flag_ventana_recepcion_razonable"].eq(1)
            & engineered_df["flag_ventana_apertura_razonable"].eq(1)
            & ~temporal_anomaly
        )

        respuestas_al_procedimiento = coerce_numeric_column(engineered_df, "respuestas_al_procedimiento")
        respuestas_externas = coerce_numeric_column(engineered_df, "respuestas_externas")
        conteo_ofertas = coerce_numeric_column(engineered_df, "conteo_de_respuestas_a_ofertas")
        proveedores_con_invitacion = coerce_numeric_column(engineered_df, "proveedores_con_invitacion")
        proveedores_unicos = coerce_numeric_column(engineered_df, "proveedores_unicos_con")
        engineered_df["total_respuestas"] = respuestas_al_procedimiento + respuestas_externas + conteo_ofertas
        engineered_df["total_interes_oferentes"] = proveedores_con_invitacion + proveedores_unicos
        engineered_df["flag_hubo_participacion"] = bool_to_int8(
            engineered_df["total_respuestas"].gt(0) | engineered_df["total_interes_oferentes"].gt(0)
        )
        engineered_df["flag_hubo_competencia_minima"] = bool_to_int8(
            proveedores_unicos.ge(2) | respuestas_al_procedimiento.ge(2)
        )
        competition_columns = [
            "respuestas_al_procedimiento",
            "respuestas_externas",
            "conteo_de_respuestas_a_ofertas",
            "proveedores_con_invitacion",
            "proveedores_unicos_con",
        ]
        engineered_df["flag_datos_competencia_disponibles"] = bool_to_int8(
            engineered_df[competition_columns].notna().any(axis=1)
        )
        competition_signal = engineered_df["total_respuestas"] + proveedores_unicos
        engineered_df["intensidad_competencia_normalizada"] = clip_series(
            np.log1p(competition_signal) / math.log1p(10)
        )

        log_message("Feature engineering completado correctamente.")
        return engineered_df
    except Exception as e:
        log_message(f"Error durante el feature engineering: {e}", level="error")
        raise
    finally:
        log_message("Finalizo el feature engineering.")


@task(name="build_scores")
@timing_decorator
@validate
def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    try:
        scored_df = df.copy()
        scored_df["score_completitud"] = weighted_score(scored_df, COMPLETENESS_WEIGHTS)
        traceability_base = weighted_score(scored_df, TRACEABILITY_WEIGHTS)
        scored_df["score_trazabilidad_base"] = traceability_base
        scored_df["score_trazabilidad"] = clip_series(
            traceability_base - (scored_df["penalizacion_duplicado"].astype(float) * 0.35)
        )
        scored_df["score_temporal"] = weighted_score(scored_df, TEMPORAL_WEIGHTS)
        scored_df["score_competencia"] = weighted_score(scored_df, COMPETITION_WEIGHTS)
        scored_df["transparency_score"] = clip_series(weighted_score(scored_df, TRANSPARENCY_SCORE_WEIGHTS))
        log_message("Construccion de scores completada correctamente.")
        return scored_df
    except Exception as e:
        log_message(f"Error durante la construccion de scores: {e}", level="error")
        raise
    finally:
        log_message("Finalizo la construccion de scores.")


@task(name="build_target")
@timing_decorator
@validate
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    try:
        target_df = df.copy()
        target_df["riesgo_baja_transparencia"] = target_df["transparency_score"].lt(0.65).astype("int8")
        target_df["nivel_riesgo_transparencia"] = np.select(
            [
                target_df["transparency_score"].lt(0.50),
                target_df["transparency_score"].lt(0.65),
            ],
            ["alto", "medio"],
            default="bajo",
        )

        traceability_evidence_count = pd.Series(0, index=target_df.index, dtype="int8")
        for column_name in [
            "id_del_proceso",
            "referencia_del_proceso",
            "urlproceso",
            "id_del_portafolio",
        ]:
            traceability_evidence_count = traceability_evidence_count.add(non_placeholder_mask(target_df[column_name]).astype("int8"))
        target_df["flag_evidencia_completitud_suficiente"] = np.int8(1)
        target_df["flag_evidencia_trazabilidad_suficiente"] = bool_to_int8(traceability_evidence_count.ge(3))

        temporal_rules_available = pd.DataFrame(
            {
                "disponibilidad": target_df["fecha_de_publicacion_del"].notna()
                & target_df["fecha_de_recepcion_de"].notna()
                & target_df["fecha_de_apertura_de_respuesta"].notna(),
                "pub_recep": target_df["fecha_de_publicacion_del"].notna() & target_df["fecha_de_recepcion_de"].notna(),
                "recep_apertura": target_df["fecha_de_recepcion_de"].notna()
                & target_df["fecha_de_apertura_de_respuesta"].notna(),
                "ventana_recep": target_df["fecha_de_publicacion_del"].notna() & target_df["fecha_de_recepcion_de"].notna(),
                "ventana_apertura": target_df["fecha_de_recepcion_de"].notna()
                & target_df["fecha_de_apertura_de_respuesta"].notna(),
            },
            index=target_df.index,
        )
        target_df["flag_evidencia_temporal_suficiente"] = bool_to_int8(temporal_rules_available.sum(axis=1).ge(3))
        target_df["flag_evidencia_competencia_suficiente"] = bool_to_int8(
            target_df[
                [
                    "respuestas_al_procedimiento",
                    "respuestas_externas",
                    "conteo_de_respuestas_a_ofertas",
                    "proveedores_con_invitacion",
                    "proveedores_unicos_con",
                ]
            ]
            .notna()
            .any(axis=1)
        )

        target_df["evidence_coverage"] = clip_series(
            (
                target_df["flag_evidencia_completitud_suficiente"].astype(float)
                + target_df["flag_evidencia_trazabilidad_suficiente"].astype(float)
                + target_df["flag_evidencia_temporal_suficiente"].astype(float)
                + target_df["flag_evidencia_competencia_suficiente"].astype(float)
            )
            / 4.0
        )
        target_df["margin_to_threshold"] = target_df["transparency_score"].sub(0.65).abs()
        target_df["confianza_label"] = np.select(
            [
                (target_df["evidence_coverage"].ge(0.75) & target_df["margin_to_threshold"].ge(0.15)),
                (target_df["evidence_coverage"].ge(0.50) & target_df["margin_to_threshold"].ge(0.07)),
            ],
            ["alta", "media"],
            default="baja",
        )

        log_message("Construccion de variables objetivo completada correctamente.")
        return target_df
    except Exception as e:
        log_message(f"Error durante la construccion de la variable objetivo: {e}", level="error")
        raise
    finally:
        log_message("Finalizo la construccion de la variable objetivo.")


@task(name="save_parquet")
@timing_decorator
@validate
def save_parquet(df: pd.DataFrame, output_path: str) -> Path:
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_file, index=False, engine="pyarrow")
        log_message(f"Archivo parquet guardado correctamente en: {output_file}")
        return output_file
    except Exception as e:
        log_message(f"Error al escribir el archivo parquet: {e}", level="error")
        raise
    finally:
        log_message("Finalizo la escritura del archivo parquet.")


@task(name="upload_postgres")
@timing_decorator
@validate
def upload_postgres(df: pd.DataFrame, db_config: dict[str, Any], table_name: str) -> int:
    connection = None
    cursor = None
    try:
        if not DEPENDENCY_STATUS.get("psycopg2", False):
            raise ImportError(
                "psycopg2 no esta disponible. Instale 'psycopg2-binary' o ejecute el script con conectividad a pip."
            )

        if not db_config_is_complete(db_config):
            raise ValueError("La configuracion de PostgreSQL esta incompleta en el archivo .env.")

        psycopg2 = __import__("psycopg2")
        upload_df = df.copy()
        for column_name in upload_df.columns:
            if pd.api.types.is_datetime64_any_dtype(upload_df[column_name]):
                upload_df[column_name] = upload_df[column_name].astype("string")

        create_postgres_table(df, db_config, table_name)
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
        rows_uploaded = len(upload_df)
        log_message(f"Se cargaron {rows_uploaded:,} filas en la tabla '{table_name}'.")
        return rows_uploaded
    except Exception as e:
        if connection is not None:
            connection.rollback()
        log_message(f"Error en la conexion o carga hacia PostgreSQL: {e}", level="error")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()
        log_message("Finalizo la carga hacia PostgreSQL.")


@flow(name="secop_feature_engineering_flow")
def secop_feature_engineering_flow(
    parquet_path: str = str(PROJECT_ROOT / INPUT_PARQUET),
    output_path: str = str(PROJECT_ROOT / OUTPUT_PARQUET),
    table_name: str = OUTPUT_TABLE,
    skip_postgres: bool = False,
) -> dict[str, Any]:
    db_config = build_db_config()
    rows_uploaded = 0

    try:
        df = extract_data(parquet_path)
        df = validate_data(df)
        df = feature_engineering(df)
        df = build_scores(df)
        df = build_target(df)
        saved_path = save_parquet(df, output_path)

        if skip_postgres:
            log_message("Carga a PostgreSQL omitida por configuracion del flujo.")
        else:
            rows_uploaded = upload_postgres(df, db_config, table_name)

        result = {
            "rows_processed": int(len(df)),
            "columns_generated": int(len(df.columns)),
            "output_path": str(saved_path),
            "postgres_table": table_name,
            "rows_uploaded": int(rows_uploaded),
            "prefect_available": PREFECT_AVAILABLE,
        }
        log_message(f"Flujo completado correctamente: {result}")
        return result
    except Exception as e:
        log_message(f"El flujo de feature engineering fallo: {e}", level="error")
        raise
    finally:
        log_message("Finalizo el flujo de feature engineering SECOP.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de feature engineering para SECOP II con Prefect y carga a PostgreSQL."
    )
    parser.add_argument("--input-path", default=str(PROJECT_ROOT / INPUT_PARQUET))
    parser.add_argument("--output-path", default=str(PROJECT_ROOT / OUTPUT_PARQUET))
    parser.add_argument("--table-name", default=OUTPUT_TABLE)
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Ejecuta el pipeline completo y guarda el parquet, pero no sube resultados a PostgreSQL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    secop_feature_engineering_flow(
        parquet_path=args.input_path,
        output_path=args.output_path,
        table_name=args.table_name,
        skip_postgres=args.skip_postgres,
    )


if __name__ == "__main__":
    main()
