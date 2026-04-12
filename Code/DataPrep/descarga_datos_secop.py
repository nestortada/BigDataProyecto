from __future__ import annotations

from functools import wraps
from pathlib import Path
import inspect
import os
import time
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from prefect import flow, get_run_logger, task
from requests.exceptions import HTTPError, RequestException
from sodapy import Socrata


def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / ".env").exists() and (candidate / "Data").exists() and (candidate / "Code").exists():
            return candidate
    return start


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

            if arg_name == "years_to_extract":
                if not isinstance(arg_value, int) or arg_value <= 0:
                    raise ValueError("'years_to_extract' debe ser un entero mayor que cero.")

            if arg_name == "page_size":
                if not isinstance(arg_value, int) or arg_value <= 0:
                    raise ValueError("'page_size' debe ser un entero mayor que cero.")

            if arg_name == "max_rows" and arg_value is not None:
                if not isinstance(arg_value, int) or arg_value <= 0:
                    raise ValueError("'max_rows' debe ser None o un entero mayor que cero.")

            if arg_name == "target_file_size_mb" and arg_value is not None:
                if not isinstance(arg_value, (int, float)) or arg_value <= 0:
                    raise ValueError("'target_file_size_mb' debe ser None o un numero mayor que cero.")

            if arg_name == "df":
                if not isinstance(arg_value, pd.DataFrame):
                    raise TypeError("'df' debe ser un pandas.DataFrame.")

            if arg_name == "output_path":
                try:
                    output_parent = Path(arg_value).expanduser().resolve().parent
                except Exception as exc:
                    raise ValueError("'output_path' no es una ruta valida.") from exc
                if not output_parent.exists():
                    output_parent.mkdir(parents=True, exist_ok=True)

        return func(*args, **kwargs)

    return wrapper


PROJECT_ROOT = find_project_root(Path.cwd())
load_dotenv(PROJECT_ROOT / ".env")

DOMAIN = os.getenv("SOCRATA_DOMAIN", "www.datos.gov.co")
DATASET_ID = os.getenv("SOCRATA_DATASET_ID", "p6dx-8zbt")
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN") or None
API_SECRET = os.getenv("SOCRATA_API_SECRET")
DATE_FILTER_COLUMN = os.getenv("SOCRATA_DATE_FILTER_COLUMN", "fecha_de_publicacion")
YEARS_TO_EXTRACT = int(os.getenv("SECOP_YEARS_TO_EXTRACT", "5"))
OUTPUT_PATH = PROJECT_ROOT / "Data" / "Raw" / "secop_procesos.parquet"
PAGE_SIZE = int(os.getenv("SECOP_PAGE_SIZE", "25000"))
TARGET_PREVIEW_ROWS = int(os.getenv("SECOP_TARGET_PREVIEW_ROWS", "25000"))
TARGET_FILE_SIZE_MB = float(os.getenv("SECOP_TARGET_FILE_SIZE_MB", "95"))
MAX_ROWS_TO_DOWNLOAD_RAW = os.getenv("SECOP_MAX_ROWS_TO_DOWNLOAD", "")
MAX_ROWS_TO_DOWNLOAD = int(MAX_ROWS_TO_DOWNLOAD_RAW) if MAX_ROWS_TO_DOWNLOAD_RAW.strip() else None

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


@task(retries=2, retry_delay_seconds=10)
@timing_decorator
@validate_inputs
def build_date_window(years_to_extract: int, date_column: str) -> dict[str, str]:
    logger = get_run_logger()
    window: dict[str, str] = {}

    try:
        end_timestamp = pd.Timestamp.now(tz="America/Bogota").floor("D")
        start_timestamp = end_timestamp - pd.DateOffset(years=years_to_extract)

        window = {
            "start_iso": start_timestamp.strftime("%Y-%m-%dT00:00:00"),
            "end_iso": end_timestamp.strftime("%Y-%m-%dT23:59:59"),
            "where_clause": (
                f"{date_column} between "
                f"'{start_timestamp.strftime('%Y-%m-%dT00:00:00')}' and "
                f"'{end_timestamp.strftime('%Y-%m-%dT23:59:59')}'"
            ),
        }
        logger.info("Ventana de consulta construida: %s", window["where_clause"])
        return window
    except Exception:
        logger.exception("No fue posible construir la ventana de fechas para la extraccion.")
        raise
    finally:
        logger.info("Finalizo la preparacion de la ventana de fechas.")


@task(retries=2, retry_delay_seconds=10)
@timing_decorator
@validate_inputs
def download_dataset(
    domain: str,
    dataset_id: str,
    app_token: str | None,
    where_clause: str,
    page_size: int = 25000,
    max_rows: int | None = None,
    target_file_size_mb: float | None = None,
) -> pd.DataFrame:
    logger = get_run_logger()
    all_rows: list[dict[str, Any]] = []
    target_file_size_bytes = int(target_file_size_mb * 1024 * 1024) if target_file_size_mb is not None else None

    def _download_with_token(token: str | None) -> list[dict[str, Any]]:
        local_rows: list[dict[str, Any]] = []
        offset = 0
        local_client: Socrata | None = None

        try:
            local_client = Socrata(domain, token, timeout=60)
            while True:
                current_limit = page_size
                if max_rows is not None:
                    remaining = max_rows - len(local_rows)
                    if remaining <= 0:
                        break
                    current_limit = min(page_size, remaining)

                batch = local_client.get(
                    dataset_id,
                    limit=current_limit,
                    offset=offset,
                    where=where_clause,
                    order=f"{DATE_FILTER_COLUMN} ASC",
                )

                if not batch:
                    break

                local_rows.extend(batch)
                logger.info("Descargados %s registros acumulados.", len(local_rows))

                if target_file_size_bytes is not None:
                    estimated_size_bytes = len(pd.DataFrame.from_records(local_rows).to_parquet(index=False))
                    logger.info(
                        "Tamano estimado actual del parquet: %.2f MB de %.2f MB objetivo.",
                        estimated_size_bytes / (1024 * 1024),
                        target_file_size_bytes / (1024 * 1024),
                    )
                    if estimated_size_bytes >= target_file_size_bytes:
                        logger.info(
                            "Se alcanzo el tamano objetivo del archivo (%.2f MB). Se detiene la descarga.",
                            target_file_size_bytes / (1024 * 1024),
                        )
                        break

                if len(batch) < current_limit:
                    break

                offset += current_limit

            return local_rows
        except Exception:
            logger.exception("Fallo la descarga paginada desde Socrata.")
            raise
        finally:
            if local_client is not None:
                try:
                    local_client.close()
                except Exception:
                    logger.warning("No se pudo cerrar limpiamente el cliente de Socrata.")

    try:
        all_rows = _download_with_token(app_token)
    except HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code == 403 and app_token:
            logger.warning("El app token fue rechazado. Reintentando sin token porque el dataset es publico.")
            all_rows = _download_with_token(None)
        else:
            logger.exception("HTTPError durante la descarga del dataset.")
            raise
    except RequestException:
        logger.exception("Error de red al consultar la API de Socrata.")
        raise
    except Exception:
        logger.exception("Se produjo un error inesperado durante la descarga del dataset.")
        raise
    finally:
        logger.info("Finalizo la tarea de descarga del dataset.")

    if not all_rows:
        logger.warning("La consulta no devolvio registros para la ventana solicitada.")

    df = pd.DataFrame.from_records(all_rows)
    logger.info("DataFrame creado con %s filas y %s columnas.", len(df), len(df.columns))
    return df


@task
@timing_decorator
@validate_inputs
def validate_and_filter_dataframe(df: pd.DataFrame, date_column: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    logger = get_run_logger()
    filtered_df = pd.DataFrame()

    try:
        if df.empty:
            logger.warning("El DataFrame recibido esta vacio; no hay filas para validar.")
            return df

        if date_column not in df.columns:
            raise KeyError(f"La columna de fecha '{date_column}' no existe en el dataset descargado.")

        filtered_df = df.copy()
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce", utc=True)
        start_ts = pd.Timestamp(start_iso, tz="UTC")
        end_ts = pd.Timestamp(end_iso, tz="UTC")
        filtered_df = filtered_df.loc[filtered_df[date_column].between(start_ts, end_ts, inclusive="both")].copy()
        filtered_df.sort_values(by=date_column, inplace=True)
        filtered_df.reset_index(drop=True, inplace=True)

        logger.info("DataFrame validado y filtrado a %s filas.", len(filtered_df))
        return filtered_df
    except Exception:
        logger.exception("No fue posible validar o filtrar el DataFrame descargado.")
        raise
    finally:
        logger.info("Finalizo la validacion del DataFrame.")


@task
@timing_decorator
@validate_inputs
def save_dataset(df: pd.DataFrame, output_path: str) -> str:
    logger = get_run_logger()

    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Parquet guardado en %s", path.resolve())
        return str(path.resolve())
    except Exception:
        logger.exception("No fue posible guardar el archivo parquet de salida.")
        raise
    finally:
        logger.info("Finalizo la tarea de persistencia del parquet.")


@flow(name="SECOP Data Prep Pipeline")
def main_pipeline() -> pd.DataFrame:
    logger = get_run_logger()
    final_df = pd.DataFrame()

    try:
        date_window = build_date_window(YEARS_TO_EXTRACT, DATE_FILTER_COLUMN)
        raw_df = download_dataset(
            DOMAIN,
            DATASET_ID,
            APP_TOKEN,
            where_clause=date_window["where_clause"],
            page_size=PAGE_SIZE,
            max_rows=MAX_ROWS_TO_DOWNLOAD,
            target_file_size_mb=TARGET_FILE_SIZE_MB,
        )
        final_df = validate_and_filter_dataframe(
            raw_df,
            DATE_FILTER_COLUMN,
            start_iso=date_window["start_iso"],
            end_iso=date_window["end_iso"],
        )
        save_dataset(final_df, str(OUTPUT_PATH))

        logger.info("Pipeline completado con %s filas finales y %s columnas.", len(final_df), len(final_df.columns))
        return final_df
    except Exception:
        logger.exception("La ejecucion del pipeline de SECOP fallo.")
        raise
    finally:
        logger.info("La ejecucion del flujo principal ha finalizado.")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT.resolve()}")
    print(f"Dominio: {DOMAIN}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Archivo de salida: {OUTPUT_PATH.resolve()}")
    print(f"App token cargado: {'si' if APP_TOKEN else 'no'}")
    print(f"Secret cargada: {'si' if API_SECRET else 'no'}")
    print(f"Columna de fecha usada para filtro: {DATE_FILTER_COLUMN}")
    print(f"Ventana de extraccion: ultimos {YEARS_TO_EXTRACT} anos")
    print(f"Tamano objetivo del archivo: {TARGET_FILE_SIZE_MB} MB")
    print("Maximo de filas configurado:", MAX_ROWS_TO_DOWNLOAD if MAX_ROWS_TO_DOWNLOAD is not None else "sin limite")

    df = main_pipeline()
    print(f"Total de filas descargadas y filtradas: {len(df):,}")
    print(f"Total de columnas: {len(df.columns)}")
    print(df.head(TARGET_PREVIEW_ROWS))
    print(df.iloc[:, :10].head())


if __name__ == "__main__":
    main()
