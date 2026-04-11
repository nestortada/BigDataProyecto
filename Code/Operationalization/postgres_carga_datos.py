from functools import wraps
from io import StringIO
from pathlib import Path
import inspect
import os
import re
import time
import unicodedata

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from prefect import flow, task
from prefect.logging import get_run_logger


def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / ".env").exists() and (candidate / "Data").exists() and (candidate / "Code").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path.cwd())
load_dotenv(PROJECT_ROOT / ".env")

PARQUET_FILE = PROJECT_ROOT / "Data" / "Raw" / "secop_procesos.parquet"
TABLE_NAME = "secop_procesos"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5433")),
    "database": os.getenv("POSTGRES_DB", "bigdatatools1"),
    "user": os.getenv("POSTGRES_USER", "psqluser"),
    "password": os.getenv("POSTGRES_PASSWORD", "psqlpass"),
}


def normalize_column_name(column_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", column_name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


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

            if arg_name == "parquet_path":
                parquet_path = Path(arg_value)
                if not parquet_path.exists():
                    raise FileNotFoundError(f"No se encontro el archivo parquet: {parquet_path}")

            if arg_name == "df" and not isinstance(arg_value, pd.DataFrame):
                raise TypeError("'df' debe ser un pandas.DataFrame.")

            if arg_name == "db_config":
                required_keys = {"host", "port", "database", "user", "password"}
                if not isinstance(arg_value, dict):
                    raise TypeError("'db_config' debe ser un diccionario.")
                if not required_keys.issubset(arg_value.keys()):
                    raise ValueError("'db_config' debe incluir host, port, database, user y password.")

            if arg_name == "table_name":
                if not isinstance(arg_value, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", arg_value):
                    raise ValueError("'table_name' debe ser un identificador SQL simple y valido.")

        return func(*args, **kwargs)

    return wrapper


@task(name="leer_parquet_secop")
@timing_decorator
@validate_inputs
def read_parquet_file(parquet_path: str) -> pd.DataFrame:
    parquet_path = Path(parquet_path)
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Archivo leido correctamente: {parquet_path}")
        print(f"Filas: {len(df):,} | Columnas: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"Error al leer el parquet: {e}")
        raise


@task(name="transformar_datos_secop")
@timing_decorator
@validate_inputs
def transform_secop_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df.columns = [normalize_column_name(col) for col in df.columns]

        if "urlproceso" in df.columns:
            df["urlproceso"] = df["urlproceso"].apply(
                lambda value: value.get("url") if isinstance(value, dict) else value
            )

        if "fecha_de_publicacion" in df.columns:
            df["fecha_de_publicacion"] = pd.to_datetime(df["fecha_de_publicacion"], utc=True)

        for column in df.columns:
            if column != "fecha_de_publicacion":
                df[column] = df[column].astype("string")

        print("Transformacion completada.")
        print("Columnas normalizadas:")
        print(df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error durante la transformacion de los datos: {e}")
        raise


@task(name="crear_tabla_en_postgres")
@timing_decorator
@validate_inputs
def create_table(df: pd.DataFrame, db_config: dict, table_name: str) -> None:
    conn = None
    cur = None
    try:
        columns_sql = []
        for column in df.columns:
            columns_sql.append(f'"{column}" {postgres_type_for_series(df[column])}')

        create_sql = f"""
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} (
            {', '.join(columns_sql)}
        );
        """

        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        print(f"Tabla '{table_name}' creada correctamente.")
    except Exception as e:
        print(f"Error al crear la tabla '{table_name}': {e}")
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@task(name="cargar_datos_en_postgres")
@timing_decorator
@validate_inputs
def load_data_to_postgres(df: pd.DataFrame, db_config: dict, table_name: str) -> int:
    conn = None
    cur = None
    try:
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
        print(f"Se cargaron {len(df):,} filas en '{table_name}'.")
        return len(df)
    except Exception as e:
        print(f"Error al cargar datos en PostgreSQL: {e}")
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@task(name="validar_carga_en_postgres")
@timing_decorator
@validate_inputs
def validate_load(db_config: dict, table_name: str) -> int:
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name};")
        total_rows = cur.fetchone()[0]
        print(f"Validacion exitosa. Total de filas en '{table_name}': {total_rows:,}")
        return total_rows
    except Exception as e:
        print(f"Error al validar la carga en PostgreSQL: {e}")
        raise
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


@flow(name="orquestacion_carga_secop_postgres")
def secop_to_postgres_flow(parquet_path: str, db_config: dict, table_name: str = TABLE_NAME) -> int:
    try:
        df = read_parquet_file(parquet_path)
        df = transform_secop_data(df)
        create_table(df, db_config, table_name)
        load_data_to_postgres(df, db_config, table_name)
        total_rows = validate_load(db_config, table_name)
        print("Flujo ejecutado correctamente.")
        return total_rows
    except Exception as e:
        print(f"El flujo de Prefect fallo: {e}")
        raise
    finally:
        print("Ejecucion del flujo finalizada.")


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Parquet path:", PARQUET_FILE)

    try:
        total_rows = secop_to_postgres_flow(str(PARQUET_FILE), DB_CONFIG, TABLE_NAME)
        print(f"Proceso finalizado. Filas cargadas en PostgreSQL: {total_rows:,}")
    except Exception as e:
        print(f"No fue posible completar la carga: {e}")
    finally:
        print("Script finalizado.")


if __name__ == "__main__":
    main()
