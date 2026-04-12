from functools import wraps
from io import StringIO
from pathlib import Path
import inspect
import os
import re
import time
import unicodedata

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


COLUMNS_TO_REMOVE = [
    "adjudicado",
    "id_adjudicacion",
    "valor_total_adjudicacion",
    "codigoproveedor",
    "departamento_proveedor",
    "ciudad_proveedor",
    "nombre_del_adjudicador",
    "nombre_del_proveedor",
    "nit_del_proveedor_adjudicado",
]


def find_project_root(start: Path) -> Path:
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / ".env").exists() and (candidate / "Data").exists() and (candidate / "Code").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path.cwd())
load_dotenv(PROJECT_ROOT / ".env")

SOURCE_PARQUET = PROJECT_ROOT / "Data" / "Raw" / "secop_procesos.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "Data" / "Raw" / "datos_analisis.parquet"
OUTPUT_TABLE = "datos_analisis"

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")) if os.getenv("POSTGRES_PORT") else None,
    "database": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
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

            if arg_name in {"source_path", "source_parquet", "output_path", "output_parquet"}:
                path_value = Path(arg_value)
                if arg_name in {"source_path", "source_parquet"} and not path_value.exists():
                    raise FileNotFoundError(f"No se encontro el archivo parquet: {path_value}")

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


@task(name="crear_parquet_datos_analisis")
@timing_decorator
@validate_inputs
def create_analysis_parquet(source_path: str, output_path: str) -> pd.DataFrame:
    return _create_analysis_parquet_impl(source_path, output_path)


def _create_analysis_parquet_impl(source_path: str, output_path: str) -> pd.DataFrame:
    source = Path(source_path)
    output = Path(output_path)

    try:
        df = pd.read_parquet(source)

        missing_columns = [column for column in COLUMNS_TO_REMOVE if column not in df.columns]
        if missing_columns:
            raise ValueError(
                "No se encontraron estas columnas en el parquet original: "
                + ", ".join(missing_columns)
            )

        filtered_df = df.drop(columns=COLUMNS_TO_REMOVE).copy()
        output.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_parquet(output, index=False)

        print(f"Parquet creado correctamente: {output}")
        print(f"Filas: {len(filtered_df):,} | Columnas: {len(filtered_df.columns)}")
        return filtered_df
    except Exception as e:
        print(f"Error al crear el parquet de analisis: {e}")
        raise
    finally:
        print("Finalizo la creacion del parquet para analisis.")


@task(name="transformar_datos_analisis_secop")
@timing_decorator
@validate_inputs
def transform_secop_data(df: pd.DataFrame) -> pd.DataFrame:
    return _transform_secop_data_impl(df)


def _transform_secop_data_impl(df: pd.DataFrame) -> pd.DataFrame:
    try:
        transformed_df = df.copy()
        transformed_df.columns = [normalize_column_name(col) for col in transformed_df.columns]

        if "urlproceso" in transformed_df.columns:
            transformed_df["urlproceso"] = transformed_df["urlproceso"].apply(
                lambda value: value.get("url") if isinstance(value, dict) else value
            )

        if "fecha_de_publicacion" in transformed_df.columns:
            transformed_df["fecha_de_publicacion"] = pd.to_datetime(
                transformed_df["fecha_de_publicacion"], utc=True
            )

        for column in transformed_df.columns:
            if column != "fecha_de_publicacion":
                transformed_df[column] = transformed_df[column].astype("string")

        print("Transformacion completada.")
        print("Columnas normalizadas:")
        print(transformed_df.columns.tolist())
        return transformed_df
    except Exception as e:
        print(f"Error durante la transformacion de los datos: {e}")
        raise
    finally:
        print("Finalizo la transformacion de datos para analisis.")


@task(name="crear_tabla_datos_analisis_en_postgres")
@timing_decorator
@validate_inputs
def create_table(df: pd.DataFrame, db_config: dict, table_name: str) -> None:
    _create_table_impl(df, db_config, table_name)


def _create_table_impl(df: pd.DataFrame, db_config: dict, table_name: str) -> None:
    conn = None
    cur = None
    try:
        import psycopg2

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
        print(f"Finalizo la creacion de la tabla '{table_name}'.")


@task(name="cargar_datos_analisis_en_postgres")
@timing_decorator
@validate_inputs
def load_data_to_postgres(df: pd.DataFrame, db_config: dict, table_name: str) -> int:
    return _load_data_to_postgres_impl(df, db_config, table_name)


def _load_data_to_postgres_impl(df: pd.DataFrame, db_config: dict, table_name: str) -> int:
    conn = None
    cur = None
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
        print(f"Finalizo la carga de datos en la tabla '{table_name}'.")


@task(name="validar_carga_datos_analisis_en_postgres")
@timing_decorator
@validate_inputs
def validate_load(db_config: dict, table_name: str) -> int:
    return _validate_load_impl(db_config, table_name)


def _validate_load_impl(db_config: dict, table_name: str) -> int:
    conn = None
    cur = None
    try:
        import psycopg2

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
        print(f"Finalizo la validacion de la tabla '{table_name}'.")


@flow(name="crear_datos_analisis_y_cargar_postgres")
@timing_decorator
@validate_inputs
def create_analysis_dataset(
    source_parquet: str = str(SOURCE_PARQUET),
    output_parquet: str = str(OUTPUT_PARQUET),
    table_name: str = OUTPUT_TABLE,
) -> int:
    return _create_analysis_dataset_impl(source_parquet, output_parquet, table_name)


def _create_analysis_dataset_impl(
    source_parquet: str = str(SOURCE_PARQUET),
    output_parquet: str = str(OUTPUT_PARQUET),
    table_name: str = OUTPUT_TABLE,
) -> int:
    try:
        filtered_df = create_analysis_parquet(source_parquet, output_parquet)
        transformed_df = transform_secop_data(filtered_df)
        create_table(transformed_df, DB_CONFIG, table_name)
        load_data_to_postgres(transformed_df, DB_CONFIG, table_name)
        total_rows = validate_load(DB_CONFIG, table_name)
        print(f"Proceso completado. Tabla '{table_name}' lista en PostgreSQL.")
        return total_rows
    except Exception as e:
        print(f"El flujo de creacion de datos de analisis fallo: {e}")
        raise
    finally:
        print("Ejecucion del flujo de datos de analisis finalizada.")


@timing_decorator
@validate_inputs
def run_without_prefect(
    source_parquet: str = str(SOURCE_PARQUET),
    output_parquet: str = str(OUTPUT_PARQUET),
    table_name: str = OUTPUT_TABLE,
) -> int:
    try:
        filtered_df = _create_analysis_parquet_impl(source_parquet, output_parquet)
        transformed_df = _transform_secop_data_impl(filtered_df)
        _create_table_impl(transformed_df, DB_CONFIG, table_name)
        _load_data_to_postgres_impl(transformed_df, DB_CONFIG, table_name)
        total_rows = _validate_load_impl(DB_CONFIG, table_name)
        print(f"Proceso completado. Tabla '{table_name}' lista en PostgreSQL.")
        return total_rows
    except Exception as e:
        print(f"La ejecucion directa del proceso fallo: {e}")
        raise
    finally:
        print("Ejecucion directa del flujo de datos de analisis finalizada.")


def main() -> None:
    try:
        total_rows = run_without_prefect()
        print(f"Filas cargadas en PostgreSQL: {total_rows:,}")
    except Exception as e:
        print(f"No fue posible completar el proceso: {e}")
    finally:
        print("Script finalizado.")


if __name__ == "__main__":
    main()
