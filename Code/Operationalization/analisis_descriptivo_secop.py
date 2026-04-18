from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_ROOT = BOOTSTRAP_ROOT / ".python_packages"
RUNTIME_TAG = f"py{sys.version_info.major}{sys.version_info.minor}-{sys.platform}-{platform.machine().lower()}"
LOCAL_PACKAGE_DIR = LOCAL_PACKAGE_ROOT / RUNTIME_TAG


def register_local_package_dir() -> None:
    if LOCAL_PACKAGE_DIR.exists() and str(LOCAL_PACKAGE_DIR) not in sys.path:
        # Append instead of prepend so the active virtualenv keeps priority.
        sys.path.append(str(LOCAL_PACKAGE_DIR))


register_local_package_dir()

DEPENDENCY_MAP: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "prefect": "prefect",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scipy": "scipy",
}

PROJECT_MARKERS = {"Code", "Data", "Processed"}
DEFAULT_INPUT_PATH = Path("Data/Processed/Limpieza/datos_feature_engineering.parquet")
DEFAULT_OUTPUT_DIR = Path("Data/Processed/Descriptivo")
PLOT_SAMPLE_SIZE = 100_000
DEFAULT_INPUT_CANDIDATES = [
    Path("Data/Processed/Limpieza/datos_feature_engineering.parquet"),
    Path("processed/datos_feature_engineering.parquet"),
    Path("Processed/Limpieza/datos_feature_engineering.parquet"),
]

ALIAS_CANDIDATES: dict[str, list[str]] = {
    "descripcion_del_procedimiento": ["descripcion_del_procedimiento", "descripci_n_del_procedimiento"],
    "fecha_de_publicacion": ["fecha_de_publicacion", "fecha_de_publicacion_del"],
    "fecha_de_recepcion": ["fecha_de_recepcion", "fecha_de_recepcion_de"],
    "fecha_de_apertura": ["fecha_de_apertura", "fecha_de_apertura_de_respuesta"],
}

PRIORITY_SECOP_COLUMNS = [
    "entidad",
    "nit_entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "ordenentidad",
    "id_del_proceso",
    "referencia_del_proceso",
    "id_del_portafolio",
    "nombre_del_procedimiento",
    "descripcion_del_procedimiento",
    "fase",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "fecha_de_publicacion",
    "fecha_de_recepcion",
    "fecha_de_apertura",
    "precio_base",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "numero_de_lotes",
    "urlproceso",
    "score_completitud",
    "score_trazabilidad",
    "score_temporal",
    "score_competencia",
    "transparency_score",
    "nivel_riesgo_transparencia",
    "confianza_label",
    "evidence_coverage",
    "margin_to_threshold",
]

PRIORITY_NUMERIC_COLUMNS = [
    "precio_base",
    "duracion",
    "duracion_dias",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_unicos_con",
    "numero_de_lotes",
    "visualizaciones_del",
    "dias_hasta_recepcion",
    "dias_publicacion_a_recepcion",
    "dias_recepcion_a_apertura",
    "dias_publicacion_a_apertura",
    "total_respuestas",
    "total_interes_oferentes",
    "longitud_descripcion",
    "score_completitud",
    "score_trazabilidad_base",
    "score_trazabilidad",
    "score_temporal",
    "score_competencia",
    "transparency_score",
    "evidence_coverage",
    "margin_to_threshold",
]

PRIORITY_CATEGORICAL_COLUMNS = [
    "entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "ordenentidad",
    "fase",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "nombre_del_procedimiento",
    "nivel_riesgo_transparencia",
    "confianza_label",
]

PRIORITY_DATETIME_COLUMNS = [
    "fecha_de_publicacion",
    "fecha_de_publicacion_del",
    "fecha_de_recepcion",
    "fecha_de_recepcion_de",
    "fecha_de_apertura",
    "fecha_de_apertura_de_respuesta",
]

TRANSPARENCY_NUMERIC_COLUMNS = [
    "score_completitud",
    "score_trazabilidad_base",
    "score_trazabilidad",
    "score_temporal",
    "score_competencia",
    "transparency_score",
    "evidence_coverage",
    "margin_to_threshold",
]

TRANSPARENCY_CATEGORICAL_COLUMNS = [
    "nivel_riesgo_transparencia",
    "confianza_label",
]

TRANSPARENCY_BINARY_COLUMNS = [
    "riesgo_baja_transparencia",
    "flag_evidencia_completitud_suficiente",
    "flag_evidencia_trazabilidad_suficiente",
    "flag_evidencia_temporal_suficiente",
    "flag_evidencia_competencia_suficiente",
]

DUPLICATE_KEY_CANDIDATES = [
    ["id_del_proceso"],
    ["referencia_del_proceso"],
    ["id_del_portafolio"],
    ["urlproceso"],
    ["id_del_proceso", "referencia_del_proceso"],
]

TECHNICAL_IDENTIFIER_COLUMNS = {
    "id_del_proceso",
    "referencia_del_proceso",
    "id_del_portafolio",
    "urlproceso",
}

TEXT_COLUMNS = {
    "descripci_n_del_procedimiento",
    "descripcion_del_procedimiento",
}


def bootstrap_log(message: str) -> None:
    print(f"[BOOTSTRAP] {message}")


def update_requirements_file(requirements_path: Path, packages: Iterable[str]) -> list[str]:
    packages_to_register = sorted({package.strip() for package in packages if package and package.strip()})
    appended_packages: list[str] = []

    try:
        requirements_path.parent.mkdir(parents=True, exist_ok=True)
        existing_lines: list[str] = []

        if requirements_path.exists():
            existing_lines = requirements_path.read_text(encoding="utf-8").splitlines()

        existing_normalized = {
            line.strip().lower()
            for line in existing_lines
            if line.strip() and not line.strip().startswith("#")
        }

        appended_packages = [
            package for package in packages_to_register if package.lower() not in existing_normalized
        ]

        if appended_packages:
            output_lines = existing_lines[:]
            if output_lines and output_lines[-1].strip():
                output_lines.append("")
            output_lines.extend(appended_packages)
            requirements_path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")

        return appended_packages
    except Exception as exc:
        bootstrap_log(f"Error while updating requirements.txt: {exc}")
        raise
    finally:
        bootstrap_log(f"requirements.txt synchronization finished for {requirements_path}.")


def install_missing_packages(packages: Iterable[str]) -> dict[str, bool]:
    install_status: dict[str, bool] = {}

    try:
        LOCAL_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        register_local_package_dir()

        for package in packages:
            try:
                bootstrap_log(f"Installing missing package: {package}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--target", str(LOCAL_PACKAGE_DIR), package],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                install_status[package] = True
            except Exception as exc:
                bootstrap_log(f"Package installation failed for {package}: {exc}")
                install_status[package] = False

        importlib.invalidate_caches()
        register_local_package_dir()
        return install_status
    except Exception as exc:
        bootstrap_log(f"Error while installing missing packages: {exc}")
        raise
    finally:
        bootstrap_log("Missing package installation finished.")


def ensure_runtime_dependencies(requirements_path: Path) -> dict[str, Any]:
    dependency_status: dict[str, bool] = {}
    missing_dependencies: list[tuple[str, str]] = []

    try:
        appended_packages = update_requirements_file(requirements_path, DEPENDENCY_MAP.values())
        for module_name, package_name in DEPENDENCY_MAP.items():
            try:
                importlib.import_module(module_name)
                dependency_status[module_name] = True
            except ImportError:
                dependency_status[module_name] = False
                missing_dependencies.append((module_name, package_name))

        installation_results: dict[str, bool] = {}
        if missing_dependencies:
            installation_results = install_missing_packages(package for _, package in missing_dependencies)
            for module_name, package_name in missing_dependencies:
                try:
                    importlib.import_module(module_name)
                    dependency_status[module_name] = True
                except ImportError:
                    dependency_status[module_name] = installation_results.get(package_name, False)

        return {
            "dependency_status": dependency_status,
            "requirements_appended": appended_packages,
            "missing_dependencies": [package for _, package in missing_dependencies],
        }
    except Exception as exc:
        bootstrap_log(f"Runtime dependency bootstrap failed: {exc}")
        raise
    finally:
        bootstrap_log("Runtime dependency bootstrap finished.")


BOOTSTRAP_STATUS = ensure_runtime_dependencies(BOOTSTRAP_ROOT / "requirements.txt")

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
    PLOTTING_IMPORT_ERROR = None
except Exception as exc:
    plt = None
    sns = None
    PLOTTING_AVAILABLE = False
    PLOTTING_IMPORT_ERROR = str(exc)

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
        raise RuntimeError("Prefect is not available in the current environment.")


def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start


PROJECT_ROOT = find_project_root(Path.cwd())


def resolve_input_parquet_path(project_root: Path, input_path: str | Path) -> Path:
    requested_path = Path(input_path)
    attempted_paths: list[Path] = []

    candidate_paths: list[Path] = []
    if requested_path.is_absolute():
        candidate_paths.append(requested_path)
    else:
        candidate_paths.append(project_root / requested_path)

    for default_candidate in DEFAULT_INPUT_CANDIDATES:
        resolved_candidate = default_candidate if default_candidate.is_absolute() else project_root / default_candidate
        if resolved_candidate not in candidate_paths:
            candidate_paths.append(resolved_candidate)

    if requested_path.name == "datos_feature_engineering.parquet":
        recursive_matches = sorted(project_root.rglob(requested_path.name))
        for match in recursive_matches:
            if match not in candidate_paths:
                candidate_paths.append(match)

    for candidate_path in candidate_paths:
        attempted_paths.append(candidate_path)
        if candidate_path.exists():
            if candidate_path != candidate_paths[0]:
                log_message(f"Input parquet not found at the requested path. Using fallback path: {candidate_path}")
            return candidate_path

    attempted_paths_text = "\n".join(f"- {path}" for path in attempted_paths)
    raise FileNotFoundError(
        "Input parquet file was not found. Attempted paths:\n"
        f"{attempted_paths_text}"
    )


def build_logger() -> logging.Logger:
    logger = logging.getLogger("secop_descriptive_analysis")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[SECOP DESCRIPTIVO] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


BASE_LOGGER = build_logger()


def log_message(message: str, level: str = "info") -> None:
    try:
        run_logger = get_run_logger()
        log_method = getattr(run_logger, level, run_logger.info)
        log_method(message)
    except Exception:
        log_method = getattr(BASE_LOGGER, level, BASE_LOGGER.info)
        log_method(message)


def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_seconds = time.perf_counter() - start_time
            log_message(f"Function '{func.__name__}' completed in {elapsed_seconds:.2f} seconds.")

    return wrapper


def validate(func: Callable[..., Any]) -> Callable[..., Any]:
    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound_arguments = signature.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()

        for arg_name, arg_value in bound_arguments.arguments.items():
            if isinstance(arg_value, str) and not arg_value.strip():
                raise ValueError(f"Argument '{arg_name}' cannot be an empty string.")

            if arg_name in {"df", "raw_df"} and not isinstance(arg_value, pd.DataFrame):
                raise TypeError(f"Argument '{arg_name}' must be a pandas.DataFrame.")

            if arg_name == "parquet_path":
                parquet_path = Path(arg_value)
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Input parquet file was not found: {parquet_path}")

            if arg_name == "output_dirs":
                required_keys = {
                    "project_root",
                    "input_path",
                    "requirements_path",
                    "base",
                    "plots",
                    "numeric_plots",
                    "categorical_plots",
                    "temporal_plots",
                    "correlation_plots",
                    "frequency_tables",
                }
                if not isinstance(arg_value, dict):
                    raise TypeError("'output_dirs' must be a dictionary.")
                missing_keys = required_keys.difference(arg_value.keys())
                if missing_keys:
                    raise ValueError(f"'output_dirs' is missing required keys: {sorted(missing_keys)}")

            if arg_name in {"top_n_categories", "max_plots", "correlation_max_features"}:
                if not isinstance(arg_value, int) or arg_value <= 0:
                    raise ValueError(f"Argument '{arg_name}' must be a positive integer.")

        return func(*args, **kwargs)

    return wrapper


def json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def make_hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    if pd.isna(value):
        return pd.NA
    return value


def build_hashable_frame(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    target_df = df[columns].copy() if columns else df.copy()
    for column in target_df.columns:
        target_df[column] = target_df[column].map(make_hashable)
    return target_df


def flatten_url_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("url")
    return value


def clean_string_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return cleaned.mask(cleaned.eq(""))


def safe_to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def is_binary_like_numeric(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False

    non_null = pd.to_numeric(series, errors="coerce").dropna()
    if non_null.empty:
        return False

    unique_values = set(non_null.unique().tolist())
    return unique_values.issubset({0, 1}) and len(unique_values) <= 2


def is_date_candidate(column_name: str, series: pd.Series) -> bool:
    normalized_name = column_name.lower()
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if "fecha" not in normalized_name:
        return False

    blocked_prefixes = ("flag_", "tiene_", "dias_", "missing_", "anomalia_")
    if normalized_name.startswith(blocked_prefixes):
        return False

    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return False

    return True


def resolve_alias_map(columns: Iterable[str]) -> dict[str, str]:
    existing_columns = set(columns)
    alias_map: dict[str, str] = {}
    for report_name, candidates in ALIAS_CANDIDATES.items():
        for candidate in candidates:
            if candidate in existing_columns:
                alias_map[report_name] = candidate
                break
    return alias_map


def invert_alias_map(alias_map: dict[str, str]) -> dict[str, list[str]]:
    inverse_map: dict[str, list[str]] = {}
    for report_name, actual_name in alias_map.items():
        inverse_map.setdefault(actual_name, []).append(report_name)
    return inverse_map


def build_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    datetime_columns = [
        column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])
    ]
    boolean_columns = [
        column for column in df.columns if pd.api.types.is_bool_dtype(df[column])
    ]
    text_columns = [column for column in df.columns if column in TEXT_COLUMNS or "descripcion" in column.lower()]
    technical_columns = [column for column in df.columns if column in TECHNICAL_IDENTIFIER_COLUMNS]

    numeric_columns: list[str] = []
    binary_numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in df.columns:
        if column in datetime_columns or column in boolean_columns or column in text_columns or column in technical_columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            if is_binary_like_numeric(df[column]):
                binary_numeric_columns.append(column)
            else:
                numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return {
        "datetime": datetime_columns,
        "boolean": boolean_columns,
        "text": text_columns,
        "technical": technical_columns,
        "numeric": numeric_columns,
        "binary_numeric": binary_numeric_columns,
        "categorical": categorical_columns,
    }


def normalize_datetime_series(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)
    return series


def remove_datetime_timezone(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if getattr(series.dt, "tz", None) is not None:
        return series.dt.tz_localize(None)
    return series


def select_priority_columns(available_columns: list[str], priority_columns: list[str], max_count: int) -> list[str]:
    ordered_columns: list[str] = []

    for column in priority_columns:
        if column in available_columns and column not in ordered_columns:
            ordered_columns.append(column)

    for column in available_columns:
        if column not in ordered_columns:
            ordered_columns.append(column)

    return ordered_columns[:max_count]


def series_mode_value(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    try:
        mode_series = non_null.mode(dropna=True)
        return mode_series.iloc[0] if not mode_series.empty else None
    except Exception:
        value_counts = non_null.value_counts(dropna=True)
        return value_counts.index[0] if not value_counts.empty else None


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        log_message(f"Failed to write CSV '{path}': {exc}", level="error")
        raise
    finally:
        log_message(f"CSV write finished: {path.name}")


def safe_write_json(payload: dict[str, Any], path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable_payload = json.loads(json.dumps(payload, default=json_safe))
        path.write_text(json.dumps(serializable_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log_message(f"Failed to write JSON '{path}': {exc}", level="error")
        raise
    finally:
        log_message(f"JSON write finished: {path.name}")


def safe_write_markdown(content: str, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:
        log_message(f"Failed to write Markdown '{path}': {exc}", level="error")
        raise
    finally:
        log_message(f"Markdown write finished: {path.name}")


def safe_save_plot(fig: Any, path: Path) -> None:
    try:
        if not PLOTTING_AVAILABLE or fig is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as exc:
        log_message(f"Failed to save plot '{path}': {exc}", level="error")
        raise
    finally:
        if PLOTTING_AVAILABLE and fig is not None:
            plt.close(fig)


@task(name="bootstrap_environment")
@timing_decorator
@validate
def bootstrap_environment(requirements_path: Path) -> dict[str, Any]:
    try:
        requirements_appended = update_requirements_file(requirements_path, DEPENDENCY_MAP.values())
        pd.set_option("display.max_columns", 200)
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 160)

        if PLOTTING_AVAILABLE:
            sns.set_theme(style="whitegrid", palette="deep")
            plt.rcParams["figure.figsize"] = (12, 6)
            plt.rcParams["axes.titlesize"] = 12
            plt.rcParams["axes.labelsize"] = 10

        return {
            "bootstrap_status": {
                **BOOTSTRAP_STATUS,
                "requirements_appended": requirements_appended,
            },
            "plotting_available": PLOTTING_AVAILABLE,
            "plotting_import_error": PLOTTING_IMPORT_ERROR,
            "prefect_available": PREFECT_AVAILABLE,
        }
    except Exception as exc:
        log_message(f"Environment bootstrap failed: {exc}", level="error")
        raise
    finally:
        log_message("Environment bootstrap step finished.")


@task(name="resolve_paths_and_create_output_dirs")
@timing_decorator
@validate
def resolve_paths_and_create_output_dirs(input_path: str, output_dir: str) -> dict[str, Path]:
    try:
        project_root = find_project_root(Path.cwd())
        resolved_input_path = resolve_input_parquet_path(project_root=project_root, input_path=input_path)
        resolved_output_dir = Path(output_dir)

        if not resolved_output_dir.is_absolute():
            resolved_output_dir = project_root / resolved_output_dir

        output_dirs = {
            "project_root": project_root,
            "input_path": resolved_input_path,
            "requirements_path": project_root / "requirements.txt",
            "base": resolved_output_dir,
            "plots": resolved_output_dir / "plots",
            "numeric_plots": resolved_output_dir / "plots" / "numeric",
            "categorical_plots": resolved_output_dir / "plots" / "categorical",
            "temporal_plots": resolved_output_dir / "plots" / "temporal",
            "correlation_plots": resolved_output_dir / "plots" / "correlation",
            "frequency_tables": resolved_output_dir / "frequency_tables",
        }

        for directory_key in [
            "base",
            "plots",
            "numeric_plots",
            "categorical_plots",
            "temporal_plots",
            "correlation_plots",
            "frequency_tables",
        ]:
            output_dirs[directory_key].mkdir(parents=True, exist_ok=True)

        return output_dirs
    except Exception as exc:
        log_message(f"Path resolution failed: {exc}", level="error")
        raise
    finally:
        log_message("Output directory resolution finished.")


@task(name="load_parquet_dataset")
@timing_decorator
@validate
def load_parquet_dataset(parquet_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        log_message(f"Dataset loaded successfully from {parquet_path}. Shape: {df.shape}")
        return df
    except Exception as exc:
        log_message(f"Failed to read parquet '{parquet_path}': {exc}", level="error")
        raise
    finally:
        log_message("Parquet loading step finished.")


@task(name="prepare_dataframe_for_analysis")
@timing_decorator
@validate
def prepare_dataframe_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        prepared_df = df.copy()
        conversion_records: list[dict[str, Any]] = []

        for column in prepared_df.columns:
            if column == "urlproceso":
                flattened_series = prepared_df[column].map(flatten_url_value)
                prepared_df[column] = clean_string_series(flattened_series)
                continue

            if pd.api.types.is_object_dtype(prepared_df[column]) or pd.api.types.is_string_dtype(prepared_df[column]):
                prepared_df[column] = clean_string_series(prepared_df[column])

        date_columns: list[str] = []
        for column in prepared_df.columns:
            if not is_date_candidate(column, prepared_df[column]):
                continue

            non_null_before = int(prepared_df[column].notna().sum())
            converted_series = pd.to_datetime(prepared_df[column], errors="coerce", utc=True)
            non_null_after = int(converted_series.notna().sum())
            prepared_df[column] = converted_series
            date_columns.append(column)
            conversion_records.append(
                {
                    "column": column,
                    "conversion_type": "datetime",
                    "non_null_before": non_null_before,
                    "non_null_after": non_null_after,
                    "coerced_to_null": max(non_null_before - non_null_after, 0),
                }
            )

        numeric_candidates = sorted(
            set(PRIORITY_NUMERIC_COLUMNS).union(
                column
                for column in prepared_df.columns
                if pd.api.types.is_numeric_dtype(prepared_df[column]) and column not in date_columns
            )
        )

        for column in numeric_candidates:
            if column not in prepared_df.columns or column in date_columns:
                continue

            if pd.api.types.is_bool_dtype(prepared_df[column]):
                continue

            original_series = prepared_df[column]
            non_null_before = int(original_series.notna().sum())
            converted_series = safe_to_numeric_series(original_series)
            non_null_after = int(converted_series.notna().sum())

            if pd.api.types.is_numeric_dtype(original_series) or non_null_after > 0:
                prepared_df[column] = converted_series
                conversion_records.append(
                    {
                        "column": column,
                        "conversion_type": "numeric",
                        "non_null_before": non_null_before,
                        "non_null_after": non_null_after,
                        "coerced_to_null": max(non_null_before - non_null_after, 0),
                    }
                )

        alias_map = resolve_alias_map(prepared_df.columns)
        column_groups = build_column_groups(prepared_df)

        context = {
            "alias_map": alias_map,
            "inverse_alias_map": invert_alias_map(alias_map),
            "column_groups": column_groups,
            "conversion_summary": conversion_records,
            "priority_columns_present": [
                column
                for column in PRIORITY_SECOP_COLUMNS
                if alias_map.get(column, column) in prepared_df.columns
            ],
            "priority_columns_missing": [
                column
                for column in PRIORITY_SECOP_COLUMNS
                if alias_map.get(column, column) not in prepared_df.columns
            ],
        }
        return prepared_df, context
    except Exception as exc:
        log_message(f"Dataframe preparation failed: {exc}", level="error")
        raise
    finally:
        log_message("Dataframe preparation step finished.")


@task(name="build_dataset_overview")
@timing_decorator
@validate
def build_dataset_overview(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    try:
        column_groups = context["column_groups"]
        overview = pd.DataFrame(
            [
                {
                    "n_rows": int(df.shape[0]),
                    "n_columns": int(df.shape[1]),
                    "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / (1024 ** 2)), 2),
                    "n_datetime_columns": len(column_groups["datetime"]),
                    "n_numeric_columns": len(column_groups["numeric"]),
                    "n_binary_numeric_columns": len(column_groups["binary_numeric"]),
                    "n_boolean_columns": len(column_groups["boolean"]),
                    "n_categorical_columns": len(column_groups["categorical"]),
                    "n_text_columns": len(column_groups["text"]),
                    "n_technical_columns": len(column_groups["technical"]),
                }
            ]
        )
        return overview
    except Exception as exc:
        log_message(f"Dataset overview generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Dataset overview step finished.")


@task(name="build_column_inventory")
@timing_decorator
@validate
def build_column_inventory(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    try:
        inverse_alias_map = context["inverse_alias_map"]
        column_groups = context["column_groups"]
        priority_present = set(context["priority_columns_present"])

        group_lookup: dict[str, str] = {}
        for group_name, columns in column_groups.items():
            for column in columns:
                group_lookup[column] = group_name

        records: list[dict[str, Any]] = []
        for position, column in enumerate(df.columns, start=1):
            records.append(
                {
                    "column_position": position,
                    "column": column,
                    "dtype": str(df[column].dtype),
                    "semantic_group": group_lookup.get(column, "unknown"),
                    "report_aliases": " | ".join(inverse_alias_map.get(column, [])) or "none",
                    "is_priority_column": column in priority_present or bool(inverse_alias_map.get(column)),
                    "is_transparency_column": column.startswith("score_")
                    or column in TRANSPARENCY_NUMERIC_COLUMNS
                    or column in TRANSPARENCY_CATEGORICAL_COLUMNS
                    or column in TRANSPARENCY_BINARY_COLUMNS,
                    "non_null_count": int(df[column].notna().sum()),
                    "null_count": int(df[column].isna().sum()),
                    "null_pct": round(float(df[column].isna().mean() * 100), 4),
                    "nunique_excluding_null": int(df[column].nunique(dropna=True)),
                }
            )

        return pd.DataFrame(records)
    except Exception as exc:
        log_message(f"Column inventory generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Column inventory step finished.")


@task(name="build_dtype_profile")
@timing_decorator
@validate
def build_dtype_profile(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    try:
        inverse_alias_map = context["inverse_alias_map"]
        group_lookup: dict[str, str] = {}
        for group_name, columns in context["column_groups"].items():
            for column in columns:
                group_lookup[column] = group_name

        dtype_profile = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(dtype) for dtype in df.dtypes],
                "semantic_group": [group_lookup.get(column, "unknown") for column in df.columns],
                "report_aliases": [" | ".join(inverse_alias_map.get(column, [])) or "none" for column in df.columns],
            }
        )
        return dtype_profile
    except Exception as exc:
        log_message(f"Dtype profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Dtype profile step finished.")


@task(name="build_null_profile")
@timing_decorator
@validate
def build_null_profile(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    try:
        group_lookup: dict[str, str] = {}
        for group_name, columns in context["column_groups"].items():
            for column in columns:
                group_lookup[column] = group_name

        null_profile = pd.DataFrame(
            {
                "column": df.columns,
                "semantic_group": [group_lookup.get(column, "unknown") for column in df.columns],
                "null_count": df.isna().sum().values,
                "null_pct": (df.isna().mean().values * 100).round(4),
                "non_null_count": df.notna().sum().values,
                "completeness_pct": (df.notna().mean().values * 100).round(4),
            }
        )
        return null_profile.sort_values(["null_pct", "null_count"], ascending=[False, False]).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Null profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Null profile step finished.")


@task(name="build_duplicate_profile")
@timing_decorator
@validate
def build_duplicate_profile(df: pd.DataFrame) -> pd.DataFrame:
    try:
        duplicate_records: list[dict[str, Any]] = []

        try:
            exact_duplicate_rows = int(df.duplicated().sum())
        except TypeError:
            hashable_df = build_hashable_frame(df)
            exact_duplicate_rows = int(hashable_df.duplicated().sum())

        duplicate_records.append(
            {
                "duplicate_type": "exact_all_columns",
                "subset_columns": "all_columns",
                "duplicate_rows": exact_duplicate_rows,
                "duplicate_groups_or_keys": None,
                "duplicate_pct": round((exact_duplicate_rows / len(df)) * 100, 4) if len(df) else 0.0,
            }
        )

        for subset in DUPLICATE_KEY_CANDIDATES:
            existing_subset = [column for column in subset if column in df.columns]
            if existing_subset != subset:
                continue

            subset_df = df[existing_subset].dropna(how="any")
            if subset_df.empty:
                duplicate_records.append(
                    {
                        "duplicate_type": "key_based",
                        "subset_columns": ", ".join(existing_subset),
                        "duplicate_rows": 0,
                        "duplicate_groups_or_keys": 0,
                        "duplicate_pct": 0.0,
                    }
                )
                continue

            try:
                duplicate_mask = subset_df.duplicated(subset=existing_subset, keep=False)
                duplicate_rows = int(duplicate_mask.sum())
                duplicate_groups = int(subset_df.loc[duplicate_mask, existing_subset].drop_duplicates().shape[0]) if duplicate_rows else 0
            except TypeError:
                hashable_subset_df = build_hashable_frame(subset_df, existing_subset)
                duplicate_mask = hashable_subset_df.duplicated(subset=existing_subset, keep=False)
                duplicate_rows = int(duplicate_mask.sum())
                duplicate_groups = int(hashable_subset_df.loc[duplicate_mask, existing_subset].drop_duplicates().shape[0]) if duplicate_rows else 0

            duplicate_records.append(
                {
                    "duplicate_type": "key_based",
                    "subset_columns": ", ".join(existing_subset),
                    "duplicate_rows": duplicate_rows,
                    "duplicate_groups_or_keys": duplicate_groups,
                    "duplicate_pct": round((duplicate_rows / len(df)) * 100, 4) if len(df) else 0.0,
                }
            )

        return pd.DataFrame(duplicate_records)
    except Exception as exc:
        log_message(f"Duplicate profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Duplicate profile step finished.")


@task(name="build_numeric_profile")
@timing_decorator
@validate
def build_numeric_profile(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    try:
        numeric_columns = context["column_groups"]["numeric"] + context["column_groups"]["binary_numeric"]
        records: list[dict[str, Any]] = []

        for column in numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce")
            non_null = series.dropna()

            if non_null.empty:
                records.append(
                    {
                        "column": column,
                        "non_null_count": 0,
                        "null_count": int(series.isna().sum()),
                        "null_pct": 100.0,
                        "nunique": 0,
                        "is_binary_like": is_binary_like_numeric(series),
                        "count": 0,
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "p01": np.nan,
                        "p05": np.nan,
                        "p25": np.nan,
                        "p50": np.nan,
                        "median": np.nan,
                        "p75": np.nan,
                        "p95": np.nan,
                        "p99": np.nan,
                        "max": np.nan,
                        "iqr": np.nan,
                        "lower_bound_iqr": np.nan,
                        "upper_bound_iqr": np.nan,
                        "outlier_count_iqr": 0,
                        "outlier_pct_iqr": 0.0,
                        "zero_count": 0,
                        "zero_pct": 0.0,
                    }
                )
                continue

            q1 = float(non_null.quantile(0.25))
            q3 = float(non_null.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)

            records.append(
                {
                    "column": column,
                    "non_null_count": int(non_null.shape[0]),
                    "null_count": int(series.isna().sum()),
                    "null_pct": round(float(series.isna().mean() * 100), 4),
                    "nunique": int(non_null.nunique(dropna=True)),
                    "is_binary_like": is_binary_like_numeric(series),
                    "count": int(non_null.shape[0]),
                    "mean": float(non_null.mean()),
                    "std": float(non_null.std()) if non_null.shape[0] > 1 else 0.0,
                    "min": float(non_null.min()),
                    "p01": float(non_null.quantile(0.01)),
                    "p05": float(non_null.quantile(0.05)),
                    "p25": q1,
                    "p50": float(non_null.quantile(0.50)),
                    "median": float(non_null.median()),
                    "p75": q3,
                    "p95": float(non_null.quantile(0.95)),
                    "p99": float(non_null.quantile(0.99)),
                    "max": float(non_null.max()),
                    "iqr": float(iqr),
                    "lower_bound_iqr": float(lower_bound),
                    "upper_bound_iqr": float(upper_bound),
                    "outlier_count_iqr": int(outlier_mask.sum()),
                    "outlier_pct_iqr": round(float(outlier_mask.mean() * 100), 4),
                    "zero_count": int(non_null.eq(0).sum()),
                    "zero_pct": round(float(non_null.eq(0).mean() * 100), 4),
                }
            )

        return pd.DataFrame(records).sort_values(["is_binary_like", "null_pct", "column"], ascending=[True, False, True]).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Numeric profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Numeric profile step finished.")


@task(name="build_categorical_profile")
@timing_decorator
@validate
def build_categorical_profile(df: pd.DataFrame, output_dirs: dict[str, Path], top_n_categories: int) -> pd.DataFrame:
    try:
        categorical_candidates = [
            column
            for column in df.columns
            if (
                pd.api.types.is_object_dtype(df[column])
                or pd.api.types.is_string_dtype(df[column])
                or isinstance(df[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(df[column])
            )
            and column not in TECHNICAL_IDENTIFIER_COLUMNS
            and column not in TEXT_COLUMNS
        ]

        records: list[dict[str, Any]] = []
        for column in categorical_candidates:
            series = df[column].map(make_hashable)
            frequencies = series.value_counts(dropna=False)
            relative_frequencies = series.value_counts(dropna=False, normalize=True)

            if frequencies.empty:
                continue

            frequency_table = pd.DataFrame(
                {
                    "category": [json_safe(value) for value in frequencies.index],
                    "count": frequencies.values,
                    "relative_pct": (relative_frequencies.values * 100).round(4),
                }
            )
            safe_write_csv(frequency_table, output_dirs["frequency_tables"] / f"{column}_frequency.csv")

            top_categories_preview = ", ".join(
                str(json_safe(value)) for value in frequencies.index[: min(top_n_categories, 5)]
            )

            records.append(
                {
                    "column": column,
                    "non_null_count": int(df[column].notna().sum()),
                    "null_count": int(df[column].isna().sum()),
                    "null_pct": round(float(df[column].isna().mean() * 100), 4),
                    "nunique_excluding_null": int(df[column].nunique(dropna=True)),
                    "top_category": json_safe(frequencies.index[0]),
                    "top_count": int(frequencies.iloc[0]),
                    "top_share_pct": round(float(relative_frequencies.iloc[0] * 100), 4),
                    "rare_categories_count": int((relative_frequencies[relative_frequencies < 0.01]).shape[0]),
                    "top_categories_preview": top_categories_preview,
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "column",
                    "non_null_count",
                    "null_count",
                    "null_pct",
                    "nunique_excluding_null",
                    "top_category",
                    "top_count",
                    "top_share_pct",
                    "rare_categories_count",
                    "top_categories_preview",
                ]
            )

        return pd.DataFrame(records).sort_values(
            ["nunique_excluding_null", "top_share_pct"],
            ascending=[False, False],
        ).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Categorical profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Categorical profile step finished.")


@task(name="build_datetime_profile")
@timing_decorator
@validate
def build_datetime_profile(df: pd.DataFrame, output_dirs: dict[str, Path], context: dict[str, Any]) -> pd.DataFrame:
    try:
        datetime_columns = context["column_groups"]["datetime"]
        records: list[dict[str, Any]] = []

        for column in datetime_columns:
            series = normalize_datetime_series(df[column])
            non_null = series.dropna()

            if non_null.empty:
                records.append(
                    {
                        "column": column,
                        "non_null_count": 0,
                        "null_count": int(series.isna().sum()),
                        "null_pct": 100.0,
                        "min_datetime": None,
                        "max_datetime": None,
                        "min_year": None,
                        "max_year": None,
                        "distinct_years": 0,
                        "most_common_year": None,
                        "most_common_month": None,
                    }
                )
                continue

            non_null_naive = remove_datetime_timezone(non_null)
            year_counts = non_null_naive.dt.year.value_counts().sort_index()
            month_counts = non_null_naive.dt.to_period("M").astype(str).value_counts().sort_index()

            year_frequency_df = pd.DataFrame({"year": year_counts.index, "count": year_counts.values})
            month_frequency_df = pd.DataFrame({"year_month": month_counts.index, "count": month_counts.values})
            safe_write_csv(year_frequency_df, output_dirs["frequency_tables"] / f"{column}_year_frequency.csv")
            safe_write_csv(month_frequency_df, output_dirs["frequency_tables"] / f"{column}_month_frequency.csv")

            records.append(
                {
                    "column": column,
                    "non_null_count": int(non_null.shape[0]),
                    "null_count": int(series.isna().sum()),
                    "null_pct": round(float(series.isna().mean() * 100), 4),
                    "min_datetime": json_safe(non_null.min()),
                    "max_datetime": json_safe(non_null.max()),
                    "min_year": int(non_null_naive.dt.year.min()),
                    "max_year": int(non_null_naive.dt.year.max()),
                    "distinct_years": int(non_null_naive.dt.year.nunique()),
                    "most_common_year": int(year_counts.idxmax()) if not year_counts.empty else None,
                    "most_common_month": str(month_counts.idxmax()) if not month_counts.empty else None,
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "column",
                    "non_null_count",
                    "null_count",
                    "null_pct",
                    "min_datetime",
                    "max_datetime",
                    "min_year",
                    "max_year",
                    "distinct_years",
                    "most_common_year",
                    "most_common_month",
                ]
            )

        return pd.DataFrame(records).sort_values(["non_null_count", "column"], ascending=[False, True]).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Datetime profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Datetime profile step finished.")


@task(name="build_correlation_outputs")
@timing_decorator
@validate
def build_correlation_outputs(
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
    correlation_max_features: int,
) -> pd.DataFrame:
    try:
        numeric_candidates = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
            and not pd.api.types.is_bool_dtype(df[column])
        ]

        eligible_columns = []
        for column in numeric_candidates:
            series = pd.to_numeric(df[column], errors="coerce")
            non_null_pct = float(series.notna().mean())
            nunique = int(series.nunique(dropna=True))

            if non_null_pct < 0.60 or nunique < 5 or is_binary_like_numeric(series):
                continue
            eligible_columns.append(column)

        selected_columns = select_priority_columns(
            eligible_columns,
            PRIORITY_NUMERIC_COLUMNS,
            correlation_max_features,
        )

        if len(selected_columns) < 2:
            log_message("Correlation matrix was skipped because fewer than two eligible numeric variables were found.", level="warning")
            return pd.DataFrame(columns=["variable"])

        correlation_matrix = df[selected_columns].corr(numeric_only=True)
        if PLOTTING_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Heatmap - SECOP Descriptive Analysis")
            safe_save_plot(fig, output_dirs["correlation_plots"] / "correlation_heatmap.png")

        return correlation_matrix.reset_index().rename(columns={"index": "variable"})
    except Exception as exc:
        log_message(f"Correlation output generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Correlation output step finished.")


@task(name="build_transparency_profile")
@timing_decorator
@validate
def build_transparency_profile(df: pd.DataFrame) -> pd.DataFrame:
    try:
        records: list[dict[str, Any]] = []

        for column in TRANSPARENCY_NUMERIC_COLUMNS:
            if column not in df.columns:
                continue

            series = pd.to_numeric(df[column], errors="coerce")
            non_null = series.dropna()
            if non_null.empty:
                continue

            records.append(
                {
                    "column": column,
                    "profile_type": "numeric",
                    "non_null_count": int(non_null.shape[0]),
                    "null_count": int(series.isna().sum()),
                    "null_pct": round(float(series.isna().mean() * 100), 4),
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "min": float(non_null.min()),
                    "p25": float(non_null.quantile(0.25)),
                    "p75": float(non_null.quantile(0.75)),
                    "max": float(non_null.max()),
                    "top_category": None,
                    "top_share_pct": None,
                    "positive_rate_pct": None,
                }
            )

        for column in TRANSPARENCY_CATEGORICAL_COLUMNS:
            if column not in df.columns:
                continue

            series = df[column].map(make_hashable)
            frequencies = series.value_counts(dropna=False, normalize=True)
            absolute_frequencies = series.value_counts(dropna=False)
            if frequencies.empty:
                continue

            records.append(
                {
                    "column": column,
                    "profile_type": "categorical",
                    "non_null_count": int(df[column].notna().sum()),
                    "null_count": int(df[column].isna().sum()),
                    "null_pct": round(float(df[column].isna().mean() * 100), 4),
                    "mean": None,
                    "median": None,
                    "min": None,
                    "p25": None,
                    "p75": None,
                    "max": None,
                    "top_category": json_safe(absolute_frequencies.index[0]),
                    "top_share_pct": round(float(frequencies.iloc[0] * 100), 4),
                    "positive_rate_pct": None,
                }
            )

        for column in TRANSPARENCY_BINARY_COLUMNS:
            if column not in df.columns:
                continue

            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().sum() == 0:
                continue

            records.append(
                {
                    "column": column,
                    "profile_type": "binary_indicator",
                    "non_null_count": int(series.notna().sum()),
                    "null_count": int(series.isna().sum()),
                    "null_pct": round(float(series.isna().mean() * 100), 4),
                    "mean": None,
                    "median": None,
                    "min": None,
                    "p25": None,
                    "p75": None,
                    "max": None,
                    "top_category": None,
                    "top_share_pct": None,
                    "positive_rate_pct": round(float(series.fillna(0).gt(0).mean() * 100), 4),
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "column",
                    "profile_type",
                    "non_null_count",
                    "null_count",
                    "null_pct",
                    "mean",
                    "median",
                    "min",
                    "p25",
                    "p75",
                    "max",
                    "top_category",
                    "top_share_pct",
                    "positive_rate_pct",
                ]
            )

        return pd.DataFrame(records).sort_values(["profile_type", "column"]).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Transparency profile generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Transparency profile step finished.")


@task(name="generate_visualizations")
@timing_decorator
@validate
def generate_visualizations(
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    datetime_summary: pd.DataFrame,
    max_plots: int,
    top_n_categories: int,
) -> dict[str, list[str]]:
    generated_plots = {
        "numeric_plots": [],
        "categorical_plots": [],
        "temporal_plots": [],
    }

    try:
        if not PLOTTING_AVAILABLE:
            log_message("Plot generation skipped because plotting dependencies are not available.", level="warning")
            return generated_plots

        numeric_candidates = numeric_summary.loc[
            (numeric_summary["non_null_count"] > 0)
            & (~numeric_summary["is_binary_like"])
            & (numeric_summary["nunique"] >= 5),
            "column",
        ].tolist()
        selected_numeric_columns = select_priority_columns(numeric_candidates, PRIORITY_NUMERIC_COLUMNS, max_plots)

        for column in selected_numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty:
                continue
            if len(series) > PLOT_SAMPLE_SIZE:
                series = series.sample(PLOT_SAMPLE_SIZE, random_state=42)

            fig, ax = plt.subplots()
            sns.histplot(series, bins=40, kde=True, ax=ax, color="#2a9d8f")
            ax.set_title(f"Histogram - {column}")
            ax.set_xlabel(column)
            numeric_hist_path = output_dirs["numeric_plots"] / f"{column}_hist.png"
            safe_save_plot(fig, numeric_hist_path)
            generated_plots["numeric_plots"].append(str(numeric_hist_path))

            fig, ax = plt.subplots()
            sns.boxplot(x=series, ax=ax, color="#e76f51")
            ax.set_title(f"Boxplot - {column}")
            ax.set_xlabel(column)
            numeric_box_path = output_dirs["numeric_plots"] / f"{column}_box.png"
            safe_save_plot(fig, numeric_box_path)
            generated_plots["numeric_plots"].append(str(numeric_box_path))

        categorical_candidates = categorical_summary.loc[
            categorical_summary["nunique_excluding_null"].between(2, max(top_n_categories * 2, 10), inclusive="both"),
            "column",
        ].tolist()
        selected_categorical_columns = select_priority_columns(
            categorical_candidates,
            PRIORITY_CATEGORICAL_COLUMNS,
            max_plots,
        )

        for column in selected_categorical_columns:
            frequency_table_path = output_dirs["frequency_tables"] / f"{column}_frequency.csv"
            if not frequency_table_path.exists():
                continue
            frequency_table = pd.read_csv(frequency_table_path).head(top_n_categories)
            if frequency_table.empty:
                continue

            fig, ax = plt.subplots(figsize=(12, max(5, 0.4 * len(frequency_table))))
            sns.barplot(data=frequency_table, y="category", x="count", ax=ax, color="#457b9d")
            ax.set_title(f"Top Categories - {column}")
            ax.set_xlabel("Count")
            ax.set_ylabel(column)
            categorical_plot_path = output_dirs["categorical_plots"] / f"{column}_top_categories.png"
            safe_save_plot(fig, categorical_plot_path)
            generated_plots["categorical_plots"].append(str(categorical_plot_path))

        if not datetime_summary.empty:
            datetime_candidates = datetime_summary.loc[datetime_summary["non_null_count"] > 0, "column"].tolist()
            selected_datetime_columns = select_priority_columns(
                datetime_candidates,
                PRIORITY_DATETIME_COLUMNS,
                max_plots,
            )

            for column in selected_datetime_columns:
                series = normalize_datetime_series(df[column]).dropna()
                if series.empty:
                    continue
                series = remove_datetime_timezone(series)

                year_counts = series.dt.year.value_counts().sort_index()
                if not year_counts.empty:
                    fig, ax = plt.subplots()
                    year_counts.plot(kind="bar", ax=ax, color="#264653")
                    ax.set_title(f"Processes by Year - {column}")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Count")
                    temporal_year_plot_path = output_dirs["temporal_plots"] / f"{column}_by_year.png"
                    safe_save_plot(fig, temporal_year_plot_path)
                    generated_plots["temporal_plots"].append(str(temporal_year_plot_path))

                month_counts = series.dt.to_period("M").astype(str).value_counts().sort_index()
                if not month_counts.empty:
                    fig, ax = plt.subplots(figsize=(14, 6))
                    month_counts.plot(ax=ax, color="#1d3557")
                    ax.set_title(f"Processes by Month - {column}")
                    ax.set_xlabel("Year-Month")
                    ax.set_ylabel("Count")
                    ax.tick_params(axis="x", rotation=45)
                    temporal_month_plot_path = output_dirs["temporal_plots"] / f"{column}_by_month.png"
                    safe_save_plot(fig, temporal_month_plot_path)
                    generated_plots["temporal_plots"].append(str(temporal_month_plot_path))

        return generated_plots
    except Exception as exc:
        log_message(f"Visualization generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Visualization generation step finished.")


@task(name="write_executive_summary")
@timing_decorator
@validate
def write_executive_summary(
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
    overview: pd.DataFrame,
    null_profile: pd.DataFrame,
    duplicates_summary: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    datetime_summary: pd.DataFrame,
    transparency_summary: pd.DataFrame,
    context: dict[str, Any],
) -> str:
    try:
        overview_row = overview.iloc[0].to_dict() if not overview.empty else {}
        top_nulls = null_profile.head(10)
        numeric_lookup = numeric_summary.set_index("column") if not numeric_summary.empty else None
        datetime_lookup = datetime_summary.set_index("column") if not datetime_summary.empty else None
        transparency_lookup = transparency_summary.set_index("column") if not transparency_summary.empty else None

        exact_duplicates = 0
        if not duplicates_summary.empty:
            exact_duplicates_series = duplicates_summary.loc[
                duplicates_summary["duplicate_type"] == "exact_all_columns",
                "duplicate_rows",
            ]
            if not exact_duplicates_series.empty:
                exact_duplicates = int(exact_duplicates_series.iloc[0])

        duplicate_key_lines: list[str] = []
        for _, row in duplicates_summary.loc[duplicates_summary["duplicate_type"] == "key_based"].iterrows():
            duplicate_key_lines.append(
                f"- `{row['subset_columns']}`: {int(row['duplicate_rows']):,} duplicate rows across {int(row['duplicate_groups_or_keys'] or 0):,} repeated keys."
            )

        numeric_lines: list[str] = []
        for column in ["precio_base", "total_respuestas", "transparency_score"]:
            if numeric_lookup is not None and column in numeric_lookup.index:
                row = numeric_lookup.loc[column]
                numeric_lines.append(
                    f"- `{column}`: mean {row['mean']:.4f}, median {row['median']:.4f}, p95 {row['p95']:.4f}, IQR outliers {int(row['outlier_count_iqr']):,} ({row['outlier_pct_iqr']:.2f}%)."
                )

        temporal_lines: list[str] = []
        for alias_name in ["fecha_de_publicacion", "fecha_de_recepcion", "fecha_de_apertura"]:
            actual_column = context["alias_map"].get(alias_name)
            if datetime_lookup is not None and actual_column and actual_column in datetime_lookup.index:
                row = datetime_lookup.loc[actual_column]
                temporal_lines.append(
                    f"- `{alias_name}` mapped to `{actual_column}` spans from {row['min_datetime']} to {row['max_datetime']} with {int(row['non_null_count']):,} non-null records."
                )

        transparency_lines: list[str] = []
        for column in ["score_completitud", "score_trazabilidad", "score_temporal", "score_competencia", "transparency_score"]:
            if transparency_lookup is not None and column in transparency_lookup.index:
                row = transparency_lookup.loc[column]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if row["profile_type"] == "numeric":
                    transparency_lines.append(
                        f"- `{column}`: mean {row['mean']:.4f}, median {row['median']:.4f}, min {row['min']:.4f}, max {row['max']:.4f}."
                    )

        for column in ["nivel_riesgo_transparencia", "confianza_label"]:
            if transparency_lookup is not None and column in transparency_lookup.index:
                row = transparency_lookup.loc[column]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                transparency_lines.append(
                    f"- `{column}`: most common category `{row['top_category']}` with {row['top_share_pct']:.2f}% share."
                )

        top_null_lines = [
            f"- `{row.column}`: {int(row.null_count):,} nulls ({row.null_pct:.2f}%)."
            for row in top_nulls.itertuples(index=False)
        ]

        categorical_lines: list[str] = []
        for row in categorical_summary.head(5).itertuples(index=False):
            categorical_lines.append(
                f"- `{row.column}`: {int(row.nunique_excluding_null):,} categories; top category `{row.top_category}` represents {row.top_share_pct:.2f}%."
            )

        summary_content = "\n".join(
            [
                "# Executive Summary - SECOP Descriptive Analysis",
                "",
                "## Context",
                "This descriptive profile was generated over `processed/datos_feature_engineering.parquet` to support evidence on traceability, completeness, coherence, competition, and information quality in public procurement processes aligned with ODS 16.6.",
                "",
                "## Dataset Overview",
                f"- Rows: {int(overview_row.get('n_rows', 0)):,}",
                f"- Columns: {int(overview_row.get('n_columns', 0)):,}",
                f"- Memory usage: {float(overview_row.get('memory_usage_mb', 0.0)):.2f} MB",
                f"- Numeric columns: {int(overview_row.get('n_numeric_columns', 0)):,}",
                f"- Binary numeric columns: {int(overview_row.get('n_binary_numeric_columns', 0)):,}",
                f"- Categorical columns: {int(overview_row.get('n_categorical_columns', 0)):,}",
                f"- Datetime columns: {int(overview_row.get('n_datetime_columns', 0)):,}",
                "",
                "## Data Quality",
                f"- Exact duplicate rows across all columns: {exact_duplicates:,}.",
                *duplicate_key_lines,
                *top_null_lines,
                "",
                "## Descriptive Highlights",
                *numeric_lines,
                *categorical_lines,
                "",
                "## Temporal Coverage",
                *(temporal_lines or ["- No usable datetime columns were available for temporal profiling."]),
                "",
                "## Transparency and Institutional Risk Signals",
                *(transparency_lines or ["- Transparency-specific variables were not available in this dataset version."]),
                "",
                "## Risks for Next Stages",
                "- High-null columns may weaken downstream modeling and can reduce reliability in institutional transparency indicators.",
                "- Repeated process identifiers should be reviewed before using the data for entity-level benchmarking or process-level aggregation.",
                "- Strong skewness and outliers in monetary and participation variables suggest using robust statistics and capped transformations in later stages.",
                "- Temporal gaps in publication, reception, and opening dates can affect traceability analyses and timeline-based controls.",
            ]
        )

        output_path = output_dirs["base"] / "executive_summary.md"
        safe_write_markdown(summary_content, output_path)
        return str(output_path)
    except Exception as exc:
        log_message(f"Executive summary generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Executive summary step finished.")


def build_run_metadata(
    output_dirs: dict[str, Path],
    environment_status: dict[str, Any],
    context: dict[str, Any],
    generated_plots: dict[str, list[str]],
    artifacts: dict[str, str],
    overview: pd.DataFrame,
    null_profile: pd.DataFrame,
    duplicates_summary: pd.DataFrame,
) -> dict[str, Any]:
    overview_row = overview.iloc[0].to_dict() if not overview.empty else {}
    top_nulls = null_profile.head(10).to_dict(orient="records") if not null_profile.empty else []

    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(output_dirs["input_path"]),
        "output_dir": str(output_dirs["base"]),
        "project_root": str(output_dirs["project_root"]),
        "plotting_available": environment_status.get("plotting_available", PLOTTING_AVAILABLE),
        "plotting_import_error": environment_status.get("plotting_import_error", PLOTTING_IMPORT_ERROR),
        "prefect_available": environment_status.get("prefect_available", PREFECT_AVAILABLE),
        "dependencies": environment_status.get("bootstrap_status", BOOTSTRAP_STATUS),
        "dataset_overview": {key: json_safe(value) for key, value in overview_row.items()},
        "priority_columns_present": context.get("priority_columns_present", []),
        "priority_columns_missing": context.get("priority_columns_missing", []),
        "alias_map": context.get("alias_map", {}),
        "conversion_summary": context.get("conversion_summary", []),
        "top_null_columns": top_nulls,
        "duplicates_summary": duplicates_summary.to_dict(orient="records"),
        "generated_plots": generated_plots,
        "artifacts": artifacts,
    }


@flow(name="secop_descriptive_analysis_flow")
def secop_descriptive_analysis_flow(
    input_path: str = str(DEFAULT_INPUT_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    top_n_categories: int = 20,
    max_plots: int = 20,
    correlation_max_features: int = 20,
) -> dict[str, str]:
    try:
        output_dirs = resolve_paths_and_create_output_dirs(input_path=input_path, output_dir=output_dir)
        environment_status = bootstrap_environment(requirements_path=output_dirs["requirements_path"])
        raw_df = load_parquet_dataset(parquet_path=output_dirs["input_path"])
        prepared_df, context = prepare_dataframe_for_analysis(df=raw_df)

        overview = build_dataset_overview(df=prepared_df, context=context)
        column_inventory = build_column_inventory(df=prepared_df, context=context)
        dtype_profile = build_dtype_profile(df=prepared_df, context=context)
        null_profile = build_null_profile(df=prepared_df, context=context)
        duplicates_summary = build_duplicate_profile(df=prepared_df)
        numeric_summary = build_numeric_profile(df=prepared_df, context=context)
        categorical_summary = build_categorical_profile(
            df=prepared_df,
            output_dirs=output_dirs,
            top_n_categories=top_n_categories,
        )
        datetime_summary = build_datetime_profile(
            df=prepared_df,
            output_dirs=output_dirs,
            context=context,
        )
        correlation_matrix = build_correlation_outputs(
            df=prepared_df,
            output_dirs=output_dirs,
            correlation_max_features=correlation_max_features,
        )
        transparency_summary = build_transparency_profile(df=prepared_df)
        generated_plots = generate_visualizations(
            df=prepared_df,
            output_dirs=output_dirs,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            datetime_summary=datetime_summary,
            max_plots=max_plots,
            top_n_categories=top_n_categories,
        )

        artifacts: dict[str, str] = {}
        dataframe_outputs = {
            "dataset_overview.csv": overview,
            "column_inventory.csv": column_inventory,
            "dtypes.csv": dtype_profile,
            "null_profile.csv": null_profile,
            "duplicates_summary.csv": duplicates_summary,
            "numeric_summary.csv": numeric_summary,
            "categorical_summary.csv": categorical_summary,
            "datetime_summary.csv": datetime_summary,
            "transparency_summary.csv": transparency_summary,
            "correlation_matrix.csv": correlation_matrix,
        }

        for filename, dataframe in dataframe_outputs.items():
            output_path = output_dirs["base"] / filename
            safe_write_csv(dataframe, output_path)
            artifacts[filename] = str(output_path)

        summary_path = write_executive_summary(
            df=prepared_df,
            output_dirs=output_dirs,
            overview=overview,
            null_profile=null_profile,
            duplicates_summary=duplicates_summary,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            datetime_summary=datetime_summary,
            transparency_summary=transparency_summary,
            context=context,
        )
        artifacts["executive_summary.md"] = summary_path

        run_metadata = build_run_metadata(
            output_dirs=output_dirs,
            environment_status=environment_status,
            context=context,
            generated_plots=generated_plots,
            artifacts=artifacts,
            overview=overview,
            null_profile=null_profile,
            duplicates_summary=duplicates_summary,
        )
        run_metadata_path = output_dirs["base"] / "run_metadata.json"
        safe_write_json(run_metadata, run_metadata_path)
        artifacts["run_metadata.json"] = str(run_metadata_path)

        return artifacts
    except Exception as exc:
        log_message(f"Descriptive SECOP flow failed: {exc}", level="error")
        raise
    finally:
        log_message("Descriptive SECOP flow finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run descriptive analysis for SECOP feature engineering dataset.")
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where descriptive outputs will be stored.",
    )
    parser.add_argument(
        "--top-n-categories",
        default=20,
        type=int,
        help="Maximum number of top categories displayed in categorical charts.",
    )
    parser.add_argument(
        "--max-plots",
        default=20,
        type=int,
        help="Maximum number of variables plotted per plot family.",
    )
    parser.add_argument(
        "--correlation-max-features",
        default=20,
        type=int,
        help="Maximum number of numeric variables included in the correlation matrix.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    secop_descriptive_analysis_flow(
        input_path=cli_args.input_path,
        output_dir=cli_args.output_dir,
        top_n_categories=cli_args.top_n_categories,
        max_plots=cli_args.max_plots,
        correlation_max_features=cli_args.correlation_max_features,
    )
