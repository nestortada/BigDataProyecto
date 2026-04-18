from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import wraps
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable


def normalize_workspace_path(path: Path) -> Path:
    path_str = str(path)
    if path_str.startswith("/mnt/c/"):
        return Path(path_str.lower())
    return path


BOOTSTRAP_ROOT = normalize_workspace_path(Path(__file__).resolve().parents[2])
LOCAL_PACKAGE_ROOT = BOOTSTRAP_ROOT / ".python_packages"
RUNTIME_TAG = f"py{sys.version_info.major}{sys.version_info.minor}-{sys.platform}-{platform.machine().lower()}"
LOCAL_PACKAGE_DIR = LOCAL_PACKAGE_ROOT / RUNTIME_TAG
TMP_LOCAL_PACKAGE_DIR = Path("/tmp") / "secop_python_packages" / RUNTIME_TAG
ACTIVE_LOCAL_PACKAGE_DIR = LOCAL_PACKAGE_DIR


def register_local_package_dir(package_dir: Path | None = None) -> None:
    target_dir = package_dir or ACTIVE_LOCAL_PACKAGE_DIR
    if target_dir.exists() and str(target_dir) not in sys.path:
        # Keep the active virtualenv ahead of the local fallback directory.
        sys.path.append(str(target_dir))


def resolve_local_package_dir() -> Path:
    global ACTIVE_LOCAL_PACKAGE_DIR

    candidate_dirs = [ACTIVE_LOCAL_PACKAGE_DIR, LOCAL_PACKAGE_DIR, TMP_LOCAL_PACKAGE_DIR]
    last_error: Exception | None = None

    for candidate_dir in candidate_dirs:
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            ACTIVE_LOCAL_PACKAGE_DIR = candidate_dir
            register_local_package_dir(candidate_dir)
            return candidate_dir
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("No writable local package directory was available.")


register_local_package_dir()

DEPENDENCY_MAP: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
}

OPTIONAL_INSTALL_DEPENDENCY_MAP: dict[str, str] = {
    "statsmodels": "statsmodels",
}

PROJECT_MARKERS = {"Code", "Data", "Processed"}
DEFAULT_INPUT_PATH = Path("Data/Processed/Limpieza/datos_feature_engineering.parquet")
DEFAULT_OUTPUT_DIR = Path("Data/Processed/Inferencial")
DEFAULT_INPUT_CANDIDATES = [
    Path("Data/Processed/Limpieza/datos_feature_engineering.parquet"),
    Path("processed/datos_feature_engineering.parquet"),
    Path("Processed/Limpieza/datos_feature_engineering.parquet"),
]

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_TOP_N_CATEGORIES = 8
DEFAULT_MAX_GROUP_COMPARISONS = 25
DEFAULT_MAX_CORRELATION_FEATURES = 15
DEFAULT_PLOT_SAMPLE_SIZE = 100_000
DEFAULT_NORMALITY_SAMPLE_SIZE = 5_000
DEFAULT_MODEL_SAMPLE_SIZE = 120_000
DEFAULT_MIN_GROUP_SIZE = 30
LARGE_SAMPLE_WARNING_THRESHOLD = 5_000

ALIAS_CANDIDATES: dict[str, list[str]] = {
    "descripcion_del_procedimiento": ["descripcion_del_procedimiento", "descripci_n_del_procedimiento"],
    "fecha_de_publicacion": ["fecha_de_publicacion", "fecha_de_publicacion_del"],
    "fecha_de_recepcion": ["fecha_de_recepcion", "fecha_de_recepcion_de"],
    "fecha_de_apertura": ["fecha_de_apertura", "fecha_de_apertura_de_respuesta"],
}

PRIORITY_OUTCOME_COLUMNS = [
    "transparency_score",
    "precio_base_log",
    "duracion_dias",
    "total_respuestas",
    "total_interes_oferentes",
    "score_temporal",
    "score_competencia",
]

PRIORITY_GROUP_COLUMNS = [
    "ordenentidad",
    "departamento_entidad",
    "fase",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "justificaci_n_modalidad_de",
]

PRIORITY_NUMERIC_COLUMNS = [
    "transparency_score",
    "precio_base_log",
    "precio_base",
    "duracion_dias",
    "duracion",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "respuestas_al_procedimiento",
    "respuestas_externas",
    "conteo_de_respuestas_a_ofertas",
    "proveedores_unicos_con",
    "numero_de_lotes",
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
    "evidence_coverage",
    "margin_to_threshold",
]

PRIORITY_CATEGORICAL_COLUMNS = [
    "ordenentidad",
    "departamento_entidad",
    "ciudad_entidad",
    "fase",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "justificaci_n_modalidad_de",
    "nivel_riesgo_transparencia",
    "confianza_label",
    "competencia_reportada",
    "anomalia_temporal",
    "fue_duplicado_logico",
]

PRIORITY_BINARY_COLUMNS = [
    "riesgo_baja_transparencia",
    "flag_hubo_participacion",
    "flag_hubo_competencia_minima",
    "competencia_reportada",
    "anomalia_temporal",
    "fue_duplicado_logico",
    "flag_evidencia_completitud_suficiente",
    "flag_evidencia_trazabilidad_suficiente",
    "flag_evidencia_temporal_suficiente",
    "flag_evidencia_competencia_suficiente",
]

PRIORITY_CATEGORICAL_ASSOC_COLUMNS = [
    "ordenentidad",
    "departamento_entidad",
    "fase",
    "modalidad_de_contratacion",
    "tipo_de_contrato",
    "justificaci_n_modalidad_de",
    "nivel_riesgo_transparencia",
    "confianza_label",
    "riesgo_baja_transparencia",
    "competencia_reportada",
    "anomalia_temporal",
    "fue_duplicado_logico",
]

PRIORITY_NUMERIC_ASSOC_COLUMNS = [
    "transparency_score",
    "precio_base_log",
    "duracion_dias",
    "dias_publicacion_a_recepcion",
    "dias_recepcion_a_apertura",
    "dias_publicacion_a_apertura",
    "total_respuestas",
    "total_interes_oferentes",
    "proveedores_invitados",
    "proveedores_con_invitacion",
    "conteo_de_respuestas_a_ofertas",
    "numero_de_lotes",
    "score_temporal",
    "score_competencia",
    "longitud_descripcion",
]

MODEL_CONTINUOUS_PREDICTORS = [
    "precio_base_log",
    "duracion_dias",
    "numero_de_lotes",
    "total_respuestas",
    "total_interes_oferentes",
    "longitud_descripcion",
]

MODEL_BINARY_PREDICTORS = [
    "fue_duplicado_logico",
    "anomalia_temporal",
]

MODEL_CATEGORICAL_PREDICTORS = [
    "ordenentidad",
    "departamento_entidad",
    "tipo_de_contrato",
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
    "nombre_del_procedimiento",
}

POST_RESULT_OR_LEAKAGE_COLUMNS = {
    "estado_del_procedimiento",
    "id_estado_del_procedimiento",
    "estado_de_apertura_del_proceso",
    "estado_resumen",
    "fecha_de_ultima_publicaci",
    "fecha_de_apertura_efectiva",
    "fecha_adjudicacion",
    "adjudicado",
    "id_adjudicacion",
    "valor_total_adjudicacion",
    "nombre_del_proveedor",
    "nit_del_proveedor_adjudicado",
    "nombre_del_adjudicador",
}

MODEL_EXCLUDED_COLUMNS = {
    "score_completitud",
    "score_trazabilidad_base",
    "score_trazabilidad",
    "score_temporal",
    "score_competencia",
    "transparency_score",
    "riesgo_baja_transparencia",
    "evidence_coverage",
    "margin_to_threshold",
    "nivel_riesgo_transparencia",
    "confianza_label",
}.union(POST_RESULT_OR_LEAKAGE_COLUMNS)

TAUTOLOGICAL_CORRELATION_PAIRS = {
    frozenset({"transparency_score", "score_completitud"}),
    frozenset({"transparency_score", "score_trazabilidad"}),
    frozenset({"transparency_score", "score_trazabilidad_base"}),
    frozenset({"transparency_score", "score_temporal"}),
    frozenset({"transparency_score", "score_competencia"}),
    frozenset({"transparency_score", "evidence_coverage"}),
    frozenset({"transparency_score", "margin_to_threshold"}),
}

warnings.filterwarnings("ignore", category=FutureWarning)


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
    packages_list = [package for package in packages if package]

    try:
        target_dir = resolve_local_package_dir()
        register_local_package_dir(target_dir)

        for package in packages_list:
            try:
                bootstrap_log(f"Installing missing package: {package}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--target", str(target_dir), package],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                install_status[package] = True
            except Exception as exc:
                bootstrap_log(f"Package installation failed for {package}: {exc}")
                install_status[package] = False

        importlib.invalidate_caches()
        register_local_package_dir(target_dir)
        return install_status
    except Exception as exc:
        bootstrap_log(f"Error while installing missing packages: {exc}")
        for package in packages_list:
            install_status.setdefault(package, False)
        return install_status
    finally:
        bootstrap_log("Missing package installation finished.")


def ensure_runtime_dependencies(requirements_path: Path) -> dict[str, Any]:
    dependency_status: dict[str, bool] = {}
    missing_dependencies: list[tuple[str, str]] = []

    try:
        packages_to_register = list(DEPENDENCY_MAP.values()) + list(OPTIONAL_INSTALL_DEPENDENCY_MAP.values())
        appended_packages = update_requirements_file(requirements_path, packages_to_register)
        for module_name, package_name in {**DEPENDENCY_MAP, **OPTIONAL_INSTALL_DEPENDENCY_MAP}.items():
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
    from scipy import stats

    SCIPY_AVAILABLE = True
    SCIPY_IMPORT_ERROR = None
except Exception as exc:
    stats = None
    SCIPY_AVAILABLE = False
    SCIPY_IMPORT_ERROR = str(exc)

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
    import statsmodels.api as sm

    STATSMODELS_AVAILABLE = True
    STATSMODELS_IMPORT_ERROR = None
except Exception as exc:
    sm = None
    STATSMODELS_AVAILABLE = False
    STATSMODELS_IMPORT_ERROR = str(exc)

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
            return normalize_workspace_path(candidate)
    return normalize_workspace_path(start)


PROJECT_ROOT = find_project_root(normalize_workspace_path(Path.cwd()))


def resolve_input_parquet_path(project_root: Path, input_path: str | Path) -> Path:
    requested_path = Path(input_path)
    attempted_paths: list[Path] = []

    candidate_paths: list[Path] = []
    if requested_path.is_absolute():
        candidate_paths.append(normalize_workspace_path(requested_path))
    else:
        candidate_paths.append(normalize_workspace_path(project_root / requested_path))

    for default_candidate in DEFAULT_INPUT_CANDIDATES:
        resolved_candidate = default_candidate if default_candidate.is_absolute() else project_root / default_candidate
        resolved_candidate = normalize_workspace_path(resolved_candidate)
        if resolved_candidate not in candidate_paths:
            candidate_paths.append(resolved_candidate)

    for candidate_path in candidate_paths:
        attempted_paths.append(candidate_path)
        if candidate_path.exists():
            if candidate_path != candidate_paths[0]:
                log_message(f"Input parquet not found at the requested path. Using fallback path: {candidate_path}")
            return candidate_path

    if requested_path.name.endswith(".parquet"):
        recursive_matches = sorted(project_root.rglob(requested_path.name))
        for match in recursive_matches:
            normalized_match = normalize_workspace_path(match)
            attempted_paths.append(normalized_match)
            if normalized_match.exists():
                log_message(f"Input parquet not found at the requested path. Using recursive fallback path: {normalized_match}")
                return normalized_match

    attempted_paths_text = "\n".join(f"- {path}" for path in attempted_paths)
    raise FileNotFoundError(
        "Input parquet file was not found. Attempted paths:\n"
        f"{attempted_paths_text}"
    )


def build_logger() -> logging.Logger:
    logger = logging.getLogger("secop_inferential_analysis")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[SECOP INFERENCIAL] %(levelname)s - %(message)s"))
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
                    "assumptions",
                    "confidence_intervals",
                    "group_comparisons",
                    "categorical_associations",
                    "numeric_associations",
                    "models",
                    "plots",
                    "group_plots",
                    "confidence_plots",
                    "correlation_plots",
                    "model_plots",
                    "contingency_tables",
                }
                if not isinstance(arg_value, dict):
                    raise TypeError("'output_dirs' must be a dictionary.")
                missing_keys = required_keys.difference(arg_value.keys())
                if missing_keys:
                    raise ValueError(f"'output_dirs' is missing required keys: {sorted(missing_keys)}")

            if arg_name in {
                "top_n_categories",
                "max_group_comparisons",
                "max_correlation_features",
                "plot_sample_size",
                "normality_sample_size",
                "model_sample_size",
                "min_group_size",
            }:
                if not isinstance(arg_value, int) or arg_value <= 0:
                    raise ValueError(f"Argument '{arg_name}' must be a positive integer.")

            if arg_name == "confidence_level":
                if not isinstance(arg_value, float | int) or not (0 < float(arg_value) < 1):
                    raise ValueError("'confidence_level' must be between 0 and 1.")

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
    if pd.api.types.is_bool_dtype(series):
        return True
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


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        path = normalize_workspace_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        log_message(f"Failed to write CSV '{path}': {exc}", level="error")
        raise
    finally:
        log_message(f"CSV write finished: {path.name}")


def safe_write_json(payload: dict[str, Any], path: Path) -> None:
    try:
        path = normalize_workspace_path(path)
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
        path = normalize_workspace_path(path)
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
        path = normalize_workspace_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as exc:
        log_message(f"Failed to save plot '{path}': {exc}", level="error")
        raise
    finally:
        if PLOTTING_AVAILABLE and fig is not None:
            plt.close(fig)


def collapse_categories(series: pd.Series, top_n_categories: int) -> tuple[pd.Series, bool, list[str]]:
    cleaned = clean_string_series(series)
    frequencies = cleaned.value_counts(dropna=True)
    if frequencies.empty or frequencies.shape[0] <= top_n_categories:
        return cleaned, False, frequencies.index.astype(str).tolist()

    keep_categories = frequencies.index[:top_n_categories].astype(str).tolist()
    collapsed = cleaned.where(cleaned.isin(keep_categories), other="Other")
    return collapsed, True, keep_categories


def calculate_t_confidence_interval(series: pd.Series, confidence_level: float) -> tuple[float, float, float]:
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    if clean_series.empty:
        return (np.nan, np.nan, np.nan)

    sample_size = clean_series.shape[0]
    mean_value = float(clean_series.mean())

    if sample_size == 1:
        return (mean_value, mean_value, mean_value)

    standard_error = float(clean_series.std(ddof=1) / math.sqrt(sample_size))
    alpha = 1 - confidence_level
    if SCIPY_AVAILABLE:
        critical_value = float(stats.t.ppf(1 - alpha / 2, df=sample_size - 1))
    else:
        critical_value = float(NormalDist().inv_cdf(1 - alpha / 2))
    margin = critical_value * standard_error
    return (mean_value - margin, mean_value + margin, standard_error)


def calculate_wilson_interval(successes: int, total: int, confidence_level: float) -> tuple[float, float, float]:
    if total <= 0:
        return (np.nan, np.nan, np.nan)

    z_value = float(NormalDist().inv_cdf(1 - (1 - confidence_level) / 2))
    proportion = successes / total
    denominator = 1 + (z_value ** 2 / total)
    center = (proportion + (z_value ** 2) / (2 * total)) / denominator
    margin = (
        z_value
        * math.sqrt((proportion * (1 - proportion) / total) + (z_value ** 2) / (4 * total ** 2))
        / denominator
    )
    standard_error = math.sqrt((proportion * (1 - proportion)) / total)
    return (center - margin, center + margin, standard_error)


def compute_cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    a = pd.to_numeric(group_a, errors="coerce").dropna()
    b = pd.to_numeric(group_b, errors="coerce").dropna()
    if a.shape[0] < 2 or b.shape[0] < 2:
        return np.nan

    pooled_variance = (
        ((a.shape[0] - 1) * a.var(ddof=1)) + ((b.shape[0] - 1) * b.var(ddof=1))
    ) / (a.shape[0] + b.shape[0] - 2)
    if pooled_variance <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / math.sqrt(pooled_variance))


def compute_rank_biserial(u_statistic: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return float((2 * u_statistic) / (n1 * n2) - 1)


def compute_eta_squared(groups: list[pd.Series]) -> float:
    non_empty_groups = [pd.to_numeric(group, errors="coerce").dropna() for group in groups]
    non_empty_groups = [group for group in non_empty_groups if not group.empty]
    if len(non_empty_groups) < 2:
        return np.nan

    combined = pd.concat(non_empty_groups, ignore_index=True)
    grand_mean = combined.mean()
    ss_between = sum(group.shape[0] * ((group.mean() - grand_mean) ** 2) for group in non_empty_groups)
    ss_total = float(((combined - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return np.nan
    return float(ss_between / ss_total)


def compute_epsilon_squared(kruskal_statistic: float, n_total: int, n_groups: int) -> float:
    denominator = n_total - n_groups
    if denominator <= 0:
        return np.nan
    return float((kruskal_statistic - n_groups + 1) / denominator)


def compute_cramers_v(chi2_statistic: float, n_total: int, rows: int, columns: int) -> float:
    if n_total <= 0 or rows <= 1 or columns <= 1:
        return np.nan
    denominator = n_total * min(rows - 1, columns - 1)
    if denominator <= 0:
        return np.nan
    return float(math.sqrt(chi2_statistic / denominator))


def apply_holm_correction(p_values: list[float]) -> list[float]:
    if not p_values:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)

    for rank, (original_index, p_value) in enumerate(indexed):
        holm_value = (total - rank) * p_value
        running_max = max(running_max, holm_value)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


def has_large_sample_warning(sample_size: int) -> bool:
    return sample_size >= LARGE_SAMPLE_WARNING_THRESHOLD


def prepare_group_analysis_frame(
    df: pd.DataFrame,
    outcome_column: str,
    group_column: str,
    top_n_categories: int,
    min_group_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = {
        "outcome_column": outcome_column,
        "group_column": group_column,
        "collapsed": False,
        "kept_categories": [],
        "skip_reason": None,
    }

    subset = df[[outcome_column, group_column]].copy()
    subset[outcome_column] = pd.to_numeric(subset[outcome_column], errors="coerce")

    group_series = subset[group_column]
    if pd.api.types.is_bool_dtype(group_series) or is_binary_like_numeric(group_series):
        subset[group_column] = group_series.astype("Int64").astype("string")
    else:
        subset[group_column] = clean_string_series(group_series)

    subset = subset.dropna(subset=[outcome_column, group_column])
    if subset.empty:
        metadata["skip_reason"] = "no_complete_cases"
        return subset, metadata

    collapsed_group, collapsed, kept_categories = collapse_categories(subset[group_column], top_n_categories)
    subset[group_column] = collapsed_group
    metadata["collapsed"] = collapsed
    metadata["kept_categories"] = kept_categories

    group_sizes = subset[group_column].value_counts(dropna=True)
    valid_groups = group_sizes[group_sizes >= min_group_size].index.astype(str).tolist()
    subset = subset[subset[group_column].isin(valid_groups)].copy()

    if subset.empty or subset[group_column].nunique(dropna=True) < 2:
        metadata["skip_reason"] = "insufficient_valid_groups"
        return subset, metadata

    subset[group_column] = subset[group_column].astype("string")
    return subset, metadata


def prepare_categorical_pair_frame(
    df: pd.DataFrame,
    left_column: str,
    right_column: str,
    top_n_categories: int,
    min_group_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = {
        "left_column": left_column,
        "right_column": right_column,
        "left_collapsed": False,
        "right_collapsed": False,
        "skip_reason": None,
    }

    subset = df[[left_column, right_column]].copy()
    for column in [left_column, right_column]:
        if pd.api.types.is_bool_dtype(subset[column]) or is_binary_like_numeric(subset[column]):
            subset[column] = subset[column].astype("Int64").astype("string")
        else:
            subset[column] = clean_string_series(subset[column])

    subset = subset.dropna(subset=[left_column, right_column])
    if subset.empty:
        metadata["skip_reason"] = "no_complete_cases"
        return subset, metadata

    collapsed_left, left_collapsed, _ = collapse_categories(subset[left_column], top_n_categories)
    collapsed_right, right_collapsed, _ = collapse_categories(subset[right_column], top_n_categories)
    subset[left_column] = collapsed_left
    subset[right_column] = collapsed_right
    metadata["left_collapsed"] = left_collapsed
    metadata["right_collapsed"] = right_collapsed

    left_sizes = subset[left_column].value_counts(dropna=True)
    right_sizes = subset[right_column].value_counts(dropna=True)
    valid_left = left_sizes[left_sizes >= min_group_size].index.astype(str).tolist()
    valid_right = right_sizes[right_sizes >= min_group_size].index.astype(str).tolist()
    subset = subset[subset[left_column].isin(valid_left) & subset[right_column].isin(valid_right)].copy()

    if subset.empty or subset[left_column].nunique(dropna=True) < 2 or subset[right_column].nunique(dropna=True) < 2:
        metadata["skip_reason"] = "insufficient_valid_levels"
        return subset, metadata

    return subset, metadata


def build_group_descriptive_stats(group_df: pd.DataFrame, outcome_column: str, group_column: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group_value, group_frame in group_df.groupby(group_column, dropna=True):
        outcome_series = pd.to_numeric(group_frame[outcome_column], errors="coerce").dropna()
        if outcome_series.empty:
            continue
        records.append(
            {
                "group_label": str(group_value),
                "n": int(outcome_series.shape[0]),
                "mean": float(outcome_series.mean()),
                "median": float(outcome_series.median()),
                "std": float(outcome_series.std(ddof=1)) if outcome_series.shape[0] > 1 else 0.0,
                "min": float(outcome_series.min()),
                "max": float(outcome_series.max()),
            }
        )
    return records


def sample_frame(frame: pd.DataFrame, max_rows: int, stratify_by: str | None = None, random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    if frame.shape[0] <= max_rows:
        return frame.copy(), False

    if stratify_by and stratify_by in frame.columns and frame[stratify_by].nunique(dropna=True) > 1:
        sampled = (
            frame.groupby(stratify_by, group_keys=False)
            .apply(
                lambda chunk: chunk.sample(
                    n=max(1, int(round(max_rows * (chunk.shape[0] / frame.shape[0])))),
                    random_state=random_state,
                )
                if chunk.shape[0] > 1
                else chunk
            )
            .reset_index(drop=True)
        )
        if sampled.shape[0] > max_rows:
            sampled = sampled.sample(max_rows, random_state=random_state)
        return sampled, True

    return frame.sample(max_rows, random_state=random_state), True


def build_model_design_matrix(
    df: pd.DataFrame,
    target_column: str,
    continuous_predictors: list[str],
    binary_predictors: list[str],
    categorical_predictors: list[str],
    top_n_categories: int,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    metadata = {
        "used_continuous_predictors": [],
        "used_binary_predictors": [],
        "used_categorical_predictors": [],
        "collapsed_predictors": {},
        "skip_reason": None,
    }

    available_columns = [target_column]
    available_columns.extend(column for column in continuous_predictors if column in df.columns)
    available_columns.extend(column for column in binary_predictors if column in df.columns)
    available_columns.extend(column for column in categorical_predictors if column in df.columns)

    model_frame = df[available_columns].copy()
    model_frame[target_column] = pd.to_numeric(model_frame[target_column], errors="coerce")

    for column in continuous_predictors:
        if column in model_frame.columns:
            model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce")
            metadata["used_continuous_predictors"].append(column)

    for column in binary_predictors:
        if column in model_frame.columns:
            model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce")
            metadata["used_binary_predictors"].append(column)

    for column in categorical_predictors:
        if column not in model_frame.columns:
            continue
        collapsed_series, collapsed, kept_categories = collapse_categories(model_frame[column], top_n_categories)
        model_frame[column] = collapsed_series
        value_counts = model_frame[column].value_counts(dropna=True)
        valid_levels = value_counts[value_counts >= min_group_size].index.astype(str).tolist()
        model_frame[column] = model_frame[column].where(model_frame[column].isin(valid_levels), other="Other")
        metadata["used_categorical_predictors"].append(column)
        metadata["collapsed_predictors"][column] = {
            "collapsed": collapsed,
            "kept_categories": kept_categories,
        }

    model_frame = model_frame.dropna()
    if model_frame.empty:
        metadata["skip_reason"] = "no_complete_cases"
        return pd.DataFrame(), pd.Series(dtype=float), metadata

    y = model_frame[target_column]
    X = model_frame.drop(columns=[target_column]).copy()
    if X.empty:
        metadata["skip_reason"] = "no_predictors_available"
        return pd.DataFrame(), pd.Series(dtype=float), metadata

    categorical_columns_present = [
        column for column in metadata["used_categorical_predictors"] if column in X.columns
    ]
    X = pd.get_dummies(X, columns=categorical_columns_present, drop_first=True, dtype=float)

    valid_columns = [column for column in X.columns if X[column].nunique(dropna=True) > 1]
    X = X[valid_columns].copy()
    if X.empty:
        metadata["skip_reason"] = "predictors_lack_variability"
        return pd.DataFrame(), pd.Series(dtype=float), metadata

    X = X.astype(float)
    if STATSMODELS_AVAILABLE:
        X = sm.add_constant(X, has_constant="add")

    return X, y.astype(float), metadata


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
            "statsmodels_available": STATSMODELS_AVAILABLE,
            "statsmodels_import_error": STATSMODELS_IMPORT_ERROR,
            "scipy_available": SCIPY_AVAILABLE,
            "scipy_import_error": SCIPY_IMPORT_ERROR,
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
        project_root = find_project_root(normalize_workspace_path(Path.cwd()))
        resolved_input_path = resolve_input_parquet_path(project_root=project_root, input_path=input_path)
        resolved_output_dir = normalize_workspace_path(Path(output_dir))

        if not resolved_output_dir.is_absolute():
            resolved_output_dir = normalize_workspace_path(project_root / resolved_output_dir)

        output_dirs = {
            "project_root": project_root,
            "input_path": resolved_input_path,
            "requirements_path": normalize_workspace_path(project_root / "requirements.txt"),
            "base": resolved_output_dir,
            "assumptions": resolved_output_dir / "assumptions",
            "confidence_intervals": resolved_output_dir / "confidence_intervals",
            "group_comparisons": resolved_output_dir / "group_comparisons",
            "categorical_associations": resolved_output_dir / "categorical_associations",
            "numeric_associations": resolved_output_dir / "numeric_associations",
            "models": resolved_output_dir / "models",
            "plots": resolved_output_dir / "plots",
            "group_plots": resolved_output_dir / "plots" / "group_comparisons",
            "confidence_plots": resolved_output_dir / "plots" / "confidence_intervals",
            "correlation_plots": resolved_output_dir / "plots" / "correlations",
            "model_plots": resolved_output_dir / "plots" / "models",
            "contingency_tables": resolved_output_dir / "categorical_associations" / "contingency_tables",
        }

        for directory_key in [
            "base",
            "assumptions",
            "confidence_intervals",
            "group_comparisons",
            "categorical_associations",
            "numeric_associations",
            "models",
            "plots",
            "group_plots",
            "confidence_plots",
            "correlation_plots",
            "model_plots",
            "contingency_tables",
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


@task(name="prepare_dataframe_for_inference")
@timing_decorator
@validate
def prepare_dataframe_for_inference(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
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

        datetime_columns_detected: list[str] = []
        for column in prepared_df.columns:
            if not is_date_candidate(column, prepared_df[column]):
                continue

            non_null_before = int(prepared_df[column].notna().sum())
            converted_series = pd.to_datetime(prepared_df[column], errors="coerce", utc=True)
            non_null_after = int(converted_series.notna().sum())
            prepared_df[column] = converted_series
            datetime_columns_detected.append(column)
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
                if pd.api.types.is_numeric_dtype(prepared_df[column]) and column not in datetime_columns_detected
            )
        )

        for column in numeric_candidates:
            if column not in prepared_df.columns or column in datetime_columns_detected:
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
        leakage_columns_present = sorted(
            column for column in MODEL_EXCLUDED_COLUMNS if column in prepared_df.columns
        )

        context = {
            "alias_map": alias_map,
            "inverse_alias_map": invert_alias_map(alias_map),
            "column_groups": column_groups,
            "conversion_summary": conversion_records,
            "model_excluded_columns_present": leakage_columns_present,
            "technical_identifier_columns_present": [
                column for column in TECHNICAL_IDENTIFIER_COLUMNS if column in prepared_df.columns
            ],
        }
        return prepared_df, context
    except Exception as exc:
        log_message(f"Dataframe preparation failed: {exc}", level="error")
        raise
    finally:
        log_message("Dataframe preparation step finished.")


@task(name="select_inferential_candidates")
@timing_decorator
@validate
def select_inferential_candidates(
    df: pd.DataFrame,
    context: dict[str, Any],
    top_n_categories: int,
    max_group_comparisons: int,
    max_correlation_features: int,
    min_group_size: int,
) -> dict[str, Any]:
    try:
        column_groups = context["column_groups"]
        group_lookup: dict[str, str] = {}
        for group_name, columns in column_groups.items():
            for column in columns:
                group_lookup[column] = group_name

        candidate_records: list[dict[str, Any]] = []
        eligible_numeric_ci: list[str] = []
        eligible_binary_ci: list[str] = []
        eligible_group_outcomes: list[str] = []
        eligible_group_vars: list[str] = []
        eligible_categorical_association_vars: list[str] = []
        eligible_numeric_association_vars: list[str] = []
        linear_continuous_predictors: list[str] = []
        linear_binary_predictors: list[str] = []
        linear_categorical_predictors: list[str] = []

        for column in df.columns:
            non_null_count = int(df[column].notna().sum())
            null_pct = round(float(df[column].isna().mean() * 100), 4)
            non_null_pct = round(100 - null_pct, 4)
            nunique = int(df[column].nunique(dropna=True))
            semantic_group = group_lookup.get(column, "unknown")
            is_binary = (
                semantic_group in {"boolean", "binary_numeric"} or is_binary_like_numeric(df[column])
            )
            is_model_excluded = column in MODEL_EXCLUDED_COLUMNS
            is_text_or_id = column in TECHNICAL_IDENTIFIER_COLUMNS or column in TEXT_COLUMNS

            eligible_mean_ci = semantic_group == "numeric" and non_null_pct >= 60 and nunique >= 5
            eligible_prop_ci = (
                is_binary
                or (
                    semantic_group == "categorical"
                    and non_null_pct >= 60
                    and 2 <= nunique <= max(top_n_categories * 4, 12)
                )
            )
            eligible_group_outcome = column in PRIORITY_OUTCOME_COLUMNS and eligible_mean_ci
            eligible_group_var = column in PRIORITY_GROUP_COLUMNS and semantic_group in {"categorical", "binary_numeric", "boolean"} and non_null_pct >= 60 and nunique >= 2
            eligible_categorical_assoc = (
                column in PRIORITY_CATEGORICAL_ASSOC_COLUMNS
                and semantic_group in {"categorical", "binary_numeric", "boolean"}
                and non_null_pct >= 60
                and nunique >= 2
            )
            eligible_numeric_assoc = (
                semantic_group == "numeric"
                and non_null_pct >= 60
                and nunique >= 5
            )
            eligible_linear_continuous = (
                column in MODEL_CONTINUOUS_PREDICTORS and eligible_numeric_assoc and not is_model_excluded
            )
            eligible_linear_binary = (
                column in MODEL_BINARY_PREDICTORS and is_binary and not is_model_excluded
            )
            eligible_linear_categorical = (
                column in MODEL_CATEGORICAL_PREDICTORS
                and semantic_group in {"categorical", "binary_numeric", "boolean"}
                and not is_model_excluded
                and non_null_pct >= 60
                and nunique >= 2
            )

            if eligible_mean_ci:
                eligible_numeric_ci.append(column)
            if eligible_prop_ci:
                eligible_binary_ci.append(column)
            if eligible_group_outcome:
                eligible_group_outcomes.append(column)
            if eligible_group_var:
                eligible_group_vars.append(column)
            if eligible_categorical_assoc:
                eligible_categorical_association_vars.append(column)
            if eligible_numeric_assoc:
                eligible_numeric_association_vars.append(column)
            if eligible_linear_continuous:
                linear_continuous_predictors.append(column)
            if eligible_linear_binary:
                linear_binary_predictors.append(column)
            if eligible_linear_categorical:
                linear_categorical_predictors.append(column)

            skip_reasons: list[str] = []
            if is_text_or_id:
                skip_reasons.append("technical_or_text_column")
            if non_null_pct < 60:
                skip_reasons.append("low_completeness")
            if semantic_group == "numeric" and nunique < 5 and not is_binary:
                skip_reasons.append("low_numeric_variability")
            if semantic_group in {"categorical", "binary_numeric", "boolean"} and nunique < 2:
                skip_reasons.append("single_level")
            if is_model_excluded:
                skip_reasons.append("excluded_from_modeling")

            candidate_records.append(
                {
                    "column": column,
                    "semantic_group": semantic_group,
                    "dtype": str(df[column].dtype),
                    "non_null_count": non_null_count,
                    "non_null_pct": non_null_pct,
                    "null_pct": null_pct,
                    "nunique": nunique,
                    "is_binary_like": is_binary,
                    "eligible_mean_ci": eligible_mean_ci,
                    "eligible_proportion_ci": eligible_prop_ci,
                    "eligible_group_outcome": eligible_group_outcome,
                    "eligible_group_var": eligible_group_var,
                    "eligible_categorical_association": eligible_categorical_assoc,
                    "eligible_numeric_association": eligible_numeric_assoc,
                    "eligible_linear_predictor": eligible_linear_continuous or eligible_linear_binary or eligible_linear_categorical,
                    "model_excluded": is_model_excluded,
                    "skip_reason": " | ".join(skip_reasons) if skip_reasons else "eligible",
                }
            )

        eligible_group_outcomes = select_priority_columns(
            list(dict.fromkeys(eligible_group_outcomes)),
            PRIORITY_OUTCOME_COLUMNS,
            len(PRIORITY_OUTCOME_COLUMNS),
        )
        eligible_group_vars = select_priority_columns(
            list(dict.fromkeys(eligible_group_vars)),
            PRIORITY_GROUP_COLUMNS,
            len(PRIORITY_GROUP_COLUMNS),
        )
        group_pairs: list[dict[str, Any]] = []
        for outcome_column in eligible_group_outcomes:
            for group_column in eligible_group_vars:
                if len(group_pairs) >= max_group_comparisons:
                    break
                group_pairs.append(
                    {
                        "outcome_column": outcome_column,
                        "group_column": group_column,
                    }
                )
            if len(group_pairs) >= max_group_comparisons:
                break

        categorical_association_vars = select_priority_columns(
            list(dict.fromkeys(eligible_categorical_association_vars)),
            PRIORITY_CATEGORICAL_ASSOC_COLUMNS,
            max(top_n_categories + 4, 10),
        )
        numeric_association_vars = select_priority_columns(
            list(dict.fromkeys(eligible_numeric_association_vars)),
            PRIORITY_NUMERIC_ASSOC_COLUMNS,
            max_correlation_features,
        )
        numeric_ci_columns = select_priority_columns(
            list(dict.fromkeys(eligible_numeric_ci)),
            PRIORITY_NUMERIC_COLUMNS,
            max(max_correlation_features + 5, 15),
        )
        proportion_ci_columns = select_priority_columns(
            list(dict.fromkeys(eligible_binary_ci)),
            PRIORITY_BINARY_COLUMNS + PRIORITY_CATEGORICAL_COLUMNS,
            max(top_n_categories + 6, 12),
        )

        candidate_inventory = pd.DataFrame(candidate_records).sort_values(
            ["eligible_group_outcome", "eligible_group_var", "eligible_numeric_association", "column"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

        return {
            "candidate_inventory": candidate_inventory,
            "numeric_ci_columns": numeric_ci_columns,
            "proportion_ci_columns": proportion_ci_columns,
            "group_pairs": group_pairs,
            "categorical_association_vars": categorical_association_vars,
            "numeric_association_vars": numeric_association_vars,
            "linear_continuous_predictors": select_priority_columns(
                list(dict.fromkeys(linear_continuous_predictors)),
                MODEL_CONTINUOUS_PREDICTORS,
                len(MODEL_CONTINUOUS_PREDICTORS),
            ),
            "linear_binary_predictors": select_priority_columns(
                list(dict.fromkeys(linear_binary_predictors)),
                MODEL_BINARY_PREDICTORS,
                len(MODEL_BINARY_PREDICTORS),
            ),
            "linear_categorical_predictors": select_priority_columns(
                list(dict.fromkeys(linear_categorical_predictors)),
                MODEL_CATEGORICAL_PREDICTORS,
                len(MODEL_CATEGORICAL_PREDICTORS),
            ),
        }
    except Exception as exc:
        log_message(f"Inferential candidate selection failed: {exc}", level="error")
        raise
    finally:
        log_message("Inferential candidate selection step finished.")


@task(name="evaluate_assumptions")
@timing_decorator
@validate
def evaluate_assumptions(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    top_n_categories: int,
    normality_sample_size: int,
    min_group_size: int,
) -> pd.DataFrame:
    try:
        if not SCIPY_AVAILABLE:
            skipped_records = [
                {
                    "outcome_column": pair["outcome_column"],
                    "group_column": pair["group_column"],
                    "n_used": 0,
                    "n_groups": 0,
                    "group_collapsed": False,
                    "normality_test": None,
                    "normality_pvalue_min": np.nan,
                    "all_groups_normal": False,
                    "homoscedasticity_test": None,
                    "homoscedasticity_pvalue": np.nan,
                    "homoscedastic": False,
                    "parametric_test_recommended": False,
                    "large_sample_warning": False,
                    "status": "skipped",
                    "skip_reason": "scipy_unavailable",
                }
                for pair in candidate_spec["group_pairs"]
            ]
            return pd.DataFrame(skipped_records)

        records: list[dict[str, Any]] = []

        for pair in candidate_spec["group_pairs"]:
            outcome_column = pair["outcome_column"]
            group_column = pair["group_column"]
            analysis_df, metadata = prepare_group_analysis_frame(
                df=df,
                outcome_column=outcome_column,
                group_column=group_column,
                top_n_categories=top_n_categories,
                min_group_size=min_group_size,
            )

            if metadata["skip_reason"]:
                records.append(
                    {
                        "outcome_column": outcome_column,
                        "group_column": group_column,
                        "n_used": int(analysis_df.shape[0]),
                        "n_groups": int(analysis_df[group_column].nunique(dropna=True)) if not analysis_df.empty else 0,
                        "group_collapsed": metadata["collapsed"],
                        "normality_test": None,
                        "normality_pvalue_min": np.nan,
                        "all_groups_normal": False,
                        "homoscedasticity_test": None,
                        "homoscedasticity_pvalue": np.nan,
                        "homoscedastic": False,
                        "parametric_test_recommended": False,
                        "large_sample_warning": False,
                        "status": "skipped",
                        "skip_reason": metadata["skip_reason"],
                    }
                )
                continue

            groups = [
                group_frame[outcome_column]
                for _, group_frame in analysis_df.groupby(group_column, dropna=True)
            ]
            normality_pvalues: list[float] = []
            for group_series in groups:
                sample_series = pd.to_numeric(group_series, errors="coerce").dropna()
                if sample_series.shape[0] < 3:
                    continue
                if sample_series.shape[0] > normality_sample_size:
                    sample_series = sample_series.sample(normality_sample_size, random_state=42)
                try:
                    shapiro_result = stats.shapiro(sample_series)
                    normality_pvalues.append(float(shapiro_result.pvalue))
                except Exception:
                    continue

            if normality_pvalues:
                normality_pvalue_min = min(normality_pvalues)
                all_groups_normal = all(pvalue >= 0.05 for pvalue in normality_pvalues)
                normality_test = "Shapiro-Wilk"
            else:
                normality_pvalue_min = np.nan
                all_groups_normal = False
                normality_test = None

            try:
                levene_result = stats.levene(*groups, center="median")
                homoscedasticity_pvalue = float(levene_result.pvalue)
                homoscedastic = homoscedasticity_pvalue >= 0.05
                homoscedasticity_test = "Levene"
            except Exception:
                homoscedasticity_pvalue = np.nan
                homoscedastic = False
                homoscedasticity_test = None

            n_groups = int(analysis_df[group_column].nunique(dropna=True))
            parametric_test_recommended = all_groups_normal if n_groups == 2 else (all_groups_normal and homoscedastic)
            large_sample_warning = has_large_sample_warning(int(analysis_df.shape[0]))

            records.append(
                {
                    "outcome_column": outcome_column,
                    "group_column": group_column,
                    "n_used": int(analysis_df.shape[0]),
                    "n_groups": n_groups,
                    "group_collapsed": metadata["collapsed"],
                    "normality_test": normality_test,
                    "normality_pvalue_min": normality_pvalue_min,
                    "all_groups_normal": all_groups_normal,
                    "homoscedasticity_test": homoscedasticity_test,
                    "homoscedasticity_pvalue": homoscedasticity_pvalue,
                    "homoscedastic": homoscedastic,
                    "parametric_test_recommended": parametric_test_recommended,
                    "large_sample_warning": large_sample_warning,
                    "status": "ok",
                    "skip_reason": None,
                }
            )

        if not records:
            return pd.DataFrame(
                columns=[
                    "outcome_column",
                    "group_column",
                    "n_used",
                    "n_groups",
                    "group_collapsed",
                    "normality_test",
                    "normality_pvalue_min",
                    "all_groups_normal",
                    "homoscedasticity_test",
                    "homoscedasticity_pvalue",
                    "homoscedastic",
                    "parametric_test_recommended",
                    "large_sample_warning",
                    "status",
                    "skip_reason",
                ]
            )

        return pd.DataFrame(records).sort_values(
            ["status", "outcome_column", "group_column"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Assumption evaluation failed: {exc}", level="error")
        raise
    finally:
        log_message("Assumption evaluation step finished.")


@task(name="build_confidence_intervals")
@timing_decorator
@validate
def build_confidence_intervals(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    confidence_level: float,
    plot_sample_size: int,
    top_n_categories: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        numeric_records: list[dict[str, Any]] = []
        proportion_records: list[dict[str, Any]] = []

        for column in candidate_spec["numeric_ci_columns"]:
            if column not in df.columns:
                continue

            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty or series.nunique(dropna=True) < 5:
                continue

            ci_series = series
            ci_lower, ci_upper, standard_error = calculate_t_confidence_interval(ci_series, confidence_level)
            numeric_records.append(
                {
                    "column": column,
                    "n_used": int(ci_series.shape[0]),
                    "estimate": float(ci_series.mean()),
                    "standard_error": standard_error,
                    "confidence_level": confidence_level,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "sampling_applied": False,
                }
            )

        for column in candidate_spec["proportion_ci_columns"]:
            if column not in df.columns:
                continue

            raw_series = df[column]
            if pd.api.types.is_bool_dtype(raw_series) or is_binary_like_numeric(raw_series):
                series = pd.to_numeric(raw_series, errors="coerce").dropna()
                if series.empty:
                    continue
                successes = int(series.gt(0).sum())
                total = int(series.shape[0])
                ci_lower, ci_upper, standard_error = calculate_wilson_interval(successes, total, confidence_level)
                proportion_records.append(
                    {
                        "column": column,
                        "category": "positive",
                        "successes": successes,
                        "n_used": total,
                        "estimate": successes / total if total else np.nan,
                        "standard_error": standard_error,
                        "confidence_level": confidence_level,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "sampling_applied": False,
                    }
                )
                continue

            series = clean_string_series(raw_series).dropna()
            if series.empty or series.nunique(dropna=True) < 2:
                continue

            collapsed_series, _, _ = collapse_categories(series, top_n_categories)
            value_counts = collapsed_series.value_counts(dropna=True)
            total = int(collapsed_series.shape[0])
            for category_value, count in value_counts.items():
                ci_lower, ci_upper, standard_error = calculate_wilson_interval(int(count), total, confidence_level)
                proportion_records.append(
                    {
                        "column": column,
                        "category": str(category_value),
                        "successes": int(count),
                        "n_used": total,
                        "estimate": int(count) / total if total else np.nan,
                        "standard_error": standard_error,
                        "confidence_level": confidence_level,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "sampling_applied": False,
                    }
                )

        numeric_df = pd.DataFrame(numeric_records)
        proportion_df = pd.DataFrame(proportion_records)

        if numeric_df.empty:
            numeric_df = pd.DataFrame(
                columns=[
                    "column",
                    "n_used",
                    "estimate",
                    "standard_error",
                    "confidence_level",
                    "ci_lower",
                    "ci_upper",
                    "sampling_applied",
                ]
            )

        if proportion_df.empty:
            proportion_df = pd.DataFrame(
                columns=[
                    "column",
                    "category",
                    "successes",
                    "n_used",
                    "estimate",
                    "standard_error",
                    "confidence_level",
                    "ci_lower",
                    "ci_upper",
                    "sampling_applied",
                ]
            )

        return numeric_df.sort_values("column").reset_index(drop=True), proportion_df.sort_values(
            ["column", "category"]
        ).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Confidence interval construction failed: {exc}", level="error")
        raise
    finally:
        log_message("Confidence interval step finished.")


@task(name="run_group_comparisons")
@timing_decorator
@validate
def run_group_comparisons(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    assumptions_summary: pd.DataFrame,
    top_n_categories: int,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not SCIPY_AVAILABLE:
            skipped_results = pd.DataFrame(
                [
                    {
                        "outcome_column": pair["outcome_column"],
                        "group_column": pair["group_column"],
                        "test_name": None,
                        "n_used": 0,
                        "n_groups": 0,
                        "group_collapsed": False,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect_size_name": None,
                        "effect_size_value": np.nan,
                        "descriptive_stats": None,
                        "large_sample_warning": False,
                        "status": "skipped",
                        "skip_reason": "scipy_unavailable",
                    }
                    for pair in candidate_spec["group_pairs"]
                ]
            )
            skipped_posthoc = pd.DataFrame(
                columns=[
                    "outcome_column",
                    "group_column",
                    "left_group",
                    "right_group",
                    "test_name",
                    "statistic",
                    "raw_p_value",
                    "holm_adjusted_p_value",
                    "significant_after_holm",
                    "n_left",
                    "n_right",
                ]
            )
            return skipped_results, skipped_posthoc

        assumption_lookup = {}
        if not assumptions_summary.empty:
            assumption_lookup = {
                (row.outcome_column, row.group_column): row
                for row in assumptions_summary.itertuples(index=False)
            }

        result_records: list[dict[str, Any]] = []
        posthoc_records: list[dict[str, Any]] = []

        for pair in candidate_spec["group_pairs"]:
            outcome_column = pair["outcome_column"]
            group_column = pair["group_column"]
            assumption_row = assumption_lookup.get((outcome_column, group_column))
            analysis_df, metadata = prepare_group_analysis_frame(
                df=df,
                outcome_column=outcome_column,
                group_column=group_column,
                top_n_categories=top_n_categories,
                min_group_size=min_group_size,
            )

            if metadata["skip_reason"] or assumption_row is None or getattr(assumption_row, "status", "skipped") != "ok":
                result_records.append(
                    {
                        "outcome_column": outcome_column,
                        "group_column": group_column,
                        "test_name": None,
                        "n_used": int(analysis_df.shape[0]),
                        "n_groups": int(analysis_df[group_column].nunique(dropna=True)) if not analysis_df.empty else 0,
                        "group_collapsed": metadata["collapsed"],
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect_size_name": None,
                        "effect_size_value": np.nan,
                        "descriptive_stats": None,
                        "large_sample_warning": bool(getattr(assumption_row, "large_sample_warning", False)),
                        "status": "skipped",
                        "skip_reason": metadata["skip_reason"] or getattr(assumption_row, "skip_reason", "assumptions_unavailable"),
                    }
                )
                continue

            group_frames = []
            group_labels = []
            for group_label, group_frame in analysis_df.groupby(group_column, dropna=True):
                group_frames.append(pd.to_numeric(group_frame[outcome_column], errors="coerce").dropna())
                group_labels.append(str(group_label))

            n_groups = len(group_frames)
            descriptive_stats = build_group_descriptive_stats(analysis_df, outcome_column, group_column)
            test_name = None
            statistic = np.nan
            p_value = np.nan
            effect_size_name = None
            effect_size_value = np.nan
            parametric = bool(getattr(assumption_row, "parametric_test_recommended"))

            if n_groups == 2:
                group_a = group_frames[0]
                group_b = group_frames[1]
                if parametric:
                    test_name = "Welch t-test"
                    test_result = stats.ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
                    statistic = float(test_result.statistic)
                    p_value = float(test_result.pvalue)
                    effect_size_name = "cohens_d"
                    effect_size_value = compute_cohens_d(group_a, group_b)
                else:
                    test_name = "Mann-Whitney U"
                    test_result = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
                    statistic = float(test_result.statistic)
                    p_value = float(test_result.pvalue)
                    effect_size_name = "rank_biserial"
                    effect_size_value = compute_rank_biserial(float(test_result.statistic), group_a.shape[0], group_b.shape[0])
            else:
                if parametric:
                    test_name = "ANOVA"
                    test_result = stats.f_oneway(*group_frames)
                    statistic = float(test_result.statistic)
                    p_value = float(test_result.pvalue)
                    effect_size_name = "eta_squared"
                    effect_size_value = compute_eta_squared(group_frames)
                else:
                    test_name = "Kruskal-Wallis"
                    test_result = stats.kruskal(*group_frames)
                    statistic = float(test_result.statistic)
                    p_value = float(test_result.pvalue)
                    effect_size_name = "epsilon_squared"
                    effect_size_value = compute_epsilon_squared(float(test_result.statistic), int(analysis_df.shape[0]), n_groups)

                if p_value < 0.05 and n_groups <= 6:
                    pairwise_results: list[dict[str, Any]] = []
                    pairwise_pvalues: list[float] = []
                    for left_index, right_index in combinations(range(n_groups), 2):
                        left_group = group_frames[left_index]
                        right_group = group_frames[right_index]
                        if parametric:
                            pairwise_test_name = "Welch t-test"
                            pairwise_test = stats.ttest_ind(left_group, right_group, equal_var=False, nan_policy="omit")
                            pairwise_statistic = float(pairwise_test.statistic)
                            pairwise_pvalue = float(pairwise_test.pvalue)
                        else:
                            pairwise_test_name = "Mann-Whitney U"
                            pairwise_test = stats.mannwhitneyu(left_group, right_group, alternative="two-sided")
                            pairwise_statistic = float(pairwise_test.statistic)
                            pairwise_pvalue = float(pairwise_test.pvalue)

                        pairwise_pvalues.append(pairwise_pvalue)
                        pairwise_results.append(
                            {
                                "outcome_column": outcome_column,
                                "group_column": group_column,
                                "left_group": group_labels[left_index],
                                "right_group": group_labels[right_index],
                                "test_name": pairwise_test_name,
                                "statistic": pairwise_statistic,
                                "raw_p_value": pairwise_pvalue,
                                "n_left": int(left_group.shape[0]),
                                "n_right": int(right_group.shape[0]),
                            }
                        )

                    adjusted_pvalues = apply_holm_correction(pairwise_pvalues)
                    for row_index, adjusted_pvalue in enumerate(adjusted_pvalues):
                        pairwise_results[row_index]["holm_adjusted_p_value"] = adjusted_pvalue
                        pairwise_results[row_index]["significant_after_holm"] = adjusted_pvalue < 0.05
                    posthoc_records.extend(pairwise_results)

            result_records.append(
                {
                    "outcome_column": outcome_column,
                    "group_column": group_column,
                    "test_name": test_name,
                    "n_used": int(analysis_df.shape[0]),
                    "n_groups": n_groups,
                    "group_collapsed": metadata["collapsed"],
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size_name": effect_size_name,
                    "effect_size_value": effect_size_value,
                    "descriptive_stats": json.dumps(descriptive_stats, ensure_ascii=False),
                    "large_sample_warning": bool(getattr(assumption_row, "large_sample_warning", False)),
                    "status": "ok",
                    "skip_reason": None,
                }
            )

        results_df = pd.DataFrame(result_records)
        posthoc_df = pd.DataFrame(posthoc_records)

        if results_df.empty:
            results_df = pd.DataFrame(
                columns=[
                    "outcome_column",
                    "group_column",
                    "test_name",
                    "n_used",
                    "n_groups",
                    "group_collapsed",
                    "statistic",
                    "p_value",
                    "effect_size_name",
                    "effect_size_value",
                    "descriptive_stats",
                    "large_sample_warning",
                    "status",
                    "skip_reason",
                ]
            )

        if posthoc_df.empty:
            posthoc_df = pd.DataFrame(
                columns=[
                    "outcome_column",
                    "group_column",
                    "left_group",
                    "right_group",
                    "test_name",
                    "statistic",
                    "raw_p_value",
                    "holm_adjusted_p_value",
                    "significant_after_holm",
                    "n_left",
                    "n_right",
                ]
            )

        return results_df.sort_values(["status", "p_value", "outcome_column", "group_column"]).reset_index(
            drop=True
        ), posthoc_df.sort_values(["outcome_column", "group_column", "holm_adjusted_p_value"]).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Group comparisons failed: {exc}", level="error")
        raise
    finally:
        log_message("Group comparison step finished.")


@task(name="run_categorical_associations")
@timing_decorator
@validate
def run_categorical_associations(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    output_dirs: dict[str, Path],
    top_n_categories: int,
    min_group_size: int,
) -> pd.DataFrame:
    try:
        association_records: list[dict[str, Any]] = []
        candidate_columns = candidate_spec["categorical_association_vars"]

        if not SCIPY_AVAILABLE:
            for left_column, right_column in combinations(candidate_columns, 2):
                association_records.append(
                    {
                        "left_column": left_column,
                        "right_column": right_column,
                        "test_name": None,
                        "n_used": 0,
                        "table_shape": None,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "degrees_of_freedom": np.nan,
                        "cramers_v": np.nan,
                        "expected_cells_below_5_pct": np.nan,
                        "large_sample_warning": False,
                        "status": "skipped",
                        "skip_reason": "scipy_unavailable",
                    }
                )
            return pd.DataFrame(association_records)

        for left_column, right_column in combinations(candidate_columns, 2):
            analysis_df, metadata = prepare_categorical_pair_frame(
                df=df,
                left_column=left_column,
                right_column=right_column,
                top_n_categories=top_n_categories,
                min_group_size=min_group_size,
            )

            if metadata["skip_reason"]:
                association_records.append(
                    {
                        "left_column": left_column,
                        "right_column": right_column,
                        "test_name": None,
                        "n_used": int(analysis_df.shape[0]),
                        "table_shape": None,
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "degrees_of_freedom": np.nan,
                        "cramers_v": np.nan,
                        "expected_cells_below_5_pct": np.nan,
                        "large_sample_warning": False,
                        "status": "skipped",
                        "skip_reason": metadata["skip_reason"],
                    }
                )
                continue

            contingency_table = pd.crosstab(analysis_df[left_column], analysis_df[right_column], dropna=False)
            contingency_path = output_dirs["contingency_tables"] / f"{left_column}_vs_{right_column}.csv"
            safe_write_csv(contingency_table.reset_index(), contingency_path)

            test_name = "Chi-square"
            statistic = np.nan
            p_value = np.nan
            degrees_of_freedom = np.nan
            expected_cells_below_5_pct = np.nan
            cramers_v = np.nan
            status = "ok"
            skip_reason = None

            try:
                chi2_statistic, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
                expected_cells_below_5_pct = float((expected < 5).mean() * 100)
                statistic = float(chi2_statistic)
                p_value = float(chi2_pvalue)
                degrees_of_freedom = int(dof)
                cramers_v = compute_cramers_v(
                    chi2_statistic=float(chi2_statistic),
                    n_total=int(contingency_table.to_numpy().sum()),
                    rows=contingency_table.shape[0],
                    columns=contingency_table.shape[1],
                )

                if contingency_table.shape == (2, 2) and expected_cells_below_5_pct > 20:
                    test_name = "Fisher exact"
                    fisher_statistic, fisher_pvalue = stats.fisher_exact(contingency_table.to_numpy())
                    statistic = float(fisher_statistic)
                    p_value = float(fisher_pvalue)
                    degrees_of_freedom = 1
            except Exception as exc:
                status = "skipped"
                skip_reason = f"categorical_association_error: {exc}"

            association_records.append(
                {
                    "left_column": left_column,
                    "right_column": right_column,
                    "test_name": test_name if status == "ok" else None,
                    "n_used": int(analysis_df.shape[0]),
                    "table_shape": f"{contingency_table.shape[0]}x{contingency_table.shape[1]}",
                    "statistic": statistic,
                    "p_value": p_value,
                    "degrees_of_freedom": degrees_of_freedom,
                    "cramers_v": cramers_v,
                    "expected_cells_below_5_pct": expected_cells_below_5_pct,
                    "large_sample_warning": has_large_sample_warning(int(analysis_df.shape[0])),
                    "status": status,
                    "skip_reason": skip_reason,
                }
            )

        if not association_records:
            return pd.DataFrame(
                columns=[
                    "left_column",
                    "right_column",
                    "test_name",
                    "n_used",
                    "table_shape",
                    "statistic",
                    "p_value",
                    "degrees_of_freedom",
                    "cramers_v",
                    "expected_cells_below_5_pct",
                    "large_sample_warning",
                    "status",
                    "skip_reason",
                ]
            )

        return pd.DataFrame(association_records).sort_values(
            ["status", "p_value", "left_column", "right_column"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    except Exception as exc:
        log_message(f"Categorical association analysis failed: {exc}", level="error")
        raise
    finally:
        log_message("Categorical association step finished.")


@task(name="run_numeric_associations")
@timing_decorator
@validate
def run_numeric_associations(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        selected_columns = candidate_spec["numeric_association_vars"]
        records: list[dict[str, Any]] = []

        matrix_columns = selected_columns[:]
        if "transparency_score" in matrix_columns:
            matrix_columns = [
                column
                for column in matrix_columns
                if column
                not in {
                    "score_completitud",
                    "score_trazabilidad_base",
                    "score_trazabilidad",
                    "score_temporal",
                    "score_competencia",
                    "evidence_coverage",
                    "margin_to_threshold",
                }
            ]
        if matrix_columns:
            correlation_matrix = (
                df[matrix_columns]
                .apply(pd.to_numeric, errors="coerce")
                .corr(method="spearman", numeric_only=True)
                .reset_index()
                .rename(columns={"index": "variable"})
            )
        else:
            correlation_matrix = pd.DataFrame(columns=["variable"])

        if not SCIPY_AVAILABLE:
            skipped_records = [
                {
                    "left_column": left_column,
                    "right_column": right_column,
                    "n_used": 0,
                    "pearson_r": np.nan,
                    "pearson_p_value": np.nan,
                    "spearman_rho": np.nan,
                    "spearman_p_value": np.nan,
                    "large_sample_warning": False,
                    "status": "skipped",
                    "skip_reason": "scipy_unavailable",
                }
                for left_column, right_column in combinations(selected_columns, 2)
                if frozenset({left_column, right_column}) not in TAUTOLOGICAL_CORRELATION_PAIRS
            ]
            return pd.DataFrame(skipped_records), correlation_matrix

        for left_column, right_column in combinations(selected_columns, 2):
            if frozenset({left_column, right_column}) in TAUTOLOGICAL_CORRELATION_PAIRS:
                continue

            subset = df[[left_column, right_column]].copy()
            subset[left_column] = pd.to_numeric(subset[left_column], errors="coerce")
            subset[right_column] = pd.to_numeric(subset[right_column], errors="coerce")
            subset = subset.dropna()
            if subset.shape[0] < min_group_size:
                records.append(
                    {
                        "left_column": left_column,
                        "right_column": right_column,
                        "n_used": int(subset.shape[0]),
                        "pearson_r": np.nan,
                        "pearson_p_value": np.nan,
                        "spearman_rho": np.nan,
                        "spearman_p_value": np.nan,
                        "large_sample_warning": False,
                        "status": "skipped",
                        "skip_reason": "insufficient_complete_cases",
                    }
                )
                continue

            pearson_result = stats.pearsonr(subset[left_column], subset[right_column])
            spearman_result = stats.spearmanr(subset[left_column], subset[right_column], nan_policy="omit")
            records.append(
                {
                    "left_column": left_column,
                    "right_column": right_column,
                    "n_used": int(subset.shape[0]),
                    "pearson_r": float(pearson_result.statistic),
                    "pearson_p_value": float(pearson_result.pvalue),
                    "spearman_rho": float(spearman_result.statistic),
                    "spearman_p_value": float(spearman_result.pvalue),
                    "large_sample_warning": has_large_sample_warning(int(subset.shape[0])),
                    "status": "ok",
                    "skip_reason": None,
                }
            )

        if not records:
            correlation_results = pd.DataFrame(
                columns=[
                    "left_column",
                    "right_column",
                    "n_used",
                    "pearson_r",
                    "pearson_p_value",
                    "spearman_rho",
                    "spearman_p_value",
                    "large_sample_warning",
                    "status",
                    "skip_reason",
                ]
            )
        else:
            correlation_results = pd.DataFrame(records).sort_values(
                ["status", "spearman_p_value", "left_column", "right_column"],
                ascending=[True, True, True, True],
            ).reset_index(drop=True)

        return correlation_results, correlation_matrix
    except Exception as exc:
        log_message(f"Numeric association analysis failed: {exc}", level="error")
        raise
    finally:
        log_message("Numeric association step finished.")


@task(name="fit_linear_regression_model")
@timing_decorator
@validate
def fit_linear_regression_model(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    top_n_categories: int,
    min_group_size: int,
    model_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        diagnostics_record = {
            "model_name": "linear_regression",
            "target_column": "transparency_score",
            "n_used": 0,
            "sampling_applied": False,
            "metric_r2": np.nan,
            "metric_adj_r2": np.nan,
            "metric_rmse": np.nan,
            "metric_mae": np.nan,
            "metric_aic": np.nan,
            "metric_bic": np.nan,
            "large_sample_warning": False,
            "status": "skipped",
            "skip_reason": None,
        }
        payload = {"fitted_values": pd.Series(dtype=float), "residuals": pd.Series(dtype=float)}

        if not STATSMODELS_AVAILABLE:
            diagnostics_record["skip_reason"] = "statsmodels_unavailable"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        if "transparency_score" not in df.columns:
            diagnostics_record["skip_reason"] = "target_not_available"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        X, y, metadata = build_model_design_matrix(
            df=df,
            target_column="transparency_score",
            continuous_predictors=candidate_spec["linear_continuous_predictors"],
            binary_predictors=candidate_spec["linear_binary_predictors"],
            categorical_predictors=candidate_spec["linear_categorical_predictors"],
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
        )

        if metadata["skip_reason"]:
            diagnostics_record["skip_reason"] = metadata["skip_reason"]
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        model_frame = X.copy()
        model_frame["transparency_score"] = y.values
        sampled_frame, sampling_applied = sample_frame(model_frame, model_sample_size, stratify_by=None)
        y_sample = sampled_frame["transparency_score"]
        X_sample = sampled_frame.drop(columns=["transparency_score"])

        if y_sample.shape[0] <= X_sample.shape[1] + 10:
            diagnostics_record["skip_reason"] = "insufficient_rows_for_model"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        model = sm.OLS(y_sample, X_sample).fit()
        confidence_intervals = model.conf_int()
        coefficients_df = pd.DataFrame(
            {
                "term": model.params.index,
                "coefficient": model.params.values,
                "standard_error": model.bse.values,
                "test_statistic": model.tvalues.values,
                "p_value": model.pvalues.values,
                "ci_lower": confidence_intervals.iloc[:, 0].values,
                "ci_upper": confidence_intervals.iloc[:, 1].values,
            }
        )

        predictions = model.predict(X_sample)
        residuals = y_sample - predictions
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        mae = float(np.mean(np.abs(residuals)))

        diagnostics_record.update(
            {
                "n_used": int(y_sample.shape[0]),
                "sampling_applied": sampling_applied,
                "metric_r2": float(model.rsquared),
                "metric_adj_r2": float(model.rsquared_adj),
                "metric_rmse": rmse,
                "metric_mae": mae,
                "metric_aic": float(model.aic),
                "metric_bic": float(model.bic),
                "large_sample_warning": has_large_sample_warning(int(y_sample.shape[0])),
                "status": "ok",
                "skip_reason": None,
            }
        )
        payload = {"fitted_values": predictions, "residuals": residuals}
        return coefficients_df, pd.DataFrame([diagnostics_record]), payload
    except Exception as exc:
        log_message(f"Linear regression model failed: {exc}", level="error")
        fallback = pd.DataFrame(
            [
                {
                    "model_name": "linear_regression",
                    "target_column": "transparency_score",
                    "n_used": 0,
                    "sampling_applied": False,
                    "metric_r2": np.nan,
                    "metric_adj_r2": np.nan,
                    "metric_rmse": np.nan,
                    "metric_mae": np.nan,
                    "metric_aic": np.nan,
                    "metric_bic": np.nan,
                    "large_sample_warning": False,
                    "status": "skipped",
                    "skip_reason": str(exc),
                }
            ]
        )
        return pd.DataFrame(columns=["term"]), fallback, {"fitted_values": pd.Series(dtype=float), "residuals": pd.Series(dtype=float)}
    finally:
        log_message("Linear regression modeling step finished.")


@task(name="fit_logistic_regression_model")
@timing_decorator
@validate
def fit_logistic_regression_model(
    df: pd.DataFrame,
    candidate_spec: dict[str, Any],
    top_n_categories: int,
    min_group_size: int,
    model_sample_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    try:
        diagnostics_record = {
            "model_name": "logistic_regression",
            "target_column": "riesgo_baja_transparencia",
            "n_used": 0,
            "sampling_applied": False,
            "metric_r2": np.nan,
            "metric_adj_r2": np.nan,
            "metric_rmse": np.nan,
            "metric_mae": np.nan,
            "metric_aic": np.nan,
            "metric_bic": np.nan,
            "large_sample_warning": False,
            "status": "skipped",
            "skip_reason": None,
        }
        payload = {"predicted_probability": pd.Series(dtype=float), "observed": pd.Series(dtype=float)}

        if not STATSMODELS_AVAILABLE:
            diagnostics_record["skip_reason"] = "statsmodels_unavailable"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        if "riesgo_baja_transparencia" not in df.columns:
            diagnostics_record["skip_reason"] = "target_not_available"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        X, y, metadata = build_model_design_matrix(
            df=df,
            target_column="riesgo_baja_transparencia",
            continuous_predictors=candidate_spec["linear_continuous_predictors"],
            binary_predictors=candidate_spec["linear_binary_predictors"],
            categorical_predictors=candidate_spec["linear_categorical_predictors"],
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
        )

        if metadata["skip_reason"]:
            diagnostics_record["skip_reason"] = metadata["skip_reason"]
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        y = y.astype(int)
        if y.nunique(dropna=True) < 2:
            diagnostics_record["skip_reason"] = "target_has_single_class"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        model_frame = X.copy()
        model_frame["riesgo_baja_transparencia"] = y.values
        sampled_frame, sampling_applied = sample_frame(
            model_frame,
            model_sample_size,
            stratify_by="riesgo_baja_transparencia",
        )
        y_sample = sampled_frame["riesgo_baja_transparencia"].astype(int)
        X_sample = sampled_frame.drop(columns=["riesgo_baja_transparencia"])

        if y_sample.shape[0] <= X_sample.shape[1] + 10:
            diagnostics_record["skip_reason"] = "insufficient_rows_for_model"
            return pd.DataFrame(columns=["term"]), pd.DataFrame([diagnostics_record]), payload

        glm_model = sm.GLM(y_sample, X_sample, family=sm.families.Binomial()).fit()
        confidence_intervals = glm_model.conf_int()
        coefficients_df = pd.DataFrame(
            {
                "term": glm_model.params.index,
                "coefficient": glm_model.params.values,
                "odds_ratio": np.exp(glm_model.params.values),
                "standard_error": glm_model.bse.values,
                "test_statistic": glm_model.tvalues.values,
                "p_value": glm_model.pvalues.values,
                "ci_lower": confidence_intervals.iloc[:, 0].values,
                "ci_upper": confidence_intervals.iloc[:, 1].values,
                "odds_ratio_ci_lower": np.exp(confidence_intervals.iloc[:, 0].values),
                "odds_ratio_ci_upper": np.exp(confidence_intervals.iloc[:, 1].values),
            }
        )

        predicted_probability = glm_model.predict(X_sample)
        predicted_class = (predicted_probability >= 0.5).astype(int)
        accuracy = float((predicted_class == y_sample).mean())
        brier_score = float(np.mean(np.square(predicted_probability - y_sample)))

        diagnostics_record.update(
            {
                "n_used": int(y_sample.shape[0]),
                "sampling_applied": sampling_applied,
                "metric_r2": accuracy,
                "metric_adj_r2": brier_score,
                "metric_rmse": np.nan,
                "metric_mae": np.nan,
                "metric_aic": float(glm_model.aic),
                "metric_bic": np.nan,
                "large_sample_warning": has_large_sample_warning(int(y_sample.shape[0])),
                "status": "ok",
                "skip_reason": None,
            }
        )
        payload = {"predicted_probability": predicted_probability, "observed": y_sample}
        return coefficients_df, pd.DataFrame([diagnostics_record]), payload
    except Exception as exc:
        log_message(f"Logistic regression model failed: {exc}", level="error")
        fallback = pd.DataFrame(
            [
                {
                    "model_name": "logistic_regression",
                    "target_column": "riesgo_baja_transparencia",
                    "n_used": 0,
                    "sampling_applied": False,
                    "metric_r2": np.nan,
                    "metric_adj_r2": np.nan,
                    "metric_rmse": np.nan,
                    "metric_mae": np.nan,
                    "metric_aic": np.nan,
                    "metric_bic": np.nan,
                    "large_sample_warning": False,
                    "status": "skipped",
                    "skip_reason": str(exc),
                }
            ]
        )
        return pd.DataFrame(columns=["term"]), fallback, {"predicted_probability": pd.Series(dtype=float), "observed": pd.Series(dtype=float)}
    finally:
        log_message("Logistic regression modeling step finished.")


@task(name="generate_inferential_visualizations")
@timing_decorator
@validate
def generate_inferential_visualizations(
    df: pd.DataFrame,
    output_dirs: dict[str, Path],
    candidate_spec: dict[str, Any],
    numeric_ci_df: pd.DataFrame,
    proportion_ci_df: pd.DataFrame,
    group_results_df: pd.DataFrame,
    correlation_matrix_df: pd.DataFrame,
    linear_model_payload: dict[str, Any],
    logistic_model_payload: dict[str, Any],
    plot_sample_size: int,
    top_n_categories: int,
    min_group_size: int,
) -> dict[str, list[str]]:
    generated_plots = {
        "confidence_plots": [],
        "group_plots": [],
        "correlation_plots": [],
        "model_plots": [],
    }

    try:
        if not PLOTTING_AVAILABLE:
            log_message("Plot generation skipped because plotting dependencies are not available.", level="warning")
            return generated_plots

        if not numeric_ci_df.empty:
            plot_df = numeric_ci_df.head(12).sort_values("estimate", ascending=True)
            fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(plot_df))))
            ax.errorbar(
                x=plot_df["estimate"],
                y=plot_df["column"],
                xerr=[
                    np.maximum(0, plot_df["estimate"] - plot_df["ci_lower"]),
                    np.maximum(0, plot_df["ci_upper"] - plot_df["estimate"]),
                ],
                fmt="o",
                color="#1d3557",
                ecolor="#457b9d",
                capsize=4,
            )
            ax.set_title("Numeric Mean Confidence Intervals")
            ax.set_xlabel("Estimate")
            ci_numeric_path = output_dirs["confidence_plots"] / "numeric_confidence_intervals.png"
            safe_save_plot(fig, ci_numeric_path)
            generated_plots["confidence_plots"].append(str(ci_numeric_path))

        if not proportion_ci_df.empty:
            plot_df = proportion_ci_df.groupby("column", as_index=False).head(1).head(12).copy()
            plot_df = plot_df.sort_values("estimate", ascending=False)
            fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(plot_df))))
            ax.barh(plot_df["column"], plot_df["estimate"], color="#2a9d8f")
            ax.errorbar(
                x=plot_df["estimate"],
                y=plot_df["column"],
                xerr=[
                    np.maximum(0, plot_df["estimate"] - plot_df["ci_lower"]),
                    np.maximum(0, plot_df["ci_upper"] - plot_df["estimate"]),
                ],
                fmt="none",
                ecolor="#264653",
                capsize=4,
            )
            ax.set_title("Proportion Confidence Intervals")
            ax.set_xlabel("Estimated proportion")
            ci_prop_path = output_dirs["confidence_plots"] / "proportion_confidence_intervals.png"
            safe_save_plot(fig, ci_prop_path)
            generated_plots["confidence_plots"].append(str(ci_prop_path))

        top_group_results = group_results_df.loc[group_results_df["status"] == "ok"].head(4)
        for row in top_group_results.itertuples(index=False):
            analysis_df, metadata = prepare_group_analysis_frame(
                df=df,
                outcome_column=row.outcome_column,
                group_column=row.group_column,
                top_n_categories=top_n_categories,
                min_group_size=min_group_size,
            )
            if metadata["skip_reason"] or analysis_df.empty:
                continue
            if analysis_df.shape[0] > plot_sample_size:
                analysis_df, _ = sample_frame(analysis_df, plot_sample_size)

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=analysis_df, x=row.group_column, y=row.outcome_column, ax=ax, color="#8ecae6")
            ax.set_title(f"Boxplot - {row.outcome_column} by {row.group_column}")
            ax.tick_params(axis="x", rotation=35)
            box_path = output_dirs["group_plots"] / f"{row.outcome_column}_by_{row.group_column}_box.png"
            safe_save_plot(fig, box_path)
            generated_plots["group_plots"].append(str(box_path))

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(data=analysis_df, x=row.group_column, y=row.outcome_column, ax=ax, color="#f4a261")
            ax.set_title(f"Violin Plot - {row.outcome_column} by {row.group_column}")
            ax.tick_params(axis="x", rotation=35)
            violin_path = output_dirs["group_plots"] / f"{row.outcome_column}_by_{row.group_column}_violin.png"
            safe_save_plot(fig, violin_path)
            generated_plots["group_plots"].append(str(violin_path))

        if not correlation_matrix_df.empty and "variable" in correlation_matrix_df.columns and correlation_matrix_df.shape[1] > 2:
            matrix = correlation_matrix_df.set_index("variable")
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Spearman Correlation Heatmap")
            heatmap_path = output_dirs["correlation_plots"] / "correlation_heatmap.png"
            safe_save_plot(fig, heatmap_path)
            generated_plots["correlation_plots"].append(str(heatmap_path))

        fitted_values = linear_model_payload.get("fitted_values", pd.Series(dtype=float))
        residuals = linear_model_payload.get("residuals", pd.Series(dtype=float))
        if not fitted_values.empty and not residuals.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=fitted_values, y=residuals, ax=ax, s=20, color="#457b9d")
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title("OLS Residuals vs Fitted")
            ax.set_xlabel("Fitted values")
            ax.set_ylabel("Residuals")
            residual_plot_path = output_dirs["model_plots"] / "linear_residuals_vs_fitted.png"
            safe_save_plot(fig, residual_plot_path)
            generated_plots["model_plots"].append(str(residual_plot_path))

            fig, ax = plt.subplots(figsize=(10, 6))
            stats.probplot(residuals, dist="norm", plot=ax)
            ax.set_title("OLS Residual QQ Plot")
            qq_path = output_dirs["model_plots"] / "linear_residuals_qq.png"
            safe_save_plot(fig, qq_path)
            generated_plots["model_plots"].append(str(qq_path))

        predicted_probability = logistic_model_payload.get("predicted_probability", pd.Series(dtype=float))
        if not predicted_probability.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(predicted_probability, bins=30, kde=True, ax=ax, color="#e76f51")
            ax.set_title("Predicted Probability Distribution - Logistic Model")
            ax.set_xlabel("Predicted probability")
            logistic_path = output_dirs["model_plots"] / "logistic_predicted_probability_distribution.png"
            safe_save_plot(fig, logistic_path)
            generated_plots["model_plots"].append(str(logistic_path))

        return generated_plots
    except Exception as exc:
        log_message(f"Inferential visualization generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Inferential visualization step finished.")


@task(name="write_executive_summary")
@timing_decorator
@validate
def write_executive_summary(
    output_dirs: dict[str, Path],
    candidate_spec: dict[str, Any],
    assumptions_summary: pd.DataFrame,
    numeric_ci_df: pd.DataFrame,
    proportion_ci_df: pd.DataFrame,
    group_results_df: pd.DataFrame,
    posthoc_df: pd.DataFrame,
    categorical_assoc_df: pd.DataFrame,
    numeric_assoc_df: pd.DataFrame,
    model_diagnostics_df: pd.DataFrame,
) -> str:
    try:
        executed_group_tests = group_results_df.loc[group_results_df["status"] == "ok"]
        executed_categorical_tests = categorical_assoc_df.loc[categorical_assoc_df["status"] == "ok"]
        executed_numeric_tests = numeric_assoc_df.loc[numeric_assoc_df["status"] == "ok"]
        executed_models = model_diagnostics_df.loc[model_diagnostics_df["status"] == "ok"]

        significant_group_tests = executed_group_tests.loc[executed_group_tests["p_value"] < 0.05].head(5)
        significant_categorical_tests = executed_categorical_tests.loc[executed_categorical_tests["p_value"] < 0.05].head(5)
        significant_numeric_tests = executed_numeric_tests.loc[
            (executed_numeric_tests["spearman_p_value"] < 0.05) | (executed_numeric_tests["pearson_p_value"] < 0.05)
        ].head(5)

        summary_lines = [
            "# Executive Summary - SECOP II Inferential Analysis",
            "",
            "## Overview",
            f"- Input dataset: `{output_dirs['input_path']}`",
            f"- Candidate variables reviewed: {int(candidate_spec['candidate_inventory'].shape[0]):,}",
            f"- Group comparisons attempted: {int(group_results_df.shape[0]):,}",
            f"- Categorical associations attempted: {int(categorical_assoc_df.shape[0]):,}",
            f"- Numeric associations attempted: {int(numeric_assoc_df.shape[0]):,}",
            f"- Models estimated successfully: {int(executed_models.shape[0]):,}",
            "",
            "## Confidence Intervals",
            f"- Numeric mean confidence intervals generated: {int(numeric_ci_df.shape[0]):,}",
            f"- Proportion confidence intervals generated: {int(proportion_ci_df.shape[0]):,}",
            "",
            "## Assumptions and Defensive Logic",
            f"- Assumption checks generated: {int(assumptions_summary.shape[0]):,}",
            f"- Skipped group comparisons: {int((group_results_df['status'] == 'skipped').sum()) if not group_results_df.empty else 0:,}",
            "- Normality was evaluated with Shapiro-Wilk on bounded samples, and homoscedasticity with Levene when applicable.",
            "- For large samples, p-values can become trivially small; effect sizes should be prioritized in interpretation.",
            "",
            "## Notable Group Differences",
        ]

        if significant_group_tests.empty:
            summary_lines.append("- No group comparison reached the significance threshold after defensive filtering, or all valid tests were skipped.")
        else:
            for row in significant_group_tests.itertuples(index=False):
                summary_lines.append(
                    f"- `{row.outcome_column}` vs `{row.group_column}`: {row.test_name} with p-value {row.p_value:.4g} and {row.effect_size_name}={row.effect_size_value:.4f}."
                )

        summary_lines.extend(["", "## Notable Categorical Associations"])
        if significant_categorical_tests.empty:
            summary_lines.append("- No categorical association showed statistically significant dependence under the executed tests.")
        else:
            for row in significant_categorical_tests.itertuples(index=False):
                summary_lines.append(
                    f"- `{row.left_column}` vs `{row.right_column}`: {row.test_name} with p-value {row.p_value:.4g} and Cramer's V={row.cramers_v:.4f}."
                )

        summary_lines.extend(["", "## Notable Numeric Associations"])
        if significant_numeric_tests.empty:
            summary_lines.append("- No numeric pair showed a statistically meaningful association after filtering and testing.")
        else:
            for row in significant_numeric_tests.itertuples(index=False):
                summary_lines.append(
                    f"- `{row.left_column}` vs `{row.right_column}`: Pearson r={row.pearson_r:.4f} (p={row.pearson_p_value:.4g}) and Spearman rho={row.spearman_rho:.4f} (p={row.spearman_p_value:.4g})."
                )

        summary_lines.extend(["", "## Model Diagnostics"])
        if executed_models.empty:
            summary_lines.append("- No inferential model was fitted successfully. Review `model_diagnostics.csv` for skip reasons.")
        else:
            for row in executed_models.itertuples(index=False):
                summary_lines.append(
                    f"- `{row.model_name}` on `{row.target_column}` used {int(row.n_used):,} rows with AIC={row.metric_aic:.4f}."
                )

        if not posthoc_df.empty:
            significant_posthoc = posthoc_df.loc[posthoc_df["significant_after_holm"]].head(5)
            summary_lines.extend(["", "## Post Hoc Signals"])
            if significant_posthoc.empty:
                summary_lines.append("- No pairwise post hoc contrast remained significant after Holm correction.")
            else:
                for row in significant_posthoc.itertuples(index=False):
                    summary_lines.append(
                        f"- `{row.outcome_column}` by `{row.group_column}`: `{row.left_group}` vs `{row.right_group}` remained significant after Holm correction (adjusted p={row.holm_adjusted_p_value:.4g})."
                    )

        summary_lines.extend(
            [
                "",
                "## Interpretation Notes",
                "- Statistical significance does not imply causality.",
                "- Results should be interpreted alongside data quality, missingness patterns, derived-score construction, and the institutional context of SECOP II.",
                "- Variables documented as leakage or post-outcome fields were excluded from inferential modeling to avoid tautological or contaminated conclusions.",
                "- Sparse temporal coverage and highly imbalanced binary outcomes remain important limitations for institutional transparency inference.",
            ]
        )

        output_path = output_dirs["base"] / "executive_summary.md"
        safe_write_markdown("\n".join(summary_lines), output_path)
        return str(output_path)
    except Exception as exc:
        log_message(f"Executive summary generation failed: {exc}", level="error")
        raise
    finally:
        log_message("Executive summary step finished.")


@task(name="build_run_metadata")
@timing_decorator
@validate
def build_run_metadata(
    output_dirs: dict[str, Path],
    environment_status: dict[str, Any],
    context: dict[str, Any],
    candidate_spec: dict[str, Any],
    assumptions_summary: pd.DataFrame,
    group_results_df: pd.DataFrame,
    categorical_assoc_df: pd.DataFrame,
    numeric_assoc_df: pd.DataFrame,
    model_diagnostics_df: pd.DataFrame,
    generated_plots: dict[str, list[str]],
    artifacts: dict[str, str],
) -> dict[str, Any]:
    return {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(output_dirs["input_path"]),
        "output_dir": str(output_dirs["base"]),
        "project_root": str(output_dirs["project_root"]),
        "plotting_available": environment_status.get("plotting_available", PLOTTING_AVAILABLE),
        "plotting_import_error": environment_status.get("plotting_import_error", PLOTTING_IMPORT_ERROR),
        "prefect_available": environment_status.get("prefect_available", PREFECT_AVAILABLE),
        "statsmodels_available": environment_status.get("statsmodels_available", STATSMODELS_AVAILABLE),
        "statsmodels_import_error": environment_status.get("statsmodels_import_error", STATSMODELS_IMPORT_ERROR),
        "scipy_available": environment_status.get("scipy_available", SCIPY_AVAILABLE),
        "scipy_import_error": environment_status.get("scipy_import_error", SCIPY_IMPORT_ERROR),
        "dependencies": environment_status.get("bootstrap_status", BOOTSTRAP_STATUS),
        "alias_map": context.get("alias_map", {}),
        "conversion_summary": context.get("conversion_summary", []),
        "model_excluded_columns_present": context.get("model_excluded_columns_present", []),
        "candidate_summary": {
            "candidate_variables": int(candidate_spec["candidate_inventory"].shape[0]),
            "group_pairs": int(len(candidate_spec["group_pairs"])),
            "numeric_association_vars": int(len(candidate_spec["numeric_association_vars"])),
            "categorical_association_vars": int(len(candidate_spec["categorical_association_vars"])),
        },
        "result_summary": {
            "assumption_checks": int(assumptions_summary.shape[0]),
            "group_tests_ok": int((group_results_df["status"] == "ok").sum()) if not group_results_df.empty else 0,
            "categorical_tests_ok": int((categorical_assoc_df["status"] == "ok").sum()) if not categorical_assoc_df.empty else 0,
            "numeric_tests_ok": int((numeric_assoc_df["status"] == "ok").sum()) if not numeric_assoc_df.empty else 0,
            "models_ok": int((model_diagnostics_df["status"] == "ok").sum()) if not model_diagnostics_df.empty else 0,
        },
        "generated_plots": generated_plots,
        "artifacts": artifacts,
    }


@flow(name="secop_inferential_analysis_flow")
def secop_inferential_analysis_flow(
    input_path: str = str(DEFAULT_INPUT_PATH),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    top_n_categories: int = DEFAULT_TOP_N_CATEGORIES,
    max_group_comparisons: int = DEFAULT_MAX_GROUP_COMPARISONS,
    max_correlation_features: int = DEFAULT_MAX_CORRELATION_FEATURES,
    plot_sample_size: int = DEFAULT_PLOT_SAMPLE_SIZE,
    normality_sample_size: int = DEFAULT_NORMALITY_SAMPLE_SIZE,
    model_sample_size: int = DEFAULT_MODEL_SAMPLE_SIZE,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> dict[str, str]:
    try:
        output_dirs = resolve_paths_and_create_output_dirs(input_path=input_path, output_dir=output_dir)
        environment_status = bootstrap_environment(requirements_path=output_dirs["requirements_path"])
        raw_df = load_parquet_dataset(parquet_path=output_dirs["input_path"])
        prepared_df, context = prepare_dataframe_for_inference(df=raw_df)

        candidate_spec = select_inferential_candidates(
            df=prepared_df,
            context=context,
            top_n_categories=top_n_categories,
            max_group_comparisons=max_group_comparisons,
            max_correlation_features=max_correlation_features,
            min_group_size=min_group_size,
        )
        assumptions_summary = evaluate_assumptions(
            df=prepared_df,
            candidate_spec=candidate_spec,
            top_n_categories=top_n_categories,
            normality_sample_size=normality_sample_size,
            min_group_size=min_group_size,
        )
        numeric_ci_df, proportion_ci_df = build_confidence_intervals(
            df=prepared_df,
            candidate_spec=candidate_spec,
            confidence_level=confidence_level,
            plot_sample_size=plot_sample_size,
            top_n_categories=top_n_categories,
        )
        group_results_df, posthoc_df = run_group_comparisons(
            df=prepared_df,
            candidate_spec=candidate_spec,
            assumptions_summary=assumptions_summary,
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
        )
        categorical_assoc_df = run_categorical_associations(
            df=prepared_df,
            candidate_spec=candidate_spec,
            output_dirs=output_dirs,
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
        )
        numeric_assoc_df, correlation_matrix_df = run_numeric_associations(
            df=prepared_df,
            candidate_spec=candidate_spec,
            min_group_size=min_group_size,
        )
        linear_coefficients_df, linear_diagnostics_df, linear_model_payload = fit_linear_regression_model(
            df=prepared_df,
            candidate_spec=candidate_spec,
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
            model_sample_size=model_sample_size,
        )
        logistic_coefficients_df, logistic_diagnostics_df, logistic_model_payload = fit_logistic_regression_model(
            df=prepared_df,
            candidate_spec=candidate_spec,
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
            model_sample_size=model_sample_size,
        )

        model_diagnostics_df = pd.concat(
            [linear_diagnostics_df, logistic_diagnostics_df],
            ignore_index=True,
        )

        generated_plots = generate_inferential_visualizations(
            df=prepared_df,
            output_dirs=output_dirs,
            candidate_spec=candidate_spec,
            numeric_ci_df=numeric_ci_df,
            proportion_ci_df=proportion_ci_df,
            group_results_df=group_results_df,
            correlation_matrix_df=correlation_matrix_df,
            linear_model_payload=linear_model_payload,
            logistic_model_payload=logistic_model_payload,
            plot_sample_size=plot_sample_size,
            top_n_categories=top_n_categories,
            min_group_size=min_group_size,
        )

        artifacts: dict[str, str] = {}
        dataframe_outputs = {
            "candidate_variables.csv": candidate_spec["candidate_inventory"],
            "assumptions_summary.csv": assumptions_summary,
            "numeric_mean_confidence_intervals.csv": numeric_ci_df,
            "proportion_confidence_intervals.csv": proportion_ci_df,
            "group_comparison_results.csv": group_results_df,
            "posthoc_pairwise_results.csv": posthoc_df,
            "chi_square_results.csv": categorical_assoc_df,
            "correlation_inference.csv": numeric_assoc_df,
            "correlation_matrix.csv": correlation_matrix_df,
            "linear_model_coefficients.csv": linear_coefficients_df,
            "logistic_model_coefficients.csv": logistic_coefficients_df,
            "model_diagnostics.csv": model_diagnostics_df,
        }

        output_location_lookup = {
            "candidate_variables.csv": output_dirs["base"],
            "assumptions_summary.csv": output_dirs["assumptions"],
            "numeric_mean_confidence_intervals.csv": output_dirs["confidence_intervals"],
            "proportion_confidence_intervals.csv": output_dirs["confidence_intervals"],
            "group_comparison_results.csv": output_dirs["group_comparisons"],
            "posthoc_pairwise_results.csv": output_dirs["group_comparisons"],
            "chi_square_results.csv": output_dirs["categorical_associations"],
            "correlation_inference.csv": output_dirs["numeric_associations"],
            "correlation_matrix.csv": output_dirs["numeric_associations"],
            "linear_model_coefficients.csv": output_dirs["models"],
            "logistic_model_coefficients.csv": output_dirs["models"],
            "model_diagnostics.csv": output_dirs["models"],
        }

        for filename, dataframe in dataframe_outputs.items():
            output_path = output_location_lookup[filename] / filename
            safe_write_csv(dataframe, output_path)
            artifacts[filename] = str(output_path)

        summary_path = write_executive_summary(
            output_dirs=output_dirs,
            candidate_spec=candidate_spec,
            assumptions_summary=assumptions_summary,
            numeric_ci_df=numeric_ci_df,
            proportion_ci_df=proportion_ci_df,
            group_results_df=group_results_df,
            posthoc_df=posthoc_df,
            categorical_assoc_df=categorical_assoc_df,
            numeric_assoc_df=numeric_assoc_df,
            model_diagnostics_df=model_diagnostics_df,
        )
        artifacts["executive_summary.md"] = summary_path

        run_metadata = build_run_metadata(
            output_dirs=output_dirs,
            environment_status=environment_status,
            context=context,
            candidate_spec=candidate_spec,
            assumptions_summary=assumptions_summary,
            group_results_df=group_results_df,
            categorical_assoc_df=categorical_assoc_df,
            numeric_assoc_df=numeric_assoc_df,
            model_diagnostics_df=model_diagnostics_df,
            generated_plots=generated_plots,
            artifacts=artifacts,
        )
        run_metadata_path = output_dirs["base"] / "run_metadata.json"
        safe_write_json(run_metadata, run_metadata_path)
        artifacts["run_metadata.json"] = str(run_metadata_path)

        return artifacts
    except Exception as exc:
        log_message(f"Inferential SECOP flow failed: {exc}", level="error")
        raise
    finally:
        log_message("Inferential SECOP flow finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inferential analysis for the SECOP II feature engineering dataset.")
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the input parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where inferential outputs will be stored.",
    )
    parser.add_argument(
        "--confidence-level",
        default=DEFAULT_CONFIDENCE_LEVEL,
        type=float,
        help="Confidence level for interval estimation.",
    )
    parser.add_argument(
        "--top-n-categories",
        default=DEFAULT_TOP_N_CATEGORIES,
        type=int,
        help="Number of top categories retained before collapsing the remainder into 'Other'.",
    )
    parser.add_argument(
        "--max-group-comparisons",
        default=DEFAULT_MAX_GROUP_COMPARISONS,
        type=int,
        help="Maximum number of prioritized outcome/group pairs evaluated.",
    )
    parser.add_argument(
        "--max-correlation-features",
        default=DEFAULT_MAX_CORRELATION_FEATURES,
        type=int,
        help="Maximum number of numeric features included in inferential correlation analysis.",
    )
    parser.add_argument(
        "--plot-sample-size",
        default=DEFAULT_PLOT_SAMPLE_SIZE,
        type=int,
        help="Maximum number of rows sampled for visualization-heavy steps.",
    )
    parser.add_argument(
        "--normality-sample-size",
        default=DEFAULT_NORMALITY_SAMPLE_SIZE,
        type=int,
        help="Maximum sample size used for Shapiro-Wilk normality checks.",
    )
    parser.add_argument(
        "--model-sample-size",
        default=DEFAULT_MODEL_SAMPLE_SIZE,
        type=int,
        help="Maximum sample size used for inferential models.",
    )
    parser.add_argument(
        "--min-group-size",
        default=DEFAULT_MIN_GROUP_SIZE,
        type=int,
        help="Minimum observations required per valid group level.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    secop_inferential_analysis_flow(
        input_path=cli_args.input_path,
        output_dir=cli_args.output_dir,
        confidence_level=cli_args.confidence_level,
        top_n_categories=cli_args.top_n_categories,
        max_group_comparisons=cli_args.max_group_comparisons,
        max_correlation_features=cli_args.max_correlation_features,
        plot_sample_size=cli_args.plot_sample_size,
        normality_sample_size=cli_args.normality_sample_size,
        model_sample_size=cli_args.model_sample_size,
        min_group_size=cli_args.min_group_size,
    )
