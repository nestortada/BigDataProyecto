from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
import logging
import os
import platform
import time
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))
os.environ.setdefault("PREFECT_HOME", str(Path("/tmp") / "prefect"))

import joblib
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Prefect
from prefect import flow, task
from prefect.logging import get_run_logger

# Scikit-Learn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Modelos opcionales
try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


# ==============================================================================
# CONFIGURACION GENERAL Y RUTAS
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "Data" / "Processed" / "Limpieza" / "datos_feature_engineering.parquet"
DEFAULT_PROCESSED_DATA_DIR = PROJECT_ROOT / "Data" / "Processed" / "Model"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "Reports" / "Model"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "Models" / "trained"

TARGET_COL = "riesgo_baja_transparencia"
DEFAULT_RANDOM_STATE = 42
DEFAULT_METRIC = "F1_score"
DEFAULT_N_FOLDS = 5
LEAKAGE_CORRELATION_THRESHOLD = 0.995
PERFECT_METRIC_WARNING_THRESHOLD = 0.995
STRICT_MODE = False
EARLY_FEATURES_ONLY = True

# Columnas estrictamente prohibidas por data leakage o por ser metadata granular.
LEAKAGE_COLS = [
    "adjudicado",
    "fecha_adjudicacion",
    "valor_total_adjudicacion",
    "estado_resumen",
    "transparency_score",
    "nivel_riesgo_transparencia",
    "confianza_label",
    "margin_to_threshold",
    "evidence_coverage",
]

METADATA_COLS = [
    "id_del_proceso",
    "referencia_del_proceso",
    "id_del_portafolio",
    "urlproceso",
    "nombre_del_procedimiento",
    "descripci_n_del_procedimiento",
    "fecha_de_publicacion_del",
    "fecha_de_recepcion_de",
    "fecha_de_apertura_de_respuesta",
]

# Componentes usados directa o indirectamente para construir el score/target en
# feature_engineering_secop.py. Se excluyen para evitar que el modelo aprenda la
# regla de etiquetado en vez de patrones defendibles ex ante.
TARGET_DERIVED_COLS = [
    "score_completitud",
    "score_trazabilidad_base",
    "score_trazabilidad",
    "score_temporal",
    "score_competencia",
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
    "flag_fechas_temporales_disponibles",
    "flag_publicacion_antes_recepcion",
    "flag_recepcion_antes_apertura",
    "flag_ventana_recepcion_razonable",
    "flag_ventana_apertura_razonable",
    "flag_coherencia_temporal_global",
    "flag_datos_competencia_disponibles",
    "flag_hubo_participacion",
    "flag_hubo_competencia_minima",
    "intensidad_competencia_normalizada",
    "flag_evidencia_completitud_suficiente",
    "flag_evidencia_trazabilidad_suficiente",
    "flag_evidencia_temporal_suficiente",
    "flag_evidencia_competencia_suficiente",
]

LEAKAGE_NAME_PATTERNS = {
    "derivada del target o score de transparencia": [
        "transparency",
        "riesgo",
        "nivel_riesgo",
        "confianza",
        "score_",
        "evidence",
        "evidencia",
        "margin_to_threshold",
    ],
    "resultado posterior a apertura/adjudicacion": [
        "adjudic",
        "respuesta",
        "respuestas",
        "proveedor",
        "proveedores",
        "oferente",
        "oferentes",
        "competencia",
        "participacion",
        "visualizaciones",
        "total_",
        "conteo_de_respuestas",
    ],
}

STRICT_TEMPORAL_EXCLUDE_COLS = [
    "dias_",
    "tiempo_",
    "duracion",
    "fecha_",
    "temporal",
    "missingindicator_dias",
    "missing_temporal",
    "tiene_fecha",
    "flag_",
]

EARLY_ALLOWED_FEATURES = [
    "entidad",
    "nit_entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "ordenentidad",
    "modalidad_de_contratacion",
    "justificacion_modalidad",
    "justificaci_n_modalidad_de",
    "tipo_de_contrato",
    "subtipo_de_contrato",
    "categoria",
    "categorias_adicionales",
    "codigo_principal_de_categoria",
    "precio_base",
    "precio_base_log",
    "fase",
    "nombre_del_procedimiento",
    "descripcion_del_procedimiento",
    "descripci_n_del_procedimiento",
]

EARLY_FORBIDDEN_PATTERNS = [
    "score",
    "transparency",
    "riesgo",
    "confianza",
    "evidence",
    "coverage",
    "completitud",
    "trazabilidad",
    "temporal",
    "competencia",
    "coherencia",
    "anomalia",
    "flag",
    "missing",
    "missingindicator",
    "count",
    "conteo",
    "total",
    "ratio",
    "proveedores",
    "respuestas",
    "visualizaciones",
    "fecha_adjudicacion",
    "valor_total_adjudicacion",
    "adjudicado",
    "estado_resumen",
    "id_del_proceso",
    "referencia_del_proceso",
    "urlproceso",
    "id_del_portafolio",
    "duracion",
    "dias",
    "tiene_",
]

EARLY_HIGH_CARDINALITY_MAX_RATIO = 0.05
EARLY_HIGH_CARDINALITY_MAX_UNIQUE = 10_000

METRIC_ALIASES = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1_score",
    "f1_score": "F1_score",
    "roc_auc": "ROC_AUC",
    "auc": "ROC_AUC",
    "pr_auc": "PR_AUC",
    "average_precision": "PR_AUC",
}


# ==============================================================================
# UTILIDADES GENERALES
# ==============================================================================


def timing_decorator(func):
    """Mide y registra el tiempo de ejecucion de una funcion."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("secop_pipeline")
        start_time = time.perf_counter()
        logger.info("Empezando ejecucion de '%s'...", func.__name__)
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info("Completado '%s' en %.2f segundos.", func.__name__, elapsed)
        return result

    return wrapper


def validate_inputs(func):
    """Valida dataframes vacios antes de ejecutar tareas criticas."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for idx, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame) and arg.empty:
                raise ValueError(f"El DataFrame en la posicion {idx} para '{func.__name__}' esta vacio.")
        return func(*args, **kwargs)

    return wrapper


def normalize_metric(metric: str) -> str:
    """Normaliza nombres de metrica recibidos desde CLI o configuracion."""
    metric_key = str(metric).strip().lower()
    if metric_key in METRIC_ALIASES:
        return METRIC_ALIASES[metric_key]
    if metric in set(METRIC_ALIASES.values()):
        return metric
    valid_metrics = sorted(set(METRIC_ALIASES.values()))
    raise ValueError(f"Metrica no soportada: {metric}. Use una de: {valid_metrics}")


def resolve_path(path_value: str | Path) -> Path:
    """Resuelve rutas relativas contra la raiz del proyecto."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_config(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_REPORTS_DIR,
    target_col: str = TARGET_COL,
    metric: str = DEFAULT_METRIC,
    strict_mode: bool = STRICT_MODE,
    early_features_only: bool = EARLY_FEATURES_ONLY,
) -> Dict[str, Any]:
    """Centraliza parametros reproducibles y rutas del pipeline."""
    reports_dir = resolve_path(output_dir)
    run_suffix = "_early" if early_features_only else "_strict" if strict_mode else ""
    config = {
        "project_root": str(PROJECT_ROOT),
        "input_path": str(resolve_path(input_path)),
        "processed_data_dir": str(DEFAULT_PROCESSED_DATA_DIR),
        "reports_dir": str(reports_dir),
        "models_dir": str(DEFAULT_MODELS_DIR),
        "target_col": target_col,
        "strict_mode": strict_mode,
        "early_features_only": early_features_only,
        "run_suffix": run_suffix,
        "experiment_name": "EARLY_FEATURES_ONLY" if early_features_only else "STRICT_TEMPORAL" if strict_mode else "NORMAL",
        "strict_temporal_exclude_patterns": STRICT_TEMPORAL_EXCLUDE_COLS,
        "early_allowed_features": EARLY_ALLOWED_FEATURES,
        "early_forbidden_patterns": EARLY_FORBIDDEN_PATTERNS,
        "early_high_cardinality_max_ratio": EARLY_HIGH_CARDINALITY_MAX_RATIO,
        "early_high_cardinality_max_unique": EARLY_HIGH_CARDINALITY_MAX_UNIQUE,
        "random_state": DEFAULT_RANDOM_STATE,
        "metric": normalize_metric(metric),
        "cv_folds": DEFAULT_N_FOLDS,
        "train_size": 0.60,
        "validation_size": 0.20,
        "test_size": 0.20,
        "leakage_correlation_threshold": LEAKAGE_CORRELATION_THRESHOLD,
        "perfect_metric_warning_threshold": PERFECT_METRIC_WARNING_THRESHOLD,
        "threshold_min": 0.05,
        "threshold_max": 0.95,
        "threshold_steps": 181,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    return config


def get_path(config: Dict[str, Any], key: str) -> Path:
    return Path(config[key])


def artifact_name(base_name: str, config: Dict[str, Any]) -> str:
    """Agrega sufijo de modo estricto antes de la extension del artefacto."""
    suffix = config.get("run_suffix", "")
    path = Path(base_name)
    if not suffix:
        return base_name
    if path.suffix:
        return f"{path.stem}{suffix}{path.suffix}"
    return f"{base_name}{suffix}"


def make_one_hot_encoder() -> OneHotEncoder:
    """Crea OneHotEncoder compatible con versiones nuevas y antiguas de sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def safe_roc_auc(y_true: pd.Series, y_score: np.ndarray | None) -> float:
    if y_score is None or pd.Series(y_true).nunique() < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_pr_auc(y_true: pd.Series, y_score: np.ndarray | None) -> float:
    if y_score is None or pd.Series(y_true).nunique() < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def get_positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray | None:
    """Obtiene probabilidades o scores continuos para la clase positiva."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def predictions_from_threshold(y_score: np.ndarray | None, threshold: float, fallback_pred: np.ndarray) -> np.ndarray:
    if y_score is None:
        return np.asarray(fallback_pred).astype(int)
    return (np.asarray(y_score) >= threshold).astype(int)


# ==============================================================================
# TAREAS DEL PIPELINE
# ==============================================================================


@task(name="1. Crear estructura de carpetas")
@timing_decorator
def check_and_create_directories(config: Dict[str, Any]) -> None:
    """Genera todas las rutas necesarias si no existen."""
    logger = get_run_logger()
    reports_dir = get_path(config, "reports_dir")
    models_dir = get_path(config, "models_dir")
    processed_data_dir = get_path(config, "processed_data_dir")

    dirs_to_create = [
        processed_data_dir,
        reports_dir,
        reports_dir / "classification_reports",
        reports_dir / "confusion_matrices",
        reports_dir / "roc_curves",
        reports_dir / "pr_curves",
        reports_dir / "feature_importance",
        reports_dir / "logs",
        models_dir,
        models_dir / "all_models",
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
    logger.info("Estructura de directorios verificada/creada exitosamente.")


@task(name="2. Guardar configuracion y entorno")
@timing_decorator
def save_reproducibility_reports(config: Dict[str, Any]) -> None:
    """Guarda configuracion de modelado y versiones principales del entorno."""
    reports_dir = get_path(config, "reports_dir")
    config_path = reports_dir / artifact_name("modeling_config.json", config)
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    packages = [
        "python",
        "pandas",
        "numpy",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "seaborn",
        "prefect",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
    lines = [
        "REPORTE DE ENTORNO",
        "=" * 50,
        f"Fecha de ejecucion: {config['created_at']}",
        f"Python: {platform.python_version()}",
        f"Plataforma: {platform.platform()}",
        "",
        "Versiones de librerias:",
    ]
    for package in packages:
        if package == "python":
            continue
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "no instalado"
        lines.append(f"- {package}: {version}")

    with open(reports_dir / artifact_name("environment_report.txt", config), "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


@task(name="3. Cargar datos")
@timing_decorator
def load_data(file_path: str | Path, target_col: str) -> pd.DataFrame:
    """Carga los datos iniciales desde parquet y revisa columna objetivo."""
    logger = get_run_logger()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo Parquet no encontrado en: {path}")

    df = pd.read_parquet(path)
    logger.info("Datos cargados. Dimensiones: %s", df.shape)

    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")

    return df


def add_drop_record(
    records: List[Dict[str, Any]],
    dropped: Dict[str, Dict[str, Any]],
    column: str,
    reason: str,
    correlation: float | None = None,
) -> None:
    """Registra una columna eliminada sin duplicar razones."""
    abs_corr = abs(correlation) if correlation is not None and not pd.isna(correlation) else np.nan
    if column not in dropped:
        dropped[column] = {
            "column": column,
            "reason": reason,
            "correlation_with_target": correlation if correlation is not None else np.nan,
            "abs_correlation_with_target": abs_corr,
            "dropped": True,
        }
        records.append(dropped[column])
    else:
        dropped[column]["reason"] = f"{dropped[column]['reason']} | {reason}"
        if correlation is not None and not pd.isna(correlation):
            dropped[column]["correlation_with_target"] = correlation
            dropped[column]["abs_correlation_with_target"] = abs_corr


def compute_numeric_correlations(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """Calcula correlaciones Pearson entre columnas numericas y target."""
    target = pd.to_numeric(df[target_col], errors="coerce")
    correlations: Dict[str, float] = {}
    for column in df.columns:
        if column == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        feature = pd.to_numeric(df[column], errors="coerce")
        valid = feature.notna() & target.notna()
        if valid.sum() < 2 or feature[valid].nunique() < 2:
            continue
        corr = feature[valid].corr(target[valid])
        if pd.notna(corr):
            correlations[column] = float(corr)
    return correlations


def normalize_column_token(value: str) -> str:
    """Normaliza texto de columnas para comparar patrones con y sin tildes."""
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().strip()


def select_early_features_only(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Selecciona solo variables tempranas defendibles y reporta la politica aplicada."""
    if target_col not in df.columns:
        raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame.")

    reports_dir = Path(df.attrs.get("reports_dir", DEFAULT_REPORTS_DIR))
    reports_dir.mkdir(parents=True, exist_ok=True)

    work_df = df.copy()
    work_df = work_df.dropna(subset=[target_col]).copy()

    allowed_tokens = {normalize_column_token(column) for column in EARLY_ALLOWED_FEATURES}
    forbidden_patterns = [normalize_column_token(pattern) for pattern in EARLY_FORBIDDEN_PATTERNS]
    n_rows = max(len(work_df), 1)

    selected_columns: List[str] = []
    selected_rows: List[Dict[str, Any]] = []
    dropped_rows: List[Dict[str, Any]] = []

    for column in work_df.columns:
        normalized_col = normalize_column_token(column)

        if column == target_col:
            selected_columns.append(column)
            selected_rows.append(
                {
                    "column": column,
                    "role": "target",
                    "dtype": str(work_df[column].dtype),
                    "non_null": int(work_df[column].notna().sum()),
                    "missing_rate": float(work_df[column].isna().mean()),
                    "unique_values": int(work_df[column].nunique(dropna=True)),
                    "unique_ratio": float(work_df[column].nunique(dropna=True) / n_rows),
                }
            )
            continue

        matched_patterns = [pattern for pattern in forbidden_patterns if pattern in normalized_col]
        is_allowed = normalized_col in allowed_tokens
        reasons = []

        if not is_allowed:
            reasons.append("no pertenece a la lista de variables tempranas permitidas")
        if matched_patterns:
            reasons.append("patron prohibido: " + "|".join(matched_patterns))

        if normalized_col == "nit_entidad" and is_allowed and not matched_patterns:
            unique_values = int(work_df[column].nunique(dropna=True))
            unique_ratio = float(unique_values / n_rows)
            if unique_ratio > EARLY_HIGH_CARDINALITY_MAX_RATIO or unique_values > EARLY_HIGH_CARDINALITY_MAX_UNIQUE:
                reasons.append(
                    "nit_entidad eliminado por posible memorizacion "
                    f"(unique_values={unique_values}, unique_ratio={unique_ratio:.4f})"
                )

        if reasons:
            dropped_rows.append(
                {
                    "column": column,
                    "reason": " | ".join(reasons),
                    "matched_pattern": "|".join(matched_patterns),
                    "allowed_early_feature": bool(is_allowed),
                }
            )
            continue

        selected_columns.append(column)
        selected_rows.append(
            {
                "column": column,
                "role": "feature",
                "dtype": str(work_df[column].dtype),
                "non_null": int(work_df[column].notna().sum()),
                "missing_rate": float(work_df[column].isna().mean()),
                "unique_values": int(work_df[column].nunique(dropna=True)),
                "unique_ratio": float(work_df[column].nunique(dropna=True) / n_rows),
            }
        )

    if target_col not in selected_columns:
        selected_columns.append(target_col)

    selected_df = pd.DataFrame(
        selected_rows,
        columns=["column", "role", "dtype", "non_null", "missing_rate", "unique_values", "unique_ratio"],
    )
    dropped_df = pd.DataFrame(
        dropped_rows,
        columns=["column", "reason", "matched_pattern", "allowed_early_feature"],
    )
    selected_df.to_csv(reports_dir / "early_features_selected.csv", index=False)
    dropped_df.to_csv(reports_dir / "early_features_dropped.csv", index=False)

    feature_names = [row["column"] for row in selected_rows if row["role"] == "feature"]
    report_lines = [
        "POLITICA DE VARIABLES TEMPRANAS - EARLY_FEATURES_ONLY",
        "=" * 70,
        "",
        "Objetivo:",
        (
            "Entrenar una version minimalista usando solo informacion disponible al inicio "
            "del proceso contractual."
        ),
        "",
        "Variables usadas:",
        *[f"- {column}" for column in feature_names],
        "",
        "Variables eliminadas y razon:",
        *[
            f"- {row['column']}: {row['reason']}"
            for _, row in dropped_df.iterrows()
        ],
        "",
        "Justificación académica del modo EARLY_FEATURES_ONLY",
        (
            "Este modo busca evaluar capacidad predictiva real usando solo informacion disponible "
            "al inicio del proceso, evitando que el modelo reconstruya reglas de completitud, "
            "trazabilidad, temporalidad o competencia usadas para construir la etiqueta."
        ),
    ]
    with open(reports_dir / "early_feature_policy_report.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    return work_df.loc[:, selected_columns].copy()


@task(name="4. Eliminar leakage y columnas no defendibles")
@validate_inputs
@timing_decorator
def remove_data_leakage(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Elimina columnas de leakage y guarda diagnostico trazable."""
    logger = get_run_logger()
    target_col = config["target_col"]
    reports_dir = get_path(config, "reports_dir")
    processed_data_dir = get_path(config, "processed_data_dir")
    threshold = float(config["leakage_correlation_threshold"])

    work_df = df.copy()
    work_df = work_df.dropna(subset=[target_col]).copy()

    records: List[Dict[str, Any]] = []
    dropped: Dict[str, Dict[str, Any]] = {}

    for column in LEAKAGE_COLS + TARGET_DERIVED_COLS:
        if column in work_df.columns and column != target_col:
            add_drop_record(records, dropped, column, "columna derivada directa/indirectamente del target")

    for column in METADATA_COLS:
        if column in work_df.columns and column != target_col:
            add_drop_record(records, dropped, column, "metadata granular o identificador no generalizable")

    for column in work_df.columns:
        lower_col = str(column).lower()
        if column == target_col:
            continue
        if lower_col.startswith("id_") or lower_col.endswith("_id") or "_id_" in lower_col:
            add_drop_record(records, dropped, column, "identificador con alto riesgo de memorizar registros")
        for reason, patterns in LEAKAGE_NAME_PATTERNS.items():
            if any(pattern in lower_col for pattern in patterns):
                add_drop_record(records, dropped, column, reason)

    correlations = compute_numeric_correlations(work_df, target_col)
    for column, corr in correlations.items():
        if abs(corr) >= threshold and column != target_col:
            add_drop_record(
                records,
                dropped,
                column,
                f"correlacion absoluta sospechosamente alta >= {threshold}",
                corr,
            )
        elif column in dropped:
            dropped[column]["correlation_with_target"] = corr
            dropped[column]["abs_correlation_with_target"] = abs(corr)

    to_drop = sorted(dropped.keys())
    df_clean = work_df.drop(columns=to_drop, errors="ignore").copy()

    diagnostics = pd.DataFrame(records)
    if diagnostics.empty:
        diagnostics = pd.DataFrame(
            columns=[
                "column",
                "reason",
                "correlation_with_target",
                "abs_correlation_with_target",
                "dropped",
            ]
        )
    diagnostics = diagnostics.sort_values(["reason", "column"]).reset_index(drop=True)
    diagnostics.to_csv(reports_dir / "leakage_diagnostics.csv", index=False)
    diagnostics[["column", "reason"]].rename(columns={"column": "dropped_columns"}).to_csv(
        processed_data_dir / "leakage_report.csv",
        index=False,
    )

    feature_summary = pd.DataFrame({"feature": df_clean.columns, "type": df_clean.dtypes.astype(str)})
    feature_summary.to_csv(processed_data_dir / "feature_summary.csv", index=False)

    logger.info("Se eliminaron %s columnas por leakage, metadata o alta correlacion.", len(to_drop))
    logger.info("Columnas restantes para modelado: %s", df_clean.shape[1] - 1)
    return df_clean


@task(name="4b. Eliminar proxies temporales estrictos")
@validate_inputs
@timing_decorator
def remove_strict_temporal_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Elimina variables temporales/flags para validar leakage indirecto."""
    logger = get_run_logger()
    target_col = config["target_col"]
    reports_dir = get_path(config, "reports_dir")
    patterns = [str(pattern).lower() for pattern in config["strict_temporal_exclude_patterns"]]

    work_df = df.copy()
    to_drop = []
    rows = []
    for column in work_df.columns:
        if column == target_col:
            continue
        lower_col = str(column).lower()
        matched_patterns = [pattern for pattern in patterns if pattern in lower_col]
        if matched_patterns:
            to_drop.append(column)
            rows.append(
                {
                    "column": column,
                    "matched_patterns": "|".join(matched_patterns),
                    "reason": "experimento estricto contra leakage temporal indirecto",
                }
            )

    dropped_df = pd.DataFrame(rows, columns=["column", "matched_patterns", "reason"])
    dropped_df.to_csv(reports_dir / "strict_temporal_dropped.csv", index=False)

    df_strict = work_df.drop(columns=to_drop, errors="ignore").copy()
    logger.info("Modo estricto activo: se eliminaron %s columnas temporales/flags adicionales.", len(to_drop))
    logger.info("Columnas restantes tras modo estricto: %s", df_strict.shape[1] - 1)
    return df_strict


@task(name="4c. Seleccionar solo variables tempranas")
@validate_inputs
@timing_decorator
def apply_early_features_only(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Aplica el experimento EARLY_FEATURES_ONLY y guarda inventarios de columnas."""
    logger = get_run_logger()
    target_col = config["target_col"]
    reports_dir = get_path(config, "reports_dir")
    processed_data_dir = get_path(config, "processed_data_dir")

    work_df = df.copy()
    work_df.attrs["reports_dir"] = str(reports_dir)
    early_df = select_early_features_only(work_df, target_col)

    feature_summary = pd.DataFrame({"feature": early_df.columns, "type": early_df.dtypes.astype(str)})
    feature_summary.to_csv(processed_data_dir / artifact_name("feature_summary.csv", config), index=False)

    logger.info("Modo EARLY_FEATURES_ONLY activo: %s variables predictoras conservadas.", early_df.shape[1] - 1)
    logger.info("Reportes de politica early guardados en: %s", reports_dir)
    return early_df


@task(name="5. Generar splits estratificados")
@validate_inputs
@timing_decorator
def split_datasets(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, ...]:
    """Genera train/validation/test sin ajustar preprocesamiento antes del split."""
    logger = get_run_logger()
    target_col = config["target_col"]
    random_state = int(config["random_state"])
    processed_data_dir = get_path(config, "processed_data_dir")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    pd.concat([X_train, y_train], axis=1).to_parquet(
        processed_data_dir / artifact_name("train.parquet", config),
        index=False,
    )
    pd.concat([X_val, y_val], axis=1).to_parquet(
        processed_data_dir / artifact_name("validation.parquet", config),
        index=False,
    )
    pd.concat([X_test, y_test], axis=1).to_parquet(
        processed_data_dir / artifact_name("test.parquet", config),
        index=False,
    )

    logger.info("Splits creados. Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def infer_column_groups(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separa columnas numericas y categoricas para ColumnTransformer."""
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = [col for col in X.select_dtypes(include=[np.number]).columns.tolist() if col not in bool_cols]
    categorical_cols = [col for col in X.columns.tolist() if col not in numeric_cols]
    return numeric_cols, categorical_cols


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Normaliza categoricas como texto y conserva nulos para imputacion."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        X_clean = pd.DataFrame(index=X_df.index)
        for column in X_df.columns:
            series = X_df[column]
            X_clean[column] = series.where(series.isna(), series.astype(str))
        return X_clean.astype("object").where(pd.notna(X_clean), np.nan)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_preprocessor(X: pd.DataFrame, kind: str) -> ColumnTransformer:
    """Construye preprocesamiento ajustable solo con train/CV."""
    numeric_cols, categorical_cols = infer_column_groups(X)

    if kind == "linear":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("cleaner", CategoricalCleaner()),
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("encoder", make_one_hot_encoder()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("cleaner", CategoricalCleaner()),
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=np.float64,
                    ),
                ),
            ]
        )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.30)


def build_model_specs(X_train: pd.DataFrame, y_train: pd.Series, config: Dict[str, Any]) -> Dict[str, Pipeline]:
    """Define modelos comparables, cada uno con su preprocesamiento correcto."""
    random_state = int(config["random_state"])
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

    early_features_only = bool(config.get("early_features_only", False))
    estimators: Dict[str, Tuple[str, Any]] = {
        "LogisticRegression": (
            "linear",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state),
        ),
        "RandomForest": (
            "tree",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    }

    if not early_features_only:
        estimators["DecisionTree"] = (
            "tree",
            DecisionTreeClassifier(class_weight="balanced", max_depth=10, random_state=random_state),
        )
        estimators["GradientBoosting"] = (
            "tree",
            GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        )

    if HAS_XGB:
        estimators["XGBoost"] = (
            "tree",
            XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        )

    if HAS_LGBM:
        estimators["LightGBM"] = (
            "tree",
            LGBMClassifier(
                class_weight="balanced",
                force_col_wise=True,
                n_jobs=-1,
                random_state=random_state,
                verbosity=-1,
            ),
        )

    if HAS_CATBOOST:
        estimators["CatBoost"] = (
            "tree",
            CatBoostClassifier(auto_class_weights="Balanced", verbose=0, random_state=random_state),
        )

    model_specs = {}
    for model_name, (preprocess_kind, estimator) in estimators.items():
        model_specs[model_name] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train, preprocess_kind)),
                ("model", estimator),
            ]
        )
    return model_specs


def evaluate_predictions(
    y_true: pd.Series,
    y_score: np.ndarray | None,
    threshold: float,
    model_name: str,
    fallback_pred: np.ndarray,
    warning_threshold: float = PERFECT_METRIC_WARNING_THRESHOLD,
) -> Dict[str, Any]:
    """Calcula metricas de clasificacion con umbral explicito."""
    y_pred = predictions_from_threshold(y_score, threshold, fallback_pred)
    metrics = {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": safe_roc_auc(y_true, y_score),
        "PR_AUC": safe_pr_auc(y_true, y_score),
    }
    warning_metrics = [
        metric_name
        for metric_name in ["F1_score", "ROC_AUC", "PR_AUC"]
        if pd.notna(metrics[metric_name]) and metrics[metric_name] > warning_threshold
    ]
    metrics["Leakage_warning"] = (
        f"Posible leakage: metrica(s) > {warning_threshold}: " + ", ".join(warning_metrics)
        if warning_metrics
        else ""
    )
    return metrics


def optimize_threshold(
    y_true: pd.Series,
    y_score: np.ndarray | None,
    model_name: str,
    config: Dict[str, Any],
) -> Tuple[float, pd.DataFrame]:
    """Busca el umbral que maximiza F1 en validacion."""
    if y_score is None:
        return 0.50, pd.DataFrame()

    thresholds = np.linspace(
        float(config["threshold_min"]),
        float(config["threshold_max"]),
        int(config["threshold_steps"]),
    )
    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "Model": model_name,
                "Threshold": float(threshold),
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1_score": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    threshold_df = pd.DataFrame(rows)
    threshold_df["distance_to_0_5"] = (threshold_df["Threshold"] - 0.50).abs()
    best_row = threshold_df.sort_values(["F1_score", "distance_to_0_5"], ascending=[False, True]).iloc[0]
    threshold_df["selected"] = threshold_df["Threshold"].eq(best_row["Threshold"])
    threshold_df = threshold_df.drop(columns=["distance_to_0_5"])
    return float(best_row["Threshold"]), threshold_df


def plot_model_results(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    model_name: str,
    reports_dir: Path,
) -> None:
    """Guarda matriz de confusion, ROC, PR y reporte de clasificacion."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(
        reports_dir / "confusion_matrices" / f"{model_name}_cm.csv"
    )
    pd.DataFrame(cm_norm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(
        reports_dir / "confusion_matrices" / f"{model_name}_cm_normalized.csv"
    )

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Matriz de confusion normalizada: {model_name}")
    plt.ylabel("Etiqueta real")
    plt.xlabel("Etiqueta predicha")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrices" / f"{model_name}_cm_normalized.png", dpi=150)
    plt.close()

    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    positive_report = report_dict.get("1", {})
    with open(reports_dir / "classification_reports" / f"{model_name}_report.txt", "w", encoding="utf-8") as file:
        file.write(f"Classification Report - {model_name}\n\n")
        file.write(classification_report(y_true, y_pred, zero_division=0))
        file.write("\n\nReporte especifico clase positiva (riesgo=1):\n")
        file.write(json.dumps(positive_report, indent=2, ensure_ascii=False))

    if y_score is None or pd.Series(y_true).nunique() < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_true, y_score):.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curves" / f"{model_name}_roc.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC = {average_precision_score(y_true, y_score):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(reports_dir / "pr_curves" / f"{model_name}_pr.png", dpi=150)
    plt.close()


def get_feature_names(pipeline: Pipeline) -> List[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"feature_{idx}" for idx in range(pipeline.named_steps["model"].n_features_in_)]


def save_feature_importance(pipeline: Pipeline, model_name: str, reports_dir: Path) -> None:
    """Guarda importancia o coeficientes si el estimador los expone."""
    estimator = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        column_name = "Importance"
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        column_name = "Coefficient"
    else:
        return

    if len(feature_names) != len(values):
        feature_names = [f"feature_{idx}" for idx in range(len(values))]

    importance = pd.DataFrame({"Feature": feature_names, column_name: values})
    importance["abs_value"] = importance[column_name].abs()
    importance = importance.sort_values("abs_value", ascending=False).drop(columns=["abs_value"])
    importance.to_csv(reports_dir / "feature_importance" / f"{model_name}_fi.csv", index=False)


@task(name="6. Validacion cruzada estratificada")
@timing_decorator
def run_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_specs: Dict[str, Pipeline],
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Ejecuta StratifiedKFold y reporta media/desviacion de metricas."""
    logger = get_run_logger()
    reports_dir = get_path(config, "reports_dir")
    cv = StratifiedKFold(
        n_splits=int(config["cv_folds"]),
        shuffle=True,
        random_state=int(config["random_state"]),
    )
    scoring = {
        "F1": make_scorer(f1_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "Precision": make_scorer(precision_score, zero_division=0),
        "ROC_AUC": "roc_auc",
        "PR_AUC": "average_precision",
    }

    rows = []
    model_suffix = config.get("run_suffix", "")
    for model_name, pipeline in model_specs.items():
        logger.info("Validacion cruzada para %s...", model_name)
        try:
            cv_result = cross_validate(
                clone(pipeline),
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score=np.nan,
            )
            row = {"Model": f"{model_name}{model_suffix}"}
            for metric_name in scoring:
                scores = cv_result[f"test_{metric_name}"]
                row[f"{metric_name}_mean"] = np.nanmean(scores)
                row[f"{metric_name}_std"] = np.nanstd(scores)
            row["CV_error"] = ""
        except Exception as exc:
            logger.exception("Fallo CV para %s", model_name)
            row = {"Model": f"{model_name}{model_suffix}", "CV_error": str(exc)}
        rows.append(row)

    cv_summary = pd.DataFrame(rows)
    cv_summary.to_csv(reports_dir / artifact_name("cross_validation_summary.csv", config), index=False)
    return cv_summary


@task(name="7. Entrenamiento y seleccion de modelos")
@timing_decorator
def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    model_specs: Dict[str, Pipeline],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], pd.DataFrame]:
    """Entrena modelos con pipelines completos y evalua en validacion."""
    logger = get_run_logger()
    reports_dir = get_path(config, "reports_dir")
    models_dir = get_path(config, "models_dir")
    metric = config["metric"]
    model_suffix = config.get("run_suffix", "")

    if not HAS_XGB:
        logger.warning("XGBoost no esta instalado. Se omitira.")
    if not HAS_LGBM:
        logger.warning("LightGBM no esta instalado. Se omitira.")
    if not HAS_CATBOOST:
        logger.warning("CatBoost no esta instalado. Se omitira.")

    results: List[Dict[str, Any]] = []
    threshold_frames: List[pd.DataFrame] = []
    trained_models: Dict[str, Pipeline] = {}

    for model_name, pipeline in model_specs.items():
        artifact_model_name = f"{model_name}{model_suffix}"
        logger.info("Entrenando modelo: %s...", artifact_model_name)
        try:
            pipeline.fit(X_train, y_train)
            trained_models[artifact_model_name] = pipeline

            y_val_score = get_positive_scores(pipeline, X_val)
            fallback_pred = pipeline.predict(X_val)
            best_threshold, threshold_df = optimize_threshold(y_val, y_val_score, artifact_model_name, config)
            if not threshold_df.empty:
                threshold_frames.append(threshold_df)

            y_val_pred = predictions_from_threshold(y_val_score, best_threshold, fallback_pred)
            metrics = evaluate_predictions(
                y_val,
                y_val_score,
                best_threshold,
                artifact_model_name,
                fallback_pred,
                float(config["perfect_metric_warning_threshold"]),
            )
            results.append(metrics)

            plot_model_results(y_val, y_val_pred, y_val_score, artifact_model_name, reports_dir)
            save_feature_importance(pipeline, artifact_model_name, reports_dir)
            joblib.dump(pipeline, models_dir / "all_models" / f"{artifact_model_name}_pipeline.joblib")

            if metrics["Leakage_warning"]:
                logger.warning("%s: %s", artifact_model_name, metrics["Leakage_warning"])
            logger.info("%s evaluado. %s=%.4f", artifact_model_name, metric, metrics.get(metric, np.nan))
        except Exception as exc:
            logger.exception("El modelo %s fallo y fue omitido. Error: %s", artifact_model_name, exc)

    if not results:
        raise RuntimeError("Ningun modelo fue entrenado exitosamente.")

    threshold_summary = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    threshold_summary.to_csv(reports_dir / artifact_name("threshold_optimization.csv", config), index=False)

    results_df = pd.DataFrame(results).sort_values(metric, ascending=False).reset_index(drop=True)
    return results_df, trained_models, threshold_summary


@task(name="8. Evaluacion final y serializacion")
@timing_decorator
def wrap_up_and_conclude(
    results_df: pd.DataFrame,
    trained_models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_summary: pd.DataFrame,
    config: Dict[str, Any],
) -> None:
    """Guarda reportes consolidados, evalua test y serializa el mejor pipeline."""
    logger = get_run_logger()
    reports_dir = get_path(config, "reports_dir")
    models_dir = get_path(config, "models_dir")
    metric = config["metric"]

    results_df.to_csv(reports_dir / artifact_name("metrics_summary.csv", config), index=False)
    results_df.to_excel(reports_dir / artifact_name("metrics_summary.xlsx", config), index=False)

    best_model_name = str(results_df.iloc[0]["Model"])
    best_threshold = float(results_df.iloc[0]["Threshold"])
    best_pipeline = trained_models[best_model_name]
    setattr(best_pipeline, "optimal_threshold_", best_threshold)

    y_test_score = get_positive_scores(best_pipeline, X_test)
    fallback_pred = best_pipeline.predict(X_test)
    y_test_pred = predictions_from_threshold(y_test_score, best_threshold, fallback_pred)
    test_metrics = evaluate_predictions(
        y_test,
        y_test_score,
        best_threshold,
        f"{best_model_name}_TEST",
        fallback_pred,
        float(config["perfect_metric_warning_threshold"]),
    )
    plot_model_results(y_test, y_test_pred, y_test_score, f"{best_model_name}_TEST", reports_dir)

    best_model_path = models_dir / artifact_name("best_model_pipeline.joblib", config)
    joblib.dump(best_pipeline, best_model_path)
    with open(models_dir / artifact_name("best_model_threshold.json", config), "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_model_name,
                "optimal_threshold": best_threshold,
                "selection_metric": metric,
                "best_model_pipeline": str(best_model_path),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    warnings = results_df["Leakage_warning"].dropna()
    warnings = [warning for warning in warnings.tolist() if warning]
    if test_metrics["Leakage_warning"]:
        warnings.append(f"TEST: {test_metrics['Leakage_warning']}")

    positive_report = classification_report(y_test, y_test_pred, zero_division=0, output_dict=True).get("1", {})
    pd.DataFrame([{"Model": best_model_name, **positive_report}]).to_csv(
        reports_dir / artifact_name("positive_class_report.csv", config),
        index=False,
    )

    conclusion_path = reports_dir / artifact_name("best_model_report.txt", config)
    with open(conclusion_path, "w", encoding="utf-8") as file:
        file.write("=" * 60 + "\n")
        file.write("CONCLUSIONES DEL PIPELINE DE MODELADO\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"1. Mejor modelo seleccionado: {best_model_name}\n")
        file.write(f"   Metrica principal: {metric}\n")
        file.write(f"   Umbral optimo en validacion: {best_threshold:.4f}\n")
        file.write(f"   Pipeline serializado: {best_model_path}\n\n")
        file.write("2. Desempeno en validacion del mejor modelo:\n")
        for key, value in results_df.iloc[0].items():
            if isinstance(value, float):
                file.write(f"   - {key}: {value:.4f}\n")
            else:
                file.write(f"   - {key}: {value}\n")
        file.write("\n3. Desempeno en conjunto de prueba (TEST):\n")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                file.write(f"   - {key}: {value:.4f}\n")
            else:
                file.write(f"   - {key}: {value}\n")
        file.write("\n4. Validacion cruzada estratificada:\n")
        file.write(cv_summary.to_string(index=False))
        file.write("\n\n5. Notas sobre confiabilidad academica:\n")
        file.write("\n   - El preprocesamiento se ajusta dentro de Pipeline/ColumnTransformer usando solo train/CV.")
        file.write("\n   - El mejor artefacto guardado incluye imputacion, codificacion, escalamiento y estimador.")
        file.write("\n   - Se eliminaron columnas derivadas del target, metadata granular y variables post-evento.")
        file.write("\n   - No se uso SMOTE; se priorizo trazabilidad sobre maximizar metricas.")
        if config.get("early_features_only"):
            selected_path = reports_dir / "early_features_selected.csv"
            dropped_path = reports_dir / "early_features_dropped.csv"
            selected_features: List[str] = []
            if selected_path.exists():
                selected_features_df = pd.read_csv(selected_path)
                selected_features = selected_features_df.loc[
                    selected_features_df["role"].eq("feature"), "column"
                ].tolist()
            file.write("\n\n6. Justificación académica del modo EARLY_FEATURES_ONLY:")
            file.write(
                "\n   - Este modo busca evaluar capacidad predictiva real usando solo informacion "
                "disponible al inicio del proceso, evitando que el modelo reconstruya reglas de "
                "completitud, trazabilidad, temporalidad o competencia usadas para construir la etiqueta."
            )
            file.write("\n   - Variables usadas en modo early:")
            for column in selected_features:
                file.write(f"\n     * {column}")
            file.write(f"\n   - Inventario de variables usadas: {selected_path}")
            file.write(f"\n   - Inventario de variables eliminadas y razon: {dropped_path}")
        if warnings:
            warning_section = "7" if config.get("early_features_only") else "6"
            file.write(f"\n\n{warning_section}. Advertencias automaticas de posible leakage:\n")
            for warning in warnings:
                file.write(f"\n   - {warning}")
        else:
            warning_section = "7" if config.get("early_features_only") else "6"
            file.write(
                f"\n\n{warning_section}. Advertencias automaticas de posible leakage: "
                "no se detectaron metricas > 0.995."
            )

    logger.info("Mejor pipeline guardado en: %s", best_model_path)
    logger.info("Pipeline ML completado y reportes almacenados correctamente.")


@task(name="9. Comparar modo estricto contra normal")
@timing_decorator
def compare_strict_with_normal(config: Dict[str, Any]) -> None:
    """Compara metricas normales vs estrictas para diagnosticar leakage indirecto."""
    logger = get_run_logger()
    reports_dir = get_path(config, "reports_dir")
    normal_path = reports_dir / "metrics_summary.csv"
    strict_path = reports_dir / "metrics_summary_strict.csv"
    comparison_path = reports_dir / "strict_vs_normal_comparison.csv"
    report_path = reports_dir / "strict_validation_report.txt"

    if not normal_path.exists() or not strict_path.exists():
        message = (
            "No fue posible comparar modo estricto vs normal porque falta "
            f"{normal_path.name if not normal_path.exists() else strict_path.name}."
        )
        pd.DataFrame(
            [
                {
                    "metric": "comparison_status",
                    "normal_value": np.nan,
                    "strict_value": np.nan,
                    "absolute_drop": np.nan,
                    "relative_drop": np.nan,
                    "message": message,
                }
            ]
        ).to_csv(comparison_path, index=False)
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(message)
        logger.warning(message)
        return

    normal_results = pd.read_csv(normal_path)
    strict_results = pd.read_csv(strict_path)
    if normal_results.empty or strict_results.empty:
        message = "No fue posible comparar modo estricto vs normal porque algun archivo de metricas esta vacio."
        pd.DataFrame([{"metric": "comparison_status", "message": message}]).to_csv(comparison_path, index=False)
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(message)
        logger.warning(message)
        return

    normal_best = normal_results.iloc[0]
    strict_best = strict_results.iloc[0]
    rows = []
    for metric_name in ["F1_score", "ROC_AUC", "PR_AUC"]:
        normal_value = float(normal_best[metric_name])
        strict_value = float(strict_best[metric_name])
        absolute_drop = normal_value - strict_value
        relative_drop = absolute_drop / normal_value if normal_value else np.nan
        rows.append(
            {
                "metric": metric_name,
                "normal_model": normal_best["Model"],
                "strict_model": strict_best["Model"],
                "normal_value": normal_value,
                "strict_value": strict_value,
                "absolute_drop": absolute_drop,
                "relative_drop": relative_drop,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_path, index=False)

    f1_drop = comparison_df.loc[comparison_df["metric"].eq("F1_score"), "absolute_drop"].iloc[0]
    roc_drop = comparison_df.loc[comparison_df["metric"].eq("ROC_AUC"), "absolute_drop"].iloc[0]
    evidence = f1_drop > 0.01 or roc_drop > 0.005

    if evidence:
        conclusion = "Existe evidencia de posible leakage indirecto a través de variables temporales derivadas."
    else:
        conclusion = "No se detecta evidencia fuerte de leakage indirecto por variables temporales."

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("VALIDACION ESTRICTA CONTRA LEAKAGE TEMPORAL INDIRECTO\n")
        file.write("=" * 60 + "\n\n")
        file.write(f"Modelo normal comparado: {normal_best['Model']}\n")
        file.write(f"Modelo estricto comparado: {strict_best['Model']}\n\n")
        file.write(comparison_df.to_string(index=False))
        file.write("\n\n")
        file.write(conclusion)
        file.write("\n\nCriterio: caida absoluta > 0.01 en F1 o > 0.005 en ROC-AUC.\n")

    logger.info(conclusion)


@task(name="10. Comparar early contra strict y normal")
@timing_decorator
def compare_early_with_baselines(config: Dict[str, Any]) -> None:
    """Compara EARLY_FEATURES_ONLY contra los resultados normal y estricto."""
    logger = get_run_logger()
    reports_dir = get_path(config, "reports_dir")
    paths = {
        "normal": reports_dir / "metrics_summary.csv",
        "strict": reports_dir / "metrics_summary_strict.csv",
        "early": reports_dir / "metrics_summary_early.csv",
    }
    comparison_path = reports_dir / "early_vs_strict_comparison.csv"
    policy_report_path = reports_dir / "early_feature_policy_report.txt"
    metrics = ["F1_score", "Recall", "Precision", "ROC_AUC", "PR_AUC"]

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        message = "No fue posible comparar normal/strict/early porque faltan: " + ", ".join(missing)
        pd.DataFrame([{"metric": "comparison_status", "message": message}]).to_csv(comparison_path, index=False)
        with open(policy_report_path, "a", encoding="utf-8") as file:
            file.write("\n\nComparacion automatica normal/strict/early\n")
            file.write(message)
        logger.warning(message)
        return

    summaries = {name: pd.read_csv(path) for name, path in paths.items()}
    if any(summary.empty for summary in summaries.values()):
        message = "No fue posible comparar normal/strict/early porque algun resumen de metricas esta vacio."
        pd.DataFrame([{"metric": "comparison_status", "message": message}]).to_csv(comparison_path, index=False)
        with open(policy_report_path, "a", encoding="utf-8") as file:
            file.write("\n\nComparacion automatica normal/strict/early\n")
            file.write(message)
        logger.warning(message)
        return

    best_rows = {name: summary.iloc[0] for name, summary in summaries.items()}
    rows = []
    for metric_name in metrics:
        normal_value = float(best_rows["normal"][metric_name])
        strict_value = float(best_rows["strict"][metric_name])
        early_value = float(best_rows["early"][metric_name])
        strict_drop = strict_value - early_value
        normal_drop = normal_value - early_value
        rows.append(
            {
                "metric": metric_name,
                "normal_model": best_rows["normal"]["Model"],
                "strict_model": best_rows["strict"]["Model"],
                "early_model": best_rows["early"]["Model"],
                "normal_value": normal_value,
                "strict_value": strict_value,
                "early_value": early_value,
                "early_vs_strict_absolute_drop": strict_drop,
                "early_vs_strict_relative_drop": strict_drop / strict_value if strict_value else np.nan,
                "early_vs_normal_absolute_drop": normal_drop,
                "early_vs_normal_relative_drop": normal_drop / normal_value if normal_value else np.nan,
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_path, index=False)

    f1_drop = comparison_df.loc[comparison_df["metric"].eq("F1_score"), "early_vs_strict_absolute_drop"].iloc[0]
    roc_drop = comparison_df.loc[comparison_df["metric"].eq("ROC_AUC"), "early_vs_strict_absolute_drop"].iloc[0]
    pr_drop = comparison_df.loc[comparison_df["metric"].eq("PR_AUC"), "early_vs_strict_absolute_drop"].iloc[0]
    important_drop = f1_drop >= 0.05 or roc_drop >= 0.02 or pr_drop >= 0.02
    early_best = best_rows["early"]
    still_high = (
        float(early_best["F1_score"]) >= 0.90
        or float(early_best["ROC_AUC"]) >= 0.95
        or float(early_best["PR_AUC"]) >= 0.95
    )

    conclusions = []
    if important_drop:
        conclusions.append(
            "Las métricas bajaron al usar solo variables tempranas, lo que indica que los resultados "
            "anteriores probablemente estaban inflados por variables derivadas cercanas al target."
        )
    if still_high:
        conclusions.append(
            "El target puede estar fuertemente determinado por variables estructurales tempranas como "
            "modalidad, justificación, tipo de contrato o entidad. Se recomienda validar el labeling y los umbrales."
        )
    if not conclusions:
        conclusions.append(
            "El modo early reduce el riesgo de reconstruccion de la etiqueta y no mantiene metricas "
            "inusualmente altas bajo los umbrales automaticos definidos."
        )

    with open(policy_report_path, "a", encoding="utf-8") as file:
        file.write("\n\nComparacion automatica normal/strict/early\n")
        file.write("=" * 70 + "\n")
        file.write(comparison_df.to_string(index=False))
        file.write("\n\nInterpretacion automatica:\n")
        for conclusion in conclusions:
            file.write(f"- {conclusion}\n")
        file.write("\nCriterio de caida importante: F1 >= 0.05, ROC-AUC >= 0.02 o PR-AUC >= 0.02 frente a strict.\n")
        file.write("Criterio de rendimiento aun muy alto: F1 >= 0.90, ROC-AUC >= 0.95 o PR-AUC >= 0.95.\n")

    logger.info("Comparacion EARLY_FEATURES_ONLY completada. %s", " ".join(conclusions))


# ==============================================================================
# PIPELINE ORQUESTADOR
# ==============================================================================


@flow(name="SECOP_Transparency_Risk_Pipeline", log_prints=True)
def main_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_dir: str | Path = DEFAULT_REPORTS_DIR,
    target_col: str = TARGET_COL,
    metric: str = DEFAULT_METRIC,
    strict_mode: bool = STRICT_MODE,
    early_features_only: bool = EARLY_FEATURES_ONLY,
) -> None:
    """Ejecuta el ciclo de vida completo de MLOps local paso a paso."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    config = build_config(
        input_path=input_path,
        output_dir=output_dir,
        target_col=target_col,
        metric=metric,
        strict_mode=strict_mode,
        early_features_only=early_features_only,
    )
    check_and_create_directories(config)
    save_reproducibility_reports(config)

    df = load_data(config["input_path"], config["target_col"])
    if config["early_features_only"]:
        df_clean = apply_early_features_only(df, config)
    else:
        df_clean = remove_data_leakage(df, config)
    if config["strict_mode"] and not config["early_features_only"]:
        df_clean = remove_strict_temporal_features(df_clean, config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_datasets(df_clean, config)

    model_specs = build_model_specs(X_train, y_train, config)
    cv_summary = run_cross_validation(X_train, y_train, model_specs, config)
    results_df, trained_models, _ = train_and_evaluate_models(X_train, X_val, y_train, y_val, model_specs, config)
    wrap_up_and_conclude(results_df, trained_models, X_test, y_test, cv_summary, config)
    if config["early_features_only"]:
        compare_early_with_baselines(config)
    elif config["strict_mode"]:
        compare_strict_with_normal(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de modelado SECOP II - ODS 16.6")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="Ruta al parquet de entrada.")
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORTS_DIR), help="Directorio para reportes de modelado.")
    parser.add_argument("--target-col", default=TARGET_COL, help="Nombre de la columna objetivo.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metrica principal: f1, recall, precision, roc_auc, pr_auc.")
    parser.add_argument(
        "--strict-mode",
        action=argparse.BooleanOptionalAction,
        default=STRICT_MODE,
        help="Activa/desactiva el experimento estricto contra leakage temporal indirecto.",
    )
    parser.add_argument(
        "--early-features-only",
        action=argparse.BooleanOptionalAction,
        default=EARLY_FEATURES_ONLY,
        help="Activa/desactiva el experimento EARLY_FEATURES_ONLY con solo variables tempranas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_pipeline(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_col=args.target_col,
        metric=args.metric,
        strict_mode=args.strict_mode,
        early_features_only=args.early_features_only,
    )
