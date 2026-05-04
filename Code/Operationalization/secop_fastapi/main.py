from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin


ROOT_DIR = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
MODEL_PATH = ROOT_DIR / "Models" / "trained" / "all_models" / "RandomForest_early_pipeline.joblib"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "Reports" / "Model" / "feature_importance" / "RandomForest_early_fi.csv"
MODEL_THRESHOLD = 0.645
REVIEW_THRESHOLD = 0.35

MODEL_FEATURES = [
    "entidad",
    "nit_entidad",
    "departamento_entidad",
    "ciudad_entidad",
    "ordenentidad",
    "nombre_del_procedimiento",
    "descripci_n_del_procedimiento",
    "fase",
    "precio_base",
    "modalidad_de_contratacion",
    "justificaci_n_modalidad_de",
    "codigo_principal_de_categoria",
    "tipo_de_contrato",
    "categorias_adicionales",
    "precio_base_log",
]


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


# The pipeline was serialized with this custom transformer in __main__.
setattr(sys.modules["__main__"], "CategoricalCleaner", CategoricalCleaner)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    entidad: str = Field(..., min_length=1)
    nit_entidad: str = Field(..., min_length=1)
    departamento_entidad: str = Field(..., min_length=1)
    ciudad_entidad: str = Field(..., min_length=1)
    ordenentidad: str = Field(..., min_length=1)
    nombre_del_procedimiento: str = Field(..., min_length=1)
    descripci_n_del_procedimiento: str = Field(..., min_length=1)
    fase: str = Field(..., min_length=1)
    precio_base: float = Field(..., ge=0)
    modalidad_de_contratacion: str = Field(..., min_length=1)
    justificaci_n_modalidad_de: str = Field(..., min_length=1)
    codigo_principal_de_categoria: str = Field(..., min_length=1)
    tipo_de_contrato: str = Field(..., min_length=1)
    categorias_adicionales: str = ""


app = FastAPI(
    title="SECOP Transparencia Temprana",
    description="Prediccion temprana de riesgo de baja transparencia en procesos SECOP II.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_key_factors() -> list[dict[str, float | str]]:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return [
            {"feature": "modalidad_de_contratacion", "label": "Modalidad", "importance": 0.0},
            {"feature": "justificaci_n_modalidad_de", "label": "Justificación", "importance": 0.0},
            {"feature": "fase", "label": "Fase", "importance": 0.0},
            {"feature": "codigo_principal_de_categoria", "label": "Categoría", "importance": 0.0},
            {"feature": "precio_base", "label": "Precio", "importance": 0.0},
        ]

    labels = {
        "modalidad_de_contratacion": "Modalidad",
        "justificaci_n_modalidad_de": "Justificación",
        "fase": "Fase",
        "codigo_principal_de_categoria": "Categoría",
        "categorias_adicionales": "Categorías adicionales",
        "precio_base": "Precio",
        "tipo_de_contrato": "Tipo de contrato",
    }
    df = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(7)
    factors: list[dict[str, float | str]] = []
    for _, row in df.iterrows():
        raw_feature = str(row["Feature"])
        feature = raw_feature.split("__", 1)[-1]
        factors.append(
            {
                "feature": feature,
                "label": labels.get(feature, feature),
                "importance": round(float(row["Importance"]), 6),
            }
        )
    return factors


MODEL = load_model()


def build_model_frame(payload: PredictionRequest) -> pd.DataFrame:
    data = payload.model_dump()
    data["precio_base"] = float(data["precio_base"])
    data["precio_base_log"] = math.log1p(data["precio_base"])
    return pd.DataFrame([{feature: data.get(feature, "") for feature in MODEL_FEATURES}], columns=MODEL_FEATURES)


def classify_probability(probability: float) -> tuple[int, str]:
    if probability >= MODEL_THRESHOLD:
        return 1, "Riesgo alto"
    if probability >= REVIEW_THRESHOLD:
        return 0, "Revisar"
    return 0, "Riesgo bajo"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    try:
        model = MODEL
        loaded = True
        model_type = type(model).__name__
    except Exception as exc:  # pragma: no cover - surfaced in response for ops use.
        loaded = False
        model_type = None
        return {
            "status": "error",
            "model_loaded": loaded,
            "model_path": str(MODEL_PATH),
            "error": str(exc),
        }

    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_type": model_type,
        "model_path": str(MODEL_PATH),
        "threshold": MODEL_THRESHOLD,
        "features": MODEL_FEATURES,
        "feature_count": len(MODEL_FEATURES),
    }


@app.post("/api/predict")
def predict(payload: PredictionRequest) -> dict[str, object]:
    model = MODEL
    frame = build_model_frame(payload)

    try:
        probabilities = model.predict_proba(frame)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo ejecutar la prediccion: {exc}") from exc

    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
    probability = float(probabilities[positive_index])
    predicted_class, label = classify_probability(probability)

    return {
        "probabilidad_riesgo": round(probability, 6),
        "umbral": MODEL_THRESHOLD,
        "clase_predicha": predicted_class,
        "etiqueta": label,
        "modelo": MODEL_PATH.name,
        "factores_clave": load_key_factors(),
    }
