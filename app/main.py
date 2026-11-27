"""FastAPI service that serves the star-type classifier.

Model source:
  * MODEL_URI set (e.g. models:/star-classifier@champion) -> loaded from MLflow Model Registry
  * otherwise / on failure                                -> loaded from models/model.joblib
"""
import json
import os
from enum import Enum
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from starml.features import CLASS_NAMES, COLORS, FEATURES, SPECTRAL_CLASSES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
META_PATH = MODEL_PATH.with_name("model_meta.json")
MODEL_URI = os.getenv("MODEL_URI")  # e.g. models:/star-classifier@champion


def load_model():
    if MODEL_URI:
        try:
            import mlflow

            model = mlflow.sklearn.load_model(MODEL_URI)
            info = {"source": "mlflow-registry", "model_uri": MODEL_URI}
            try:
                name, alias = MODEL_URI.removeprefix("models:/").split("@")
                mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
                info.update(model_name=name, model_version=mv.version, alias=alias, run_id=mv.run_id,
                            algorithm=mv.tags.get("algorithm"))
            except Exception:
                pass
            return model, info
        except Exception as exc:  # registry unreachable -> fall back to the local file
            print(f"[warn] could not load {MODEL_URI} from MLflow ({exc}); falling back to {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    info = {"source": "local-file", "model_path": str(MODEL_PATH)}
    if META_PATH.exists():
        info.update(json.loads(META_PATH.read_text()))
    return model, info


model, model_info = load_model()

app = FastAPI(
    title="Star Type Classifier API",
    description="Predicts whether a star is a Brown Dwarf, Red Dwarf, White Dwarf, "
    "Main Sequence star, Supergiant or Hypergiant from its physical properties.",
    version="1.0.0",
)

StarColor = Enum("StarColor", {c.replace("-", "_"): c for c in COLORS}, type=str)
SpectralClass = Enum("SpectralClass", {c: c for c in SPECTRAL_CLASSES}, type=str)


class StarFeatures(BaseModel):
    temperature: float = Field(..., gt=0, le=60000, description="Surface temperature, K")
    luminosity: float = Field(..., gt=0, description="Luminosity relative to the Sun (L/Lo)")
    radius: float = Field(..., gt=0, description="Radius relative to the Sun (R/Ro)")
    abs_magnitude: float = Field(..., ge=-15, le=25, description="Absolute visual magnitude (Mv)")
    star_color: StarColor = Field(..., description="Observed colour")
    spectral_class: SpectralClass = Field(..., description="Spectral class O B A F G K M")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"temperature": 5778, "luminosity": 1.0, "radius": 1.0, "abs_magnitude": 4.83,
                 "star_color": "yellow", "spectral_class": "G"}
            ]
        }
    }


class Prediction(BaseModel):
    star_type: int
    star_type_name: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str | None = None


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def get_model_info():
    return model_info


@app.post("/predict", response_model=Prediction)
def predict(star: StarFeatures):
    row = star.model_dump()
    row["star_color"] = star.star_color.value
    row["spectral_class"] = star.spectral_class.value
    X = pd.DataFrame([row])[FEATURES]
    try:
        proba = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    classes = [int(c) for c in model.classes_]
    best = int(proba.argmax())
    return Prediction(
        star_type=classes[best],
        star_type_name=CLASS_NAMES[classes[best]],
        confidence=round(float(proba[best]), 4),
        probabilities={CLASS_NAMES[c]: round(float(p), 4) for c, p in zip(classes, proba)},
        model_version=str(model_info.get("model_version")) if model_info.get("model_version") else None,
    )
