"""FastAPI service for the MLflow diabetes progression model."""

import os
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

BASE_DIR = Path(__file__).resolve().parent
FEATURES = [
    "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"
]


def find_model() -> Path:
    configured = os.getenv("MODEL_PATH")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = BASE_DIR / path
        if (path / "MLmodel").is_file():
            return path
        raise RuntimeError(f"MODEL_PATH does not contain an MLflow model: {path}")

    candidates = [
        path.parent for root in (BASE_DIR / "mlruns", BASE_DIR / "mlartifacts")
        if root.exists() for path in root.rglob("MLmodel")
    ]
    if not candidates:
        raise RuntimeError("No saved MLflow model found. Set MODEL_PATH to its artifacts folder.")
    return max(candidates, key=lambda path: (path / "MLmodel").stat().st_mtime)


model = mlflow.pyfunc.load_model(str(find_model()))
app = FastAPI(title="Diabetes Progression Predictor", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}


@app.post("/predict")
def predict(payload: PredictionInput):
    values = [getattr(payload, name) for name in FEATURES]
    try:
        result = float(np.asarray(model.predict([values])).reshape(-1)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model prediction failed") from exc
    return {"prediction": result, "unit": "dataset progression score"}
