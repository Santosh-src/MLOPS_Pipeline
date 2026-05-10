import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease using the Cleveland UCI dataset model.",
    version="1.0.0",
)

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics"],
).instrument(app).expose(app)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(PROJECT_ROOT / "model" / "heart_disease_model.joblib"),
)
model = None


class HeartDiseaseInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., gt=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=0, le=3)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=0, le=7)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63, "sex": 1, "cp": 3, "trestbps": 145.0, "chol": 233.0,
                    "fbs": 1, "restecg": 0, "thalach": 150.0, "exang": 0,
                    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 6,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    disease_probability: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(json.dumps({"event": "model_loaded", "model_path": MODEL_PATH}))
    except FileNotFoundError:
        logger.error(json.dumps({"event": "model_load_failed", "model_path": MODEL_PATH}))


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.get("/health", response_model=HealthResponse)
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="healthy", model_loaded=True)


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HeartDiseaseInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    features = pd.DataFrame([input_data.model_dump()])
    prediction = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    disease_prob = float(proba[1])
    confidence = float(proba[prediction])
    latency_ms = (time.time() - start) * 1000

    logger.info(json.dumps({
        "event": "prediction",
        "input": input_data.model_dump(),
        "prediction": prediction,
        "disease_probability": round(disease_prob, 4),
        "confidence": round(confidence, 4),
        "latency_ms": round(latency_ms, 2),
    }))

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        disease_probability=round(disease_prob, 4),
    )
