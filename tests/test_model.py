import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.preprocess import ALL_FEATURE_COLS, TARGET_COL

MODEL_PATH = PROJECT_ROOT / "model" / "heart_disease_model.joblib"
DATA_PATH = PROJECT_ROOT / "dataprocessing" / "processed_cleveland.csv"

SAMPLE_INPUT = pd.DataFrame([{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145.0, "chol": 233.0,
    "fbs": 1, "restecg": 0, "thalach": 150.0, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}])


@pytest.fixture
def model():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}. Train first."
    return joblib.load(MODEL_PATH)


@pytest.fixture
def data():
    return pd.read_csv(DATA_PATH)


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_loads(model):
    assert model is not None


def test_model_has_predict(model):
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_prediction_shape(model):
    assert len(model.predict(SAMPLE_INPUT)) == 1


def test_prediction_is_binary(model):
    assert model.predict(SAMPLE_INPUT)[0] in (0, 1)


def test_predict_proba_shape(model):
    assert model.predict_proba(SAMPLE_INPUT).shape == (1, 2)


def test_predict_proba_sums_to_one(model):
    assert abs(model.predict_proba(SAMPLE_INPUT).sum() - 1.0) < 1e-5


def test_model_accuracy_threshold(model, data):
    accuracy = model.score(data[ALL_FEATURE_COLS], data[TARGET_COL])
    assert accuracy >= 0.75, f"Accuracy {accuracy:.4f} below 0.75"


def test_batch_prediction(model, data):
    preds = model.predict(data[ALL_FEATURE_COLS].head(50))
    assert len(preds) == 50
    assert set(preds).issubset({0, 1})
