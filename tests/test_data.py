import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.preprocess import ALL_FEATURE_COLS, TARGET_COL

DATA_PATH = PROJECT_ROOT / "dataprocessing" / "processed_cleveland.csv"
EXPECTED_COLUMNS = ALL_FEATURE_COLS + [TARGET_COL]


@pytest.fixture
def df():
    if not DATA_PATH.exists():
        os.system(f"python {PROJECT_ROOT / 'dataprocessing' / 'process_data.py'}")
    return pd.read_csv(DATA_PATH)


def test_data_exists():
    assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"


def test_data_shape(df):
    assert df.shape[0] >= 290
    assert df.shape[1] == 14


def test_no_missing_values(df):
    assert df.isna().sum().sum() == 0


def test_correct_columns(df):
    for col in EXPECTED_COLUMNS:
        assert col in df.columns, f"Missing column: {col}"


def test_target_is_binary(df):
    assert set(df[TARGET_COL].unique()).issubset({0, 1})


def test_feature_ranges(df):
    assert 0 <= df["age"].min() <= df["age"].max() <= 120
    assert df["chol"].min() >= 0
    assert df["thalach"].min() >= 0
    assert df["trestbps"].min() >= 0


def test_dtypes(df):
    int_cols = ["age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
    for col in int_cols:
        assert np.issubdtype(df[col].dtype, np.integer), f"{col} should be int, got {df[col].dtype}"


def test_class_balance(df):
    counts = df[TARGET_COL].value_counts(normalize=True)
    assert counts[0] >= 0.30 and counts[1] >= 0.30
