import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target",
]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_COLS = ["sex", "fbs", "exang"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal"]


def ensure_raw_data(project_root: Path) -> None:
    """Idempotent: fetch + verify UCI raw data via dataprocessing/download_data.py."""
    script = project_root / "dataprocessing" / "download_data.py"
    subprocess.run([sys.executable, str(script)], check=True)


def load_cleveland(data_dir: str) -> pd.DataFrame:
    proc_path = os.path.join(data_dir, "processed.cleveland.data")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(
            f"Raw dataset not found at {proc_path}. "
            "Run `python dataprocessing/download_data.py` to fetch it from UCI."
        )
    df = pd.read_csv(proc_path, header=None, names=COLUMN_NAMES, na_values="?")
    print(f"Loaded {len(df)} records from processed.cleveland.data")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print(f"Missing values before cleaning: {df.isna().sum().sum()}")
    for col in df.columns:
        nulls = df[col].isna().sum()
        if nulls > 0 and col != "target":
            print(f"  {col}: {nulls} missing")

    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS + BINARY_COLS:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    df["target"] = (df["target"] > 0).astype(int)

    int_cols = ["age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "target"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    return df.reset_index(drop=True)


def main():
    project_root = Path(__file__).resolve().parent.parent
    ensure_raw_data(project_root)

    data_dir = project_root / "data"
    output_path = Path(__file__).resolve().parent / "processed_cleveland.csv"

    df = clean_data(load_cleveland(str(data_dir)))
    df.to_csv(output_path, index=False)

    print(f"\nFinal: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target distribution: {dict(df['target'].value_counts())}")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    main()
