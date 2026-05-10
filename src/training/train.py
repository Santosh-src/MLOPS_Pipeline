import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.data  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.preprocess import ALL_FEATURE_COLS, TARGET_COL, build_pipeline  # noqa: E402
from src.evaluation.evaluate import (  # noqa: E402
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_report,
)

RANDOM_STATE = 42
CV_FOLDS = 5
TEST_SIZE = 0.2
EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"
CHAMPION_ALIAS = "champion"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
TARGET_NAMES = ["No Disease", "Disease"]

MODEL_CONFIGS = {
    "LogisticRegression": {
        "classifier": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "param_grid": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__l1_ratio": [0],
            "classifier__solver": ["lbfgs"],
        },
    },
    "RandomForest": {
        "classifier": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 5],
        },
    },
}


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        return out or "unknown"
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _sha256_of(path: Path) -> str:
    if not path.is_file():
        return "unknown"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_tags(dataset_path: Path) -> dict[str, str]:
    return {
        "git_commit": _git_commit(),
        "dataset_sha256": _sha256_of(dataset_path),
        "env": os.environ.get("MLOPS_ENV", "local"),
        "python_version": platform.python_version(),
        "cv_folds": str(CV_FOLDS),
        "test_size": str(TEST_SIZE),
    }


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Path]:
    data_path = PROJECT_ROOT / "dataprocessing" / "processed_cleveland.csv"
    if not data_path.exists():
        os.system(f"python {PROJECT_ROOT / 'dataprocessing' / 'process_data.py'}")

    df = pd.read_csv(data_path)
    X, y = df[ALL_FEATURE_COLS], df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, data_path


def _log_dataset_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_path: Path,
) -> None:
    source = str(dataset_path.relative_to(PROJECT_ROOT))
    train_df = X_train.assign(**{TARGET_COL: y_train.values})
    test_df = X_test.assign(**{TARGET_COL: y_test.values})

    train_ds = mlflow.data.from_pandas(
        train_df, source=source, name="cleveland-train", targets=TARGET_COL
    )
    test_ds = mlflow.data.from_pandas(
        test_df, source=source, name="cleveland-test", targets=TARGET_COL
    )
    mlflow.log_input(train_ds, context="training")
    mlflow.log_input(test_ds, context="evaluation")


def _log_per_fold_metrics(cv_results: dict) -> None:
    for fold_idx in range(CV_FOLDS):
        for m in SCORING:
            mlflow.log_metric(
                f"fold_{m}", cv_results[f"test_{m}"][fold_idx], step=fold_idx
            )


def _log_classification_report(y_true, y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, target_names=TARGET_NAMES, output_dict=True
    )
    mlflow.log_dict(report, "classification_report.json")
    for cls_label, vals in report.items():
        if not isinstance(vals, dict):
            continue
        safe = cls_label.replace(" ", "_").replace("/", "_")
        for k, v in vals.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"per_class.{safe}.{k}", float(v))
    return report


def _run_evaluate(model_uri: str, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    eval_df = X_test.copy()
    eval_df[TARGET_COL] = y_test.values
    try:
        mlflow.evaluate(
            model=model_uri,
            data=eval_df,
            targets=TARGET_COL,
            model_type="classifier",
            evaluators=["default"],
        )
    except (MlflowException, RuntimeError, ValueError) as exc:
        print(f"mlflow.evaluate skipped: {exc}")


def train_and_evaluate(
    model_name: str,
    config: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_path: Path,
    tags: dict,
) -> dict:
    print(f"\n{'=' * 60}\nTraining: {model_name}\n{'=' * 60}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        build_pipeline(config["classifier"]),
        config["param_grid"],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        return_train_score=True,
    )

    with mlflow.start_run(run_name=f"{model_name}-grid") as run:
        mlflow.set_tags({**tags, "model_type": model_name})
        mlflow.log_params({
            "cv_folds": CV_FOLDS,
            "scoring": "roc_auc",
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        _log_dataset_inputs(X_train, X_test, y_train, y_test, dataset_path)

        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_

        cv_results = cross_validate(best_pipeline, X_train, y_train, cv=cv, scoring=SCORING)
        cv_metrics = {f"cv_{m}": float(np.mean(cv_results[f"test_{m}"])) for m in SCORING}
        cv_std = {f"cv_{m}_std": float(np.std(cv_results[f"test_{m}"])) for m in SCORING}
        _log_per_fold_metrics(cv_results)
        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(cv_std)

        y_train_pred = best_pipeline.predict(X_train)
        y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        y_test_pred = best_pipeline.predict(X_test)
        y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        report = _log_classification_report(y_test, y_test_pred)

        predictions = pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_test_pred,
            "disease_proba": y_test_proba,
        })
        mlflow.log_table(predictions, artifact_file="predictions.json")

        print(f"\nBest params: {grid.best_params_}")
        print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
        print("\nCross-validation metrics (mean +/- std):")
        for m in SCORING:
            print(f"  {m:12s}: {cv_metrics[f'cv_{m}']:.4f} +/- {cv_std[f'cv_{m}_std']:.4f}")
        print("\nHoldout test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print_report(y_test, y_test_pred)

        cm_fig = plot_confusion_matrix(y_test, y_test_pred, title=f"{model_name} - CM (test)")
        mlflow.log_figure(cm_fig, "plots/confusion_matrix_test.png")
        plt.close(cm_fig)

        roc_fig = plot_roc_curve(y_test, y_test_proba, title=f"{model_name} - ROC (test)")
        mlflow.log_figure(roc_fig, "plots/roc_curve_test.png")
        plt.close(roc_fig)

        sample = X_train.head(5)
        signature = infer_signature(sample, best_pipeline.predict(sample))
        logged = mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        _run_evaluate(logged.model_uri, X_test, y_test)

        run_id = run.info.run_id
        model_uri = logged.model_uri

    return {
        "model_name": model_name,
        "pipeline": best_pipeline,
        "best_params": grid.best_params_,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "cv_std": cv_std,
        "classification_report": report,
        "run_id": run_id,
        "model_uri": model_uri,
    }


def _promote_champion(model_uri: str, run_id: str) -> None:
    try:
        mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    except MlflowException as exc:
        print(f"Skipping model registry (backend may not support it): {exc}")
        return

    client = mlflow.tracking.MlflowClient()
    try:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version
        )
        print(
            f"Registered: {REGISTERED_MODEL_NAME} v{mv.version} -> "
            f"@{CHAMPION_ALIAS} (run {run_id[:8]})"
        )
    except MlflowException as exc:
        print(f"Registered v{mv.version}, but alias set failed: {exc}")


def main():
    default_uri = f"file://{PROJECT_ROOT / 'mlruns'}"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {tracking_uri}")

    mlflow.sklearn.autolog(
        log_models=False,
        log_input_examples=False,
        log_model_signatures=False,
        log_post_training_metrics=True,
        silent=True,
        max_tuning_runs=10,
    )

    X_train, X_test, y_train, y_test, dataset_path = load_data()
    print(f"Train: {X_train.shape[0]} samples  Test: {X_test.shape[0]} samples")
    print(f"Train target distribution: {dict(y_train.value_counts())}")
    print(f"Test target distribution:  {dict(y_test.value_counts())}")

    tags = _run_tags(dataset_path)

    results = {
        name: train_and_evaluate(
            name, cfg, X_train, X_test, y_train, y_test, dataset_path, tags
        )
        for name, cfg in MODEL_CONFIGS.items()
    }

    best_name = max(results, key=lambda k: results[k]["cv_metrics"]["cv_roc_auc"])
    best = results[best_name]

    print(f"\n{'=' * 60}\nBEST MODEL: {best_name}")
    print(f"  CV ROC-AUC:   {best['cv_metrics']['cv_roc_auc']:.4f}")
    print(f"  Test ROC-AUC: {best['test_metrics']['roc_auc']:.4f}")
    print(f"  Run ID:       {best['run_id']}")
    print(f"  Params:       {best['best_params']}\n{'=' * 60}")

    model_dir = PROJECT_ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], model_dir / "heart_disease_model.joblib")

    summary = {
        "best_model": best_name,
        "best_run_id": best["run_id"],
        "best_params": {
            k.replace("classifier__", ""): v for k, v in best["best_params"].items()
        },
        "cv_metrics": best["cv_metrics"],
        "train_metrics": best["train_metrics"],
        "test_metrics": best["test_metrics"],
        "tags": tags,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "tracking_uri": tracking_uri,
    }
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2, default=str))

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("status", "champion")
        mlflow.log_dict(summary, "metrics_summary.json")
        mlflow.log_artifact(str(metrics_path))

    _promote_champion(best["model_uri"], best["run_id"])

    print(f"\nSaved model to:   {model_dir / 'heart_disease_model.joblib'}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"\nView experiments at: {tracking_uri}")
    if tracking_uri.startswith("file://"):
        print("  (or `mlflow ui --port 5001`)")

    return results


if __name__ == "__main__":
    main()
