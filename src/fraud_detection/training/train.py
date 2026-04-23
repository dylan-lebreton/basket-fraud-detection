"""Training entrypoint: load from S3, train XGBoost, log to MLflow."""

import json
import tempfile
from pathlib import Path

import mlflow
from mlflow import xgboost as mlflow_xgboost
import polars as pl
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from fraud_detection.preprocessing import build_features
from fraud_detection.schema import FEATURE_COLUMNS
from fraud_detection.training.config import settings
from fraud_detection.training.io import download_csv


def load_training_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    x = download_csv(settings.raw_bucket, settings.x_key)
    y = download_csv(settings.raw_bucket, settings.y_key)
    return x, y


def build_training_set(
    raw_x: pl.DataFrame, raw_y: pl.DataFrame
) -> tuple[pd.DataFrame, pd.Series, dict]:
    features_df, artifacts = build_features(raw_x, raw_y)
    bool_cols = [c for c, dt in features_df.schema.items() if dt == pl.Boolean]
    features_df = features_df.with_columns(
        pl.col(c).cast(pl.Int8) for c in bool_cols)

    x = features_df.select(FEATURE_COLUMNS).to_pandas()
    y = features_df["target"].to_pandas()
    return x, y, artifacts


def build_model(y) -> XGBClassifier:
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **settings.xgb.model_dump())


def cross_validate(model, x, y) -> dict[str, float]:
    kf = StratifiedKFold(n_splits=settings.cv_splits,
                         shuffle=True, random_state=settings.random_state)
    scores = cross_val_score(
        model, x, y, cv=kf, scoring="average_precision", n_jobs=-1)
    return {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}


def evaluate_train(model, x, y) -> float:
    probas = model.predict_proba(x)[:, 1]
    return float(average_precision_score(y, probas))


def main() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        print(f"MLflow run: {run.info.run_id}")

        raw_x, raw_y = load_training_data()
        x, y, artifacts = build_training_set(raw_x, raw_y)

        mlflow.log_params({
            "raw_bucket": settings.raw_bucket,
            "x_key": settings.x_key,
            "y_key": settings.y_key,
            "n_samples": len(x),
            "n_frauds": int(y.sum()),
            "fraud_rate": float(y.mean()),
            **settings.xgb.model_dump(),
        })

        cv_model = build_model(y)
        cv_scores = cross_validate(cv_model, x, y)
        mlflow.log_metrics(cv_scores)
        print(
            f"CV PR-AUC: {cv_scores['cv_mean']:.4f} +/- {cv_scores['cv_std']:.4f}")

        final_model = build_model(y)
        final_model.fit(x, y)
        train_score = evaluate_train(final_model, x, y)
        mlflow.log_metric("train_pr_auc", train_score)
        print(f"Train PR-AUC: {train_score:.4f}")

        with tempfile.TemporaryDirectory() as tmp:
            artifacts_path = Path(tmp) / "artifacts.json"
            artifacts_path.write_text(json.dumps(artifacts))
            mlflow.log_artifact(str(artifacts_path))

        mlflow_xgboost.log_model(
            final_model,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
        )
        print(f"Model registered: {settings.mlflow_model_name}")


if __name__ == "__main__":
    main()
