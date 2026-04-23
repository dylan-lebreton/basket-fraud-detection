"""Load the champion model and its artifacts from MLflow."""

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow import xgboost as mlflow_xgboost
from mlflow.tracking import MlflowClient

from fraud_detection.inference.config import settings


def load_model_and_artifacts() -> tuple[Any, dict, str]:
    """Load the champion model, its artifacts.json, and return (model, artifacts, version)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    version = client.get_model_version_by_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias
    )

    model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    model = mlflow_xgboost.load_model(model_uri)

    with tempfile.TemporaryDirectory() as tmp:
        assert version.run_id
        local_dir = client.download_artifacts(
            version.run_id, "artifacts.json", tmp)
        artifacts = json.loads(Path(local_dir).read_text())

    return model, artifacts, version.version
