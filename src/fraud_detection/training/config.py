"""Training configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class XGBParams(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="XGB_")

    n_estimators: int = 500
    max_depth: int = 10
    learning_rate: float = 0.05
    min_child_weight: int = 5
    eval_metric: str = "aucpr"
    tree_method: str = "hist"
    n_jobs: int = -1
    random_state: int = 0


class TrainingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    raw_bucket: str = "fraud-raw-data"
    x_key: str = Field(default="X_train.csv", alias="X_TRAIN_KEY")
    y_key: str = Field(default="Y_train.csv", alias="Y_TRAIN_KEY")

    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "fraud-detection"
    mlflow_model_name: str = "fraud-detector"

    cv_splits: int = 10
    random_state: int = 0

    xgb: XGBParams = Field(default_factory=XGBParams)


settings = TrainingSettings()