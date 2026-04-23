"""Inference configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_model_name: str = "fraud-detector"
    mlflow_model_alias: str = "champion"


settings = InferenceSettings()