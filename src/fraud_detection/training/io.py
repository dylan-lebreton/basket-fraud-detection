"""S3/MinIO I/O for training data."""

import io
import os
from pathlib import Path

import boto3
import polars as pl


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )


def download_csv(bucket: str, key: str) -> pl.DataFrame:
    """Download a CSV from S3/MinIO and return as polars DataFrame."""
    client = get_s3_client()
    response = client.get_object(Bucket=bucket, Key=key)
    data = response["Body"].read()
    return pl.read_csv(io.BytesIO(data), infer_schema_length=None)


def download_to_file(bucket: str, key: str, dest: Path) -> Path:
    """Download an object to a local file."""
    client = get_s3_client()
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return dest


def upload_file(bucket: str, key: str, src: Path) -> None:
    """Upload a local file to S3/MinIO."""
    client = get_s3_client()
    client.upload_file(str(src), bucket, key)