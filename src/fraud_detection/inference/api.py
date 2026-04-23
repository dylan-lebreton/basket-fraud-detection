"""FastAPI inference service."""

from contextlib import asynccontextmanager
import io
from typing import Any

import pandas as pd
import polars as pl
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from fraud_detection.contract import validate_raw_basket
from fraud_detection.inference.model_loader import load_model_and_artifacts
from fraud_detection.preprocessing import build_features
from fraud_detection.schema import FEATURE_COLUMNS, ID_COLUMN, ITEM_COLUMNS, MAX_ITEMS, N_ITEMS_COLUMN


state: dict[str, Any] = {"model": None, "artifacts": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model, artifacts, version = load_model_and_artifacts()
        state.update(model=model, artifacts=artifacts, version=version)
    except Exception as e:
        print(f"Warning: failed to load model at startup: {e}")
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


class Item(BaseModel):
    item: str | None = None
    cash_price: float | None = None
    make: str | None = None
    model: str | None = None
    goods_code: str | None = None
    nbr_of_prod_purchas: int | None = None


class BasketRequest(BaseModel):
    id: int
    items: list[Item]


class PredictionResponse(BaseModel):
    id: int
    fraud_probability: float
    model_version: str


def basket_to_raw_df(basket: BasketRequest) -> pl.DataFrame:
    """Convert a basket request to the wide raw format expected by preprocessing."""
    if len(basket.items) > MAX_ITEMS:
        raise ValueError(f"Basket has more than {MAX_ITEMS} items")

    row: dict[str, Any] = {ID_COLUMN: basket.id,
                           N_ITEMS_COLUMN: len(basket.items)}
    for rank in range(1, MAX_ITEMS + 1):
        item = basket.items[rank - 1] if rank <= len(basket.items) else None
        row[f"item{rank}"] = item.item if item else None
        row[f"cash_price{rank}"] = item.cash_price if item else None
        row[f"make{rank}"] = item.make if item else None
        row[f"model{rank}"] = item.model if item else None
        row[f"goods_code{rank}"] = item.goods_code if item else None
        row[f"Nbr_of_prod_purchas{rank}"] = item.nbr_of_prod_purchas if item else None

    return pl.DataFrame([row])


def predict_from_raw(raw_df: pl.DataFrame) -> pd.DataFrame:
    """Run preprocessing + prediction on a raw wide DataFrame. Returns (ID, fraud_flag)."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    validate_raw_basket(raw_df)
    features_df, _ = build_features(raw_df, artifacts=state["artifacts"])

    bool_cols = [c for c, dt in features_df.schema.items() if dt == pl.Boolean]
    features_df = features_df.with_columns(
        pl.col(c).cast(pl.Int8) for c in bool_cols)

    x = features_df.select(FEATURE_COLUMNS).to_pandas()
    probas = state["model"].predict_proba(x)[:, 1]

    return pd.DataFrame({"ID": features_df["ID"].to_pandas(), "fraud_flag": probas})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if state["model"] is not None else "model_not_loaded",
        "model_version": state["version"],
    }


@app.post("/reload")
def reload() -> dict:
    try:
        model, artifacts, version = load_model_and_artifacts()
        state.update(model=model, artifacts=artifacts, version=version)
        return {"status": "reloaded", "model_version": version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(basket: BasketRequest) -> PredictionResponse:
    try:
        raw_df = basket_to_raw_df(basket)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = predict_from_raw(raw_df)
    return PredictionResponse(
        id=int(result["ID"].iloc[0]),
        fraud_probability=float(result["fraud_flag"].iloc[0]),
        model_version=state["version"],
    )


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        raw_df = pl.read_csv(io.BytesIO(content), infer_schema_length=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    result = predict_from_raw(raw_df)
    return {
        "model_version": state["version"],
        "predictions": result.to_dict(orient="records"),
    }
