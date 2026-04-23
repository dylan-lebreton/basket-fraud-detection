"""Functions that compute artifacts from training data."""

import polars as pl


def compute_make_filling(df: pl.DataFrame) -> dict[str, str]:
    """For each goods_code, we take the mode of `make` (excluding RETAILER when possible)."""
    lookup_df = (
        df.filter(pl.col("make").is_not_null())
        .group_by("goods_code", "make")
        .agg(pl.len())
        .with_columns(is_non_retailer=(pl.col("make") != "RETAILER").cast(pl.Int8))
        .sort(["goods_code", "is_non_retailer", "len"], descending=[False, True, True])
        .unique("goods_code", keep="first")
    )
    return {
        row["goods_code"]: row["make"]
        for row in lookup_df.iter_rows(named=True)
    }


def compute_model_filling(df: pl.DataFrame) -> dict[str, str]:
    """For each goods_code, we take the mode of `model` (excluding RETAILER when possible)."""
    lookup_df = (
        df.filter(pl.col("model").is_not_null())
        .group_by("goods_code", "model")
        .agg(pl.len())
        .with_columns(is_non_retailer=(pl.col("model") != "RETAILER").cast(pl.Int8))
        .sort(["goods_code", "is_non_retailer", "len"], descending=[False, True, True])
        .unique("goods_code", keep="first")
    )
    return {
        row["goods_code"]: row["model"]
        for row in lookup_df.iter_rows(named=True)
    }