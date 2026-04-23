"""Preprocessing functions shared between training and inference."""

import polars as pl

from fraud_detection.schema import (
    COLUMN_RENAMES,
    DIGIT_PREFIX_REGEX,
    EXPECTED_DTYPES,
    ID_COLUMN,
    ITEM_COLUMNS,
    ITEMS_OUTLIERS,
    N_ITEMS_COLUMN,
    NORMAL_CHARS_REGEX,
)


def unpivot_items(df: pl.DataFrame) -> pl.DataFrame:
    """Transform wide basket format to long format (one row per item)."""
    item_ranks = sorted(
        int(col[len("item"):]) for col in df.columns
        if col.startswith("item") and col[len("item"):].isdigit()
    )
    if not item_ranks:
        raise ValueError("No item columns detected")

    frames = [
        df.select(
            pl.col(ID_COLUMN),
            pl.col(N_ITEMS_COLUMN),
            pl.lit(rank).alias("item_rank"),
            *[pl.col(f"{col}{rank}").alias(col) for col in ITEM_COLUMNS],
        )
        for rank in item_ranks
    ]
    return pl.concat(frames, how="vertical_relaxed")


def remove_empty_items(df: pl.DataFrame) -> pl.DataFrame:
    """Remove rows where all item columns are null (leftovers of the unpivot)."""
    return df.filter(pl.any_horizontal(pl.col(ITEM_COLUMNS).is_not_null()))


def rename_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename raw columns to snake_case."""
    return df.rename(COLUMN_RENAMES)


def cast_types(df: pl.DataFrame) -> pl.DataFrame:
    """Cast columns to their expected types."""
    return df.with_columns(
        pl.col(col).cast(dtype) for col, dtype in EXPECTED_DTYPES.items() if col in df.columns
    )


def clean_item(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the item column: uppercase, strip non-alphanumeric, remap outliers."""
    return df.with_columns(
        pl.col("item")
        .str.to_uppercase()
        .str.replace_all(NORMAL_CHARS_REGEX, "")
    ).with_columns(
        pl.col("item").replace_strict(ITEMS_OUTLIERS, default=pl.col("item"))
    ).with_columns(
        pl.when(pl.col("item").str.contains(DIGIT_PREFIX_REGEX))
        .then(pl.lit("OTHER"))
        .otherwise(pl.col("item"))
        .alias('item')
    )


def apply_filling(df: pl.DataFrame, column: str, filling: dict[str, str]) -> pl.DataFrame:
    """Fill missing values in column using a goods_code lookup."""
    return df.with_columns(
        pl.col(column)
        .fill_null(pl.col("goods_code").replace_strict(filling, default=None))
        .fill_null("OTHER")
    )


def aggregate_basket(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate items into one row per basket with computed features."""
    aggregations = [
        pl.col("n_items").first().alias("n_items"),
        pl.col("n_prods").sum().alias("n_prods"),
        pl.col("cash_price").sum().alias("price"),
        pl.col("cash_price").max().alias("max_price"),
        pl.col("cash_price").mean().alias("mean_price"),
        pl.col("cash_price").std().fill_null(0.0).alias("std_price"),
        pl.col("cash_price").max().is_between(
            1000, 2000).alias("max_price_in_fraud_zone"),
        pl.col("item").eq("COMPUTERS").any().alias("has_computer"),
        pl.col("item").eq("FULFILMENTCHARGE").any().alias("has_fulfilment"),
        pl.col("item").eq("SERVICE").any().alias("has_service"),
        pl.col("make").eq("APPLE").any().alias("has_apple"),
        pl.col("model").str.contains("MACBOOK").any().alias("has_macbook"),
        pl.col("model").str.contains("IPAD").any().alias("has_ipad"),
    ]

    if "target" in df.columns:
        aggregations.append(pl.col("target").first().alias("target"))

    return df.group_by(ID_COLUMN).agg(aggregations)
