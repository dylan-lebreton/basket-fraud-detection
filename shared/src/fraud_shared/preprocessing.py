"""Preprocessing functions shared between training and inference."""

import polars as pl

from fraud_shared.schema import (
    COLUMN_RENAMES,
    EXPECTED_DTYPES,
    ID_COLUMN,
    ITEM_COLUMNS,
    N_ITEMS_COLUMN,
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