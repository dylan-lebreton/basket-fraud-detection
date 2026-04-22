"""Data contract for raw basket input."""

import polars as pl

from fraud_shared.schema import ID_COLUMN, ITEM_COLUMNS, MAX_ITEMS, N_ITEMS_COLUMN

EXPECTED_COLUMNS = (
    {ID_COLUMN, N_ITEMS_COLUMN}
    | {f"{col}{rank}" for rank in range(1, MAX_ITEMS + 1) for col in ITEM_COLUMNS}
)


def validate_raw_basket(df: pl.DataFrame) -> pl.DataFrame:
    """Validate raw basket DataFrame against the expected schema."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    extra = set(df.columns) - EXPECTED_COLUMNS
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")

    if df.select(pl.col(N_ITEMS_COLUMN).min()).item() < 1:
        raise ValueError(f"{N_ITEMS_COLUMN} must be >= 1")

    if df[ID_COLUMN].n_unique() != df.height:
        raise ValueError(f"{ID_COLUMN} must be unique")

    return df