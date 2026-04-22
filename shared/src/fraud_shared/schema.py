"""Column names and schema constants for the input data."""

import polars as pl

# Raw schema
ID_COLUMN = "ID"
N_ITEMS_COLUMN = "Nb_of_items"
TARGET_COLUMN = "fraud_flag"
ITEM_COLUMNS = ["item", "cash_price", "make", "model", "goods_code", "Nbr_of_prod_purchas"]
MAX_ITEMS = 24

# Column renames applied after unpivot
COLUMN_RENAMES = {
    N_ITEMS_COLUMN: "n_items",
    "Nbr_of_prod_purchas": "n_prods",
}

# Dtypes
EXPECTED_DTYPES = {
    ID_COLUMN: pl.Utf8,
    "n_items": pl.Int64,
    "n_prods": pl.Int64,
    "item_rank": pl.Int64,
    "item": pl.Utf8,
    "cash_price": pl.Float64,
    "make": pl.Utf8,
    "model": pl.Utf8,
    "goods_code": pl.Utf8,
}

# Regex
NORMAL_CHARS_REGEX = r"[^A-Z0-9]"
DIGIT_PREFIX_REGEX = r"^\d+[A-Z]"

# Values to replace found in EDA
ITEMS_OUTLIERS = {
    "BATHROOMACCESSORIES": "BATHROOM",
    "BATHROOMFIXTURES": "BATHROOM",
    "APPLEPRODUCTDESCRIPTION": "OTHER",
    "APPLES": "OTHER",
    "HPELITEBOOK850V6": "OTHER",
    "LOGITECHPEBBLEM350BLUETOOTHWIRELESSMOUSE": "OTHER",
    "MICROSOFTOFFICEHOMEANDSTUDENT2019": "OTHER",
    "PRODUCT": "OTHER",
    "TARGUSGEOLITEESSENTIALCASE": "OTHER",
    "TOSHIBAPORTABLEHARDDRIVE": "OTHER",
    "UNKNOWN": "OTHER",
}