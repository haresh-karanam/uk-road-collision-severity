"""Tiny SQLite layer: load the cleaned collisions into an in-memory database and
run the named queries in sql/queries.sql. Pure standard library + pandas."""
import re
import sqlite3
from pathlib import Path
import pandas as pd

QUERY_FILE = Path(__file__).parent / "sql" / "queries.sql"
COLS = ["collision_year", "ksi", "hour", "speed_limit", "urban_or_rural_area", "light_conditions"]


def connect(df: pd.DataFrame) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    df[COLS].to_sql("collisions", con, index=False)
    return con


def queries(path: Path = QUERY_FILE) -> dict[str, str]:
    text = Path(path).read_text()
    parts = re.split(r"^-- name:\s*(\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    return pd.read_sql_query(queries()[name], con)
