"""Phase 1 audit: the SQL in ``sql/`` run on the transactions view, plus a missingness profile."""
from __future__ import annotations

import pandas as pd

from .data import SQL_DIR, TRAIN_END, VALID_END


def sql(name: str, **params) -> str:
    text = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
    return text.format(train_end=params.get("train_end", TRAIN_END), valid_end=params.get("valid_end", VALID_END))


def create_base(con, train_end: int = TRAIN_END, valid_end: int = VALID_END) -> None:
    """Create the ``base`` view (day, approximate customer key, split) over ``tx``."""
    con.execute(sql("00_base", train_end=train_end, valid_end=valid_end))


def query(con, name: str) -> pd.DataFrame:
    return con.sql(sql(name)).df()


def column_groups(con, table: str = "tx") -> dict[str, list[str]]:
    cols = [r[0] for r in con.sql(f"DESCRIBE {table}").fetchall()]
    groups = {"V": [], "C": [], "D": [], "M": [], "id": []}
    for c in cols:
        for prefix in ("V", "C", "D", "M"):
            if c.startswith(prefix) and c[len(prefix):].isdigit():
                groups[prefix].append(c)
        if c.startswith("id_"):
            groups["id"].append(c)
    return groups


def missingness(con, columns: list[str], table: str = "tx") -> pd.Series:
    """Share of rows missing, per column (one pass over the table)."""
    parts = ", ".join(f'AVG(CASE WHEN "{c}" IS NULL THEN 1.0 ELSE 0 END) AS "{c}"' for c in columns)
    row = con.sql(f"SELECT {parts} FROM {table}").df().iloc[0]
    return row.astype(float)


def missingness_blocks(miss: pd.Series, decimals: int = 6) -> pd.DataFrame:
    """Group columns that are missing at exactly the same rate: in this data such columns are
    missing on the same rows (they come from the same source), so each block is one signal."""
    key = miss.round(decimals)
    blocks = (pd.DataFrame({"column": miss.index, "missing": key.to_numpy()})
              .groupby("missing")["column"].agg(["count", "first", "last"])
              .rename(columns={"count": "columns"}).reset_index().sort_values("missing"))
    return blocks.reset_index(drop=True)


def same_rows_check(con, columns: list[str], table: str = "tx") -> bool:
    """True when every column in ``columns`` is missing on exactly the same rows."""
    if len(columns) < 2:
        return True
    first = columns[0]
    cond = " OR ".join(f'(("{first}" IS NULL) <> ("{c}" IS NULL))' for c in columns[1:])
    return con.sql(f"SELECT COUNT(*) FROM {table} WHERE {cond}").fetchall()[0][0] == 0
