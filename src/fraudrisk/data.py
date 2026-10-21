"""Paths, the time split, and loading the prepared table (or a synthetic stand-in).

The real table (``data/interim/transactions.parquet``) is built by ``scripts/prepare_data.py``
from competition data that may not be redistributed, so it never reaches GitHub. CI and anyone
without the data get the synthetic table from :mod:`fraudrisk.synthetic`, which has the same
columns and the same labelling mechanism.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "interim" / "transactions.parquet"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
SQL_DIR = ROOT / "sql"

DAY = 86_400  # TransactionDT is in seconds

# Time split, in days since the first transaction (TransactionDT // DAY - first day).
# Fixed before modelling; notebook 01 checks each block holds enough fraud to be useful.
TRAIN_END = 120   # days [0, 120): fit
VALID_END = 150   # days [120, 150): choose models, thresholds and policies
# days [150, end]: test, scored once per model family


def real_data_available(path: Path = PARQUET) -> bool:
    """True when the prepared real table exists and FRAUDRISK_SYNTHETIC is not set."""
    return Path(path).exists() and os.environ.get("FRAUDRISK_SYNTHETIC", "") not in ("1", "true", "yes")


def source() -> str:
    return "real" if real_data_available() else "synthetic"


def load(columns: list[str] | None = None, path: Path = PARQUET, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """Return ``(transactions, source)`` where source is ``"real"`` or ``"synthetic"``."""
    if real_data_available(path):
        return pd.read_parquet(path, columns=columns), "real"
    from .synthetic import make_transactions

    df = make_transactions(seed=seed)
    return (df[columns] if columns else df), "synthetic"


def connect(df: pd.DataFrame | None = None):
    """A DuckDB connection with a ``tx`` view over the real Parquet file or the given frame."""
    import duckdb

    con = duckdb.connect()
    if df is not None:
        con.register("tx_frame", df)
        con.execute("CREATE VIEW tx AS SELECT * FROM tx_frame")
    elif real_data_available():
        con.execute(f"CREATE VIEW tx AS SELECT * FROM read_parquet('{PARQUET.as_posix()}')")
    else:
        from .synthetic import make_transactions

        con.register("tx_frame", make_transactions())
        con.execute("CREATE VIEW tx AS SELECT * FROM tx_frame")
    return con


def savefig(fig, name: str, source_: str) -> None:
    """Save a figure under reports/figures, but only for real-data runs."""
    if source_ != "real":
        return None
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
