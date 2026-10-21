"""Phase 2: point-in-time features, built in SQL (``sql/10_features.sql``), and the check that
they cannot see the future."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import audit, data

FEATURES = ["hour", "amount_cents", "cust_prior_tx", "cust_tx_1h", "cust_tx_24h", "cust_tx_7d", "cust_amt_24h",
            "hours_since_prev", "cust_mean_amt", "amt_vs_cust_mean", "amt_vs_cust_max", "email_changed",
            "device_changed", "card_tx_24h", "card_tx_1h"]


def build(con) -> None:
    """Create the ``features`` table from ``tx`` (``base`` is created if missing)."""
    if not con.sql("SELECT COUNT(*) FROM duckdb_views() WHERE view_name = 'base'").fetchone()[0]:
        audit.create_base(con)
    con.execute(audit.sql("10_features"))


def frame(con, columns: list[str] | None = None) -> pd.DataFrame:
    cols = ", ".join(columns) if columns else "*"
    return con.sql(f"SELECT {cols} FROM features ORDER BY TransactionDT, TransactionID").df()


def point_in_time_violations(raw: pd.DataFrame, cutoffs: list[float], columns: list[str] = FEATURES) -> pd.DataFrame:
    """Rebuild the features on the data truncated at each cutoff (everything after it deleted) and
    compare with the features built on all the data. A feature that changes for a transaction
    before the cutoff was using information from after it. Returns one row per cutoff."""
    full = _features_of(raw)
    rows = []
    for cut in cutoffs:
        part = _features_of(raw[raw["TransactionDT"] < cut])
        f = full.loc[part.index]
        diffs = {}
        for c in columns:
            a, b = f[c].to_numpy(dtype=float), part[c].to_numpy(dtype=float)
            same = np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
            diffs[c] = int((~same).sum())
        rows.append({"cutoff": cut, "transactions_checked": len(part), **diffs})
    return pd.DataFrame(rows)


def _features_of(raw: pd.DataFrame) -> pd.DataFrame:
    con = data.connect(raw)
    build(con)
    return frame(con).set_index("TransactionID")
