"""Build data/interim/transactions.parquet from the two competition CSVs.

    python scripts/prepare_data.py

Reads data/raw/train_transaction.csv and data/raw/train_identity.csv with DuckDB (streamed, so
it fits in a few GB of memory), checks them against the published shape, left-joins identity
onto transactions by TransactionID, and writes one Parquet file. Counts are written to
data/interim/prepare_manifest.json. Neither file leaves your machine: data/ is gitignored.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraudrisk.data import PARQUET, RAW_DIR  # noqa: E402

# Published shape of the training files. 144,233 identity rows is the commonly reported count;
# it is checked as a warning only, since identity coverage is measured in phase 1 anyway.
EXPECTED = {"transaction_rows": 590_540, "transaction_columns": 394, "identity_columns": 41}
EXPECTED_IDENTITY_ROWS = 144_233


def prepare(raw_dir: Path = RAW_DIR, out: Path = PARQUET, expected: dict | None = EXPECTED,
            memory_limit: str = "2GB") -> dict:
    import duckdb

    tx_csv, id_csv = raw_dir / "train_transaction.csv", raw_dir / "train_identity.csv"
    for f in (tx_csv, id_csv):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found; see data/raw/README.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    # sample_size = -1: infer each column's type from the whole file, not the first rows
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv('{tx_csv.as_posix()}', header = true, sample_size = -1)")
    con.execute(f"CREATE VIEW i AS SELECT * FROM read_csv('{id_csv.as_posix()}', header = true, sample_size = -1)")
    info = {
        "transaction_rows": con.sql("SELECT COUNT(*) FROM t").fetchone()[0],
        "transaction_columns": len(con.sql("DESCRIBE t").fetchall()),
        "identity_rows": con.sql("SELECT COUNT(*) FROM i").fetchone()[0],
        "identity_columns": len(con.sql("DESCRIBE i").fetchall()),
        "duplicate_transaction_ids": con.sql(
            "SELECT COUNT(*) - COUNT(DISTINCT TransactionID) FROM t").fetchone()[0],
        "identity_without_transaction": con.sql(
            "SELECT COUNT(*) FROM i ANTI JOIN t USING (TransactionID)").fetchone()[0],
    }
    if expected:
        bad = {k: (info[k], v) for k, v in expected.items() if info[k] != v}
        if bad:
            raise ValueError(f"files do not match the published shape (found, expected): {bad}")
    if expected and info["identity_rows"] != EXPECTED_IDENTITY_ROWS:
        print(f"note: {info['identity_rows']:,} identity rows (commonly reported: {EXPECTED_IDENTITY_ROWS:,})")
    if info["duplicate_transaction_ids"] or info["identity_without_transaction"]:
        raise ValueError(f"key problems: {info}")
    t0 = time.time()
    con.execute(f"""COPY (SELECT * FROM t LEFT JOIN i USING (TransactionID) ORDER BY TransactionDT, TransactionID)
                    TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    info["joined_rows"] = con.sql(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    info["joined_columns"] = len(con.sql(f"DESCRIBE SELECT * FROM read_parquet('{out.as_posix()}')").fetchall())
    info["fraud_rate"] = con.sql(f"SELECT AVG(isFraud) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    info["seconds"] = round(time.time() - t0, 1)
    info["parquet_mb"] = round(out.stat().st_size / 1e6, 1)
    if info["joined_rows"] != info["transaction_rows"]:
        raise ValueError("the join changed the number of transactions")
    (out.parent / "prepare_manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--memory-limit", default="2GB")
    args = ap.parse_args(argv)
    info = prepare(memory_limit=args.memory_limit)
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
