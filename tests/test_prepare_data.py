import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_data import prepare  # noqa: E402

from fraudrisk.synthetic import IDENTITY_COLS_ALL as _ID  # noqa: E402


def test_prepare_joins_identity_and_keeps_every_transaction(tmp_path, synthetic):
    raw = tmp_path / "raw"
    raw.mkdir()
    tx = synthetic.drop(columns=_ID)
    ident = synthetic.loc[synthetic["DeviceType"].notna(), ["TransactionID", *_ID]]
    tx.to_csv(raw / "train_transaction.csv", index=False)
    ident.to_csv(raw / "train_identity.csv", index=False)
    out = tmp_path / "interim" / "transactions.parquet"
    info = prepare(raw, out, expected=None, memory_limit="1GB")
    assert info["joined_rows"] == len(synthetic)
    assert info["identity_rows"] == len(ident)
    back = pd.read_parquet(out)
    assert back["TransactionDT"].is_monotonic_increasing
    assert back["DeviceType"].notna().sum() == len(ident)
    assert abs(back["isFraud"].mean() - synthetic["isFraud"].mean()) < 1e-12
    assert json.loads((out.parent / "prepare_manifest.json").read_text())["joined_columns"] == synthetic.shape[1]
