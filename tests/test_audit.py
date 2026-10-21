import numpy as np
import pandas as pd
import pytest

from fraudrisk import audit, data


def tiny():
    """Two customers: A defrauded from its third transaction on, B never. Days 0-9 train, 10-14 valid, 15+ test."""
    rows = []
    tid = 1
    for day, cust, fraud in [(0, "A", 0), (2, "A", 0), (5, "A", 1), (12, "A", 1), (16, "A", 1),
                             (1, "B", 0), (11, "B", 0), (17, "B", 0), (18, "B", 1)]:
        card1, addr1, first = (1000, 300.0, -5) if cust == "A" else (2000, np.nan, 0)
        rows.append({"TransactionID": tid, "isFraud": fraud, "TransactionDT": 86_400 * (1 + day) + 3600,
                     "TransactionAmt": 10.0 * tid, "ProductCD": "W", "card1": card1, "addr1": addr1,
                     "D1": float(1 + day - (1 + first)), "P_emaildomain": "gmail.com", "DeviceType": None,
                     "id_01": np.nan})
        tid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def con():
    c = data.connect(tiny())
    audit.create_base(c, train_end=10, valid_end=15)
    return c


def test_uid_is_constant_within_a_customer(con):
    uids = con.sql("SELECT card1, COUNT(DISTINCT uid) AS n FROM base GROUP BY card1").df()
    assert (uids["n"] == 1).all()
    assert con.sql("SELECT uid FROM base WHERE card1 = 2000 LIMIT 1").fetchone()[0].split("_")[1] == "na"


def test_splits_follow_days(con):
    s = con.sql("SELECT split, COUNT(*) AS n FROM base GROUP BY split").df().set_index("split")["n"]
    assert s.to_dict() == {"train": 4, "valid": 2, "test": 3}


def test_label_propagation_counts(con):
    p = audit.query(con, "04_label_propagation").set_index("previous")
    assert p.loc["first transaction of customer", "transactions"] == 2
    # A: prev fraud -> next fraud twice; B's last fraud follows a non-fraud
    assert p.loc["previous transaction fraud", "transactions"] == 2
    assert p.loc["previous transaction fraud", "fraud_rate"] == 1.0
    assert p.loc["previous transaction not fraud", "fraud_rate"] == pytest.approx(2 / 5)


def test_time_split_memorization(con):
    leak = audit.query(con, "06_leakage").set_index("split")
    # A is labelled fraud in train, so its valid and test transactions are flagged; B's test fraud is not
    assert leak.loc["valid", "recall"] == 1.0
    assert leak.loc["test", "recall"] == pytest.approx(0.5)
    assert leak.loc["test", "precision"] == 1.0


def test_missingness_blocks():
    df = pd.DataFrame({"V1": [1, None, 3, None], "V2": [1, None, 2, None], "V3": [None, None, None, 1]})
    c = data.connect(df)
    m = audit.missingness(c, ["V1", "V2", "V3"])
    blocks = audit.missingness_blocks(m)
    assert blocks["columns"].tolist() == [2, 1]
    assert audit.same_rows_check(c, ["V1", "V2"]) and not audit.same_rows_check(c, ["V1", "V3"])
