import numpy as np
import pandas as pd

from fraudrisk import data, features

RAW = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "addr1", "D1",
       "P_emaildomain", "DeviceType", "DeviceInfo", "dist1", "id_01"]


def test_no_feature_sees_the_future(synthetic):
    raw = synthetic[RAW]
    cut = list(raw["TransactionDT"].quantile([0.4, 0.8]))
    pit = features.point_in_time_violations(raw, cut)
    assert pit[features.FEATURES].to_numpy().sum() == 0
    assert (pit["transactions_checked"] > 0).all()


def test_the_check_catches_a_leaky_feature(synthetic, monkeypatch):
    raw = synthetic[RAW]
    real = features._features_of

    def leaky(df):
        f = real(df)
        # total transactions of the customer over the WHOLE table, future included
        f["cust_prior_tx"] = f.groupby("day")["day"].transform("size").astype(float)
        return f

    monkeypatch.setattr(features, "_features_of", leaky)
    pit = features.point_in_time_violations(raw, [raw["TransactionDT"].median()], ["cust_prior_tx"])
    assert pit["cust_prior_tx"].iloc[0] > 0


def test_velocity_counts_known_answer():
    rows = []
    for i, (t, amt) in enumerate([(0, 10.0), (1800, 20.0), (1800, 30.0), (80_000, 40.0)]):
        rows.append({"TransactionID": i + 1, "isFraud": 0, "TransactionDT": 86_400 + t, "TransactionAmt": amt,
                     "ProductCD": "W", "card1": 5, "addr1": 1.0, "D1": float(t // 86_400), "P_emaildomain": "a.com",
                     "DeviceType": None, "DeviceInfo": None, "dist1": np.nan, "id_01": np.nan})
    con = data.connect(pd.DataFrame(rows))
    features.build(con)
    f = features.frame(con).set_index("TransactionID")
    assert f["cust_prior_tx"].tolist() == [0, 1, 1, 3]          # same-second rows do not count each other
    assert f["cust_tx_1h"].tolist() == [0, 1, 1, 0]
    assert f["cust_tx_24h"].tolist() == [0, 1, 1, 3]
    assert f.loc[4, "amt_vs_cust_mean"] == 40.0 / 20.0
    assert np.isnan(f.loc[1, "hours_since_prev"]) and f.loc[4, "hours_since_prev"] == (80_000 - 1800) / 3600
