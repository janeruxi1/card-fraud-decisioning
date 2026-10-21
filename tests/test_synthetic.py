from fraudrisk.synthetic import N_V

TRANSACTION_COLUMNS = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
                       + [f"card{i}" for i in range(1, 7)] + ["addr1", "addr2", "dist1", "dist2",
                                                               "P_emaildomain", "R_emaildomain"]
                       + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
                       + [f"M{i}" for i in range(1, 10)] + [f"V{i}" for i in range(1, N_V + 1)])
IDENTITY_COLUMNS = [f"id_{i:02d}" for i in range(1, 39)] + ["DeviceType", "DeviceInfo"]


def test_same_columns_as_the_competition_files(synthetic):
    assert len(TRANSACTION_COLUMNS) == 394
    assert set(synthetic.columns) == set(TRANSACTION_COLUMNS) | set(IDENTITY_COLUMNS)


def test_plausible_shape(synthetic):
    assert synthetic["TransactionID"].is_unique
    assert synthetic["TransactionDT"].is_monotonic_increasing
    assert 0.02 < synthetic["isFraud"].mean() < 0.07
    assert 0.1 < synthetic["DeviceType"].notna().mean() < 0.5


def test_deterministic():
    from fraudrisk.synthetic import make_transactions

    a, b = make_transactions(n_customers=300, seed=7), make_transactions(n_customers=300, seed=7)
    assert a.equals(b)
