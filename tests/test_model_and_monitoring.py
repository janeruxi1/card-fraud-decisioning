import numpy as np
import pandas as pd

from fraudrisk import anomaly, model, monitoring


def test_at_capacity_known_answer():
    score = np.array([0.9, 0.8, 0.1, 0.2] * 25)
    y = np.array([1, 0, 0, 0] * 25)
    r = model.at_capacity(score, y, np.ones(100) * 10, capacity=0.25)
    assert r["reviewed"] == 25 and r["recall"] == 1.0 and r["precision"] == 1.0


def test_design_encodes_from_training_months_only(synthetic):
    from fraudrisk import data, features

    con = data.connect(synthetic)
    features.build(con)
    F = features.frame(con)
    raw = con.sql(f"SELECT {', '.join(model.raw_columns(with_v=False))} FROM tx").df()
    d = model.design(F, raw, with_v=False)
    tr = d.split == "train"
    for c in model.CATEGORICAL:
        cats = set(d.X[c].cat.categories)
        assert cats <= set(raw.set_index("TransactionID").loc[d.transaction_id[tr], c].dropna().astype(str))
    assert d.X["card1_freq"][tr].notna().all()


def test_psi_is_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert monitoring.psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert monitoring.psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_label_arrival_median():
    d = monitoring.label_arrival(20000, 30)
    assert abs(np.median(d) - 30) < 1.5


def test_blend_capacity_keeps_total_and_adds_new():
    primary = np.arange(100, dtype=float)
    secondary = -np.arange(100, dtype=float)
    picked = anomaly.blend_capacity(primary, secondary, 0.1, 0.3)
    assert picked.sum() == 10
    assert picked[:3].all() and picked[-7:].all()


def test_robust_distance_ranks_outliers_high():
    Z = pd.DataFrame({"a": np.r_[np.zeros(99), 50.0], "b": np.r_[np.ones(99), 1.0]})
    s = anomaly.fit_robust_distance(Z)(Z)
    assert s.argmax() == 99


def test_tuning_stays_inside_the_training_months(synthetic):
    from fraudrisk import data, features

    con = data.connect(synthetic)
    features.build(con)
    F = features.frame(con)
    raw = con.sql(f"SELECT {', '.join(model.raw_columns(with_v=False))} FROM tx").df()
    d = model.design(F, raw, with_v=False)
    seen = []
    real_fit = model.fit_lightgbm

    def spy(sub, *a, **k):
        seen.append(set(np.unique(sub.split)))
        assert sub.day[sub.split == "valid"].min() >= model.TUNE_START
        assert sub.day[np.isin(sub.split, ["train", "valid"])].max() < 120
        return real_fit(sub, *a, **k)

    model.fit_lightgbm = spy
    try:
        out = model.tune_lightgbm(d, n_trials=2, rounds=20, patience=5)
    finally:
        model.fit_lightgbm = real_fit
    assert len(seen) == 2 and out["default_value"] > 0
    assert len(out["trials"]) == 2
