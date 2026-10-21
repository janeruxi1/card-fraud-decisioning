"""Phase 3: design matrix, models and evaluation on later months.

Models are fitted on the training months; early stopping and calibration use the validation
months; the test months are scored once per model family.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config
from .features import FEATURES

CATEGORICAL = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
               "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "id_12", "id_15", "id_16", "id_28", "id_29",
               "id_31", "id_35", "id_36", "id_37", "id_38", "DeviceInfo"]
FREQ_ENCODED = ["card1", "card2", "card5", "addr1"]           # high-cardinality codes: frequency in the training months
NUMERIC = (["TransactionAmt", "card3", "dist1", "dist2"] + [f"C{i}" for i in range(1, 15)]
           + [f"D{i}" for i in range(1, 16)] + [f"id_{i:02d}" for i in range(1, 12)])
V_COLUMNS = [f"V{i}" for i in range(1, 340)]
# D columns count days since an event; subtracting them from the day turns them into the date of
# that event, which is stable for a customer (the same idea as the customer key in phase 1).
D_NORMALIZED = ["D1", "D2", "D4", "D10", "D15"]


def raw_columns(with_v: bool) -> list[str]:
    cols = ["TransactionID"] + CATEGORICAL + FREQ_ENCODED + NUMERIC
    return cols + (V_COLUMNS if with_v else [])


@dataclass
class Design:
    X: pd.DataFrame
    y: np.ndarray
    split: np.ndarray
    amount: np.ndarray
    day: np.ndarray
    transaction_id: np.ndarray


def design(features: pd.DataFrame, raw: pd.DataFrame, with_v: bool) -> Design:
    """Join phase-2 features to the raw fields and encode them using the training months only."""
    df = features.merge(raw, on="TransactionID", how="left", validate="one_to_one")
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    tr = (df["split"] == "train").to_numpy()
    X = pd.DataFrame(index=df.index)
    for c in FEATURES:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    X["log_amount"] = np.log1p(df["TransactionAmt"].astype("float32"))
    for c in NUMERIC:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in D_NORMALIZED:
        X[f"{c}_date"] = (df["day"] - pd.to_numeric(df[c], errors="coerce")).astype("float32")
    for c in FREQ_ENCODED:
        freq = df.loc[tr, c].value_counts(normalize=True)
        X[f"{c}_freq"] = df[c].map(freq).astype("float32")
    for c in CATEGORICAL:
        cats = pd.Index(df.loc[tr, c].dropna().astype(str).unique()).sort_values()
        X[c] = pd.Categorical(df[c].astype("string"), categories=cats)
    if with_v:
        X = pd.concat([X, df[V_COLUMNS].astype("float32")], axis=1)
    return Design(X=X, y=df["isFraud"].to_numpy().astype(int), split=df["split"].to_numpy(),
                  amount=df["TransactionAmt"].to_numpy(dtype=float), day=df["day"].to_numpy(),
                  transaction_id=df["TransactionID"].to_numpy())


def with_v(d: Design, con) -> Design:
    """Add the 339 ``V`` columns to a design, read as 32-bit floats straight from ``tx`` (reading
    them as 64-bit through a wide pandas frame needs several GB)."""
    cols = ", ".join(f'CAST("{c}" AS FLOAT) AS "{c}"' for c in V_COLUMNS)
    v = con.sql(f"SELECT TransactionID, {cols} FROM tx").df().set_index("TransactionID").loc[d.transaction_id]
    X = pd.concat([d.X, v.reset_index(drop=True).set_axis(d.X.index)], axis=1)
    return Design(X=X, y=d.y, split=d.split, amount=d.amount, day=d.day, transaction_id=d.transaction_id)


def rules_score(frame: pd.DataFrame) -> np.ndarray:
    """Today's rules (config.RULES): number of rules tripped, ties broken by amount."""
    import duckdb

    con = duckdb.connect()
    con.register("f", frame)
    hits = " + ".join(f"CAST(COALESCE({cond}, FALSE) AS INTEGER)" for cond in config.RULES.values())
    s = con.sql(f"SELECT ({hits}) AS hits, TransactionAmt FROM f").df()
    return s["hits"].to_numpy() + s["TransactionAmt"].rank(pct=True).to_numpy() * 0.5


def fit_lightgbm(d: Design, params: dict | None = None, rounds: int = 2000, seed: int = 0, patience: int = 50):
    import lightgbm as lgb

    p = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
         "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
         "cat_smooth": 20, "max_cat_to_onehot": 8, "max_bin": 127, "verbose": -1, "seed": seed,
         # same result on any machine: a fixed thread count and LightGBM's deterministic mode
         "num_threads": 4, "deterministic": True, "force_row_wise": True}
    p.update(params or {})
    tr, va = d.split == "train", d.split == "valid"
    dtr = lgb.Dataset(d.X[tr], d.y[tr], free_raw_data=True)
    dva = lgb.Dataset(d.X[va], d.y[va], reference=dtr)
    booster = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
    return booster


def fit_logistic(d: Design):
    """Logistic regression on the numeric columns: median-imputed, missing flags, standardized."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in d.X.columns if str(d.X[c].dtype) != "category"]
    tr = d.split == "train"
    med = d.X.loc[tr, cols].median()

    def prep(X):
        Z = X[cols].astype(float)
        flags = Z.isna().astype(float).add_suffix("_na")
        flags = flags.loc[:, [c for c in flags.columns if c in keep_flags]]
        return pd.concat([Z.fillna(med).clip(lower=lo, upper=hi, axis=1), flags], axis=1)

    lo, hi = d.X.loc[tr, cols].quantile(0.001), d.X.loc[tr, cols].quantile(0.999)
    keep_flags = [f"{c}_na" for c in cols if d.X.loc[tr, c].isna().mean() > 0.01]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    pipe.fit(prep(d.X[tr]), d.y[tr])
    return lambda X: pipe.predict_proba(prep(X))[:, 1]


def calibrate(scores_valid: np.ndarray, y_valid: np.ndarray):
    """Isotonic map from score to probability, fitted on the validation months."""
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(scores_valid, y_valid)
    return iso.predict


def at_capacity(score: np.ndarray, y: np.ndarray, amount: np.ndarray, capacity: float = config.REVIEW_CAPACITY) -> dict:
    """Recall, dollar recall and precision when the top `capacity` share of scores is reviewed."""
    k = max(1, int(round(capacity * len(score))))
    top = np.argsort(-score, kind="stable")[:k]
    fraud_dollars = (amount * y).sum()
    return {"recall": y[top].sum() / max(y.sum(), 1), "dollar_recall": (amount[top] * y[top]).sum() / max(fraud_dollars, 1e-9),
            "precision": y[top].mean(), "reviewed": k}


def evaluate(score: np.ndarray, y: np.ndarray, amount: np.ndarray, prob: np.ndarray | None = None) -> dict:
    out = {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
           "base_rate": y.mean(), **{f"{k}_at_capacity": v for k, v in at_capacity(score, y, amount).items()}}
    if prob is not None:
        out["brier"] = brier_score_loss(y, prob)
        out["mean_predicted"] = float(np.mean(prob))
    return out


def bootstrap(score, y, amount, groups, n_boot: int = 200, seed: int = 0, metrics=("pr_auc", "recall_at_capacity")):
    """95% intervals by resampling customers (labels are shared within a customer, so rows are not independent)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx_by_group = pd.Series(np.arange(len(y))).groupby(codes).apply(np.array).to_numpy()
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate(idx_by_group[pick])
        if y[idx].sum() == 0:
            continue
        e = evaluate(score[idx], y[idx], amount[idx])
        for m in metrics:
            draws[m].append(e[m])
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in draws.items()}


def calibration_table(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(prob).rank(method="first"), bins, labels=False)
    return (pd.DataFrame({"bin": q, "pred": prob, "obs": y}).groupby("bin")
            .agg(predicted=("pred", "mean"), observed=("obs", "mean"), n=("obs", "size")).reset_index())


def reason_codes(booster, X: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """Top contributions (SHAP values from LightGBM's tree path) per row: the 'why' a reviewer sees."""
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]
    names = np.array(X.columns)
    order = np.argsort(-contrib, axis=1)[:, :top]
    return pd.DataFrame({f"reason_{i + 1}": names[order[:, i]] for i in range(top)}, index=X.index)


def global_importance(booster, X: pd.DataFrame, sample: int = 20_000, seed: int = 0) -> pd.Series:
    """Mean absolute SHAP contribution per feature, on a sample of rows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    contrib = booster.predict(X.iloc[idx], pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=X.columns).sort_values(ascending=False)


def load_or_fit_scores(con, source: str, path=None) -> pd.DataFrame:
    """Phase-3 scores for every transaction. Real runs read ``data/interim/scores.parquet`` (written
    by notebook 03); synthetic runs refit a quick version: rules, gradient boosting without V,
    isotonic calibration on the validation months."""
    from . import data as _data
    from . import features as _features

    path = path or _data.PARQUET.parent / "scores.parquet"
    if source == "real":
        return pd.read_parquet(path)
    _features.build(con)
    F = _features.frame(con)
    raw = con.sql(f"SELECT {', '.join(raw_columns(with_v=False))} FROM tx").df()
    d = design(F, raw, with_v=False)
    g = fit_lightgbm(d, {"learning_rate": 0.2, "num_leaves": 15}, 150, patience=30)
    score = g.predict(d.X, num_iteration=g.best_iteration)
    va = d.split == "valid"
    prob = calibrate(score[va], d.y[va])(score)
    rf = F.merge(con.sql("SELECT TransactionID, ProductCD, TransactionAmt FROM tx").df(), on="TransactionID")
    rf = rf.set_index("TransactionID").loc[d.transaction_id].reset_index()
    uid = con.sql("SELECT TransactionID, COALESCE(uid, 'tx_' || CAST(TransactionID AS VARCHAR)) AS uid FROM base").df()
    out = pd.DataFrame({"TransactionID": d.transaction_id, "split": d.split, "day": d.day, "isFraud": d.y,
                        "amount": d.amount, "score": score, "prob": prob, "rules_score": rules_score(rf)})
    return out.merge(uid, on="TransactionID", how="left")


TUNE_START = 90   # inner split for tuning: fit on days [0, 90), score on [90, TRAIN_END)
SEARCH_SPACE = {
    "num_leaves": (15, 255, "log_int"), "min_child_samples": (20, 1000, "log_int"),
    "feature_fraction": (0.3, 1.0, "float"), "bagging_fraction": (0.5, 1.0, "float"),
    "lambda_l2": (1e-3, 50.0, "log_float"), "cat_smooth": (1.0, 100.0, "log_float"),
}
DEFAULTS = {"num_leaves": 63, "min_child_samples": 100, "feature_fraction": 0.5, "bagging_fraction": 0.8,
            "lambda_l2": 5.0, "cat_smooth": 20.0}


def tune_lightgbm(d: Design, n_trials: int = 40, seed: int = 0, learning_rate: float = 0.1,
                  rounds: int = 1000, patience: int = 30) -> dict:
    """Optuna (TPE) search over tree-shape and regularization parameters, inside the training months only.

    Trials are fitted on days [0, TUNE_START) and scored by PR-AUC on [TUNE_START, train end). The
    validation months stay untouched for early stopping, calibration and the decline threshold, and the
    test months for the final score. The current defaults are evaluated first, as trial 0, so the search
    is compared with them on the same footing.
    """
    import optuna

    tr = d.split == "train"
    inner = np.where(tr & (d.day < TUNE_START), "train", np.where(tr & (d.day >= TUNE_START), "valid", "other"))
    sub = Design(X=d.X, y=d.y, split=inner, amount=d.amount, day=d.day, transaction_id=d.transaction_id)

    def suggest(trial):
        out = {}
        for k, (lo, hi, kind) in SEARCH_SPACE.items():
            if kind == "log_int":
                out[k] = trial.suggest_int(k, lo, hi, log=True)
            elif kind == "log_float":
                out[k] = trial.suggest_float(k, lo, hi, log=True)
            else:
                out[k] = trial.suggest_float(k, lo, hi)
        return out

    def objective(trial):
        params = suggest(trial)
        booster = fit_lightgbm(sub, {**params, "learning_rate": learning_rate}, rounds, seed=seed, patience=patience)
        trial.set_user_attr("rounds", booster.best_iteration)
        return booster.best_score["valid_0"]["average_precision"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(DEFAULTS)
    study.optimize(objective, n_trials=n_trials)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return {"best_params": study.best_params, "best_value": study.best_value,
            "default_value": study.trials[0].value, "trials": trials}
