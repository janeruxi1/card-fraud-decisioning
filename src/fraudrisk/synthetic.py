"""A synthetic stand-in for the IEEE-CIS training table: same columns, same labelling rule.

It exists so CI and readers without the competition data can run every notebook. Nothing in
``reports/`` comes from it. What it reproduces on purpose:

- customers who reappear over the period, identifiable (approximately) by card, address and the
  day the card was first seen (``day - D1``);
- the organisers' labelling rule: once a customer is defrauded, their later transactions are
  labelled fraud too, so the label is partly a property of the customer;
- fraud that looks different (larger amounts, product C, night hours, new devices, bursts), with
  the pattern shifting in the last two months so that decay can be measured;
- blocks of ``V`` columns that are missing together, and an identity table for a subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DAY

N_V = 339
ID_NUMERIC = [f"id_{i:02d}" for i in range(1, 12)]
ID_CATEGORICAL = [f"id_{i:02d}" for i in range(12, 39)]
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "outlook.com",
          "comcast.net", "icloud.com", "protonmail.com", "mail.com"]
GOOD_EMAIL_P = np.array([0.40, 0.18, 0.08, 0.07, 0.07, 0.05, 0.05, 0.06, 0.02, 0.02])
FRAUD_EMAIL_P = np.array([0.30, 0.08, 0.12, 0.08, 0.02, 0.10, 0.02, 0.06, 0.12, 0.10])
PRODUCTS = ["W", "C", "R", "H", "S"]


def _norm(p):
    p = np.asarray(p, float)
    return p / p.sum()


def make_transactions(n_customers: int = 4_000, days: int = 182, seed: int = 0,
                      compromised_share: float = 0.028) -> pd.DataFrame:
    """Return a table with the columns of train_transaction joined to train_identity."""
    rng = np.random.default_rng(seed)

    # --- customers --------------------------------------------------------------------------
    c = pd.DataFrame({
        "cust": np.arange(n_customers),
        "card1": rng.integers(1000, 18397, n_customers),
        "card2": rng.choice(np.r_[np.arange(100, 600, 7), np.nan], n_customers).astype(float),
        "card3": rng.choice([150.0, 185.0, 144.0], n_customers, p=[0.88, 0.09, 0.03]),
        "card4": rng.choice(["visa", "mastercard", "american express", "discover"], n_customers,
                            p=[0.65, 0.30, 0.03, 0.02]),
        "card5": rng.choice([226.0, 224.0, 166.0, 102.0, 117.0, 138.0], n_customers),
        "card6": rng.choice(["debit", "credit"], n_customers, p=[0.75, 0.25]),
        "addr1": np.where(rng.random(n_customers) < 0.1, np.nan, rng.integers(100, 541, n_customers)),
        "addr2": np.where(rng.random(n_customers) < 0.1, np.nan, 87.0),
        "email": rng.choice(EMAILS, n_customers, p=_norm(GOOD_EMAIL_P)),
        "first_day": rng.integers(-400, days - 5, n_customers),
        "rate": rng.gamma(1.2, 6.0, n_customers) + 1.0,          # transactions over the period
        "amount_scale": np.exp(rng.normal(3.9, 0.7, n_customers)),
        "has_device": rng.random(n_customers) < 0.25,
    })
    c["start"] = np.maximum(c["first_day"], 0)
    c["active_days"] = days - c["start"]

    # --- legitimate activity -----------------------------------------------------------------
    n_tx = rng.poisson(c["rate"] * c["active_days"] / days)
    cust = np.repeat(c["cust"].to_numpy(), n_tx)
    start = np.repeat(c["start"].to_numpy(), n_tx)
    span = np.repeat(c["active_days"].to_numpy(), n_tx)
    t = (start + rng.random(len(cust)) * span) * DAY
    legit = pd.DataFrame({"cust": cust, "t": t, "fraud_event": False})

    # --- fraud: compromised customers, then bursts by the fraudster ----------------------------
    comp = c.sample(frac=compromised_share, random_state=seed)
    f_start = (comp["start"] + rng.random(len(comp)) * comp["active_days"] * 0.8).to_numpy() * DAY
    burst_n = rng.integers(2, 9, len(comp))
    f_cust = np.repeat(comp["cust"].to_numpy(), burst_n)
    f_t = np.repeat(f_start, burst_n) + rng.exponential(6 * 3600, burst_n.sum()).cumsum() % (5 * DAY)
    fraud = pd.DataFrame({"cust": f_cust, "t": np.minimum(f_t, days * DAY - 1), "fraud_event": True})

    # a few one-off frauds on customers who are never compromised (no propagation)
    one_off = c.sample(n=max(1, n_customers // 150), random_state=seed + 1)
    one = pd.DataFrame({"cust": one_off["cust"].to_numpy(),
                        "t": ((one_off["start"] + rng.random(len(one_off)) * one_off["active_days"]) * DAY).to_numpy(),
                        "fraud_event": True})

    tx = pd.concat([legit, fraud, one], ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    tx = tx.merge(c, on="cust", how="left", sort=False)
    tx = tx.sort_values("t", kind="stable").reset_index(drop=True)
    n = len(tx)
    day = np.floor(tx["t"] / DAY).astype(int)
    late = (day >= days - 60).to_numpy()
    fe = tx["fraud_event"].to_numpy()

    # --- labels: the organisers' rule. After a customer's first fraud, everything is fraud. ----
    first_fraud = tx.loc[fe].groupby("cust")["t"].min()
    ff = tx["cust"].map(first_fraud).to_numpy()
    compromised = tx["cust"].isin(comp["cust"]).to_numpy()
    propagated = compromised & ~np.isnan(ff) & (tx["t"].to_numpy() >= ff)
    is_fraud = (fe | propagated).astype(np.int8)
    # label noise: a few chargebacks never reported, a few good orders disputed
    flip = rng.random(n)
    is_fraud = np.where((is_fraud == 1) & (flip < 0.03), 0, is_fraud)
    is_fraud = np.where((is_fraud == 0) & (flip > 0.9985), 1, is_fraud).astype(np.int8)

    # --- transaction fields -------------------------------------------------------------------
    hour = np.where(fe, rng.choice(24, n, p=_norm(np.r_[np.full(6, 3.0), np.full(18, 1.0)])),
                    rng.choice(24, n, p=_norm(np.r_[np.full(6, 0.4), np.full(18, 1.0)])))
    secs = day * DAY + hour * 3600 + rng.integers(0, 3600, n)
    amount_mult = np.where(fe, np.where(late, 0.6, 2.2) * np.exp(rng.normal(0.3, 0.8, n)), np.exp(rng.normal(0, 0.6, n)))
    amount = np.round(tx["amount_scale"].to_numpy() * amount_mult, 2).clip(0.25, 30_000)
    product = np.where(fe, rng.choice(PRODUCTS, n, p=[0.35, 0.45, 0.05, 0.08, 0.07]),
                       rng.choice(PRODUCTS, n, p=[0.76, 0.10, 0.07, 0.05, 0.02]))
    p_email = np.where(fe & (rng.random(n) < 0.5), rng.choice(EMAILS, n, p=_norm(FRAUD_EMAIL_P)), tx["email"])
    p_email = np.where(rng.random(n) < 0.15, None, p_email)
    r_email = np.where((product == "C") | (rng.random(n) < 0.1), p_email, None)

    out = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n),
        "isFraud": is_fraud,
        "TransactionDT": (DAY + secs).astype(np.int64),
        "TransactionAmt": amount,
        "ProductCD": product,
    })
    for col in ("card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"):
        out[col] = tx[col].to_numpy()
    out["dist1"] = np.where(rng.random(n) < 0.6, np.nan, np.round(rng.exponential(np.where(fe, 300, 60), n)))
    out["dist2"] = np.where(rng.random(n) < 0.93, np.nan, np.round(rng.exponential(200, n)))
    out["P_emaildomain"] = p_email
    out["R_emaildomain"] = r_email

    # counts per customer up to now (C1-C14), and time deltas (D1-D15)
    order = np.lexsort((secs, tx["cust"].to_numpy()))
    seq = np.empty(n, int)
    seq[order] = pd.Series(tx["cust"].to_numpy()[order]).groupby(tx["cust"].to_numpy()[order]).cumcount().to_numpy()
    for i in range(1, 15):
        base = seq + 1 if i in (1, 2, 11, 13, 14) else rng.poisson(1.0, n)
        out[f"C{i}"] = (base + np.where(fe & (i in (1, 2, 4, 8, 10)), rng.poisson(3, n), 0)).astype(float)
    out["D1"] = (day - tx["first_day"].to_numpy()).clip(0).astype(float)
    out.loc[rng.random(n) < 0.02, "D1"] = np.nan
    for i in range(2, 16):
        miss = rng.random(n) < (0.45 if i in (4, 10, 15) else 0.85)
        out[f"D{i}"] = np.where(miss, np.nan, np.round(rng.exponential(np.where(fe, 15, 90), n)))
    for i in range(1, 10):
        p_t = 0.55 if i not in (4,) else 0.3
        val = np.where(rng.random(n) < np.where(fe & (i in (4, 5, 6)), p_t - 0.25, p_t), "T", "F")
        out[f"M{i}"] = np.where(rng.random(n) < 0.45, None, val)

    # V1-V339: blocks of columns that are missing together; a few carry signal
    v = {}
    block_starts = list(range(1, N_V + 1, 30))
    for b0 in block_starts:
        miss = rng.random(n) < rng.uniform(0.0, 0.8)
        for j in range(b0, min(b0 + 30, N_V + 1)):
            signal = (j % 37 == 0) | (j in (258, 294, 317))
            val = rng.poisson(np.where(fe & signal, 3.0, 0.5), n).astype(np.float32)
            v[f"V{j}"] = np.where(miss, np.nan, val).astype(np.float32)
    out = pd.concat([out, pd.DataFrame(v)], axis=1)

    # identity table: a subset of transactions (more often for fraud, and for customers with devices)
    has_id = (tx["has_device"].to_numpy() | (fe & (rng.random(n) < 0.6))) & (rng.random(n) < 0.95)
    for col in ID_NUMERIC:
        out[col] = np.where(has_id, np.round(rng.normal(np.where(fe, -5, 0), 10, n)), np.nan)
    for col in ID_CATEGORICAL:
        out[col] = np.where(has_id, rng.choice(["Found", "NotFound", "New"], n, p=[0.6, 0.3, 0.1]), None)
    new_device = fe & ~late
    out["id_31"] = np.where(has_id, np.where(new_device, "chrome 63.0", rng.choice(["chrome 70.0", "safari 12.0", "edge 17.0"], n)), None)
    out["DeviceType"] = np.where(has_id, rng.choice(["mobile", "desktop"], n, p=[0.4, 0.6]), None)
    out["DeviceInfo"] = np.where(has_id, rng.choice(["Windows", "iOS Device", "MacOS", "SM-G960U"], n), None)
    return out.sort_values("TransactionDT", kind="stable").reset_index(drop=True)


IDENTITY_COLS_ALL = ID_NUMERIC + ID_CATEGORICAL + ["DeviceType", "DeviceInfo"]
