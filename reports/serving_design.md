# Serving design: how this would run at checkout

**Status: design, not built.** This repository trains, calibrates, prices and monitors the model offline.
This note explains what it would take to make the same decisions live, which parts of the offline work
carry over unchanged, and which parts would have to change. A working API with Docker and CI/CD is in
[Ride-hailing Driver Pay](https://github.com/janeruxi1/ridehail-driver-pay-optimization); it is not
repeated here. The model trained on the IEEE-CIS data is also not published, because the data belong to
the provider (see [`data/raw/README.md`](../data/raw/README.md)).

## 1. What has to happen, and when

| Step | When | Where it lives today |
|---|---|---|
| Build the features for one payment | At checkout, before the merchant answers | `sql/10_features.sql` (batch, over all payments) |
| Score it and convert the score to a fraud probability | At checkout | LightGBM booster + isotonic map (`model.py`) |
| Approve, decline, or hold for review | At checkout | `policy.choose` (batch, one day at a time) |
| Review held orders | Within the day, by analysts | Assumed in `config.py` (1.5% capacity) |
| Retrain, recalibrate, re-check the decline threshold | Monthly, offline | Notebooks 03 and 05 |
| Monitor | Daily and monthly | Notebook 06, `monitoring.py` |

Volume is not the hard part. The data average about 3,200 payments a day, so a review capacity of
1.5% is roughly 50 reviews a day, and one small service could score every payment. The hard parts are
the **features** (section 2) and the **review queue** (section 4). They are where the live system and
the offline analysis can quietly disagree.

## 2. Features: the main risk

Of the 15 model features, two need only the payment itself (`hour`, `amount_cents`). The other 13 need
history:

- **per customer** (the approximate key built from `card1`, `addr1` and `D1`): payment counts in the
  past 1 hour, 24 hours and 7 days and over all time, the 24-hour amount, the all-time mean and maximum
  amount, the time since the previous payment, and whether the email domain or device changed since then;
- **per card number** (`card1`): payment counts in the past 1 hour and 24 hours.

Offline, these are SQL window functions over the full table. Live, they need a small **online feature
store** keyed by customer and by card. It holds the last 7 days of `(timestamp, amount)` events plus
running all-time aggregates (count, sum, max, last timestamp, last email domain, last device). For each
payment, the service:

1. reads the state for the payment's customer and card;
2. computes the features from events **at least one second earlier**. This matches the SQL frames, which
   end at `1 PRECEDING` and so leave out same-second payments;
3. scores and decides;
4. writes the payment into the state, keyed by `TransactionID` so that a retried request is not counted
   twice.

**Parity test (the acceptance criterion).** Replay the test months through the online feature code in
time order, one payment at a time, and require it to reproduce the features built by
`sql/10_features.sql` exactly. This extends the truncation test from notebook 02, which already shows
that the SQL features never use the future. Without the parity test, the model would be scored on
features it was never trained on, and every number in the memo would describe a different system.

**An open question for the payments team: which events count?** The training data count every payment in
the release. Live, someone has to decide whether declined and abandoned attempts update the velocity
state. Counting attempts is usually the stronger fraud signal, but the training data must be built the
same way the store is filled. If the definitions differ, the model has to be retrained, not just
redeployed.

40% of training payments are a customer's first, with no history. That is not a serving failure: the
store has no state for them, and the features take the same empty-history values they had in training.

## 3. Model bundle

Deploy one versioned bundle, never its parts separately:

- the LightGBM booster and the exact list and order of its features;
- the isotonic calibration map, which was fitted on the validation months;
- the decline guardrail (fraud probability ≥ 0.94) and the cost assumptions from `config.py`;
- a hash of `sql/10_features.sql` and of the online feature code that passed the parity test.

Every decision is logged with the bundle version, the features as served, the probability, the action,
and the top three SHAP reason codes (`model.reason_codes`) that reviewers see. These logs are the
monitoring data in section 5, and they are what later chargebacks are joined to.

## 4. The decision layer: the change from the offline policy

Approving and declining work live as they do offline: for each payment, take the cheaper of the two
expected costs, and decline only above the guardrail.

**The review step does not carry over as written.** `policy.choose` looks at a whole day, ranks every
payment by how much a review would save, and reviews the top 1.5%. At checkout, the rest of the day has
not happened yet. Holding for review does not need an instant final answer, though, because the order is
held, not refused. That allows a close live equivalent:

- **Admit** a payment to the review queue when its expected saving from a review is above a threshold.
  Set the threshold from the saving distribution of recent days, so that about 1.5% of payments clear
  it.
- **Order** the queue by saving, as offline. Analysts always take the most valuable held order next.
- **At the end of the review window**, any held order not reviewed gets its default action (approve, or
  decline above the guardrail).

On a quiet day the threshold over-admits and the end-of-window rule absorbs it. On a busy day it
under-admits, which costs some value. **The gap between this rule and the daily ranking has not been
measured.** Before trusting the memo's $3,496 per 1,000 payments as the live figure, measure it by
replaying the test months through the live rule.

**Fallback.** If the feature store or the model cannot answer in time, use today's rules (the `RULES`
stand-in in `config.py`). Do not score with missing history: the model was never trained on that.

## 5. Monitoring job

Chargebacks arrive weeks late. A week after a period ends, only 29% of its fraud labels are in. So the
job is split by how fast each signal arrives. The functions already exist in `monitoring.py` and
notebook 06.

**Daily (no chargebacks needed):**
- **Review-queue hit rate.** Reviewers' verdicts arrive the same day, so this is the leading indicator.
  In notebook 06 it fell from 94% to 76% as recall decayed, while the score distribution barely moved
  (PSI ≤ 0.012).
- **Reviews within capacity**, and the share of held orders that timed out to their default action.
- **Score PSI and input PSI** against the validation months. Use PSI > 0.1 as "investigate" and leave
  out the `_date` inputs, which drift by construction. Treat a stable PSI as reassurance, not as proof
  that the model still works.
- **Feature skew.** Each night, rebuild the previous day's features with `sql/10_features.sql` from the
  logged payments, and compare them with the features as served. Any mismatch is a bug, not drift.

**Monthly, as chargebacks arrive (always reported with label completeness):**
- recall and dollar recall at capacity, overall and for customers new to fraud;
- realized cost per 1,000 payments against the rules;
- calibration by decile, and the good-customer decline rate under the guardrail. Re-solve the threshold
  if the decline rate rises above today's level.

**What the alerts lead to.** Notebook 06 found that refitting on the labels available at the time did
not stop the decay. A falling hit rate should therefore trigger a feature and data investigation, not an
automatic retrain.

## 6. Rollout

This follows the memo's shadow-mode trial:

1. **Shadow, four weeks.** Score every payment with both leading models and log what the policy would
   have done, but act on the rules only. Check feature parity and the live-versus-daily review gap
   (section 4) on real traffic.
2. **Switch** to the cheaper model's policy if its realized cost is below the rules' and good-customer
   declines stay at or below today's level.
3. **Monthly:** recalibrate, re-check the guardrail, and re-run the parity test on any change to the
   features.

## If this were built in this repository

1. An `online_features.py` with the read-compute-write logic, plus the replay parity test against
   `sql/10_features.sql`. This is most of the value and most of the risk.
2. A replay of the live review rule (section 4) on the test months, reported next to the $3,496 daily
   figure.
3. A thin FastAPI `/score` endpoint and a Dockerfile following the P5 pattern, run in CI on the
   synthetic table only.
4. A scheduled job (`scripts/monitor.py`) running the daily checks in section 5 over the decision log.
