# Model card: card-not-present fraud score and decision policy

| | |
|---|---|
| **What it does** | Scores each payment's probability of fraud, then picks approve / review / decline by expected cost |
| **Model** | LightGBM on 15 point-in-time features and the provider's raw fields (not its 339 `V` columns), parameters from an Optuna search inside the training months; isotonic calibration |
| **Data** | IEEE-CIS release: 590,540 card-not-present payments over 182 days, 3.50% fraud; trained on days 0–119, tuned on 120–149, tested on 150–181 |
| **Status** | A scenario on public data, not a deployed system |

## Inputs

Amount, product code, card and address codes, email domains, the provider's counts (`C`), time deltas
(`D`, also as the dates they imply), match flags (`M`), device fields, the `V` columns, and features from
the customer's earlier payments (velocity, amount against their history, email or device change). No
input uses a fraud label: labels arrive weeks after a payment.

## Evaluation (test months, scored once)

- Chosen among four boosting models (default or tuned parameters, with or without the `V` columns) by the
  cost of the expected-cost policy on the validation months ($4,138 vs $4,184 for the runner-up). On the
  test months the runner-up (defaults, with `V`) was cheaper, $3,869 vs $4,056: the choice is close.
- Top 1.5% of scores: 35% of fraud (95% interval 32–38%), 27% of fraud dollars, 82% precision.
- PR-AUC 0.562 (logistic regression 0.249).
- Fraud from customers new to fraud: 32% caught at capacity; from customers already labelled fraud: 50%.
- Calibration: Brier 0.0213 (base rate alone 0.0336); predicted and observed fraud rates agree closely by decile.
- Policy: $4,056 per 1,000 payments vs $4,566 for reviewing the top scores (see the decision memo).

## Known limitations

- **Labels are shared within a customer** (a chargeback labels the customer's later payments), so the
  label measures fraud on the account, not on the single payment.
- **Anonymized inputs** limit what a reviewer can be told; top reason codes are provider counts (`C1`, `C14`).
- **Leans on returning fraudsters**: recall on customers already labelled fraud (50%) is well above recall on
  new fraud (32%).
- **Six months of data.** Decay was measured over two months only; seasonality is not covered.
- **Costs, review accuracy and chargeback delays are assumptions** from the brief, with sensitivity shown.
- **No fairness evaluation is possible**: the data has no protected attributes.

## Monitoring (if deployed)

- Daily: review-queue hit rate, reviews within capacity, score PSI, input PSI (the `_date` inputs drift
  by construction and are excluded from alerts).
- Monthly, as chargebacks arrive: recall and dollar recall at capacity, realized cost per 1,000 payments,
  calibration by decile.
