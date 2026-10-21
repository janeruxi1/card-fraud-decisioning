# Model card: card-not-present fraud score and decision policy

| | |
|---|---|
| **What it does** | Scores each payment's probability of fraud, then picks approve / review / decline by expected cost |
| **Model** | LightGBM on 15 point-in-time features, the provider's raw fields and its 339 `V` columns; isotonic calibration |
| **Data** | IEEE-CIS release: 590,540 card-not-present payments over 182 days, 3.50% fraud; trained on days 0–119, tuned on 120–149, tested on 150–181 |
| **Status** | A scenario on public data, not a deployed system |

## Inputs

Amount, product code, card and address codes, email domains, the provider's counts (`C`), time deltas
(`D`, also as the dates they imply), match flags (`M`), device fields, the `V` columns, and features from
the customer's earlier payments (velocity, amount against their history, email or device change). No
input uses a fraud label: labels arrive weeks after a payment.

## Evaluation (test months, scored once)

- Top 1.5% of scores: 34% of fraud (95% interval 32–37%), 26% of fraud dollars, 79% precision.
- PR-AUC 0.552 (logistic regression 0.249; without the `V` columns 0.526).
- Fraud from customers new to fraud: 35% caught at capacity; from customers already labelled fraud: 32%.
- Calibration: Brier 0.0219 (base rate alone 0.0336); predicted and observed fraud rates agree by decile.
- Policy: $3,869 per 1,000 payments vs $4,559 for reviewing the top scores (see the decision memo).

## Known limitations

- **Labels are shared within a customer** (a chargeback labels the customer's later payments), so the
  label measures fraud on the account, not on the single payment.
- **Anonymized inputs** limit what a reviewer can be told; top reason codes are provider counts.
- **Six months of data.** Decay was measured over two months only; seasonality is not covered.
- **Costs, review accuracy and chargeback delays are assumptions** from the brief, with sensitivity shown.
- **No fairness evaluation is possible**: the data has no protected attributes.

## Monitoring (if deployed)

- Daily: review-queue hit rate, reviews within capacity, score PSI, input PSI (the `_date` inputs drift
  by construction and are excluded from alerts).
- Monthly, as chargebacks arrive: recall and dollar recall at capacity, realized cost per 1,000 payments,
  calibration by decile.
