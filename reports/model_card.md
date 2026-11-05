# Model card: card-not-present fraud score and decision policy

| | |
|---|---|
| **What it does** | Scores each payment's probability of fraud, then picks approve / review / decline by expected cost |
| **Model** | LightGBM on 15 point-in-time features, the provider's raw fields and its 339 engineered `V` columns, parameters from an Optuna search inside the training months; isotonic calibration |
| **Data** | IEEE-CIS release: 590,540 card-not-present payments over 182 days, 3.50% fraud; trained and tuned on days 0–119, chosen and calibrated on 120–149, tested on 150–181 |
| **Status** | A scenario on public data, not a deployed system |

## Inputs

Amount, product code, card and address codes, email domains, the provider's counts (`C`), time deltas
(`D`, also as the dates they imply), match flags (`M`), device fields, and features from the customer's
earlier payments (velocity, amount against their history, email or device change), and the provider's
339 engineered `V` columns. No input uses a fraud label: labels arrive weeks after a payment.

## Evaluation (test months, scored once)

- Chosen among four boosting models (default or tuned parameters, with or without the `V` columns) by the
  cost of the expected-cost policy on the validation months ($3,247 vs $3,259 for the runner-up, the
  default model with `V`). Priced on the same payments with customers resampled, the lead is $12 per
  1,000 payments (95% interval -$131 to $151) on validation and $16 (-$130 to $162) on test: the two are
  indistinguishable.
- Top 1.5% of scores: 35% of fraud (95% interval 32–38%), 26% of fraud dollars, 81% precision.
- PR-AUC 0.565 (logistic regression 0.249).
- Fraud from customers new to fraud: 35% caught at capacity; from customers already labelled fraud: 34%.
- Calibration: Brier 0.0214 (base rate alone 0.0336); predicted and observed fraud rates agree within 0.5 percentage points in the
  nine lower deciles; in the top decile 27.8% is predicted and 25.3% observed.
- Policy: $3,430 per 1,000 payments vs $4,581 for reviewing the top scores (see the decision memo).
- By segment (top 1.5% of scores): product `W` 9% of its fraud (42% of all fraud), product `C` 51%;
  without device data 19%, with it 50%; first payments 42%, returning customers 30%; orders over
  $1,000 2%.

## Known limitations

- **Labels are shared within a customer** (a chargeback labels the customer's later payments), so the
  label measures fraud on the account, not on the single payment.
- **Anonymized inputs** limit what a reviewer can be told; top reason codes are a provider count and an
  engineered column (`C1`, `V258`, `C14`) whose meaning is masked.
- **Weak on the largest product** (`W`) and on payments without device data, see above.
- **The model choice is not stable**: the top two candidates cannot be told apart on these data, and a
  rerun with different library versions picked a different one. Treat the choice as a tie.
- **Six months of data.** Decay was measured over two months only; seasonality is not covered.
- **Costs, review accuracy and chargeback delays are assumptions** from the brief, with sensitivity shown.
- **No fairness evaluation is possible**: the data has no protected attributes.

## Monitoring (if deployed)

- Daily: review-queue hit rate, reviews within capacity, score PSI, input PSI (the `_date` inputs drift
  by construction and are excluded from alerts).
- Monthly, as chargebacks arrive: recall and dollar recall at capacity, realized cost per 1,000 payments,
  calibration by decile.
