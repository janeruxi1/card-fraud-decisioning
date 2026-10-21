# Metric framework

Unit: the **transaction**. Validation is by time: later months are scored by models trained only
on earlier ones.

## Primary

| Metric | Definition | Why |
|---|---|---|
| Expected cost per 1,000 payments | Fraud approved + good payments declined + reviews, under the brief's cost assumptions | The decision's objective |

## Model quality (secondary)

| Metric | Definition | Why |
|---|---|---|
| PR-AUC | Area under precision–recall | With ~3.5% fraud, ROC-AUC flatters every model |
| Recall at review capacity | Share of fraud in the top 1.5% of scores | What the review team can actually act on |
| Dollar recall at capacity | Share of fraud *dollars* in the top 1.5% | Large frauds matter more than small ones |
| Calibration | Predicted vs observed fraud rate by score decile; Brier score | Expected-cost decisions need probabilities, not ranks |

## Guardrails

| Guardrail | Threshold |
|---|---|
| Good-customer decline rate | Not above today's rules |
| Reviews | Within capacity every day of the test months |
| Fraud approved, in dollars | Not above today's rules |

## Rules

- Features use only transactions strictly before the one scored.
- Thresholds and policies are chosen on validation months and reported once on test months.
- Intervals: bootstrap over approximate customers, not over rows, because labels are shared within
  a customer.
