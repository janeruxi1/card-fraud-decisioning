# Card-not-present fraud: plan and design decisions

Records the decisions made before building, so the reasoning stays on record next to the code.
Numbers about the data below are from the competition's published description; phase 1 checks
every one of them and records what it finds in `reports/metrics/01_audit.json`.

## What this project is

An online merchant approves, declines or sends to manual review every card payment in real time.
Approving fraud costs the order value plus a chargeback fee; declining a good customer loses the
sale and some of the customer; a manual review costs analyst time, and the review team has a fixed
daily capacity. **The decision is what to do with each transaction, and the model is only an input
to it.**

Questions, in order:

1. What does the data really contain: how are fraud labels assigned, over what period, and what
   would leak the label into a model?
2. Which signals separate fraud, once features are built only from what was known at the time of
   the payment?
3. How well can fraud be ranked on later, unseen months, and how far does a model trained on
   labels beat rules and an unsupervised anomaly detector?
4. What approve / review / decline policy minimizes expected cost under the review team's capacity,
   and how sensitive is it to the cost assumptions?
5. How fast does the model decay, and how would we know in production, when fraud labels arrive
   weeks after the payment?

## Data

**IEEE-CIS Fraud Detection** (Kaggle competition, 2019; the data come from Vesta's e-commerce
payment platform). Real transactions, anonymized:

- `train_transaction.csv`: about 590,000 card-not-present transactions, about 3.5% labelled fraud;
  amount, product code, anonymized card fields (`card1`–`card6`), billing address codes, email
  domains, counts (`C1`–`C14`), time deltas (`D1`–`D15`), match flags (`M1`–`M9`) and 339 features
  engineered by the provider (`V1`–`V339`).
- `train_identity.csv`: device and network fields for a subset of transactions.
- `TransactionDT` is seconds from an undisclosed reference time, about six months of data. There
  are no calendar dates, card numbers or customer IDs.
- The test files carry no labels and are not used.

The data belong to the provider and are used under the competition rules: never committed,
never redistributed. CI runs everything on a synthetic table with the same schema, as in the hotel
search project; real-data results are recorded as aggregates in `reports/metrics/`.

**How labels work (phase 1 verifies this):** according to the competition organisers, a
transaction is labelled fraud when its card had a reported chargeback, and later transactions
linked to the same account, email or billing address are labelled fraud too. That makes the label
partly a property of the *customer*, not the transaction, which has two consequences designed in
from the start:

- **Validation is by time, never random.** A random split puts the same customer on both sides and
  rewards memorizing customers.
- **Customer-level leakage is measured, not assumed away.** Phase 1 reconstructs approximate
  customers from card and address fields and reports how much of the test fraud belongs to
  customers already seen as fraud in training.

## Phases

Each phase ends with an executed notebook ending in a verdict, a JSON record in `reports/metrics/`,
and tests for the code it adds. Phase N = notebook `0N_*`.

| # | Phase | Main techniques | Output |
|---|---|---|---|
| 0 | Brief, metrics, plan | Decision framing: costs, review capacity, guardrails | This file, `reports/scenario_brief.md`, `reports/metrics_framework.md` |
| 1 | Audit | DuckDB SQL on the raw tables; fraud rate over time; missingness blocks; customer reconstruction; label-propagation and leakage checks | `01_audit.json` |
| 2 | Point-in-time features | Velocity and history features with SQL window functions over strictly earlier transactions; amount relative to the customer's history; tests that no feature can see the future | Feature table and leakage tests |
| 3 | Supervised model | Rules and logistic regression baselines; LightGBM; time split with a gap; PR-AUC, recall at review capacity, dollar-weighted recall; calibration; SHAP reason codes for reviewers | `03_model.json` |
| 4 | Unsupervised anomaly detection | Isolation Forest and a robust distance score fitted without labels; what they catch that the supervised model misses; value for new fraud patterns | `04_anomaly.json` |
| 5 | Decision policy | Three-way approve / review / decline by expected cost, review capacity as a constraint, amount-dependent thresholds; sensitivity of the policy to every cost assumption | `05_policy.json`, decision memo |
| 6 | Decay and monitoring | Performance month by month; feature drift (PSI); simulated label delay: how many weeks until a decline in recall is visible; retraining cadence | `06_monitoring.json` |
| 7 | Packaging | README, model card, scoring function, Streamlit demo on aggregates only, CI on synthetic data | Repository |

## Rules

- No number reaches the README or memo unless it is read from `reports/metrics/`, written by the
  code that produced it; CI checks the key ones.
- Every feature is computed from transactions strictly before the one being scored.
- The test months are touched once per model family, after choices are fixed on validation months.
- Cost assumptions live in one config file and every headline is shown with its sensitivity.

## Scope boundaries

- No real cardholder data; the anonymized fields are not de-anonymized beyond the approximate
  customer keys needed to measure leakage.
- Chargeback disputes, 3-D Secure and step-up authentication are discussed as policy options, not
  modelled.
- The dataset's own engineered `V` features are used, but the headline model is also reported
  without them, since a reviewer cannot explain them.
