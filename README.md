# Card-Not-Present Fraud: From a Score to a Decision

[![CI](https://github.com/janeruxi1/card-fraud-decisioning/actions/workflows/ci.yml/badge.svg)](https://github.com/janeruxi1/card-fraud-decisioning/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/Streamlit-Live%20demo-FF4B4B?logo=streamlit)](https://janeruxi1-card-fraud-decisioning.streamlit.app/)

**[Decision memo](reports/decision_memo.md)** · **[Live demo](https://janeruxi1-card-fraud-decisioning.streamlit.app/)** ·
**[Model card](reports/model_card.md)** · **[Serving design](reports/serving_design.md)** · **[Notebooks](notebooks/)** · Every number below is read from
`reports/metrics/`, and CI checks the key ones in this README and the memo against those records.

An online merchant has to approve, decline or send to manual review every card payment, with a review
team that can look at 1.5% of them. This project builds that decision on **590,540 real e-commerce
payments**: it finds the trap in how fraud is labelled, builds features that only use the past, trains
and calibrates a model on earlier months and judges it on later ones, prices every decision in dollars,
and measures how the model decays when fraud labels arrive weeks late.

**Data:** the IEEE-CIS Fraud Detection release (Kaggle, 2019), anonymized card-not-present transactions
from an e-commerce payments provider: 590,540 payments over 182 days, 3.50% fraud. The data belong to the
provider and are not redistributed here; see [`data/raw/README.md`](data/raw/README.md). The merchant,
its costs and its review team are a scenario ([brief](reports/scenario_brief.md)).

## The answer in short

- **Don't trust a random train/test split here:** fraud labels follow the customer, so a model can
  look good by recognizing past victims. Every result is judged on later months.
- **The model finds 35% of fraud** in the 1.5% of payments the team can review, against 2% for
  today's rules, and as much of the fraud from customers new to fraud (35%) as from repeat victims.
- **Deciding by cost, not by score, saves the most:** $3,430 per 1,000 payments against $5,299
  under today's rules, while declining fewer good customers than the rules do (3 against 8).
- **Watch it after launch:** recall ends 9 points lower two months on, the score's own drift stays
  small, and chargebacks take weeks, so the reviewers' hit rate is the alarm to watch.

## Results

| Question | Answer | Notebook |
|---|---|---|
| How is fraud labelled, and what would leak? | Labels follow the customer: after a payment labelled fraud, the customer's next payment is fraud **77%** of the time, against **0.96%** otherwise. Recognizing customers already labelled fraud finds **74%** of fraud in a random hold-out but **16%** in later months, so every result here is measured on later months. | [01](notebooks/01_audit.ipynb) |
| Can the features see the future? | No: rebuilding them with everything after three cutoffs deleted changed **0** values. 40% of training payments are a customer's first, with no history to use. | [02](notebooks/02_features.ipynb) |
| How well can fraud be ranked on unseen months? | The chosen model puts **35% of fraud** [95% interval 32–38%] and 26% of fraud dollars in the top 1.5% of scores, at 81% precision; logistic regression 19%. On fraud from customers new to fraud it finds **35%**. Calibrated: Brier 0.0214 vs 0.0336 for the base rate. | [03](notebooks/03_model.ipynb) |
| Which model, and does tuning help? | 40 Optuna trials inside the training months raised PR-AUC there from 0.586 to 0.624. Four boosting models (default or tuned, with or without the provider's `V` columns) were compared by **the cost of their decisions on the validation months**; the tuned model with `V` won by **$12** per 1,000 payments. Priced on the same payments with customers resampled, its lead over the runner-up is $12 [95% interval -$131 to $151] on the validation months and $16 [-$130 to $162] on the test months: **the two are indistinguishable**, so the choice between them should rest on something else, such as explainability. | [03](notebooks/03_model.ipynb) |
| Do anomaly detectors that need no labels help? | No better than chance here (Isolation Forest PR-AUC **0.038** at a 0.035 fraud rate). Its most "unusual" payments come from heavy legitimate customers (median **702** earlier payments). With just 111 fraud labels, a supervised model already reaches PR-AUC 0.37. | [04](notebooks/04_anomaly.ipynb) |
| What should we do with each payment? | An expected-cost policy costs **$3,430 per 1,000 payments** against $5,795 for approving everything and $4,581 for reviewing the model's top scores: the decision layer alone saves **$1,152** [95% interval $794–$1,534] with the same model and the same reviewers. | [05](notebooks/05_policy.ipynb) |
| What does the good-customer guardrail cost? | Without it the policy would decline 1.32% of good customers and cost $2,855 per 1,000 payments. By the brief's costs the guardrail costs **$575 per 1,000 payments**; it pays off only if a wrongly declined customer costs more than the brief assumes. That is a decision for finance, made visible. | [05](notebooks/05_policy.ipynb) |
| Is the guardrail met, and can we tell? | Met: the policy declined **3** good payments out of 89,214 against **8** under the rules. But the difference, -0.6 per 10,000 good payments [95% interval -1.3 to +0.1], rests on a handful of customers; a strict "no more than the rules" test can flip on one person, so finance should set a margin. | [05](notebooks/05_policy.ipynb) |
| Where is the model weak? | Product `W`, 42% of the fraud: the top 1.5% finds only **9%** of it (product `C`: 51%). Payments without device data: **19%** (with it: 50%). Above $1,000 today's rules beat the policy (on only 45 frauds), so keep the large-order rule as a floor. First payments are not a weak spot (42% against 30% for returning customers). | [05](notebooks/05_policy.ipynb) |
| Is more review capacity worth it? | Yes, with falling returns: around today's 1.5%, each extra review a day saves about **$39** after its $6 cost; reviewing 2% of payments would bring the cost to $3,234 per 1,000 payments. | [05](notebooks/05_policy.ipynb) |
| How fast does it decay, and when would we know? | Recall at capacity is **37%** in the first 15 days after training and **28%** two months later, a change of -9 points [95% interval -13 to -5], though not in a straight line (30% and 37% in between); refitting on the labels available at the time does not help. A week after a period ends only **29%** of its fraud labels are in. The score's distribution moves little (PSI ≤ 0.069), but reviewers' same-day verdicts do: the queue's hit rate fell from **88%** to **72%**. | [06](notebooks/06_monitoring.ipynb) |
| How long should the shadow trial run? | Payments are not the limit: 4 days would detect the saving against the rules. Chargebacks are: 80% of a one-week trial's fraud labels are in **67 days** after it ends, so the readout comes about 11 weeks after the start. Telling the two leading models apart would take about **10,000 days** of payments, so the trial cannot settle that. | [05](notebooks/05_policy.ipynb) |

![Cost per 1,000 payments under each policy](reports/figures/00_readme_summary.png)

*What each way of handling payments costs on the test months, in fraud losses plus review and
decline costs. The model already helps when it only picks what to review; deciding each payment
by expected cost helps most. Drawn from the recorded results by `scripts/make_readme_figure.py`.*

![Model comparison on the test months](reports/figures/03_model.png)

*Model detail. Left: precision against recall for each model. Middle: fraud (and fraud dollars)
caught when reviewing the top 1.5% of scores. Right: predicted against observed fraud rate.*

![What drives the score](reports/figures/03b_explain.png)

*Explanations, as aggregates. Left: the inputs that move the score most (SHAP). Middle: whether low
or high values of the three strongest numeric inputs push towards fraud. Right: the first reason a
reviewer would see on the payments sent to review. Most inputs are masked by the provider.*

## The decision

| Policy, test months | Cost per 1,000 payments | Fraud dollars stopped | Good customers declined |
|---|---|---|---|
| Approve everything | $5,795 | 0% | 0% |
| Rules stand-in (review / decline on fixed rules) | $5,299 | 12% | 0.009% |
| Model: review the top 1.5% of scores | $4,581 | 23% | 0% |
| **Model: expected-cost policy** | **$3,430** | **43%** | **0.003%** |
| Model: expected cost, no guardrail | $2,855 | 60% | 1.32% |

Each payment gets the action with the lowest expected cost from its calibrated fraud probability and the
scenario's costs (lost order + chargeback fee; lost margin and customer; review time). Reviews go, day by
day and within capacity, where a review saves the most dollars, which means large orders: about 15% of
orders over $500 are reviewed. Declines are allowed only above a probability threshold (0.96) chosen on
the validation months so that good customers are declined no more often than under the rules; on the
test months the policy declines 0.003% of good customers against 0.009% for the rules. Every cost
assumption halved or doubled leaves the saving against the rules stand-in positive (smallest: $1,575 per
1,000 payments).

![Policy costs and sensitivity](reports/figures/05_policy.png)

![Weak spots, review capacity and the shadow trial](reports/figures/05b_policy_checks.png)

*Left: the share of each group's fraud found in the top 1.5%. Middle: the policy's cost if the team
could review more or fewer payments a day. Right: how long after a shadow trial ends its chargebacks
are mostly in, which is when its cost can be read.*

## What makes this data hard

- **The label is partly a property of the customer.** The provider labels a customer's later payments
  fraud after a chargeback, so a random split rewards memorizing customers (74% vs 16% above). The model
  is judged on later months, and its recall is reported separately for customers new to fraud.
- **It is heavily anonymized.** Card, address and device fields are codes, and 339 `V` columns are the
  provider's engineered features with no documentation. The chosen model uses them, so its second most
  common reason code, `V258`, cannot be explained to a reviewer; the models without them cost $107–188
  more per 1,000 payments on the validation months.
- **Most customers have little history.** The median approximate customer has one payment, which caps
  what history features can do.
- **Costs and label delays are assumptions.** The data has no margins, review outcomes or chargeback
  dates; they live in one config file and every headline is shown with its sensitivity.
- **The rules stand-in is weak** (ROC AUC 0.51 on this data); the merchant's real rules are not in the
  data, so the policy is compared with approving everything and with the model's own top scores as well.

## How it was built

| Phase | What | Techniques |
|---|---|---|
| 0 | [Brief, metrics, plan](PROJECT_PLAN.md) | Decision framing: costs, review capacity, guardrails; time split fixed before modelling |
| 1 | [Audit](notebooks/01_audit.ipynb) | DuckDB SQL on the raw tables ([`sql/`](sql/)); approximate customer keys from card, address and `D1`; label-propagation and leakage measurement; missing-together column blocks |
| 2 | [Point-in-time features](notebooks/02_features.ipynb) | SQL window functions over strictly earlier transactions (velocity, amount vs history, email/device changes); a truncation test that no feature sees the future |
| 3 | [Supervised models](notebooks/03_model.ipynb) | Rules stand-in, logistic regression, LightGBM with and without the provider columns; Optuna search inside the training months; model chosen by decision cost on the validation months, with a paired customer bootstrap of the margin; PR-AUC, recall and dollar recall at capacity; customer-level bootstrap; isotonic calibration; SHAP reason codes |
| 4 | [Anomaly detection](notebooks/04_anomaly.ipynb) | Isolation Forest and a robust-distance score without labels; who the anomalies are; capacity blending; label-scarcity curve |
| 5 | [Decision policy](notebooks/05_policy.ipynb) | Expected-cost approve / review / decline; daily capacity; guardrail threshold chosen on validation; bootstrap; sensitivity to every assumption; guardrail counts with an interval; errors by segment; value of review capacity; shadow-trial length from power and label delay |
| 6 | [Decay and monitoring](notebooks/06_monitoring.ipynb) | Frozen vs refitted on available labels, with customer-resampled intervals per window; PSI drift; simulated chargeback delay; same-day signals |
| 7 | Packaging | Package, tests, CI on a synthetic table with the same schema, model card, demo on aggregates only |

## Run it

```bash
pip install -r requirements.txt
python scripts/prepare_data.py                 # needs data/raw/train_*.csv, see data/raw/README.md
python scripts/build_notebooks.py --execute    # runs 01-06 on the real data, writes reports/metrics
pytest
```

Each analysis lives in a `.py` file in `notebooks/` (the source of truth, run by CI) and a generated
`.ipynb` beside it, executed on the real data so GitHub shows every table and figure. Without the data,
`FRAUDRISK_SYNTHETIC=1 python notebooks/01_audit.py` runs on a synthetic table with the same columns and
the same labelling rule; this is what CI does, and synthetic runs never overwrite `reports/metrics/`.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py             # the demo: reads only reports/metrics
```

The same demo runs online: [janeruxi1-card-fraud-decisioning.streamlit.app](https://janeruxi1-card-fraud-decisioning.streamlit.app/).

## Repository layout

```
reports/decision_memo.md     the recommendation
reports/model_card.md        what the model is, how it was checked, where it fails
reports/metrics/*.json       every result, written by the notebooks (aggregates only)
reports/serving_design.md    how the model and policy would run live at checkout (design, not built)
notebooks/0N_*.py / .ipynb   one notebook per phase
sql/                         DuckDB SQL: audit queries and point-in-time features
src/fraudrisk/               data, synthetic table, audit, features, model, anomaly, policy, monitoring, demo
app/streamlit_app.py         the demo, including a policy calculator on aggregated cells
tests/                       known-answer tests, the leakage check, policy optimality, memo/README numbers vs records
```

## Related projects

- [A/B testing on real experiments](https://github.com/janeruxi1/ab-testing-real-experiments)
- [Churn and retention on real data](https://github.com/janeruxi1/churn-uplift-decisioning)
- [StreamFlix RAG Evaluation](https://github.com/janeruxi1/StreamFlix-RAG-evaluation)
- [Hotel Search Ranking](https://github.com/janeruxi1/Hotel-Search-Ranking)
- [Ride-hailing Driver Pay](https://github.com/janeruxi1/ridehail-driver-pay-optimization)
