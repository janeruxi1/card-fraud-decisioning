# Notebooks: reading guide

Read them in order. Each notebook opens with a short table (question, answer, next) and ends
with a **Verdict**; every figure has a "how to read it" note under it.

| # | Notebook | The question in plain words | Main figure |
|---|---|---|---|
| 01 | [Audit](01_audit.ipynb) | What does the data contain, and how could a model cheat? | Fraud rate by week, by the customer's previous label, by amount |
| 02 | [Point-in-time features](02_features.ipynb) | Which features use only what was known when the payment arrived? | Signal of each feature on its own |
| 03 | [Supervised models](03_model.ipynb) | Which model should rank payments for review? | Precision–recall, fraud caught at capacity, calibration |
| 04 | [Anomaly detection](04_anomaly.ipynb) | Does label-free anomaly detection catch what the model misses? | Labels vs no labels; sharing the review queue |
| 05 | [The decision](05_policy.ipynb) | Approve, review or decline, by expected cost | Cost per 1,000 payments; sensitivity |
| 06 | [Monitoring](06_monitoring.ipynb) | How fast does the model decay, and what do late labels hide? | Recall by window, drift, label arrival |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it (the `.py` is the
source); see the main [README](../README.md) for the data and the build command.
