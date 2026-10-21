# Business brief: fraud screening for card-not-present payments

**From:** Head of Payments, an online merchant (scenario)
**To:** Data Science (Risk)
**Status:** phase 0, written before any modelling

The transactions are real (an anonymized e-commerce payment dataset, see `PROJECT_PLAN.md`); the
merchant, its costs and its review team are the scenario.

## Background

Every card payment is approved, declined or held for a manual review before approval. Today a
small set of rules sends payments to review and the rest are approved.

- **Fraud that gets through** costs the order value, a chargeback fee, and goods that are gone.
- **A good customer declined** costs the sale's margin, and some of those customers do not come back.
- **A manual review** costs analyst time and delays the order; the team can review a fixed number
  of payments a day, and payments waiting beyond that are approved unreviewed.

## The question

> **For each payment, should we approve, review or decline, so that total expected cost (fraud
> losses + lost good sales + review time) is as low as possible, within the review team's capacity?**

And, before trusting any answer:

- How much better than today's rules is a model, measured on later months it never saw?
- Do anomaly detectors that need no fraud labels add anything, especially for new fraud patterns?
- How quickly does performance decay, and how would we notice when fraud labels arrive weeks late?

## Cost assumptions (scenario; every headline is shown with its sensitivity)

| Item | Assumption |
|---|---|
| Fraud approved | order amount + $15 chargeback fee |
| Good payment declined | 30% margin on the order + 10% chance of losing a customer worth $50 in future margin |
| Manual review | $6 per review; reviewers catch 90% of fraud they see and approve 99% of good orders |
| Review capacity | 1.5% of daily payments |

## What would change the recommendation

- If the model's lift over the rules on later months is small, keep the rules and spend on reviews.
- If the policy's savings depend mostly on one cost assumption, that assumption needs a finance
  owner before anything ships.
- If performance decays within weeks, retraining cadence matters more than model choice.
