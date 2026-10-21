# Decision memo: fraud screening for card-not-present payments

**To:** Head of Payments · **From:** Xi Ru, Data Science (Risk) · **Status:** recommendation, pending a shadow-mode trial
**Scenario on real, anonymized e-commerce payments (IEEE-CIS release); costs and review team are assumptions (see the brief).**

## Recommendation

Replace the fixed screening rules with a **calibrated fraud model and an expected-cost policy**:
approve, review or decline each payment by which costs least, send reviews to the payments where a
review saves the most dollars, and decline only when the fraud probability is at least 0.94. Run it in
**shadow mode for four weeks** first, then switch.

## Why

1. **It costs less, on months the model never saw.** On the test months the policy costs **$3,869 per
   1,000 payments**, against $5,795 for approving everything and $5,299 under the rules stand-in.
2. **The decision layer earns its keep on its own.** With the same model and the same 1.5% review
   capacity, simply reviewing the highest scores costs $4,559. Pricing each decision saves a further
   **$690 per 1,000 payments** (95% interval $318–$1,116), by sending large orders to review
   (22% of orders over $1,000) instead of the most suspicious small ones.
3. **It recognizes fraud, not just known fraudsters.** The model finds 35% of fraud from customers with
   no fraud history in the top 1.5% of scores, against 2% for the rules stand-in.
4. **The saving survives the assumptions.** Halving or doubling any cost, or the review capacity, leaves
   the saving positive (smallest: $1,035 per 1,000 payments against the rules stand-in).

## The decision we need from you

**How much is a wrongly declined good customer worth?** Declines are held to today's good-customer
decline rate (0.01%). Allowing the policy to decline wherever expected cost says so would cut cost to
**$2,811 per 1,000 payments** (a further **$1,058**), at the price of declining **1.25%** of good
customers. By the brief's assumptions that trade is worth it; it is not if a declined customer costs
much more than 30% of the order plus a 10% chance of losing $50 of future margin. Finance owns that
number, and the policy should be re-solved when it changes.

## Risks

- **Decay.** Recall at capacity fell from 39% to 28% in the two months after training. Refitting on
  the labels available at the time did not help, so expect to re-engineer features, not just retrain.
- **Late labels.** Chargebacks arrive weeks late: a week after a period ends only 29% of its fraud
  labels are in. Monitor the review queue's hit rate daily; it fell from 92% to 72% as performance
  declined, while the score distribution barely moved.
- **Anomaly detectors do not substitute for labels here.** The unusual payments are heavy legitimate
  customers; keep them for new products with no labels yet, not in the review queue.
- **Anonymized inputs.** Reason codes point to provider fields (counts such as `C1`, `C13`) that
  reviewers cannot interpret directly; reviewers need a translation of the top reasons.

## Shadow-mode trial

- **Run** the policy alongside the rules for four weeks; act on the rules only.
- **Measure daily:** reviews within capacity; the policy's review-queue hit rate on the payments both
  would review; score PSI. **At four and eight weeks:** realized cost per 1,000 payments for both, as
  chargebacks arrive.
- **Switch** if the policy's realized cost is lower with good-customer declines at or below today's.

*Analysis: notebooks 01–06 in this repository; the key numbers above are checked against
`reports/metrics/` in CI.*
