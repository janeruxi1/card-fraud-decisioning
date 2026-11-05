# Decision memo: fraud screening for card-not-present payments

**To:** Head of Payments · **From:** Xi Ru, Data Science (Risk) · **Status:** recommendation, pending a shadow-mode trial
**Scenario on real, anonymized e-commerce payments (IEEE-CIS release); costs and review team are assumptions (see the brief).**

## Recommendation

Replace the fixed screening rules with a **calibrated fraud model and an expected-cost policy**:
approve, review or decline each payment by which costs least, send reviews to the payments where a
review saves the most dollars, and decline only when the fraud probability is at least 0.96. Keep
today's large-order rule as a floor above $1,000. Run it in **shadow mode for one to two weeks of
traffic**, read the result once most chargebacks are in (about ten weeks later), then switch.

## Why

1. **It costs less, on months the model never saw.** On the test months the policy costs **$3,430 per
   1,000 payments**, against $5,795 for approving everything and $5,299 under the rules stand-in.
2. **The decision layer earns its keep on its own.** With the same model and the same 1.5% review
   capacity, simply reviewing the highest scores costs $4,581. Pricing each decision saves a further
   **$1,152 per 1,000 payments** (95% interval $794–$1,534), by sending large orders to review
   (about 15% of orders over $500) instead of the most suspicious small ones.
3. **It recognizes new fraud, not only known fraudsters.** The model finds 35% of fraud from customers
   with no fraud history in the top 1.5% of scores, against 2% for the rules stand-in.
4. **It declines fewer good customers than the rules.** 3 good payments out of 89,214 on the test
   months, against 8 under the rules.
5. **The saving survives the assumptions.** Halving or doubling any cost, or the review capacity, leaves
   the saving positive (smallest: $1,575 per 1,000 payments against the rules stand-in).

## The decisions we need from you

**How much is a wrongly declined good customer worth?** Declines are held to today's good-customer
decline rate (about 0.01%). Allowing the policy to decline wherever expected cost says so would cut cost
to **$2,855 per 1,000 payments** (a further **$575**), at the price of declining **1.32%** of good
customers. By the brief's assumptions that trade is worth it; it is not if a declined customer costs
much more than 30% of the order plus a 10% chance of losing $50 of future margin. Finance owns that
number, and the policy should be re-solved when it changes.

**What margin should the guardrail allow?** At today's volumes good-customer declines are counted in
single digits a month, so "no more than the rules" can be decided by one customer. A stated margin
(for example, at most one extra decline per 10,000 good payments) makes the monthly check meaningful.

**Is more review capacity worth buying?** Around today's 1.5%, each extra review a day saves about $39
after its $6 cost, falling as capacity grows (about $30 at 3%).

## Risks

- **Model choice is a coin toss.** Four candidate models were compared by the cost of their decisions on
  the validation months; the chosen one (tuned, with the provider's `V` columns) led the runner-up by $12
  per 1,000 payments, 95% interval -$131 to $151, and by $16 on the test months. If reason codes matter
  to the review team, the models without the `V` columns cost $107–188 more on the validation months.
- **Weak spots.** Product `W` carries 42% of the fraud but the top 1.5% of scores finds only 9% of it,
  and payments without device data 19%. Above $1,000 the rules did better than the policy, on 45 frauds.
- **Decay.** Recall at capacity was 37% just after training and 28% two months later (change -9 points,
  95% interval -13 to -5), with a dip and recovery in between: four 15-day windows cannot tell steady
  decay from fraud arriving in waves. Refitting on the labels available at the time did not help, so expect to
  re-engineer features, not just retrain.
- **Late labels.** Chargebacks arrive weeks late: a week after a period ends only 29% of its fraud
  labels are in. Monitor the review queue's hit rate daily; it fell from 88% to 72% as performance
  declined, while the score distribution moved little.
- **Anomaly detectors do not substitute for labels here.** The unusual payments are heavy legitimate
  customers; keep them for new products with no labels yet, not in the review queue.

## Shadow-mode trial

- **Why not four weeks of traffic:** with both policies deciding the same payments, 4 days of traffic
  would detect the measured saving against the rules (16 days for half of it), at 80% power. What takes
  time is the chargebacks: with a 30-day median delay, 80% of a one-week trial's fraud labels are in
  67 days after it ends.
- **Run** the policy alongside the rules for one to two weeks, scoring with both leading models; act on
  the rules only.
- **Measure daily:** reviews within capacity; the review-queue hit rate on payments the policy would
  review; score PSI. **At four and eight weeks:** realized cost on the labels in so far, with their
  completeness; **final read about ten weeks after the trial ends**.
- **Switch** to the policy if its realized cost is lower than the rules' with good-customer declines
  within the agreed margin. The trial cannot tell the two leading models apart (that would take about
  10,000 days of traffic); choose between them on explainability.

*Analysis: notebooks 01–06 in this repository; the key numbers above are checked against
`reports/metrics/` in CI.*
