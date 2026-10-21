"""Decision assumptions in one place (scenario values from reports/scenario_brief.md).

Every headline that depends on these is reported with its sensitivity in phase 5.
"""

REVIEW_CAPACITY = 0.015          # share of payments the review team can look at

CHARGEBACK_FEE = 15.0            # on top of the order amount, when fraud is approved
MARGIN = 0.30                    # margin lost when a good order is declined
CHURN_PROB_AFTER_DECLINE = 0.10  # chance a declined good customer does not come back
CUSTOMER_FUTURE_MARGIN = 50.0    # future margin lost if they don't
REVIEW_COST = 6.0                # analyst cost per review
REVIEW_CATCH_RATE = 0.90         # share of fraud a reviewer stops
REVIEW_GOOD_PASS_RATE = 0.99     # share of good orders a reviewer approves

# A stand-in for "today's rules", written before any model was fitted. A payment that trips
# RULES_DECLINE_HITS or more rules is declined; the others that trip a rule go to review, most
# rules first, then larger amounts, up to capacity.
RULES = {
    "product C over $100": "ProductCD = 'C' AND TransactionAmt > 100",
    "2+ payments by the customer in the past hour": "cust_tx_1h >= 2",
    "email changed and over $200": "email_changed AND TransactionAmt > 200",
    "over $1,000": "TransactionAmt > 1000",
    "new device and over $300": "device_changed AND TransactionAmt > 300",
}
RULES_DECLINE_HITS = 3

# Chargebacks arrive weeks after the payment. Scenario assumption for phase 6: delays are
# exponential with this median; reviewers' verdicts on reviewed payments arrive the same day.
LABEL_DELAY_MEDIAN_DAYS = 30
