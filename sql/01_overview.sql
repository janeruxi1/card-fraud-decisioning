-- Size, fraud rate, time span and identity coverage, overall and by split.
SELECT
    split,
    COUNT(*)                                   AS transactions,
    SUM(isFraud)                               AS frauds,
    AVG(isFraud)                               AS fraud_rate,
    MIN(day)                                   AS first_day,
    MAX(day)                                   AS last_day,
    AVG(CAST(has_identity AS DOUBLE))          AS identity_share,
    SUM(TransactionAmt * isFraud) / SUM(TransactionAmt) AS fraud_dollar_share
FROM base
GROUP BY ROLLUP (split)
ORDER BY split NULLS FIRST;
