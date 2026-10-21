-- Is the label a property of the customer? Within each approximate customer, in time order:
-- how often is a transaction fraud when the customer's previous transaction was fraud, vs not?
WITH seq AS (
    SELECT uid, isFraud,
           LAG(isFraud) OVER (PARTITION BY uid ORDER BY TransactionDT, TransactionID) AS prev_fraud
    FROM base WHERE uid IS NOT NULL
)
SELECT
    CASE WHEN prev_fraud IS NULL THEN 'first transaction of customer'
         WHEN prev_fraud = 1 THEN 'previous transaction fraud'
         ELSE 'previous transaction not fraud' END AS previous,
    COUNT(*) AS transactions,
    AVG(isFraud) AS fraud_rate
FROM seq GROUP BY 1 ORDER BY 1;
