-- Fraud rate and fraud dollars by amount decile.
WITH d AS (SELECT *, NTILE(10) OVER (ORDER BY TransactionAmt) AS decile FROM base)
SELECT decile, MIN(TransactionAmt) AS min_amount, MAX(TransactionAmt) AS max_amount,
       COUNT(*) AS transactions, AVG(isFraud) AS fraud_rate, SUM(TransactionAmt * isFraud) AS fraud_dollars
FROM d GROUP BY decile ORDER BY decile;
