-- Fraud rate by product code and by identity coverage.
SELECT 'product' AS dimension, ProductCD AS level, COUNT(*) AS transactions, AVG(isFraud) AS fraud_rate
FROM base GROUP BY ProductCD
UNION ALL
SELECT 'identity', CASE WHEN has_identity THEN 'device data' ELSE 'no device data' END, COUNT(*), AVG(isFraud)
FROM base GROUP BY has_identity
ORDER BY dimension, transactions DESC;
