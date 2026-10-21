-- The same memorization rule under a random 80/20 split of transactions (hash of the ID):
-- customers' fraud labels from the future sit in the training part.
WITH r AS (SELECT *, hash(TransactionID) % 5 = 0 AS held_out FROM base),
fraud_customers AS (SELECT DISTINCT uid FROM r WHERE NOT held_out AND isFraud = 1 AND uid IS NOT NULL)
SELECT
    SUM(r.isFraud) AS frauds,
    SUM(CASE WHEN f.uid IS NOT NULL THEN r.isFraud ELSE 0 END) / NULLIF(SUM(r.isFraud), 0) AS recall,
    SUM(CASE WHEN f.uid IS NOT NULL THEN r.isFraud ELSE 0 END)
        / NULLIF(SUM(CASE WHEN f.uid IS NOT NULL THEN 1 ELSE 0 END), 0) AS precision
FROM r LEFT JOIN fraud_customers f USING (uid)
WHERE r.held_out;
