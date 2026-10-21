-- What a model could get by memorizing customers. For each evaluation split, flag a transaction
-- when its customer already had a fraud label in the training split, and measure how much of the
-- split's fraud that rule finds. Under a time split this is what a deployed model could know
-- (if labels arrived instantly); under a random split it is pure leakage of the future.
WITH fraud_customers AS (
    SELECT DISTINCT uid FROM base WHERE split = 'train' AND isFraud = 1 AND uid IS NOT NULL
)
SELECT
    b.split,
    SUM(b.isFraud)                                                        AS frauds,
    AVG(CASE WHEN f.uid IS NOT NULL THEN 1.0 ELSE 0 END)                  AS flagged_share,
    SUM(CASE WHEN f.uid IS NOT NULL THEN b.isFraud ELSE 0 END) / NULLIF(SUM(b.isFraud), 0) AS recall,
    SUM(CASE WHEN f.uid IS NOT NULL THEN b.isFraud ELSE 0 END)
        / NULLIF(SUM(CASE WHEN f.uid IS NOT NULL THEN 1 ELSE 0 END), 0)  AS precision
FROM base b LEFT JOIN fraud_customers f USING (uid)
WHERE b.split IN ('valid', 'test')
GROUP BY b.split ORDER BY b.split;
