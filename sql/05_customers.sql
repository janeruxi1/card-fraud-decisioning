-- Customers: how many, how many transactions each, and how labels are shared within them.
WITH c AS (
    SELECT uid, COUNT(*) AS n, SUM(isFraud) AS frauds
    FROM base WHERE uid IS NOT NULL GROUP BY uid
)
SELECT
    COUNT(*)                                                     AS customers,
    AVG(n)                                                       AS transactions_per_customer,
    MEDIAN(n)                                                    AS median_transactions,
    AVG(CASE WHEN frauds > 0 THEN 1.0 ELSE 0 END)                AS customers_with_fraud,
    SUM(CASE WHEN frauds = n AND n > 1 THEN n ELSE 0 END)
        / NULLIF(SUM(CASE WHEN frauds > 0 AND n > 1 THEN n ELSE 0 END), 0) AS share_in_all_fraud_customers,
    (SELECT AVG(CASE WHEN uid IS NULL THEN 1.0 ELSE 0 END) FROM base) AS no_uid_share
FROM c;
