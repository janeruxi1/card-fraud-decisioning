-- Volume and fraud rate by week: is the fraud rate stable enough for a time split?
SELECT day // 7 AS week, COUNT(*) AS transactions, AVG(isFraud) AS fraud_rate
FROM base GROUP BY 1 ORDER BY 1;
