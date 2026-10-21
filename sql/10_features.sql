-- Point-in-time features: every value uses only transactions STRICTLY EARLIER than the one being
-- scored (window frames end at "1 PRECEDING" second, so same-second transactions are excluded too).
-- No feature uses a fraud label: labels arrive weeks later (chargebacks), so a deployed model
-- could not know them at payment time. Phase 6 treats label delay separately.
CREATE OR REPLACE TABLE features AS
WITH b AS (
    SELECT
        base.TransactionID, base.isFraud, base.TransactionDT, base.day, base.split, base.uid,
        base.TransactionAmt, tx.card1, tx.P_emaildomain, tx.DeviceInfo, tx.addr1, tx.dist1,
        COALESCE(base.uid, 'tx_' || CAST(base.TransactionID AS VARCHAR)) AS cust
    FROM base JOIN tx USING (TransactionID)
)
SELECT
    TransactionID, isFraud, TransactionDT, day, split,
    uid IS NOT NULL                                                         AS has_uid,
    -- time of day, relative to the undisclosed start (the same offset for every transaction)
    CAST((TransactionDT // 3600) % 24 AS INTEGER)                           AS hour,
    TransactionAmt - FLOOR(TransactionAmt)                                  AS amount_cents,
    -- customer velocity
    COUNT(*) OVER w_all                                                     AS cust_prior_tx,
    COUNT(*) OVER w_1h                                                      AS cust_tx_1h,
    COUNT(*) OVER w_24h                                                     AS cust_tx_24h,
    COUNT(*) OVER w_7d                                                      AS cust_tx_7d,
    COALESCE(SUM(TransactionAmt) OVER w_24h, 0)                             AS cust_amt_24h,
    (TransactionDT - MAX(TransactionDT) OVER w_all) / 3600.0                AS hours_since_prev,
    -- amount against the customer's own history
    AVG(TransactionAmt) OVER w_all                                          AS cust_mean_amt,
    TransactionAmt / NULLIF(AVG(TransactionAmt) OVER w_all, 0)              AS amt_vs_cust_mean,
    TransactionAmt / NULLIF(MAX(TransactionAmt) OVER w_all, 0)              AS amt_vs_cust_max,
    -- changes from the customer's previous transaction
    CASE WHEN LAG(TransactionID) OVER w_seq IS NULL THEN NULL
         ELSE COALESCE(P_emaildomain, '') <> COALESCE(LAG(P_emaildomain) OVER w_seq, '') END AS email_changed,
    CASE WHEN DeviceInfo IS NULL OR LAG(DeviceInfo) OVER w_seq IS NULL THEN NULL
         ELSE DeviceInfo <> LAG(DeviceInfo) OVER w_seq END                 AS device_changed,
    -- card-level velocity (a card number shared across customers is itself a signal)
    COUNT(*) OVER (PARTITION BY card1 ORDER BY TransactionDT RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS card_tx_24h,
    COUNT(*) OVER (PARTITION BY card1 ORDER BY TransactionDT RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING)  AS card_tx_1h
FROM b
WINDOW
    w_all AS (PARTITION BY cust ORDER BY TransactionDT RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w_1h  AS (PARTITION BY cust ORDER BY TransactionDT RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING),
    w_24h AS (PARTITION BY cust ORDER BY TransactionDT RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING),
    w_7d  AS (PARTITION BY cust ORDER BY TransactionDT RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING),
    w_seq AS (PARTITION BY cust ORDER BY TransactionDT, TransactionID);
