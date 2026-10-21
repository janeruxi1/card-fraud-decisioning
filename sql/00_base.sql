-- One row per transaction with the fields every audit query needs.
--   day   : days since the first day in the data (TransactionDT is seconds from an undisclosed start)
--   uid   : an approximate customer key. D1 counts days since the card was first seen, so
--           (calendar day - D1) is the same for every transaction of one card. Combined with the
--           card and billing-address codes it groups a customer's transactions.
--   split : the fixed time split (train / valid / test)
CREATE OR REPLACE VIEW base AS
WITH t AS (SELECT MIN(TransactionDT // 86400) AS d0 FROM tx)
SELECT
    tx.TransactionID,
    tx.isFraud,
    tx.TransactionDT,
    tx.TransactionAmt,
    tx.ProductCD,
    tx.card1,
    tx.addr1,
    tx.D1,
    tx.P_emaildomain,
    tx.id_01 IS NOT NULL                            AS has_identity,   -- every identity row has id_01
    CAST(tx.TransactionDT // 86400 - t.d0 AS INTEGER) AS day,
    CASE WHEN tx.D1 IS NULL THEN NULL
         ELSE CAST(tx.card1 AS VARCHAR) || '_' || COALESCE(CAST(CAST(tx.addr1 AS INTEGER) AS VARCHAR), 'na')
              || '_' || CAST(CAST(tx.TransactionDT // 86400 - tx.D1 AS BIGINT) AS VARCHAR)
    END                                              AS uid,
    CASE WHEN tx.TransactionDT // 86400 - t.d0 < {train_end} THEN 'train'
         WHEN tx.TransactionDT // 86400 - t.d0 < {valid_end} THEN 'valid'
         ELSE 'test' END                             AS split
FROM tx, t;
