-- stg_claims.sql
-- Clean the severity table.
-- Key transforms:
--   1. Exclude zero or negative claim amounts (data errors)
--   2. Flag large claims (> 100k) for separate treatment
--   3. Cap extreme claims at 99th percentile for severity modelling
--      (extreme tail is handled separately in reinsurance — not our model's job)

WITH percentiles AS (
    SELECT
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY ClaimAmount) AS p99
    FROM {{ source('raw', 'raw_sev') }}
    WHERE ClaimAmount > 0
)

SELECT
    s.IDpol                                         AS policy_id,
    s.ClaimAmount                                   AS claim_amount_raw,
    -- Cap at p99 for modelling — extreme tail distorts gamma GLM
    LEAST(s.ClaimAmount, p.p99)                     AS claim_amount,
    CASE WHEN s.ClaimAmount > p.p99 THEN 1 ELSE 0 END AS is_large_claim,
    p.p99                                           AS p99_threshold

FROM {{ source('raw', 'raw_sev') }} s
CROSS JOIN percentiles p
WHERE s.ClaimAmount > 0  -- exclude zero/negative amounts