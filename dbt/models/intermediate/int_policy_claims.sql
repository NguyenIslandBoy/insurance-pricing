-- int_policy_claims.sql
-- Join policies with their aggregated claim data.
-- This is the critical join — it recomputes ClaimNb from severity records
-- for the corrupt policies (IDpol <= 24500), fixing the known data issue.

WITH claim_aggs AS (
    -- Aggregate severity table: one row per policy
    SELECT
        policy_id,
        COUNT(*)            AS claim_nb_from_sev,   -- true claim count from severity
        SUM(claim_amount)   AS total_claim_amount,   -- capped amounts
        SUM(claim_amount_raw) AS total_claim_amount_raw,
        AVG(claim_amount)   AS avg_claim_amount,
        MAX(claim_amount_raw) AS max_claim_amount,
        SUM(is_large_claim) AS n_large_claims
    FROM {{ ref('stg_claims') }}
    GROUP BY policy_id
)

SELECT
    p.policy_id,
    p.exposure,
    p.claimnb_corrupted,

    -- Use severity-derived count for corrupt policies, freq count otherwise
    CASE
        WHEN p.claimnb_corrupted = 1
        THEN COALESCE(c.claim_nb_from_sev, 0)
        ELSE p.claim_nb
    END                                             AS claim_nb,

    -- Claim financials (NULL for zero-claim policies)
    COALESCE(c.total_claim_amount, 0)               AS total_claim_amount,
    COALESCE(c.total_claim_amount_raw, 0)           AS total_claim_amount_raw,
    c.avg_claim_amount,                             -- NULL if no claims
    c.max_claim_amount,
    COALESCE(c.n_large_claims, 0)                   AS n_large_claims,

    -- Derived
    CASE WHEN c.policy_id IS NOT NULL THEN 1 ELSE 0 END AS has_claim,

    -- Policy features (pass through)
    p.veh_power,
    p.veh_age,
    p.veh_brand,
    p.veh_gas,
    p.driv_age,
    p.bonus_malus,
    p.area,
    p.log_density,
    p.region

FROM {{ ref('stg_policies') }} p
LEFT JOIN claim_aggs c ON p.policy_id = c.policy_id