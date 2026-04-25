-- mart_sev_features.sql
-- Modelling-ready table for the SEVERITY model (Gamma GLM + LightGBM).
-- Target: avg_claim_amount (expected cost per claim)
-- IMPORTANT: Only policies WITH at least one claim are included here.

SELECT
    policy_id,

    -- Target
    avg_claim_amount,
    total_claim_amount,
    claim_nb,

    -- Vehicle features
    veh_power,
    veh_age,
    veh_brand,
    veh_gas,

    -- Driver features
    driv_age,
    bonus_malus,
    driv_age * bonus_malus                          AS age_x_bonus,

    -- Location features
    area,
    log_density,
    region,

    -- Risk indicators (computed inline — not inherited from int layer)
    CASE WHEN driv_age < 25 THEN 1 ELSE 0 END      AS is_young_driver,
    CASE WHEN driv_age > 70 THEN 1 ELSE 0 END      AS is_senior_driver,
    CASE WHEN bonus_malus > 100 THEN 1 ELSE 0 END  AS has_malus,
    CASE WHEN veh_age > 10 THEN 1 ELSE 0 END       AS is_old_vehicle,
    CASE WHEN veh_power >= 9 THEN 1 ELSE 0 END     AS is_high_power,

    -- Severity-specific
    n_large_claims,
    CASE WHEN n_large_claims > 0 THEN 1 ELSE 0 END AS has_large_claim

FROM {{ ref('int_policy_claims') }}
WHERE has_claim = 1
  AND avg_claim_amount IS NOT NULL
  AND avg_claim_amount > 0