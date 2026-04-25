-- mart_freq_features.sql
-- Modelling-ready table for the FREQUENCY model (Poisson GLM + LightGBM).
-- Target: claim_nb (count of claims)
-- Offset: log(exposure) — mandatory for Poisson with varying exposure
-- All policies included (zero-claim policies are the majority — ~96%).

SELECT
    policy_id,

    -- Target and offset
    claim_nb,
    exposure,
    LN(exposure)            AS log_exposure,    -- offset term for Poisson GLM

    -- Vehicle features
    veh_power,
    veh_age,
    veh_brand,
    veh_gas,

    -- Driver features
    driv_age,
    bonus_malus,
    -- Age-BonusMalus interaction: correlated but model benefits from explicit term
    driv_age * bonus_malus  AS age_x_bonus,

    -- Location features
    area,
    log_density,
    region,

    -- Derived risk indicators
    -- Young driver flag (high risk segment in MTPL)
    CASE WHEN driv_age < 25 THEN 1 ELSE 0 END  AS is_young_driver,
    -- Senior driver flag
    CASE WHEN driv_age > 70 THEN 1 ELSE 0 END  AS is_senior_driver,
    -- High bonus-malus (>100 = malus — previous claims)
    CASE WHEN bonus_malus > 100 THEN 1 ELSE 0 END AS has_malus,
    -- Old vehicle
    CASE WHEN veh_age > 10 THEN 1 ELSE 0 END   AS is_old_vehicle,
    -- High power vehicle
    CASE WHEN veh_power >= 9 THEN 1 ELSE 0 END AS is_high_power,

    -- Metadata
    claimnb_corrupted

FROM {{ ref('int_policy_claims') }}