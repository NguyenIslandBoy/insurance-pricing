-- mart_freq_features.sql
-- Modelling-ready frequency table — clean pass-through after int layer.
-- Feature engineering (flags, interactions, encodings) is owned by the
-- PySpark Silver layer (src/features/spark_features.py).
SELECT
    policy_id,
    claim_nb,
    exposure,
    veh_power,
    veh_age,
    veh_brand,
    veh_gas,
    driv_age,
    bonus_malus,
    area,
    log_density,
    region,
    claimnb_corrupted
FROM {{ ref('int_policy_claims') }}