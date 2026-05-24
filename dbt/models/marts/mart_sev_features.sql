-- mart_sev_features.sql
-- Modelling-ready severity table — claims-only subset, clean pass-through.
-- Feature engineering (flags, interactions, encodings) is owned by the
-- PySpark Silver layer (src/features/spark_features.py).
SELECT
    policy_id,
    avg_claim_amount,
    total_claim_amount,
    claim_nb,
    veh_power,
    veh_age,
    veh_brand,
    veh_gas,
    driv_age,
    bonus_malus,
    area,
    log_density,
    region,
    n_large_claims
FROM {{ ref('int_policy_claims') }}
WHERE has_claim = 1
  AND avg_claim_amount IS NOT NULL
  AND avg_claim_amount > 0