-- stg_policies.sql
-- Clean the frequency table.
-- Key transforms:
--   1. Clip Exposure to (0, 1] — values > 1 are data entry errors
--   2. Clip ClaimNb to >= 0
--   3. Cap BonusMalus at 150 (values above are extreme outliers, <0.01% of data)
--   4. Standardise VehGas casing
--   5. Retain claimnb_corrupted flag for downstream awareness

SELECT
    IDpol                                           AS policy_id,
    -- Fix exposure: must be in (0, 1] for a one-year observation period
    LEAST(GREATEST(Exposure, 0.001), 1.0)           AS exposure,
    -- ClaimNb already zeroed for corrupt policies in Python loader
    GREATEST(ClaimNb, 0)                            AS claim_nb,
    claimnb_corrupted,

    -- Vehicle features
    VehPower                                        AS veh_power,
    VehAge                                          AS veh_age,
    TRIM(UPPER(VehBrand))                           AS veh_brand,
    TRIM(UPPER(VehGas))                             AS veh_gas,

    -- Driver features
    DrivAge                                         AS driv_age,
    -- BonusMalus: 50=max bonus, 350=max malus. Cap at 150 to reduce outlier influence
    LEAST(BonusMalus, 150)                          AS bonus_malus,

    -- Location features
    TRIM(UPPER(Area))                               AS area,
    -- Log density: right-skewed, log transform stabilises it
    LN(GREATEST(Density, 1))                        AS log_density,
    TRIM(Region)                                    AS region

FROM {{ source('raw', 'raw_freq') }}
WHERE Exposure > 0  -- exclude zero-exposure policies (not at risk)