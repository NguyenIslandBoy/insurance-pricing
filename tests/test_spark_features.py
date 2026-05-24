"""
tests/test_spark_features.py
-----------------------------
Unit tests for src/features/spark_features.py.

Tests cover every transformation function in isolation using a shared
SparkSession fixture. No DuckDB, no file I/O, no data files required.

Design principles:
  - One SparkSession for the entire module (session scope) — startup is
    expensive; reusing it keeps the suite fast.
  - Each test builds a minimal synthetic DataFrame with only the columns
    the function under test needs.
  - Assertions use .collect() on small DataFrames — correct for unit tests.
  - Tests are grouped by transformation function for readability.

Run:
    pytest tests/test_spark_features.py -v
"""

import pytest
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.features.spark_features import (
    apply_freq_transforms,
    apply_sev_transforms,
    build_driver_flags,
    build_interactions,
    build_log_transforms,
    build_ordinal_encodings,
    build_vehicle_flags,
    AREA_MAP,
    REGION_MAP,
    VEH_BRAND_MAP,
    VEH_GAS_MAP,
)


# ===========================================================================
# SparkSession — shared across all tests
# ===========================================================================

@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder
        .appName("Insurance_Features_Tests")
        .master("local[1]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


# ===========================================================================
# Helpers
# ===========================================================================

def make_df(spark, rows: list[dict], schema: StructType):
    """Build a small Spark DataFrame from a list of dicts with explicit schema."""
    pdf = pd.DataFrame(rows)
    for field in schema.fields:
        if field.name not in pdf.columns:
            pdf[field.name] = None
    return spark.createDataFrame(
        pdf[[f.name for f in schema.fields]], schema=schema
    )


def get_col(df, col: str):
    """Collect a single column value from a single-row DataFrame."""
    return df.collect()[0][col]


# Minimal schemas per transform group
DRIVER_SCHEMA = StructType([
    StructField("driv_age",    IntegerType(), True),
    StructField("bonus_malus", IntegerType(), True),
])

VEHICLE_SCHEMA = StructType([
    StructField("veh_age",   IntegerType(), True),
    StructField("veh_power", IntegerType(), True),
])

INTERACTION_SCHEMA = StructType([
    StructField("driv_age",    IntegerType(), True),
    StructField("bonus_malus", IntegerType(), True),
])

LOG_SCHEMA = StructType([
    StructField("exposure",    DoubleType(), True),
    StructField("log_density", DoubleType(), True),
])

ENCODING_SCHEMA = StructType([
    StructField("veh_brand", StringType(), True),
    StructField("veh_gas",   StringType(), True),
    StructField("area",      StringType(), True),
    StructField("region",    StringType(), True),
])

# Full schema for integration tests
FULL_FREQ_SCHEMA = StructType([
    StructField("policy_id",       IntegerType(), True),
    StructField("claim_nb",        IntegerType(), True),
    StructField("exposure",        DoubleType(),  True),
    StructField("log_exposure",    DoubleType(),  True),
    StructField("veh_power",       IntegerType(), True),
    StructField("veh_age",         IntegerType(), True),
    StructField("veh_brand",       StringType(),  True),
    StructField("veh_gas",         StringType(),  True),
    StructField("driv_age",        IntegerType(), True),
    StructField("bonus_malus",     IntegerType(), True),
    StructField("area",            StringType(),  True),
    StructField("log_density",     DoubleType(),  True),
    StructField("region",          StringType(),  True),
    StructField("claimnb_corrupted", IntegerType(), True),
])

FULL_SEV_SCHEMA = StructType([
    StructField("policy_id",          IntegerType(), True),
    StructField("avg_claim_amount",   DoubleType(),  True),
    StructField("total_claim_amount", DoubleType(),  True),
    StructField("claim_nb",           IntegerType(), True),
    StructField("veh_power",          IntegerType(), True),
    StructField("veh_age",            IntegerType(), True),
    StructField("veh_brand",          StringType(),  True),
    StructField("veh_gas",            StringType(),  True),
    StructField("driv_age",           IntegerType(), True),
    StructField("bonus_malus",        IntegerType(), True),
    StructField("area",               StringType(),  True),
    StructField("log_density",        DoubleType(),  True),
    StructField("region",             StringType(),  True),
    StructField("n_large_claims",     IntegerType(), True),
    StructField("has_large_claim",    IntegerType(), True),
])


def make_full_freq_row(spark, **overrides):
    defaults = {
        "policy_id": 1, "claim_nb": 0, "exposure": 1.0,
        "log_exposure": 0.0, "veh_power": 7, "veh_age": 3,
        "veh_brand": "B1", "veh_gas": "REGULAR", "driv_age": 35,
        "bonus_malus": 90, "area": "C", "log_density": 6.2,
        "region": "R11", "claimnb_corrupted": 0,
    }
    defaults.update(overrides)
    return make_df(spark, [defaults], FULL_FREQ_SCHEMA)


def make_full_sev_row(spark, **overrides):
    defaults = {
        "policy_id": 1, "avg_claim_amount": 2500.0,
        "total_claim_amount": 2500.0, "claim_nb": 1,
        "veh_power": 7, "veh_age": 3, "veh_brand": "B1",
        "veh_gas": "REGULAR", "driv_age": 35, "bonus_malus": 90,
        "area": "C", "log_density": 6.2, "region": "R11",
        "n_large_claims": 0, "has_large_claim": 0,
    }
    defaults.update(overrides)
    return make_df(spark, [defaults], FULL_SEV_SCHEMA)


# ===========================================================================
# build_driver_flags
# ===========================================================================

class TestBuildDriverFlags:

    def _get(self, spark, driv_age, bonus_malus):
        df = make_df(spark, [{"driv_age": driv_age, "bonus_malus": bonus_malus}], DRIVER_SCHEMA)
        return build_driver_flags(df).collect()[0]

    def test_young_driver_under_25(self, spark):
        row = self._get(spark, 22, 90)
        assert row["is_young_driver"] == 1

    def test_not_young_driver_at_25(self, spark):
        row = self._get(spark, 25, 90)
        assert row["is_young_driver"] == 0

    def test_not_young_driver_over_25(self, spark):
        row = self._get(spark, 35, 90)
        assert row["is_young_driver"] == 0

    def test_senior_driver_over_70(self, spark):
        row = self._get(spark, 75, 80)
        assert row["is_senior_driver"] == 1

    def test_not_senior_driver_at_70(self, spark):
        row = self._get(spark, 70, 80)
        assert row["is_senior_driver"] == 0

    def test_not_senior_driver_under_70(self, spark):
        row = self._get(spark, 35, 80)
        assert row["is_senior_driver"] == 0

    def test_has_malus_over_100(self, spark):
        row = self._get(spark, 35, 110)
        assert row["has_malus"] == 1

    def test_no_malus_at_100(self, spark):
        row = self._get(spark, 35, 100)
        assert row["has_malus"] == 0

    def test_no_malus_under_100(self, spark):
        row = self._get(spark, 35, 90)
        assert row["has_malus"] == 0

    def test_all_three_flags_present(self, spark):
        df = make_df(spark, [{"driv_age": 22, "bonus_malus": 110}], DRIVER_SCHEMA)
        result = build_driver_flags(df)
        assert {"is_young_driver", "is_senior_driver", "has_malus"}.issubset(
            set(result.columns)
        )

    def test_high_risk_profile_all_flags(self, spark):
        # Young driver with malus — both flags should fire
        row = self._get(spark, 22, 130)
        assert row["is_young_driver"] == 1
        assert row["has_malus"] == 1
        assert row["is_senior_driver"] == 0


# ===========================================================================
# build_vehicle_flags
# ===========================================================================

class TestBuildVehicleFlags:

    def _get(self, spark, veh_age, veh_power):
        df = make_df(spark, [{"veh_age": veh_age, "veh_power": veh_power}], VEHICLE_SCHEMA)
        return build_vehicle_flags(df).collect()[0]

    def test_old_vehicle_over_10(self, spark):
        assert self._get(spark, 12, 7)["is_old_vehicle"] == 1

    def test_not_old_vehicle_at_10(self, spark):
        assert self._get(spark, 10, 7)["is_old_vehicle"] == 0

    def test_not_old_vehicle_under_10(self, spark):
        assert self._get(spark, 3, 7)["is_old_vehicle"] == 0

    def test_high_power_at_9(self, spark):
        assert self._get(spark, 3, 9)["is_high_power"] == 1

    def test_high_power_over_9(self, spark):
        assert self._get(spark, 3, 12)["is_high_power"] == 1

    def test_not_high_power_under_9(self, spark):
        assert self._get(spark, 3, 7)["is_high_power"] == 0

    def test_both_flags_present(self, spark):
        df = make_df(spark, [{"veh_age": 15, "veh_power": 10}], VEHICLE_SCHEMA)
        result = build_vehicle_flags(df)
        assert {"is_old_vehicle", "is_high_power"}.issubset(set(result.columns))


# ===========================================================================
# build_interactions
# ===========================================================================

class TestBuildInteractions:

    def test_age_x_bonus_correct(self, spark):
        df = make_df(spark, [{"driv_age": 30, "bonus_malus": 100}], INTERACTION_SCHEMA)
        result = build_interactions(df)
        assert get_col(result, "age_x_bonus") == 3000

    def test_age_x_bonus_young_driver_malus(self, spark):
        df = make_df(spark, [{"driv_age": 22, "bonus_malus": 130}], INTERACTION_SCHEMA)
        result = build_interactions(df)
        assert get_col(result, "age_x_bonus") == 22 * 130

    def test_age_x_bonus_column_added(self, spark):
        df = make_df(spark, [{"driv_age": 35, "bonus_malus": 90}], INTERACTION_SCHEMA)
        result = build_interactions(df)
        assert "age_x_bonus" in result.columns

    def test_age_x_bonus_zero_bonus(self, spark):
        # bonus_malus minimum is 50 in real data but test the arithmetic
        df = make_df(spark, [{"driv_age": 40, "bonus_malus": 50}], INTERACTION_SCHEMA)
        result = build_interactions(df)
        assert get_col(result, "age_x_bonus") == 2000


# ===========================================================================
# build_log_transforms
# ===========================================================================

class TestBuildLogTransforms:

    def test_log_exposure_computed(self, spark):
        import math
        df = make_df(spark, [{"exposure": 1.0, "log_density": 5.0}], LOG_SCHEMA)
        result = build_log_transforms(df)
        assert get_col(result, "log_exposure") == pytest.approx(math.log(1.0))

    def test_log_exposure_partial_year(self, spark):
        import math
        df = make_df(spark, [{"exposure": 0.5, "log_density": 5.0}], LOG_SCHEMA)
        result = build_log_transforms(df)
        assert get_col(result, "log_exposure") == pytest.approx(math.log(0.5))

    def test_log_exposure_column_added(self, spark):
        df = make_df(spark, [{"exposure": 1.0, "log_density": 5.0}], LOG_SCHEMA)
        result = build_log_transforms(df)
        assert "log_exposure" in result.columns

    def test_log_density_recomputed(self, spark):
        import math
        # log_density comes in as the raw log value from dbt
        # build_log_transforms takes log of it again for safety floor at 1
        df = make_df(spark, [{"exposure": 1.0, "log_density": 5.0}], LOG_SCHEMA)
        result = build_log_transforms(df)
        # log(max(5.0, 1.0)) = log(5.0)
        assert get_col(result, "log_density") == pytest.approx(math.log(5.0))


# ===========================================================================
# build_ordinal_encodings
# ===========================================================================

class TestBuildOrdinalEncodings:

    def _get(self, spark, veh_brand, veh_gas, area, region):
        df = make_df(spark, [{
            "veh_brand": veh_brand, "veh_gas": veh_gas,
            "area": area, "region": region,
        }], ENCODING_SCHEMA)
        return build_ordinal_encodings(df).collect()[0]

    def test_known_brand_encoded(self, spark):
        row = self._get(spark, "B1", "REGULAR", "A", "R11")
        assert row["veh_brand_enc"] == VEH_BRAND_MAP["B1"]

    def test_unknown_brand_gives_minus1(self, spark):
        row = self._get(spark, "B99", "REGULAR", "A", "R11")
        assert row["veh_brand_enc"] == -1

    def test_diesel_encoded(self, spark):
        row = self._get(spark, "B1", "DIESEL", "A", "R11")
        assert row["veh_gas_enc"] == VEH_GAS_MAP["DIESEL"]

    def test_regular_encoded(self, spark):
        row = self._get(spark, "B1", "REGULAR", "A", "R11")
        assert row["veh_gas_enc"] == VEH_GAS_MAP["REGULAR"]

    def test_area_a_encoded(self, spark):
        row = self._get(spark, "B1", "REGULAR", "A", "R11")
        assert row["area_enc"] == AREA_MAP["A"]

    def test_area_f_encoded(self, spark):
        row = self._get(spark, "B1", "REGULAR", "F", "R11")
        assert row["area_enc"] == AREA_MAP["F"]

    def test_unknown_area_gives_minus1(self, spark):
        row = self._get(spark, "B1", "REGULAR", "Z", "R11")
        assert row["area_enc"] == -1

    def test_known_region_encoded(self, spark):
        row = self._get(spark, "B1", "REGULAR", "A", "R11")
        assert row["region_enc"] == REGION_MAP["R11"]

    def test_unknown_region_gives_minus1(self, spark):
        row = self._get(spark, "B1", "REGULAR", "A", "R99")
        assert row["region_enc"] == -1

    def test_all_enc_columns_added(self, spark):
        df = make_df(spark, [{
            "veh_brand": "B1", "veh_gas": "REGULAR",
            "area": "A", "region": "R11",
        }], ENCODING_SCHEMA)
        result = build_ordinal_encodings(df)
        assert {"veh_brand_enc", "veh_gas_enc", "area_enc", "region_enc"}.issubset(
            set(result.columns)
        )

    def test_encoding_ordering_area(self, spark):
        # Area A < B < C < ... < F — verify monotonic ordering
        areas = list(AREA_MAP.keys())
        codes = [AREA_MAP[a] for a in areas]
        assert codes == sorted(codes)

    def test_gas_has_two_values(self, spark):
        assert len(VEH_GAS_MAP) == 2
        assert "DIESEL" in VEH_GAS_MAP
        assert "REGULAR" in VEH_GAS_MAP


# ===========================================================================
# apply_freq_transforms — integration
# ===========================================================================

class TestApplyFreqTransforms:

    def test_all_derived_columns_present(self, spark):
        df = apply_freq_transforms(make_full_freq_row(spark))
        expected = {
            "is_young_driver", "is_senior_driver", "has_malus",
            "is_old_vehicle", "is_high_power", "age_x_bonus",
            "log_exposure", "veh_brand_enc", "veh_gas_enc",
            "area_enc", "region_enc", "has_claim",
        }
        assert expected.issubset(set(df.columns))

    def test_has_claim_derived_from_claim_nb_zero(self, spark):
        df = apply_freq_transforms(make_full_freq_row(spark, claim_nb=0))
        assert get_col(df, "has_claim") == 0

    def test_has_claim_derived_from_claim_nb_positive(self, spark):
        df = apply_freq_transforms(make_full_freq_row(spark, claim_nb=2))
        assert get_col(df, "has_claim") == 1

    def test_row_count_unchanged(self, spark):
        df = make_full_freq_row(spark)
        result = apply_freq_transforms(df)
        assert result.count() == df.count()

    def test_typical_low_risk_profile(self, spark):
        # Experienced driver, low bonus_malus, modest vehicle
        row = apply_freq_transforms(
            make_full_freq_row(spark, driv_age=45, bonus_malus=70,
                               veh_age=5, veh_power=6)
        ).collect()[0]
        assert row["is_young_driver"] == 0
        assert row["is_senior_driver"] == 0
        assert row["has_malus"] == 0
        assert row["is_old_vehicle"] == 0
        assert row["is_high_power"] == 0

    def test_typical_high_risk_profile(self, spark):
        # Young driver, malus, old high-power vehicle
        row = apply_freq_transforms(
            make_full_freq_row(spark, driv_age=21, bonus_malus=140,
                               veh_age=15, veh_power=11)
        ).collect()[0]
        assert row["is_young_driver"] == 1
        assert row["has_malus"] == 1
        assert row["is_old_vehicle"] == 1
        assert row["is_high_power"] == 1

    def test_all_nulls_does_not_error(self, spark):
        df = make_full_freq_row(
            spark,
            driv_age=35, bonus_malus=90, veh_age=3, veh_power=7,
            veh_brand=None, veh_gas=None, area=None, region=None,
        )
        result = apply_freq_transforms(df)
        assert result.count() == 1
        # Unknown categoricals → -1
        row = result.collect()[0]
        assert row["veh_brand_enc"] == -1
        assert row["veh_gas_enc"] == -1


# ===========================================================================
# apply_sev_transforms — integration
# ===========================================================================

class TestApplySevTransforms:

    def test_all_derived_columns_present(self, spark):
        df = apply_sev_transforms(make_full_sev_row(spark))
        expected = {
            "is_young_driver", "is_senior_driver", "has_malus",
            "is_old_vehicle", "is_high_power", "age_x_bonus",
            "veh_brand_enc", "veh_gas_enc", "area_enc", "region_enc",
        }
        assert expected.issubset(set(df.columns))

    def test_row_count_unchanged(self, spark):
        df = make_full_sev_row(spark)
        result = apply_sev_transforms(df)
        assert result.count() == df.count()

    def test_sev_has_no_log_exposure(self, spark):
        # Severity model has no exposure offset — log_exposure not needed
        df = apply_sev_transforms(make_full_sev_row(spark))
        assert "log_exposure" not in df.columns

    def test_avg_claim_amount_preserved(self, spark):
        df = apply_sev_transforms(make_full_sev_row(spark, avg_claim_amount=3500.0))
        assert get_col(df, "avg_claim_amount") == pytest.approx(3500.0)

    def test_typical_sev_profile(self, spark):
        row = apply_sev_transforms(
            make_full_sev_row(spark, driv_age=35, bonus_malus=90,
                              veh_age=3, veh_power=7, veh_brand="B1",
                              veh_gas="REGULAR", area="C", region="R11")
        ).collect()[0]
        assert row["is_young_driver"] == 0
        assert row["has_malus"] == 0
        assert row["veh_brand_enc"] == VEH_BRAND_MAP["B1"]
        assert row["area_enc"] == AREA_MAP["C"]


# ===========================================================================
# Encoding map integrity
# ===========================================================================

class TestEncodingMaps:

    def test_veh_brand_map_has_expected_brands(self):
        for brand in ["B1", "B2", "B12"]:
            assert brand in VEH_BRAND_MAP

    def test_veh_brand_map_values_unique(self):
        values = list(VEH_BRAND_MAP.values())
        assert len(values) == len(set(values))

    def test_area_map_has_6_areas(self):
        assert set(AREA_MAP.keys()) == {"A", "B", "C", "D", "E", "F"}

    def test_area_map_values_zero_indexed(self):
        assert min(AREA_MAP.values()) == 0
        assert max(AREA_MAP.values()) == 5

    def test_region_map_values_unique(self):
        values = list(REGION_MAP.values())
        assert len(values) == len(set(values))

    def test_region_map_zero_indexed(self):
        assert min(REGION_MAP.values()) == 0

    def test_gas_map_two_values(self):
        assert len(VEH_GAS_MAP) == 2

    def test_all_maps_start_at_zero(self):
        for name, mapping in [
            ("veh_brand", VEH_BRAND_MAP), ("veh_gas", VEH_GAS_MAP),
            ("area", AREA_MAP), ("region", REGION_MAP),
        ]:
            assert min(mapping.values()) == 0, f"{name} map does not start at 0"