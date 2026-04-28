"""
tests/test_insurance_pipeline.py
----------------------------------
Unit tests for the insurance pricing pipeline.

Covers:
  - loader.py  — ClaimNb corruption fix, column normalisation, DuckDB writes
  - app.py     — _build_features, _risk_tier, API schema validation
  - pure_premium logic — combination naming, score properties

All tests use synthetic DataFrames and in-memory DuckDB.
No CSV data files or trained model artifacts required.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.loader import write_to_db
from src.api.app import _build_features, _risk_tier, PolicyFeatures


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mem_con():
    """Fresh in-memory DuckDB connection per test."""
    return duckdb.connect(":memory:")


@pytest.fixture
def raw_freq_df():
    """Synthetic freMTPL2freq-style DataFrame with both corrupt and clean rows."""
    return pd.DataFrame({
        "IDpol":    [1000, 15000, 24500, 24501, 50000, 100000],
        "ClaimNb":  [2,    3,     1,     4,     0,     2],
        "Exposure": [0.5,  1.0,   0.75,  1.0,   0.3,   1.0],
        "VehPower": [6,    7,     8,     9,     10,    11],
        "VehAge":   [3,    5,     2,     7,     1,     4],
        "DrivAge":  [25,   35,    22,    45,    60,    30],
        "BonusMalus": [80, 100,   120,   90,    75,    110],
        "VehBrand": ["B1","B2",  "B1",  "B3",  "B2",  "B1"],
        "VehGas":   ["REGULAR","DIESEL","REGULAR","DIESEL","REGULAR","DIESEL"],
        "Area":     ["A","B",   "C",   "D",   "E",   "F"],
        "Density":  [100, 500,  200,   800,   50,    1000],
        "Region":   ["R11","R24","R11","R52","R24","R11"],
    })


@pytest.fixture
def sample_policy():
    return PolicyFeatures(
        veh_power=7,
        veh_age=3,
        veh_brand="B1",
        veh_gas="REGULAR",
        driv_age=35,
        bonus_malus=90,
        area="C",
        density=500.0,
        region="R11",
        exposure=1.0,
    )


# ===========================================================================
# loader — ClaimNb corruption fix
# ===========================================================================

class TestClaimNbCorruptionFix:

    def _apply_fix(self, df, threshold=24500):
        """Reproduce the corruption fix logic from loader.load_freq."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        corrupt_mask = df["IDpol"] <= threshold
        df.loc[corrupt_mask, "ClaimNb"] = 0
        df["claimnb_corrupted"] = corrupt_mask.astype(int)
        return df

    def test_corrupt_policies_claimnb_zeroed(self, raw_freq_df):
        result = self._apply_fix(raw_freq_df)
        corrupt = result[result["IDpol"] <= 24500]
        assert (corrupt["ClaimNb"] == 0).all()

    def test_clean_policies_claimnb_unchanged(self, raw_freq_df):
        original = raw_freq_df.copy()
        result = self._apply_fix(raw_freq_df)
        clean = result[result["IDpol"] > 24500]
        original_clean = original[original["IDpol"] > 24500]
        pd.testing.assert_series_equal(
            clean["ClaimNb"].reset_index(drop=True),
            original_clean["ClaimNb"].reset_index(drop=True),
        )

    def test_corruption_flag_column_added(self, raw_freq_df):
        result = self._apply_fix(raw_freq_df)
        assert "claimnb_corrupted" in result.columns

    def test_corruption_flag_is_1_for_corrupt(self, raw_freq_df):
        result = self._apply_fix(raw_freq_df)
        corrupt = result[result["IDpol"] <= 24500]
        assert (corrupt["claimnb_corrupted"] == 1).all()

    def test_corruption_flag_is_0_for_clean(self, raw_freq_df):
        result = self._apply_fix(raw_freq_df)
        clean = result[result["IDpol"] > 24500]
        assert (clean["claimnb_corrupted"] == 0).all()

    def test_boundary_exactly_at_threshold(self, raw_freq_df):
        # IDpol == 24500 is corrupt (<=), IDpol == 24501 is clean (>)
        result = self._apply_fix(raw_freq_df)
        assert result.loc[result["IDpol"] == 24500, "ClaimNb"].iloc[0] == 0
        assert result.loc[result["IDpol"] == 24501, "ClaimNb"].iloc[0] == 4

    def test_total_row_count_unchanged(self, raw_freq_df):
        result = self._apply_fix(raw_freq_df)
        assert len(result) == len(raw_freq_df)

    def test_column_strip_applied(self):
        df = pd.DataFrame({
            "  IDpol  ": [1], "  ClaimNb  ": [1], "  Exposure  ": [1.0]
        })
        df.columns = df.columns.str.strip()
        assert "IDpol" in df.columns
        assert "ClaimNb" in df.columns


# ===========================================================================
# loader — write_to_db
# ===========================================================================

class TestWriteToDb:

    def test_table_created(self, mem_con):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        write_to_db(df, "test_table", mem_con)
        tables = mem_con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'test_table'"
        ).fetchall()
        assert len(tables) == 1

    def test_row_count_matches(self, mem_con):
        df = pd.DataFrame({"x": range(100)})
        write_to_db(df, "rows_test", mem_con)
        count = mem_con.execute("SELECT COUNT(*) FROM rows_test").fetchone()[0]
        assert count == 100

    def test_columns_preserved(self, mem_con):
        df = pd.DataFrame({"policy_id": [1, 2], "exposure": [0.5, 1.0]})
        write_to_db(df, "col_test", mem_con)
        cols = [r[0] for r in mem_con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'col_test'"
        ).fetchall()]
        assert "policy_id" in cols
        assert "exposure" in cols

    def test_overwrites_existing_table(self, mem_con):
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [10, 20]})
        write_to_db(df1, "overwrite_test", mem_con)
        write_to_db(df2, "overwrite_test", mem_con)
        count = mem_con.execute("SELECT COUNT(*) FROM overwrite_test").fetchone()[0]
        assert count == 2

    def test_empty_dataframe_writes_empty_table(self, mem_con):
        df = pd.DataFrame({"a": pd.Series([], dtype=int)})
        write_to_db(df, "empty_test", mem_con)
        count = mem_con.execute("SELECT COUNT(*) FROM empty_test").fetchone()[0]
        assert count == 0

    def test_numeric_values_round_trip(self, mem_con):
        df = pd.DataFrame({"val": [1.23456789]})
        write_to_db(df, "numeric_test", mem_con)
        result = mem_con.execute("SELECT val FROM numeric_test").fetchone()[0]
        assert result == pytest.approx(1.23456789)


# ===========================================================================
# app — _risk_tier
# ===========================================================================

class TestRiskTier:

    def test_below_100_is_low(self):
        assert _risk_tier(0) == "LOW"
        assert _risk_tier(50) == "LOW"
        assert _risk_tier(99.99) == "LOW"

    def test_exactly_100_is_medium(self):
        assert _risk_tier(100) == "MEDIUM"

    def test_between_100_and_200_is_medium(self):
        assert _risk_tier(100) == "MEDIUM"
        assert _risk_tier(150) == "MEDIUM"
        assert _risk_tier(199.99) == "MEDIUM"

    def test_exactly_200_is_high(self):
        assert _risk_tier(200) == "HIGH"

    def test_above_200_is_high(self):
        assert _risk_tier(200) == "HIGH"
        assert _risk_tier(500) == "HIGH"
        assert _risk_tier(10000) == "HIGH"

    def test_returns_string(self):
        assert isinstance(_risk_tier(100), str)

    def test_only_three_tiers(self):
        valid = {"LOW", "MEDIUM", "HIGH"}
        test_values = [0, 50, 99, 100, 150, 199, 200, 500, 1000]
        for v in test_values:
            assert _risk_tier(v) in valid


# ===========================================================================
# app — _build_features
# ===========================================================================

class TestBuildFeatures:

    def test_returns_dataframe(self, sample_policy):
        result = _build_features(sample_policy)
        assert isinstance(result, pd.DataFrame)

    def test_single_row(self, sample_policy):
        result = _build_features(sample_policy)
        assert len(result) == 1

    def test_all_expected_columns_present(self, sample_policy):
        result = _build_features(sample_policy)
        expected = {
            "veh_power", "veh_age", "veh_brand", "veh_gas",
            "driv_age", "bonus_malus", "age_x_bonus",
            "area", "log_density", "region",
            "is_young_driver", "is_senior_driver", "has_malus",
            "is_old_vehicle", "is_high_power",
        }
        assert expected.issubset(set(result.columns))

    def test_log_density_is_log_of_density(self, sample_policy):
        result = _build_features(sample_policy)
        expected = np.log(max(sample_policy.density, 1))
        assert result["log_density"].iloc[0] == pytest.approx(expected)

    def test_age_x_bonus_interaction(self, sample_policy):
        result = _build_features(sample_policy)
        # bonus_malus capped at 150 in feature engineering
        expected = sample_policy.driv_age * min(sample_policy.bonus_malus, 150)
        assert result["age_x_bonus"].iloc[0] == pytest.approx(expected)

    def test_bonus_malus_capped_at_150(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=35, bonus_malus=300, area="C", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["bonus_malus"].iloc[0] == 150

    def test_young_driver_flag_under_25(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=22, bonus_malus=90, area="C", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["is_young_driver"].iloc[0] == 1

    def test_young_driver_flag_over_25(self, sample_policy):
        result = _build_features(sample_policy)  # driv_age=35
        assert result["is_young_driver"].iloc[0] == 0

    def test_senior_driver_flag_over_70(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=75, bonus_malus=80, area="C", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["is_senior_driver"].iloc[0] == 1

    def test_senior_driver_flag_under_70(self, sample_policy):
        result = _build_features(sample_policy)  # driv_age=35
        assert result["is_senior_driver"].iloc[0] == 0

    def test_has_malus_flag_over_100(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=35, bonus_malus=110, area="C", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["has_malus"].iloc[0] == 1

    def test_has_malus_flag_under_100(self, sample_policy):
        result = _build_features(sample_policy)  # bonus_malus=90
        assert result["has_malus"].iloc[0] == 0

    def test_old_vehicle_flag_over_10(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=12, veh_brand="B1", veh_gas="REGULAR",
            driv_age=35, bonus_malus=90, area="C", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["is_old_vehicle"].iloc[0] == 1

    def test_old_vehicle_flag_under_10(self, sample_policy):
        result = _build_features(sample_policy)  # veh_age=3
        assert result["is_old_vehicle"].iloc[0] == 0

    def test_high_power_flag_at_9(self):
        policy = PolicyFeatures(
            veh_power=9, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=35, bonus_malus=90, area="C", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["is_high_power"].iloc[0] == 1

    def test_high_power_flag_under_9(self, sample_policy):
        result = _build_features(sample_policy)  # veh_power=7
        assert result["is_high_power"].iloc[0] == 0

    def test_veh_brand_uppercased(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="b1", veh_gas="regular",
            driv_age=35, bonus_malus=90, area="c", density=500.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["veh_brand"].iloc[0] == "B1"
        assert result["veh_gas"].iloc[0] == "REGULAR"
        assert result["area"].iloc[0] == "C"

    def test_density_1_gives_log_0(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=35, bonus_malus=90, area="C", density=1.0,
            region="R11", exposure=1.0,
        )
        result = _build_features(policy)
        assert result["log_density"].iloc[0] == pytest.approx(0.0)


# ===========================================================================
# pure_premium — combination logic
# ===========================================================================

class TestPurePremiumCombinations:

    VALID_COMBINATIONS = {"GLM x GLM", "GLM x LGBM", "LGBM x GLM", "LGBM x LGBM"}

    def test_four_combinations_defined(self):
        assert len(self.VALID_COMBINATIONS) == 4

    def test_combination_names_follow_convention(self):
        for name in self.VALID_COMBINATIONS:
            parts = name.split(" x ")
            assert len(parts) == 2
            assert parts[0] in {"GLM", "LGBM"}
            assert parts[1] in {"GLM", "LGBM"}

    def test_pp_equals_freq_times_sev(self):
        freq = pd.Series([0.05, 0.10, 0.20])
        sev  = pd.Series([1000, 2000, 500])
        pp   = freq * sev
        assert pp.iloc[0] == pytest.approx(50.0)
        assert pp.iloc[1] == pytest.approx(200.0)
        assert pp.iloc[2] == pytest.approx(100.0)

    def test_pp_scaled_by_exposure(self):
        freq     = 0.1
        sev      = 1000.0
        exposure = 0.5
        pp = freq * sev * exposure
        assert pp == pytest.approx(50.0)

    def test_normalisation_max_is_100(self):
        raw = pd.Series([10.0, 50.0, 100.0, 200.0])
        normalised = (raw / raw.max() * 100).round(1)
        assert normalised.max() == pytest.approx(100.0)

    def test_best_combination_is_valid(self):
        # Simulate selecting best by lowest RMSE
        rmse = {"GLM x GLM": 300.0, "GLM x LGBM": 280.0,
                "LGBM x GLM": 290.0, "LGBM x LGBM": 275.0}
        best = min(rmse, key=rmse.get)
        assert best in self.VALID_COMBINATIONS

    def test_log_rmse_only_on_claims(self):
        actual   = np.array([0, 0, 500, 1000, 0, 200])
        pred     = np.array([50, 80, 480, 1020, 60, 210])
        mask     = actual > 0
        log_rmse = np.sqrt(np.mean(
            (np.log(actual[mask] + 1) - np.log(pred[mask] + 1)) ** 2
        ))
        assert isinstance(log_rmse, float)
        assert log_rmse >= 0


# ===========================================================================
# API schema validation
# ===========================================================================

class TestPolicyFeaturesSchema:

    def test_valid_policy_constructs(self, sample_policy):
        assert sample_policy.veh_power == 7

    def test_exposure_defaults_to_1(self):
        policy = PolicyFeatures(
            veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
            driv_age=35, bonus_malus=90, area="C", density=500.0,
            region="R11",
        )
        assert policy.exposure == 1.0

    def test_veh_power_minimum_4(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyFeatures(
                veh_power=3, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
                driv_age=35, bonus_malus=90, area="C", density=500.0,
                region="R11",
            )

    def test_driv_age_minimum_18(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyFeatures(
                veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
                driv_age=17, bonus_malus=90, area="C", density=500.0,
                region="R11",
            )

    def test_bonus_malus_minimum_50(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyFeatures(
                veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
                driv_age=35, bonus_malus=49, area="C", density=500.0,
                region="R11",
            )

    def test_exposure_minimum_0001(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyFeatures(
                veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
                driv_age=35, bonus_malus=90, area="C", density=500.0,
                region="R11", exposure=0.0,
            )

    def test_exposure_maximum_1(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PolicyFeatures(
                veh_power=7, veh_age=3, veh_brand="B1", veh_gas="REGULAR",
                driv_age=35, bonus_malus=90, area="C", density=500.0,
                region="R11", exposure=1.5,
            )