"""
tests/test_pipeline.py
-----------------------
Tests for the Prefect pipeline structure.

Verifies the flow and task definitions are correct without
running the full training pipeline (which requires data + GPU time).
All heavy tasks are mocked.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineStructure:

    def test_pipeline_module_importable(self):
        import pipeline
        assert hasattr(pipeline, "insurance_pipeline")

    def test_flow_is_callable(self):
        from pipeline import insurance_pipeline
        assert callable(insurance_pipeline)

    def test_all_tasks_defined(self):
        from pipeline import (
            task_ingest, task_dbt_run, task_dbt_test,
            task_train_frequency, task_train_severity,
            task_pure_premium,
        )
        for task_fn in [task_ingest, task_dbt_run, task_dbt_test,
                        task_train_frequency, task_train_severity,
                        task_pure_premium]:
            assert callable(task_fn)

    def test_flow_has_correct_name(self):
        from pipeline import insurance_pipeline
        assert insurance_pipeline.name == "insurance-pricing-pipeline"

    def test_ingest_task_has_retries(self):
        from pipeline import task_ingest
        assert task_ingest.retries == 2

    def test_frequency_task_has_retries(self):
        from pipeline import task_train_frequency
        assert task_train_frequency.retries == 1

    def test_severity_task_has_retries(self):
        from pipeline import task_train_severity
        assert task_train_severity.retries == 1

    def test_dbt_test_task_no_retries(self):
        # dbt_test should fail fast — no retries
        from pipeline import task_dbt_test
        assert task_dbt_test.retries == 0

    def test_flow_accepts_skip_ingest_flag(self):
        import inspect
        from pipeline import insurance_pipeline
        sig = inspect.signature(insurance_pipeline.fn)
        assert "skip_ingest" in sig.parameters

    def test_flow_accepts_skip_training_flag(self):
        import inspect
        from pipeline import insurance_pipeline
        sig = inspect.signature(insurance_pipeline.fn)
        assert "skip_training" in sig.parameters

    def test_flow_skip_flags_default_false(self):
        import inspect
        from pipeline import insurance_pipeline
        sig = inspect.signature(insurance_pipeline.fn)
        assert sig.parameters["skip_ingest"].default is False
        assert sig.parameters["skip_training"].default is False


class TestPipelineRun:

    def test_pipeline_runs_with_all_skipped(self):
        """
        Run the full flow with skip_ingest=True and mock dbt + training.
        Verifies the flow orchestration logic without any real computation.
        """
        with patch("pipeline.task_ingest") as mock_ingest, \
             patch("pipeline.task_dbt_run") as mock_dbt_run, \
             patch("pipeline.task_dbt_test") as mock_dbt_test, \
             patch("pipeline.task_train_frequency") as mock_freq, \
             patch("pipeline.task_train_severity") as mock_sev, \
             patch("pipeline.task_pure_premium") as mock_pp:

            mock_dbt_run.return_value = {"elapsed_s": 1.0}
            mock_dbt_test.return_value = {"elapsed_s": 0.5}
            mock_freq.return_value = {
                "glm_rmse": 0.1, "lgbm_rmse": 0.09,
                "glm_deviance": 100.0, "lgbm_deviance": 90.0,
                "elapsed_s": 30.0
            }
            mock_sev.return_value = {
                "glm_log_rmse": 0.5, "lgbm_log_rmse": 0.45,
                "glm_deviance": 200.0, "lgbm_deviance": 180.0,
                "elapsed_s": 25.0
            }
            mock_pp.return_value = {
                "best_combination": "GLM x LGBM",
                "best_rmse": 280.0,
                "elapsed_s": 5.0,
            }

            from pipeline import insurance_pipeline
            result = insurance_pipeline(skip_ingest=True, skip_training=False)

            assert isinstance(result, dict)
            assert "total_elapsed_s" in result
            assert result["total_elapsed_s"] >= 0

    def test_pipeline_skip_training_skips_model_tasks(self):
        """When skip_training=True, model tasks must not be called."""
        with patch("pipeline.task_ingest") as mock_ingest, \
             patch("pipeline.task_dbt_run") as mock_dbt_run, \
             patch("pipeline.task_dbt_test") as mock_dbt_test, \
             patch("pipeline.task_train_frequency") as mock_freq, \
             patch("pipeline.task_train_severity") as mock_sev, \
             patch("pipeline.task_pure_premium") as mock_pp:

            mock_dbt_run.return_value = {"elapsed_s": 1.0}
            mock_dbt_test.return_value = {"elapsed_s": 0.5}

            from pipeline import insurance_pipeline
            result = insurance_pipeline(skip_ingest=True, skip_training=True)

            mock_freq.assert_not_called()
            mock_sev.assert_not_called()
            mock_pp.assert_not_called()

    def test_pipeline_returns_summary_dict(self):
        with patch("pipeline.task_ingest") as mock_ingest, \
             patch("pipeline.task_dbt_run") as mock_dbt_run, \
             patch("pipeline.task_dbt_test") as mock_dbt_test, \
             patch("pipeline.task_train_frequency") as mock_freq, \
             patch("pipeline.task_train_severity") as mock_sev, \
             patch("pipeline.task_pure_premium") as mock_pp:

            mock_dbt_run.return_value = {"elapsed_s": 1.0}
            mock_dbt_test.return_value = {"elapsed_s": 0.5}

            from pipeline import insurance_pipeline
            result = insurance_pipeline(skip_ingest=True, skip_training=True)

            assert isinstance(result, dict)
            assert "dbt_run" in result
            assert "dbt_test" in result

    def test_dbt_always_runs_regardless_of_skip_flags(self):
        """dbt must always run — skip flags only affect ingest and training."""
        with patch("pipeline.task_ingest"), \
             patch("pipeline.task_dbt_run") as mock_dbt_run, \
             patch("pipeline.task_dbt_test") as mock_dbt_test, \
             patch("pipeline.task_train_frequency"), \
             patch("pipeline.task_train_severity"), \
             patch("pipeline.task_pure_premium"):

            mock_dbt_run.return_value = {"elapsed_s": 1.0}
            mock_dbt_test.return_value = {"elapsed_s": 0.5}

            from pipeline import insurance_pipeline
            insurance_pipeline(skip_ingest=True, skip_training=True)

            mock_dbt_run.assert_called_once()
            mock_dbt_test.assert_called_once()