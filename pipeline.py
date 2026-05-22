"""
pipeline.py — Prefect orchestration for the insurance pricing pipeline.

Stages:
  1. ingest       — load CSVs → DuckDB (loader.py)
  2. dbt_run      — staging → intermediate → marts
  3. dbt_test     — data quality tests on all dbt models
  4. train_freq   — Poisson GLM + LightGBM frequency model
  5. train_sev    — Gamma GLM + LightGBM severity model
  6. pure_premium — combine models, evaluate 4 combinations, save artifact

Run locally (one-off):
    python pipeline.py

Run with Prefect UI (requires `prefect server start` in a separate terminal):
    prefect server start          # in terminal 1
    python pipeline.py            # in terminal 2
    # then open http://localhost:4200

Schedule (daily at 2am):
    from prefect.schedules import CronSchedule
    # see bottom of this file for deployment example
"""

import subprocess
import sys
import time
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.states import Failed

sys.path.insert(0, str(Path(__file__).parent))

from src.ingest.loader import run as ingest_run
from src.models.frequency import run as freq_run
from src.models.severity import run as sev_run
from src.models.pure_premium import load_models, load_data, predict_frequency
from src.models.pure_premium import predict_severity, evaluate_pure_premium
from src.models.pure_premium import plot_distribution, save_combined


# ===========================================================================
# Tasks
# ===========================================================================

@task(name="ingest", retries=2, retry_delay_seconds=10)
def task_ingest() -> dict:
    """Load raw CSVs into DuckDB. Retries twice on transient IO errors."""
    logger = get_run_logger()
    logger.info("Stage 1 — Ingesting raw CSVs into DuckDB")
    t0 = time.perf_counter()

    ingest_run()

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"Ingest complete in {elapsed}s")
    return {"elapsed_s": elapsed}


@task(name="dbt_run", retries=1, retry_delay_seconds=5)
def task_dbt_run() -> dict:
    """
    Run dbt models: staging → intermediate → marts.
    Uses subprocess so dbt CLI output is fully visible in Prefect logs.
    """
    logger = get_run_logger()
    logger.info("Stage 2 — Running dbt models")
    t0 = time.perf_counter()

    result = subprocess.run(
        ["dbt", "run", "--profiles-dir", str(Path(__file__).parent / "dbt")],
        cwd=str(Path(__file__).parent / "dbt"),
        capture_output=True,
        text=True,
    )

    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"dbt run failed:\n{result.stderr}")

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"dbt run complete in {elapsed}s")
    return {"elapsed_s": elapsed}


@task(name="dbt_test", retries=0)
def task_dbt_test() -> dict:
    """
    Run dbt data quality tests.
    Fails the pipeline if any test fails — enforces data contract.
    """
    logger = get_run_logger()
    logger.info("Stage 3 — Running dbt tests")
    t0 = time.perf_counter()

    result = subprocess.run(
        ["dbt", "test", "--profiles-dir", str(Path(__file__).parent / "dbt")],
        cwd=str(Path(__file__).parent / "dbt"),
        capture_output=True,
        text=True,
    )

    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"dbt test failed:\n{result.stderr}")

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(f"dbt tests passed in {elapsed}s")
    return {"elapsed_s": elapsed}


@task(name="train_frequency", retries=1, retry_delay_seconds=30)
def task_train_frequency() -> dict:
    """Train Poisson GLM + LightGBM frequency model. Saves freq_model.pkl."""
    logger = get_run_logger()
    logger.info("Stage 4 — Training frequency model")
    t0 = time.perf_counter()

    results = freq_run()

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(
        f"Frequency model trained in {elapsed}s | "
        f"GLM RMSE={results['glm_rmse']:.6f} | "
        f"LGBM RMSE={results['lgbm_rmse']:.6f}"
    )
    return {**results, "elapsed_s": elapsed}


@task(name="train_severity", retries=1, retry_delay_seconds=30)
def task_train_severity() -> dict:
    """Train Gamma GLM + LightGBM severity model. Saves sev_model.pkl."""
    logger = get_run_logger()
    logger.info("Stage 5 — Training severity model")
    t0 = time.perf_counter()

    results = sev_run()

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(
        f"Severity model trained in {elapsed}s | "
        f"GLM log-RMSE={results['glm_log_rmse']:.4f} | "
        f"LGBM log-RMSE={results['lgbm_log_rmse']:.4f}"
    )
    return {**results, "elapsed_s": elapsed}


@task(name="pure_premium", retries=0)
def task_pure_premium(
    freq_results: dict,
    sev_results: dict,
) -> dict:
    """
    Combine frequency and severity models into pure premium.
    Evaluates all 4 combinations, saves combined artifact.
    Depends on freq and sev completing first (enforced by arguments).
    """
    logger = get_run_logger()
    logger.info("Stage 6 — Computing pure premium")
    t0 = time.perf_counter()

    freq_payload, sev_payload = load_models()
    df = load_data()

    freq_glm, freq_lgbm = predict_frequency(freq_payload, df)
    sev_glm,  sev_lgbm  = predict_severity(sev_payload,  df)

    results_df, combinations, df_eval = evaluate_pure_premium(
        df, freq_glm, freq_lgbm, sev_glm, sev_lgbm
    )

    best_name = results_df["rmse"].astype(float).idxmin()
    best_pp   = combinations[best_name].reset_index(drop=True)

    plot_distribution(df_eval, best_pp, best_name)
    save_combined(freq_payload, sev_payload, best_name)

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info(
        f"Pure premium complete in {elapsed}s | "
        f"Best: {best_name} | "
        f"RMSE={results_df.loc[best_name, 'rmse']:.4f}"
    )
    return {
        "best_combination": best_name,
        "best_rmse": float(results_df.loc[best_name, "rmse"]),
        "elapsed_s": elapsed,
    }


# ===========================================================================
# Flow
# ===========================================================================

@flow(
    name="insurance-pricing-pipeline",
    description=(
        "End-to-end French MTPL insurance pricing pipeline. "
        "Ingest → dbt → train frequency + severity → pure premium."
    ),
    log_prints=True,
)
def insurance_pipeline(
    skip_ingest: bool = False,
    skip_training: bool = False,
) -> dict:
    """
    Main pipeline flow.

    Args:
        skip_ingest:   Skip CSV ingestion (use existing DuckDB).
        skip_training: Skip model training (use existing model artifacts).

    Returns:
        Summary dict with timing and key metrics per stage.
    """
    logger = get_run_logger()
    logger.info("=" * 55)
    logger.info("Insurance Pricing Pipeline — starting")
    logger.info("=" * 55)

    summary = {}
    pipeline_start = time.perf_counter()

    # Stage 1 — Ingest
    if not skip_ingest:
        ingest_result = task_ingest()
        summary["ingest"] = ingest_result
    else:
        logger.info("Stage 1 — Skipped (skip_ingest=True)")

    # Stages 2 + 3 — dbt (always runs to enforce data contract)
    dbt_result  = task_dbt_run()
    test_result = task_dbt_test(wait_for=[dbt_result])
    summary["dbt_run"]  = dbt_result
    summary["dbt_test"] = test_result

    # Stages 4 + 5 — Train models (can run after dbt_test passes)
    if not skip_training:
        freq_result = task_train_frequency(wait_for=[test_result])
        sev_result  = task_train_severity(wait_for=[test_result])
        summary["frequency"] = freq_result
        summary["severity"]  = sev_result

        # Stage 6 — Pure premium (depends on both models)
        pp_result = task_pure_premium(freq_result, sev_result)
        summary["pure_premium"] = pp_result
    else:
        logger.info("Stages 4-6 — Skipped (skip_training=True)")

    total = round(time.perf_counter() - pipeline_start, 1)
    summary["total_elapsed_s"] = total

    logger.info("=" * 55)
    logger.info(f"Pipeline complete in {total}s")
    for stage, result in summary.items():
        if isinstance(result, dict) and "elapsed_s" in result:
            logger.info(f"  {stage:<20} {result['elapsed_s']}s")
    logger.info("=" * 55)

    return summary


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Insurance pricing pipeline")
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip CSV ingestion (use existing DuckDB)"
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training (use existing artifacts)"
    )
    args = parser.parse_args()

    result = insurance_pipeline(
        skip_ingest=args.skip_ingest,
        skip_training=args.skip_training,
    )
    print("\nSummary:")
    for k, v in result.items():
        print(f"  {k}: {v}")