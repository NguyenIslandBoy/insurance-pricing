"""
src/features/spark_features.py
--------------------------------
PySpark Silver feature engineering layer for the insurance pricing pipeline.

Reads modelling-ready tables from DuckDB (produced by dbt Bronze/Intermediate
layer), applies all feature engineering using PySpark DataFrame operations,
and writes partitioned Parquet Silver tables consumed by model training.

Architecture:
    Bronze (DuckDB/dbt):  raw ingestion, ClaimNb fix, joins, cleaning
    Silver (PySpark):     feature engineering — flags, interactions, encodings
    Gold (model files):   training, evaluation, artifact saving

Why PySpark here:
    At 678k policies the feature engineering is tractable in pandas. The
    PySpark implementation demonstrates the pattern that scales to tens of
    millions of policies (e.g. a national insurer's full book) without
    rewriting the logic. Feature pipelines are a primary PySpark use case
    in production insurance and fintech DE environments.

Single responsibility:
    dbt  — SQL transformations: joins, cleaning, ClaimNb fix, filtering
    Spark — feature engineering: binary flags, interactions, log transforms,
             categorical encodings for both GLM (one-hot) and LightGBM (ordinal)
    Model files — load Silver parquet, train, evaluate, save artifacts

Silver outputs (data/silver/):
    freq_features/  — all 678k policies, partitioned by has_claim
    sev_features/   — claims-only subset (~25k rows)

Usage:
    python -m src.features.spark_features
    python -m src.features.spark_features --dry-run
    python -m src.features.spark_features --db path/to/insurance.duckdb
"""

import argparse
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DB_PATH

SILVER_DIR = Path("data/silver")

# ---------------------------------------------------------------------------
# Source table names (dbt mart outputs in DuckDB)
# ---------------------------------------------------------------------------
TABLE_FREQ = "mart_freq_features"
TABLE_SEV  = "mart_sev_features"

# ---------------------------------------------------------------------------
# Categorical columns — encoded two ways:
#   GLM path:     one-hot via pd.get_dummies (done at train time with Silver cols)
#   LightGBM path: ordinal encoding stored as integer columns in Silver
# ---------------------------------------------------------------------------
CAT_COLS = ["veh_brand", "veh_gas", "area", "region"]

# Ordinal encoding maps — consistent across freq and sev
# Unknown values map to -1 (matches OrdinalEncoder unknown_value=-1)
VEH_BRAND_MAP = {
    "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4,
    "B6": 5, "B7": 6, "B8": 7, "B9": 8, "B10": 9,
    "B11": 10, "B12": 11, "B13": 12, "B14": 13,
}
VEH_GAS_MAP  = {"DIESEL": 0, "REGULAR": 1}
AREA_MAP     = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
REGION_MAP   = {
    "R11": 0, "R21": 1, "R22": 2, "R23": 3, "R24": 4,
    "R25": 5, "R26": 6, "R31": 7, "R41": 8, "R42": 9,
    "R43": 10, "R52": 11, "R53": 12, "R54": 13, "R72": 14,
    "R73": 15, "R74": 16, "R82": 17, "R83": 18, "R91": 19,
    "R93": 20, "R94": 21,
}


# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------

def get_spark(app_name: str = "Insurance_Silver_Features") -> SparkSession:
    """
    Local SparkSession for single-machine feature engineering.
    On Databricks: remove .master() — cluster handles it.
    shuffle.partitions=8: proportional to 678k rows; default 200 is for 100GB+.
    """
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Bronze read: DuckDB → pandas → Spark
# ---------------------------------------------------------------------------

def load_bronze(
    spark: SparkSession,
    db_path: Path,
    table: str,
) -> DataFrame:
    """
    Load a dbt mart table from DuckDB into a Spark DataFrame.

    DuckDB → pandas → Spark is the correct bridge pattern.
    Direct JDBC to DuckDB is not production-stable.
    Arrow-based conversion avoids pickle serialisation issues on Windows.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB not found at {db_path}. "
            "Run loader.py and dbt first."
        )
    con = duckdb.connect(str(db_path), read_only=True)
    pdf = con.execute(f"SELECT * FROM {table}").df()
    con.close()

    # Ensure clean dtypes before handing to Spark via Arrow
    int_cols = ["policy_id", "claim_nb", "veh_power", "veh_age",
                "driv_age", "bonus_malus", "has_claim", "has_large_claim",
                "claimnb_corrupted", "n_large_claims"]
    float_cols = ["exposure", "log_density", "avg_claim_amount",
                  "total_claim_amount", "n_large_claims", "log_exposure"]

    for col in int_cols:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype("int32")
    for col in float_cols:
        if col in pdf.columns:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype("float64")

    return spark.createDataFrame(pdf)


# ---------------------------------------------------------------------------
# Feature engineering transforms
# All functions: DataFrame in → DataFrame out (pure, composable)
# ---------------------------------------------------------------------------

def build_driver_flags(df: DataFrame) -> DataFrame:
    """
    Binary driver risk flags.
    Thresholds match actuarial literature and original dbt mart definitions.
    """
    return (
        df
        .withColumn(
            "is_young_driver",
            F.when(F.col("driv_age") < 25, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "is_senior_driver",
            F.when(F.col("driv_age") > 70, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "has_malus",
            F.when(F.col("bonus_malus") > 100, F.lit(1)).otherwise(F.lit(0))
        )
    )


def build_vehicle_flags(df: DataFrame) -> DataFrame:
    """Binary vehicle risk flags."""
    return (
        df
        .withColumn(
            "is_old_vehicle",
            F.when(F.col("veh_age") > 10, F.lit(1)).otherwise(F.lit(0))
        )
        .withColumn(
            "is_high_power",
            F.when(F.col("veh_power") >= 9, F.lit(1)).otherwise(F.lit(0))
        )
    )


def build_interactions(df: DataFrame) -> DataFrame:
    """
    Feature interactions.
    age_x_bonus: correlated but the model benefits from the explicit cross-term.
    bonus_malus is capped at 150 upstream in dbt (stg_policies) before this runs.
    """
    return df.withColumn(
        "age_x_bonus",
        F.col("driv_age") * F.col("bonus_malus")
    )


def build_log_transforms(df: DataFrame) -> DataFrame:
    """
    Log transforms for right-skewed numeric features.
    log_exposure: mandatory offset for Poisson GLM (log of observation period).
    log_density: stabilises right-skewed population density.
    Both already present in dbt mart as pass-throughs — recomputed here
    so Silver is self-contained and dbt marts become pure join/filter layers.
    """
    df = df.withColumn(
        "log_density",
        F.log(F.greatest(F.col("log_density").cast(DoubleType()), F.lit(1.0)))
    ) if "log_density" in df.columns else df

    if "exposure" in df.columns:
        df = df.withColumn(
            "log_exposure",
            F.log(F.col("exposure").cast(DoubleType()))
        )
    return df


def build_ordinal_encodings(df: DataFrame) -> DataFrame:
    """
    Ordinal encode categorical columns for LightGBM.
    Stored as integer columns with _enc suffix.
    Unknown values → -1 (consistent with OrdinalEncoder unknown_value=-1).

    GLM one-hot encoding is NOT done here — pd.get_dummies at train time
    is correct for GLM because the dummy columns depend on the training
    data categories seen, which must be consistent between train and val.
    """
    brand_map  = F.create_map([F.lit(k) for pair in VEH_BRAND_MAP.items()  for k in pair])
    gas_map    = F.create_map([F.lit(k) for pair in VEH_GAS_MAP.items()    for k in pair])
    area_map   = F.create_map([F.lit(k) for pair in AREA_MAP.items()       for k in pair])
    region_map = F.create_map([F.lit(k) for pair in REGION_MAP.items()     for k in pair])

    return (
        df
        .withColumn(
            "veh_brand_enc",
            F.coalesce(brand_map[F.col("veh_brand")], F.lit(-1))
        )
        .withColumn(
            "veh_gas_enc",
            F.coalesce(gas_map[F.col("veh_gas")], F.lit(-1))
        )
        .withColumn(
            "area_enc",
            F.coalesce(area_map[F.col("area")], F.lit(-1))
        )
        .withColumn(
            "region_enc",
            F.coalesce(region_map[F.col("region")], F.lit(-1))
        )
    )


def apply_freq_transforms(df: DataFrame) -> DataFrame:
    """
    Full feature engineering chain for the frequency table.
    Applied in dependency order: flags → interactions → log → encodings.
    """
    df = build_driver_flags(df)
    df = build_vehicle_flags(df)
    df = build_interactions(df)
    df = build_log_transforms(df)
    df = build_ordinal_encodings(df)
    df = df.withColumn(
        "has_claim",
        F.when(F.col("claim_nb") > 0, F.lit(1)).otherwise(F.lit(0))
    )
    return df


def apply_sev_transforms(df: DataFrame) -> DataFrame:
    """
    Full feature engineering chain for the severity table.
    Same transforms as frequency — severity uses identical feature set.
    No log_exposure needed (severity model has no exposure offset).
    """
    df = build_driver_flags(df)
    df = build_vehicle_flags(df)
    df = build_interactions(df)
    df = build_log_transforms(df)
    df = build_ordinal_encodings(df)
    return df


# ---------------------------------------------------------------------------
# Write Silver layer (pyarrow — avoids winutils on Windows)
# ---------------------------------------------------------------------------

def write_silver(
    df: DataFrame,
    name: str,
    silver_dir: Path,
    partition_col: str,
    dry_run: bool,
) -> int:
    """
    Write a Spark DataFrame to the Silver parquet layer via pyarrow.

    Spark handles all transformation logic.
    pyarrow handles file write — avoids winutils requirement on Windows.
    Partitioned by partition_col for downstream read efficiency.
    """
    out_path = silver_dir / name
    count = df.count()

    if dry_run:
        print(f"  [DRY RUN] {name}: {count:,} rows → {out_path}/")
        return count

    pdf = df.toPandas()
    out_path.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(pdf, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(out_path),
        partition_cols=[partition_col],
        compression="snappy",
        existing_data_behavior="delete_matching",
    )
    print(f"  ✓ {name:<25} {count:>10,} rows → {out_path}/")
    return count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    db_path: Path = DB_PATH,
    silver_dir: Path = SILVER_DIR,
    dry_run: bool = False,
) -> dict:
    """
    Execute the Bronze → Silver feature engineering pipeline.
    Returns {table_name: row_count} for testability.
    """
    silver_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Insurance Pricing — PySpark Silver Feature Engineering")
    print(f"Source:   {db_path}")
    print(f"Output:   {silver_dir}/")
    print(f"Dry run:  {dry_run}")
    print("=" * 60)

    t0 = time.perf_counter()
    spark = get_spark()

    # ── Frequency features ────────────────────────────────────────────────
    print(f"\n[1/4] Loading Bronze frequency table ({TABLE_FREQ})...")
    freq_bronze = load_bronze(spark, db_path, TABLE_FREQ)
    print(f"      {freq_bronze.count():,} rows loaded")

    print("\n[2/4] Applying frequency feature transforms...")
    freq_silver = apply_freq_transforms(freq_bronze)

    # ── Severity features ─────────────────────────────────────────────────
    print(f"\n[3/4] Loading Bronze severity table ({TABLE_SEV})...")
    sev_bronze = load_bronze(spark, db_path, TABLE_SEV)
    print(f"      {sev_bronze.count():,} rows loaded")

    print("\n[4/4] Applying severity feature transforms + writing Silver...")
    sev_silver = apply_sev_transforms(sev_bronze)

    # ── Write Silver ──────────────────────────────────────────────────────
    # This is for debugging since "KeyError: 'Field "has_claim" does not exist in schema'"
    # print("\nFreq silver columns:", freq_silver.columns)
    print("\nWriting Silver layer...")
    results = {}
    results["freq_features"] = write_silver(
        freq_silver, "freq_features", silver_dir,
        partition_col="has_claim", dry_run=dry_run,
    )
    results["sev_features"] = write_silver(
        sev_silver, "sev_features", silver_dir,
        partition_col="has_large_claim", dry_run=dry_run,
    )

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\n{'=' * 60}")
    print(f"Silver feature layer complete in {elapsed}s")
    for name, count in results.items():
        print(f"  {name:<25} {count:,} rows")
    print("=" * 60)

    spark.stop()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PySpark Silver feature engineering for insurance pricing"
    )
    parser.add_argument("--db",      default=str(DB_PATH),    help="Path to insurance.duckdb")
    parser.add_argument("--silver",  default=str(SILVER_DIR), help="Silver output directory")
    parser.add_argument("--dry-run", action="store_true",     help="Count rows without writing")
    args = parser.parse_args()

    run(
        db_path=Path(args.db),
        silver_dir=Path(args.silver),
        dry_run=args.dry_run,
    )