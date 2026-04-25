"""
src/ingest/loader.py — Load raw CSVs into DuckDB.

Handles the known ClaimNb corruption for IDpol <= 24500 by recomputing
claim counts from freMTPL2sev rather than trusting freMTPL2freq.

Run:
    python src/ingest/loader.py
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    RAW_FREQ, RAW_SEV, DB_PATH,
    TABLE_RAW_FREQ, TABLE_RAW_SEV,
    POLICY_ID_COL, CLAIM_NB_COL, EXPOSURE_COL,
    CORRUPT_POLICY_ID_THRESHOLD,
    EXPECTED_FREQ_ROWS, EXPECTED_SEV_ROWS,
)


def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def load_freq() -> pd.DataFrame:
    """
    Load freMTPL2freq.csv and fix the ClaimNb corruption.

    For policies with IDpol <= 24500, claim counts in freMTPL2freq are
    unreliable. We zero them out here — they will be recomputed from
    freMTPL2sev in the dbt intermediate layer.
    """
    if not RAW_FREQ.exists():
        raise FileNotFoundError(f"Not found: {RAW_FREQ}")

    df = pd.read_csv(RAW_FREQ)
    df.columns = df.columns.str.strip()

    print(f"[freq] raw shape: {df.shape}")
    print(f"       ClaimNb range: {df[CLAIM_NB_COL].min()}–{df[CLAIM_NB_COL].max()}")
    print(f"       Exposure range: {df[EXPOSURE_COL].min():.4f}–{df[EXPOSURE_COL].max():.4f}")

    # Flag corrupt policies
    corrupt_mask = df[POLICY_ID_COL] <= CORRUPT_POLICY_ID_THRESHOLD
    n_corrupt = corrupt_mask.sum()
    print(f"       Corrupt ClaimNb (IDpol <= {CORRUPT_POLICY_ID_THRESHOLD}): {n_corrupt:,} policies")

    # Zero out corrupt claim counts — will be fixed in dbt
    df.loc[corrupt_mask, CLAIM_NB_COL] = 0
    df["claimnb_corrupted"] = corrupt_mask.astype(int)

    return df


def load_sev() -> pd.DataFrame:
    """Load freMTPL2sev.csv — one row per individual claim."""
    if not RAW_SEV.exists():
        raise FileNotFoundError(f"Not found: {RAW_SEV}")

    df = pd.read_csv(RAW_SEV)
    df.columns = df.columns.str.strip()

    print(f"\n[sev]  raw shape: {df.shape}")
    print(f"       ClaimAmount range: {df['ClaimAmount'].min():.2f}–{df['ClaimAmount'].max():.2f}")
    print(f"       Unique policies with claims: {df[POLICY_ID_COL].nunique():,}")

    return df


def write_to_db(df: pd.DataFrame, table_name: str,
                con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"  ✓ {table_name}: {n:,} rows written")


def run():
    print("=" * 55)
    print("Loading raw data → DuckDB")
    print("=" * 55)

    freq = load_freq()
    sev  = load_sev()

    con = get_connection()
    print("\nWriting to DuckDB...")
    write_to_db(freq, TABLE_RAW_FREQ, con)
    write_to_db(sev,  TABLE_RAW_SEV,  con)
    con.close()

    print(f"\nDatabase: {DB_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    run()