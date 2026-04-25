"""
config.py — Central configuration for the insurance pricing pipeline.
"""

from pathlib import Path

ROOT = Path(__file__).parent

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

RAW_FREQ = DATA_RAW / "freMTPL2freq.csv"
RAW_SEV  = DATA_RAW / "freMTPL2sev.csv"

# ── DuckDB ────────────────────────────────────────────────────────────────────
DB_PATH = DATA_PROCESSED / "insurance.duckdb"

# ── Table names ───────────────────────────────────────────────────────────────
# Raw ingestion
TABLE_RAW_FREQ = "raw_freq"
TABLE_RAW_SEV  = "raw_sev"

# dbt produces these (marts layer — what models consume)
TABLE_FREQ_FEATURES = "mart_freq_features"
TABLE_SEV_FEATURES  = "mart_sev_features"

# ── Known data quality issue (from Wüthrich & Loser) ─────────────────────────
# Policies with IDpol <= 24500 have ClaimNb entries that don't match
# severity records — their claim counts must be recomputed from freMTPL2sev.
CORRUPT_POLICY_ID_THRESHOLD = 24500

# ── Column definitions ────────────────────────────────────────────────────────
POLICY_ID_COL = "IDpol"
CLAIM_NB_COL  = "ClaimNb"
EXPOSURE_COL  = "Exposure"
CLAIM_AMT_COL = "ClaimAmount"

# Categorical features — need encoding before modelling
CAT_COLS = ["VehBrand", "VehGas", "Area", "Region"]

# Ordinal features — keep as numeric but treat carefully
ORD_COLS = ["VehPower"]

# Numeric features
NUM_COLS = ["VehAge", "DrivAge", "BonusMalus", "Density"]

# ── Model output ──────────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "src" / "models" / "artifacts"
FREQ_MODEL_PATH = MODELS_DIR / "freq_model.pkl"
SEV_MODEL_PATH  = MODELS_DIR / "sev_model.pkl"

# ── Expected row counts ───────────────────────────────────────────────────────
EXPECTED_FREQ_ROWS = 678_013
EXPECTED_SEV_ROWS  = 26_444