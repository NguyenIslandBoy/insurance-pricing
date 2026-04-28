"""
scripts/seed_ci.py
------------------
Seeds minimal synthetic data into DuckDB for CI dbt runs.
Creates raw_freq (100 policies) and raw_sev (20 claims).
Run from repo root: python scripts/seed_ci.py
"""

import os
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# Use GITHUB_WORKSPACE in CI, fall back to local path
workspace = os.environ.get("GITHUB_WORKSPACE", str(Path(__file__).parent.parent))
db_path = Path(workspace) / "data" / "processed" / "insurance.duckdb"
db_path.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
n = 100

freq = pd.DataFrame({
    "IDpol":             list(range(1, n + 1)),
    "ClaimNb":           ([0] * 80 + [1] * 15 + [2] * 5),
    "Exposure":          np.random.uniform(0.1, 1.0, n).round(4),
    "VehPower":          np.random.randint(4, 12, n),
    "VehAge":            np.random.randint(0, 20, n),
    "DrivAge":           np.random.randint(18, 80, n),
    "BonusMalus":        np.random.randint(50, 200, n),
    "VehBrand":          np.random.choice(["B1", "B2", "B3", "B12"], n),
    "VehGas":            np.random.choice(["REGULAR", "DIESEL"], n),
    "Area":              np.random.choice(["A", "B", "C", "D", "E", "F"], n),
    "Density":           np.random.randint(50, 5000, n).astype(float),
    "Region":            np.random.choice(["R11", "R24", "R52", "R72"], n),
    "claimnb_corrupted": [0] * n,
})
freq["ClaimNb"] = freq["ClaimNb"].astype(int)

sev = pd.DataFrame({
    "IDpol":       [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                    15, 25, 35, 45, 55, 65, 75, 85, 95, 5],
    "ClaimAmount": np.random.uniform(500, 15000, 20).round(2),
})

con = duckdb.connect(str(db_path))
con.execute("DROP TABLE IF EXISTS raw_freq")
con.execute("DROP TABLE IF EXISTS raw_sev")
con.execute("CREATE TABLE raw_freq AS SELECT * FROM freq")
con.execute("CREATE TABLE raw_sev  AS SELECT * FROM sev")
con.close()

print(f"Seed complete: {db_path}")
print(f"  raw_freq: {n} policies")
print(f"  raw_sev:  {len(sev)} claims")