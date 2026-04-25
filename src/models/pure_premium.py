"""
src/models/pure_premium.py — Pure premium calculation.

Pure premium = E[frequency] × E[severity]
             = expected claims per year × expected cost per claim

This is the technical price of insuring a policy for one year.
It represents what the insurer expects to pay out on average,
before adding loadings for expenses, profit, and risk margin.

We compute four versions for comparison:
  1. GLM x GLM        — fully interpretable, industry standard
  2. GLM x LGBM       — hybrid: interpretable frequency, flexible severity
  3. LGBM x GLM       — hybrid: flexible frequency, interpretable severity
  4. LGBM x LGBM      — fully flexible challenger

The best combination for this dataset is reported.

Run:
    python src/models/pure_premium.py
"""

import sys
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DB_PATH, TABLE_FREQ_FEATURES, TABLE_SEV_FEATURES

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FREQ_MODEL_PATH = ARTIFACTS_DIR / "freq_model.pkl"
SEV_MODEL_PATH  = ARTIFACTS_DIR / "sev_model.pkl"

FREQ_FEATURE_COLS = [
    "veh_power", "veh_age", "veh_brand", "veh_gas",
    "driv_age", "bonus_malus", "age_x_bonus",
    "area", "log_density", "region",
    "is_young_driver", "is_senior_driver", "has_malus",
    "is_old_vehicle", "is_high_power",
]
SEV_FEATURE_COLS = FREQ_FEATURE_COLS  # same features for both models


# ── Load models ───────────────────────────────────────────────────────────────

def load_models() -> tuple:
    for path in [FREQ_MODEL_PATH, SEV_MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run frequency.py and severity.py first."
            )
    with open(FREQ_MODEL_PATH, "rb") as f:
        freq = pickle.load(f)
    with open(SEV_MODEL_PATH, "rb") as f:
        sev = pickle.load(f)
    print("Models loaded.")
    return freq, sev


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load frequency table — all policies.
    Pure premium is computed for every policy regardless of claim history.
    """
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df  = con.execute(f"SELECT * FROM {TABLE_FREQ_FEATURES}").df()
    con.close()
    print(f"Loaded {len(df):,} policies for pure premium computation")
    return df


# ── Predict frequency ─────────────────────────────────────────────────────────

def predict_frequency(freq_payload: dict, df: pd.DataFrame) -> pd.Series:
    """Predict expected claim count per policy per year."""
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    num_cols = ["veh_power", "veh_age", "driv_age", "bonus_malus",
                "age_x_bonus", "log_density", "is_young_driver",
                "is_senior_driver", "has_malus", "is_old_vehicle",
                "is_high_power"]

    # GLM frequency
    X_glm = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
    X_glm = sm.add_constant(X_glm.astype(float))
    X_glm = X_glm.reindex(
        columns=freq_payload["glm_feature_cols"], fill_value=0
    )
    # Predict at exposure=1 (full year) for pure premium
    offset_full_year = pd.Series(np.zeros(len(df)))
    freq_glm = freq_payload["glm"].predict(X_glm, offset=offset_full_year)

    # LGBM frequency — predict rate (claims/year), already per unit exposure
    enc = freq_payload["encoder"]
    df_enc = df.copy()
    df_enc[cat_cols] = enc.transform(df_enc[cat_cols])
    freq_lgbm = pd.Series(
        freq_payload["lgbm"].predict(df_enc[FREQ_FEATURE_COLS]),
        index=df.index,
    )

    return freq_glm, freq_lgbm


# ── Predict severity ──────────────────────────────────────────────────────────

def predict_severity(sev_payload: dict, df: pd.DataFrame) -> pd.Series:
    """Predict expected cost per claim for every policy."""
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    num_cols = ["veh_power", "veh_age", "driv_age", "bonus_malus",
                "age_x_bonus", "log_density", "is_young_driver",
                "is_senior_driver", "has_malus", "is_old_vehicle",
                "is_high_power"]

    # GLM severity
    X_glm = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
    X_glm = sm.add_constant(X_glm.astype(float))
    X_glm = X_glm.reindex(
        columns=sev_payload["glm_feature_cols"], fill_value=0
    )
    sev_glm = sev_payload["glm"].predict(X_glm)

    # LGBM severity
    enc = sev_payload["encoder"]
    df_enc = df.copy()
    df_enc[cat_cols] = enc.transform(df_enc[cat_cols])
    sev_lgbm = pd.Series(
        sev_payload["lgbm"].predict(df_enc[SEV_FEATURE_COLS]),
        index=df.index,
    )

    return sev_glm, sev_lgbm


# ── Evaluate pure premium ─────────────────────────────────────────────────────

def evaluate_pure_premium(
    df: pd.DataFrame,
    freq_glm: pd.Series,
    freq_lgbm: pd.Series,
    sev_glm: pd.Series,
    sev_lgbm: pd.Series,
) -> pd.DataFrame:
    """
    Compute four pure premium combinations and evaluate against
    actual observed loss cost (claim_nb * avg_claim would be ideal,
    but we use total_stakes proxy: actual_loss = claim_nb / exposure
    as a rate for comparison).

    We use a held-out 20% split for honest evaluation.
    """
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df = val_df.reset_index(drop=True)

    # Reload freq/sev predictions aligned to val_df
    _, freq_glm_val   = train_test_split(freq_glm,  test_size=0.2, random_state=42)
    _, freq_lgbm_val  = train_test_split(freq_lgbm, test_size=0.2, random_state=42)
    _, sev_glm_val    = train_test_split(sev_glm,   test_size=0.2, random_state=42)
    _, sev_lgbm_val   = train_test_split(sev_lgbm,  test_size=0.2, random_state=42)

    freq_glm_val  = freq_glm_val.reset_index(drop=True)
    freq_lgbm_val = freq_lgbm_val.reset_index(drop=True)
    sev_glm_val   = sev_glm_val.reset_index(drop=True)
    sev_lgbm_val  = sev_lgbm_val.reset_index(drop=True)

    # Load actual claim amounts
    con = duckdb.connect(str(DB_PATH), read_only=True)
    sev_df = con.execute(f"""
        SELECT policy_id, total_claim_amount
        FROM {TABLE_SEV_FEATURES}
    """).df()
    con.close()

    df_eval = val_df.merge(sev_df, on="policy_id", how="left")
    df_eval["total_claim_amount"] = df_eval["total_claim_amount"].fillna(0)
    df_eval["actual_loss_cost"] = (
        df_eval["total_claim_amount"] / df_eval["exposure"]
    )
    df_eval = df_eval.reset_index(drop=True)

    results = {}
    combinations = {
        "GLM x GLM":   freq_glm_val  * sev_glm_val,
        "GLM x LGBM":  freq_glm_val  * sev_lgbm_val,
        "LGBM x GLM":  freq_lgbm_val * sev_glm_val,
        "LGBM x LGBM": freq_lgbm_val * sev_lgbm_val,
    }

    print("\n── Pure premium comparison ───────────────────────────")
    print(f"  {'Combination':<15} {'Mean PP':>10} {'RMSE':>12} {'log-RMSE':>10}")
    print(f"  {'-'*50}")

    actual = df_eval["actual_loss_cost"].values
    best_name, best_rmse = None, np.inf

    for name, pp in combinations.items():
        pp_vals = pp.values
        rmse = np.sqrt(mean_squared_error(actual, pp_vals))

        # log-RMSE only on policies with actual claims
        mask = actual > 0
        log_rmse = np.sqrt(mean_squared_error(
            np.log(actual[mask] + 1),
            np.log(pp_vals[mask] + 1)
        ))
        mean_pp = pp_vals.mean()

        print(f"  {name:<15} {mean_pp:>10.2f} {rmse:>12.4f} {log_rmse:>10.4f}")
        results[name] = {"mean_pp": mean_pp, "rmse": rmse, "log_rmse": log_rmse}

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name

    print(f"\n  Best combination: {best_name}")
    return pd.DataFrame(results).T, combinations, df_eval


# ── Distribution plot ─────────────────────────────────────────────────────────

def plot_distribution(
    df_eval: pd.DataFrame,
    best_pp: pd.Series,
    best_name: str,
) -> None:
    """Compare predicted vs actual pure premium distribution."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: distribution of pure premium by risk segment
    axes[0].hist(
        np.log1p(best_pp.values), bins=50,
        color="steelblue", alpha=0.7, label="Predicted PP"
    )
    axes[0].set_title(f"Predicted Pure Premium Distribution\n({best_name})")
    axes[0].set_xlabel("log(1 + Pure Premium) €")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Right: predicted vs actual (claims-only for visibility)
    mask = df_eval["actual_loss_cost"] > 0
    axes[1].scatter(
        np.log1p(df_eval.loc[mask, "actual_loss_cost"]),
        np.log1p(best_pp[mask]),
        alpha=0.2, s=5, color="coral",
    )
    max_val = max(
        np.log1p(df_eval.loc[mask, "actual_loss_cost"]).max(),
        np.log1p(best_pp[mask]).max(),
    )
    axes[1].plot([0, max_val], [0, max_val], "k--", lw=1, label="Perfect fit")
    axes[1].set_title("Predicted vs Actual Loss Cost\n(claims-only, log scale)")
    axes[1].set_xlabel("log(1 + Actual) €")
    axes[1].set_ylabel("log(1 + Predicted) €")
    axes[1].legend()

    plt.tight_layout()
    path = ARTIFACTS_DIR / "pure_premium.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {path}")


# ── Save final combined model ─────────────────────────────────────────────────

def save_combined(
    freq_payload: dict,
    sev_payload: dict,
    best_name: str,
) -> None:
    """Save a single combined payload for the API."""
    payload = {
        "freq": freq_payload,
        "sev": sev_payload,
        "best_combination": best_name,
        "freq_feature_cols": FREQ_FEATURE_COLS,
        "sev_feature_cols": SEV_FEATURE_COLS,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / "pure_premium_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Combined model saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Pure Premium = Frequency × Severity")
    print("=" * 55)

    freq_payload, sev_payload = load_models()
    df = load_data()

    print("\nPredicting frequency...")
    freq_glm, freq_lgbm = predict_frequency(freq_payload, df)

    print("Predicting severity...")
    sev_glm, sev_lgbm = predict_severity(sev_payload, df)

    print("\nDescriptive stats (full portfolio):")
    for name, pp in [
        ("GLM x GLM",   freq_glm  * sev_glm),
        ("LGBM x LGBM", freq_lgbm * sev_lgbm),
    ]:
        print(f"  {name}: mean=€{pp.mean():.2f}  "
              f"p25=€{pp.quantile(0.25):.2f}  "
              f"p75=€{pp.quantile(0.75):.2f}  "
              f"p99=€{pp.quantile(0.99):.2f}")

    results_df, combinations, df_eval = evaluate_pure_premium(
        df, freq_glm, freq_lgbm, sev_glm, sev_lgbm
    )

    # Get best combination
    best_name = results_df["rmse"].astype(float).idxmin()
    best_pp   = combinations[best_name].reset_index(drop=True)

    plot_distribution(df_eval, best_pp, best_name)
    save_combined(freq_payload, sev_payload, best_name)

    print("\n" + "=" * 55)
    print("Done.")
    print("=" * 55)