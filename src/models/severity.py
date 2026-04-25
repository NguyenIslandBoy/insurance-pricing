"""
src/models/severity.py — Claim severity model.

Models the expected cost per claim, given that a claim occurred.
Only the ~4% of policies with at least one claim are used.

Two approaches:
  1. Gamma GLM  — industry standard for right-skewed, positive claim amounts
  2. LightGBM   — challenger, captures non-linearities in claim costs

Key points:
  - Gamma GLM with log link: E[cost] = exp(X*beta)
  - No exposure offset needed — we model cost per claim, not per year
  - Target is avg_claim_amount (capped at p99 to reduce tail distortion)
  - Heavy right tail is normal for insurance — log-transform for diagnostics only

Target: avg_claim_amount
Metric: Gamma deviance (standard for severity), also RMSE on log scale

Run:
    python src/models/severity.py
"""

import sys
import pickle
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DB_PATH, TABLE_SEV_FEATURES

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

SEV_FEATURE_COLS = [
    "veh_power", "veh_age", "veh_brand", "veh_gas",
    "driv_age", "bonus_malus", "age_x_bonus",
    "area", "log_density", "region",
    "is_young_driver", "is_senior_driver", "has_malus",
    "is_old_vehicle", "is_high_power",
]
TARGET = "avg_claim_amount"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"SELECT * FROM {TABLE_SEV_FEATURES}").df()
    con.close()
    print(f"Loaded: {df.shape[0]:,} rows (claims-only subset)")
    print(f"  Mean claim:   €{df[TARGET].mean():,.2f}")
    print(f"  Median claim: €{df[TARGET].median():,.2f}")
    print(f"  Max claim:    €{df[TARGET].max():,.2f}  (capped at p99)")
    return df


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    train, val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train):,}  Val: {len(val):,}")
    return train, val


# ── Gamma GLM ─────────────────────────────────────────────────────────────────

def train_glm(train: pd.DataFrame):
    """
    Gamma GLM with log link.
    Gamma is appropriate for positive, right-skewed continuous amounts.
    Log link ensures predictions are always positive.
    """
    print("\n── Gamma GLM ────────────────────────────────────────")

    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    num_cols = ["veh_power", "veh_age", "driv_age", "bonus_malus",
                "age_x_bonus", "log_density", "is_young_driver",
                "is_senior_driver", "has_malus", "is_old_vehicle",
                "is_high_power"]

    X = pd.get_dummies(train[cat_cols + num_cols], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = train[TARGET]

    model = sm.GLM(
        y, X,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit()

    print(f"  Converged: {model.converged}")
    print(f"  Deviance:  {model.deviance:.4f}")
    print(f"  AIC:       {model.aic:.2f}")
    print(f"  Params:    {len(model.params)}")
    return model, X.columns.tolist()


def evaluate_glm(
    model,
    glm_feature_cols: list,
    val: pd.DataFrame,
) -> dict:
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    num_cols = ["veh_power", "veh_age", "driv_age", "bonus_malus",
                "age_x_bonus", "log_density", "is_young_driver",
                "is_senior_driver", "has_malus", "is_old_vehicle",
                "is_high_power"]

    X_val = pd.get_dummies(val[cat_cols + num_cols], drop_first=True)
    X_val = sm.add_constant(X_val.astype(float))
    X_val = X_val.reindex(columns=glm_feature_cols, fill_value=0)

    y_pred = model.predict(X_val)
    y_true = val[TARGET]

    # RMSE on log scale (standard for severity)
    log_rmse = np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))
    # Gamma deviance: 2 * sum(-log(y/mu) + (y-mu)/mu)
    deviance = 2 * (
        -np.log(y_true / y_pred) + (y_true - y_pred) / y_pred
    ).sum()

    print(f"\n  GLM Val log-RMSE: {log_rmse:.4f}")
    print(f"  GLM Val Deviance: {deviance:.4f}")
    return {"log_rmse": log_rmse, "deviance": deviance, "y_pred": y_pred}


# ── LightGBM ──────────────────────────────────────────────────────────────────

def encode_categoricals(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple:
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train[cat_cols] = enc.fit_transform(df_train[cat_cols])
    df_val[cat_cols]   = enc.transform(df_val[cat_cols])
    return df_train, df_val, enc


def train_lgbm(train: pd.DataFrame, val: pd.DataFrame):
    """
    LightGBM with Gamma regression objective.
    Models log(claim_amount) implicitly via the Gamma loss.
    """
    print("\n── LightGBM Gamma ───────────────────────────────────")

    train_enc, val_enc, enc = encode_categoricals(train, val)

    X_train = train_enc[SEV_FEATURE_COLS]
    y_train = train_enc[TARGET]
    X_val   = val_enc[SEV_FEATURE_COLS]
    y_val   = val_enc[TARGET]

    model = lgb.LGBMRegressor(
        objective="gamma",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="gamma",
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"  Best iteration: {model.best_iteration_}")
    return model, enc


def evaluate_lgbm(
    model: lgb.LGBMRegressor,
    enc: OrdinalEncoder,
    val: pd.DataFrame,
) -> dict:
    val_enc = val.copy()
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    val_enc[cat_cols] = enc.transform(val_enc[cat_cols])

    X_val  = val_enc[SEV_FEATURE_COLS]
    y_pred = model.predict(X_val)
    y_true = val[TARGET]

    log_rmse = np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))
    deviance = 2 * (
        -np.log(y_true / y_pred) + (y_true - y_pred) / y_pred
    ).sum()

    print(f"\n  LGBM Val log-RMSE: {log_rmse:.4f}")
    print(f"  LGBM Val Deviance: {deviance:.4f}")
    return {"log_rmse": log_rmse, "deviance": deviance, "y_pred": y_pred}


# ── SHAP ──────────────────────────────────────────────────────────────────────

def shap_importance(
    model: lgb.LGBMRegressor,
    enc: OrdinalEncoder,
    val: pd.DataFrame,
) -> None:
    print("\n── SHAP (LightGBM) ──────────────────────────────────")
    val_enc = val.copy()
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    val_enc[cat_cols] = enc.transform(val_enc[cat_cols])

    X_val = val_enc[SEV_FEATURE_COLS].sample(
        min(2000, len(val_enc)), random_state=42
    )
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=SEV_FEATURE_COLS,
    ).sort_values(ascending=False)

    print("\n  Top features by mean |SHAP|:")
    for feat, val_s in mean_shap.items():
        bar = "█" * int(val_s / mean_shap.iloc[0] * 20)
        print(f"  {feat:<30} {val_s:.4f}  {bar}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_shap.sort_values().plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Severity Model — Mean |SHAP| Value")
    ax.set_xlabel("Mean |SHAP|")
    plt.tight_layout()
    path = ARTIFACTS_DIR / "sev_shap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  SHAP plot saved: {path}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save(
    lgbm_model: lgb.LGBMRegressor,
    glm_model,
    enc: OrdinalEncoder,
    glm_feature_cols: list,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "lgbm": lgbm_model,
        "glm": glm_model,
        "encoder": enc,
        "glm_feature_cols": glm_feature_cols,
        "feature_cols": SEV_FEATURE_COLS,
    }
    path = ARTIFACTS_DIR / "sev_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Severity model saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Severity model (Gamma GLM + LightGBM)")
    print("=" * 55)

    df = load_data()
    train, val = split_data(df)

    glm_model, glm_cols = train_glm(train)
    glm_results = evaluate_glm(glm_model, glm_cols, val)

    lgbm_model, enc = train_lgbm(train, val)
    lgbm_results = evaluate_lgbm(lgbm_model, enc, val)

    print("\n── Model comparison ─────────────────────────────────")
    print(f"  {'Model':<12} {'log-RMSE':>10} {'Deviance':>12}")
    print(f"  {'GLM':<12} {glm_results['log_rmse']:>10.4f} "
          f"{glm_results['deviance']:>12.4f}")
    print(f"  {'LightGBM':<12} {lgbm_results['log_rmse']:>10.4f} "
          f"{lgbm_results['deviance']:>12.4f}")

    shap_importance(lgbm_model, enc, val)
    save(lgbm_model, glm_model, enc, glm_cols)

    print("\n" + "=" * 55)
    print("Done.")
    print("=" * 55)