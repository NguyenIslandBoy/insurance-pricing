"""
src/models/frequency.py — Claim frequency model.

Models the expected number of claims per policy per year.
Two approaches compared:
  1. Poisson GLM  — actuarial industry standard, interpretable coefficients
  2. LightGBM     — challenger model, captures non-linearities

Key actuarial concept: EXPOSURE OFFSET
  Policies are observed for different durations (0 to 1 year).
  A policy observed for 0.5 years has half the expected claims of a
  full-year policy, all else equal. We handle this by:
  - GLM: log(exposure) as an offset term (not a coefficient)
  - LightGBM: exposure as a feature + predict rate then scale

Target: claim_nb (count, >= 0)
Metric: Poisson deviance (standard actuarial metric), also report RMSE

Run:
    python src/models/frequency.py
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
from config import (
    DB_PATH, TABLE_FREQ_FEATURES,
    CAT_COLS, MODELS_DIR,
    POLICY_ID_COL,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FREQ_FEATURE_COLS = [
    "veh_power", "veh_age", "veh_brand", "veh_gas",
    "driv_age", "bonus_malus", "age_x_bonus",
    "area", "log_density", "region",
    "is_young_driver", "is_senior_driver", "has_malus",
    "is_old_vehicle", "is_high_power",
]
TARGET     = "claim_nb"
OFFSET_COL = "log_exposure"
EXPOSURE   = "exposure"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}. Run loader + dbt first.")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(f"SELECT * FROM {TABLE_FREQ_FEATURES}").df()
    con.close()
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Claim rate: {df[TARGET].mean():.4f} claims/policy")
    print(f"  Zero claims: {(df[TARGET] == 0).mean():.1%} of policies")
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def encode_categoricals(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Ordinal encode categoricals for LightGBM."""
    cat_cols_present = [c for c in ["veh_brand", "veh_gas", "area", "region"]
                        if c in df_train.columns]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train = df_train.copy()
    df_val   = df_val.copy()
    df_train[cat_cols_present] = enc.fit_transform(df_train[cat_cols_present])
    df_val[cat_cols_present]   = enc.transform(df_val[cat_cols_present])
    return df_train, df_val, enc


def split_data(df: pd.DataFrame):
    train, val = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train: {len(train):,}  Val: {len(val):,}")
    return train, val


# ── Poisson GLM ───────────────────────────────────────────────────────────────

def train_glm(train: pd.DataFrame) -> sm.GLM:
    """
    Poisson GLM with log link and exposure offset.
    Categorical variables are one-hot encoded via pd.get_dummies.
    Exposure enters as offset = log(exposure), not as a feature.
    """
    print("\n── Poisson GLM ──────────────────────────────────────")

    # One-hot encode categoricals for GLM
    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    num_cols = ["veh_power", "veh_age", "driv_age", "bonus_malus",
                "age_x_bonus", "log_density", "is_young_driver",
                "is_senior_driver", "has_malus", "is_old_vehicle",
                "is_high_power"]

    X = pd.get_dummies(train[cat_cols + num_cols], drop_first=True)
    X = sm.add_constant(X.astype(float))
    y = train[TARGET]
    offset = train[OFFSET_COL]

    model = sm.GLM(
        y, X,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=offset,
    ).fit()

    print(f"  Converged: {model.converged}")
    print(f"  Deviance:  {model.deviance:.2f}")
    print(f"  AIC:       {model.aic:.2f}")
    print(f"  Params:    {len(model.params)}")
    return model, X.columns.tolist()


def evaluate_glm(
    model: sm.GLM,
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
    # Align columns — val may have fewer dummies than train
    X_val = X_val.reindex(columns=glm_feature_cols, fill_value=0)

    y_pred = model.predict(X_val, offset=val[OFFSET_COL])
    y_true = val[TARGET]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))
    mask = y_true > 0
    deviance = 2 * (
        (y_true[mask] * np.log(y_true[mask] / y_pred[mask])).sum()
        - (y_true - y_pred).sum()
    )
    print(f"\n  GLM Val RMSE:     {rmse:.6f}")
    print(f"  GLM Val Deviance: {deviance:.2f}")
    return {"rmse": rmse, "deviance": deviance, "y_pred": y_pred}


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> lgb.LGBMRegressor:
    """
    LightGBM Poisson regressor.
    Uses exposure as a feature AND as an offset via the built-in
    objective='poisson' which models log(claim_rate) = log(claims/exposure).
    """
    print("\n── LightGBM Poisson ─────────────────────────────────")

    train_enc, val_enc, enc = encode_categoricals(train, val)

    X_train = train_enc[FREQ_FEATURE_COLS]
    y_train = train_enc[TARGET] / train_enc[EXPOSURE]  # claim rate (per year)
    X_val   = val_enc[FREQ_FEATURE_COLS]
    y_val   = val_enc[TARGET] / val_enc[EXPOSURE]

    model = lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=50,
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
        eval_metric="poisson",
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

    X_val  = val_enc[FREQ_FEATURE_COLS]
    # Predict rate then scale by exposure to get expected claim count
    y_pred_rate = model.predict(X_val)
    y_pred = y_pred_rate * val[EXPOSURE]
    y_true = val[TARGET]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 0
    deviance = 2 * (
        (y_true[mask] * np.log(y_true[mask] / y_pred[mask])).sum()
        - (y_true - y_pred).sum()
    )
    print(f"\n  LGBM Val RMSE:     {rmse:.6f}")
    print(f"  LGBM Val Deviance: {deviance:.2f}")
    return {"rmse": rmse, "deviance": deviance, "y_pred": y_pred}


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

    X_val = val_enc[FREQ_FEATURE_COLS].sample(2000, random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=FREQ_FEATURE_COLS,
    ).sort_values(ascending=False)

    print("\n  Top features by mean |SHAP|:")
    for feat, val_s in mean_shap.items():
        bar = "█" * int(val_s / mean_shap.iloc[0] * 20)
        print(f"  {feat:<30} {val_s:.4f}  {bar}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_shap.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Frequency Model — Mean |SHAP| Value")
    ax.set_xlabel("Mean |SHAP|")
    plt.tight_layout()
    path = ARTIFACTS_DIR / "freq_shap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  SHAP plot saved: {path}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save(
    lgbm_model: lgb.LGBMRegressor,
    glm_model: sm.GLM,
    enc: OrdinalEncoder,
    glm_feature_cols: list,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "lgbm": lgbm_model,
        "glm": glm_model,
        "encoder": enc,
        "glm_feature_cols": glm_feature_cols,
        "feature_cols": FREQ_FEATURE_COLS,
    }
    path = ARTIFACTS_DIR / "freq_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Frequency model saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Frequency model (Poisson GLM + LightGBM)")
    print("=" * 55)

    df = load_data()
    train, val = split_data(df)

    # GLM
    glm_model, glm_cols = train_glm(train)
    glm_results = evaluate_glm(glm_model, glm_cols, val)

    # LightGBM
    lgbm_model, enc = train_lgbm(train, val)
    lgbm_results = evaluate_lgbm(lgbm_model, enc, val)

    # Comparison
    print("\n── Model comparison ─────────────────────────────────")
    print(f"  {'Model':<12} {'RMSE':>10} {'Deviance':>12}")
    print(f"  {'GLM':<12} {glm_results['rmse']:>10.6f} {glm_results['deviance']:>12.2f}")
    print(f"  {'LightGBM':<12} {lgbm_results['rmse']:>10.6f} {lgbm_results['deviance']:>12.2f}")

    shap_importance(lgbm_model, enc, val)
    save(lgbm_model, glm_model, enc, glm_cols)

    print("\n" + "=" * 55)
    print("Done.")
    print("=" * 55)