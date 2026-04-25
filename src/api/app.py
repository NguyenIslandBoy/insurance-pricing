"""
src/api/app.py — FastAPI inference endpoint for the insurance pricing pipeline.

Endpoints:
  GET  /health         — liveness check
  GET  /model/info     — model metadata
  POST /quote          — pure premium for a single policy
  POST /quote/batch    — pure premium for multiple policies (max 1000)

Run:
    uvicorn src.api.app:app --reload --port 8000
"""

import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

ARTIFACTS_DIR = Path(__file__).parent.parent / "models" / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "pure_premium_model.pkl"

FREQ_FEATURE_COLS = [
    "veh_power", "veh_age", "veh_brand", "veh_gas",
    "driv_age", "bonus_malus", "age_x_bonus",
    "area", "log_density", "region",
    "is_young_driver", "is_senior_driver", "has_malus",
    "is_old_vehicle", "is_high_power",
]

_payload = None


# ── Startup ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _payload
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run src/models/pure_premium.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        _payload = pickle.load(f)
    print(f"Model loaded. Best combination: {_payload['best_combination']}")
    yield


app = FastAPI(
    title="Insurance Pricing API",
    description=(
        "Computes the pure premium (expected loss cost) for a French MTPL "
        "motor insurance policy based on risk features. "
        "Pure premium = E[frequency] × E[severity]. "
        "Built on freMTPL2 dataset (677,991 policies)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PolicyFeatures(BaseModel):
    """Risk features for a single motor insurance policy."""

    veh_power: int = Field(
        ge=4, le=15,
        description="Vehicle power (ordinal 4–15, higher = more powerful)"
    )
    veh_age: int = Field(
        ge=0, le=100,
        description="Vehicle age in years"
    )
    veh_brand: str = Field(
        description="Vehicle brand code (e.g. B1, B2, B12)"
    )
    veh_gas: str = Field(
        description="Fuel type: REGULAR or DIESEL"
    )
    driv_age: int = Field(
        ge=18, le=100,
        description="Driver age in years"
    )
    bonus_malus: int = Field(
        ge=50, le=350,
        description="Bonus-malus level (50=max bonus, >100=malus)"
    )
    area: str = Field(
        description="Area code (A–F, A=most rural, F=most urban)"
    )
    density: float = Field(
        ge=1,
        description="Population density of the policy holder's municipality"
    )
    region: str = Field(
        description="French region code (e.g. R11, R24)"
    )
    exposure: float = Field(
        default=1.0, ge=0.001, le=1.0,
        description="Observation period in years (default=1.0 for annual quote)"
    )


class QuoteResponse(BaseModel):
    pure_premium_annual: float = Field(
        description="Expected loss cost per year in euros"
    )
    expected_claim_freq: float = Field(
        description="Expected number of claims per year"
    )
    expected_claim_cost: float = Field(
        description="Expected cost per claim in euros"
    )
    risk_tier: str = Field(
        description="LOW / MEDIUM / HIGH based on pure premium percentile"
    )
    model_combination: str = Field(
        description="Which freq × sev combination was used"
    )
    latency_ms: float


class BatchRequest(BaseModel):
    policies: list[PolicyFeatures] = Field(max_length=1000)


class BatchResponse(BaseModel):
    results: list[dict]
    portfolio_mean_pp: float
    n_high_risk: int
    n_medium_risk: int
    n_low_risk: int
    latency_ms: float


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(policy: PolicyFeatures) -> pd.DataFrame:
    """Convert policy input to model feature DataFrame."""
    d = {
        "veh_power":        policy.veh_power,
        "veh_age":          policy.veh_age,
        "veh_brand":        policy.veh_brand.upper().strip(),
        "veh_gas":          policy.veh_gas.upper().strip(),
        "driv_age":         policy.driv_age,
        "bonus_malus":      min(policy.bonus_malus, 150),  # cap as in dbt
        "age_x_bonus":      policy.driv_age * min(policy.bonus_malus, 150),
        "area":             policy.area.upper().strip(),
        "log_density":      np.log(max(policy.density, 1)),
        "region":           policy.region.strip(),
        "is_young_driver":  int(policy.driv_age < 25),
        "is_senior_driver": int(policy.driv_age > 70),
        "has_malus":        int(policy.bonus_malus > 100),
        "is_old_vehicle":   int(policy.veh_age > 10),
        "is_high_power":    int(policy.veh_power >= 9),
    }
    return pd.DataFrame([d])


def _predict_one(df: pd.DataFrame, exposure: float) -> tuple:
    """Run frequency and severity predictions for one policy row."""
    freq_payload = _payload["freq"]
    sev_payload  = _payload["sev"]
    best         = _payload["best_combination"]

    cat_cols = ["veh_brand", "veh_gas", "area", "region"]
    num_cols = ["veh_power", "veh_age", "driv_age", "bonus_malus",
                "age_x_bonus", "log_density", "is_young_driver",
                "is_senior_driver", "has_malus", "is_old_vehicle",
                "is_high_power"]

    # GLM predictions
    X_glm = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
    X_glm = sm.add_constant(X_glm.astype(float), has_constant="add")

    X_freq_glm = X_glm.reindex(
        columns=freq_payload["glm_feature_cols"], fill_value=0
    )
    X_sev_glm = X_glm.reindex(
        columns=sev_payload["glm_feature_cols"], fill_value=0
    )
    freq_glm = float(freq_payload["glm"].predict(
        X_freq_glm, offset=pd.Series([0.0])
    ).iloc[0])
    sev_glm = float(sev_payload["glm"].predict(X_sev_glm).iloc[0])

    # LGBM predictions
    enc_freq = freq_payload["encoder"]
    enc_sev  = sev_payload["encoder"]

    df_enc_freq = df.copy()
    df_enc_freq[cat_cols] = enc_freq.transform(df_enc_freq[cat_cols])
    freq_lgbm = float(freq_payload["lgbm"].predict(
        df_enc_freq[FREQ_FEATURE_COLS]
    )[0])

    df_enc_sev = df.copy()
    df_enc_sev[cat_cols] = enc_sev.transform(df_enc_sev[cat_cols])
    sev_lgbm = float(sev_payload["lgbm"].predict(
        df_enc_sev[FREQ_FEATURE_COLS]
    )[0])

    # Select freq and sev based on best combination
    freq_map = {"GLM x GLM": freq_glm, "GLM x LGBM": freq_glm,
                "LGBM x GLM": freq_lgbm, "LGBM x LGBM": freq_lgbm}
    sev_map  = {"GLM x GLM": sev_glm,  "GLM x LGBM": sev_lgbm,
                "LGBM x GLM": sev_glm,  "LGBM x LGBM": sev_lgbm}

    freq = freq_map[best]
    sev  = sev_map[best]
    pp   = freq * sev * exposure  # scale by exposure for partial-year quotes

    return pp, freq, sev


def _risk_tier(pp: float) -> str:
    """
    Assign risk tier based on pure premium.
    Thresholds derived from portfolio distribution:
      LOW    < 100  (~p40 of portfolio)
      MEDIUM < 200  (~p80 of portfolio)
      HIGH   >= 200
    """
    if pp < 100:
        return "LOW"
    elif pp < 200:
        return "MEDIUM"
    else:
        return "HIGH"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _payload is not None,
        "best_combination": _payload["best_combination"] if _payload else None,
    }


@app.get("/model/info")
def model_info():
    if _payload is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "best_combination": _payload["best_combination"],
        "n_freq_features":  len(_payload["freq_feature_cols"]),
        "n_sev_features":   len(_payload["sev_feature_cols"]),
        "training_data":    "freMTPL2 — 677,991 French MTPL policies",
        "freq_model":       "Poisson GLM + LightGBM (GLM used in best combination)",
        "sev_model":        "Gamma GLM + LightGBM (LGBM used in best combination)",
        "note": (
            "Pure premium = E[frequency] × E[severity]. "
            "Does not include expense loading, profit margin, or tax."
        ),
    }


@app.post("/quote", response_model=QuoteResponse)
def quote(policy: PolicyFeatures):
    if _payload is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    try:
        df = _build_features(policy)
        pp, freq, sev = _predict_one(df, policy.exposure)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    latency = round((time.perf_counter() - t0) * 1000, 2)

    return QuoteResponse(
        pure_premium_annual=round(pp, 2),
        expected_claim_freq=round(freq, 6),
        expected_claim_cost=round(sev, 2),
        risk_tier=_risk_tier(pp),
        model_combination=_payload["best_combination"],
        latency_ms=latency,
    )


@app.post("/quote/batch", response_model=BatchResponse)
def quote_batch(batch: BatchRequest):
    if _payload is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not batch.policies:
        raise HTTPException(status_code=400, detail="Empty batch")

    t0 = time.perf_counter()
    results = []
    for policy in batch.policies:
        try:
            df = _build_features(policy)
            pp, freq, sev = _predict_one(df, policy.exposure)
            tier = _risk_tier(pp)
            results.append({
                "pure_premium_annual": round(pp, 2),
                "expected_claim_freq": round(freq, 6),
                "expected_claim_cost": round(sev, 2),
                "risk_tier": tier,
            })
        except Exception as e:
            results.append({"error": str(e)})

    latency = round((time.perf_counter() - t0) * 1000, 2)
    valid   = [r for r in results if "error" not in r]
    tiers   = [r["risk_tier"] for r in valid]
    pps     = [r["pure_premium_annual"] for r in valid]

    return BatchResponse(
        results=results,
        portfolio_mean_pp=round(np.mean(pps), 2) if pps else 0,
        n_high_risk=tiers.count("HIGH"),
        n_medium_risk=tiers.count("MEDIUM"),
        n_low_risk=tiers.count("LOW"),
        latency_ms=latency,
    )