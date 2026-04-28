# Insurance Pricing Pipeline

![CI](https://github.com/NguyenIslandBoy/insurance-pricing/actions/workflows/ci.yml/badge.svg)

End-to-end insurance pricing system for French Motor Third-Party Liability (MTPL) policies. Implements the actuarial **frequency-severity decomposition** using both GLM (industry standard) and LightGBM (challenger), with a dbt transformation layer and a FastAPI pricing endpoint.

**Pure premium range:** €95 (low-risk) → €904 (high-risk) across tested profiles.

---

## Background

Built on the **freMTPL2** dataset from the R `CASdatasets` package - real French MTPL insurance records covering 677,991 policies. Standard benchmark dataset in actuarial science and insurance ML research.

> Dutang, C. & Charpentier, A. (2020). *CASdatasets R package.*

The pricing methodology follows:
> Wüthrich, M. & Buser, C. *Data Analytics for Non-Life Insurance Pricing.* (SSRN, free)

---

## Architecture

```
insurance-pricing/
├── data/
│   ├── raw/                        # freMTPL2freq.csv, freMTPL2sev.csv
│   └── processed/                  # DuckDB database (generated)
├── src/
│   ├── ingest/loader.py            # CSV → DuckDB, fixes corrupt ClaimNb
│   ├── models/
│   │   ├── frequency.py            # Poisson GLM + LightGBM frequency model
│   │   ├── severity.py             # Gamma GLM + LightGBM severity model
│   │   ├── pure_premium.py         # Combines models, evaluates 4 combinations
│   │   └── artifacts/              # Saved model files (generated)
│   └── api/app.py                  # FastAPI pricing endpoint
├── dbt/
│   ├── dbt_project.yml
│   ├── profiles.yml
│   └── models/
│       ├── staging/                # stg_policies, stg_claims
│       ├── intermediate/           # int_policy_claims (critical join + ClaimNb fix)
│       └── marts/                  # mart_freq_features, mart_sev_features
└── requirements.txt
```

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place raw data in `data/raw/`**
```
freMTPL2freq.csv    # 677,991 rows - policy features + claim counts
freMTPL2sev.csv     #  26,444 rows - individual claim amounts
```
Download from [Kaggle](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq) or [Hugging Face](https://huggingface.co/datasets/mabilton/fremtpl2).

**3. Ingest**
```bash
python src/ingest/loader.py
```

**4. dbt transformations** (from inside `dbt/` folder)
```bash
cd dbt
dbt run      # staging → intermediate → marts
dbt test     # 29 data quality tests
cd ..
```

**5. Train models**
```bash
python src/models/frequency.py    # ~2 min
python src/models/severity.py     # ~1 min
python src/models/pure_premium.py # combines + evaluates
```

**6. Start API**
```bash
uvicorn src.api.app:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

---

## Data quality: known ClaimNb issue

A documented flaw in freMTPL2freq: policies with `IDpol <= 24,500` have claim counts that don't match their severity records. The pipeline handles this explicitly:

1. **`loader.py`** zeros out ClaimNb for corrupt policies and adds a `claimnb_corrupted` flag
2. **`int_policy_claims.sql`** recomputes ClaimNb from the severity table for those policies

This is the correct actuarial treatment and matches the approach in Wüthrich & Buser (2020).

---

## dbt layer

Three-layer medallion architecture:

| Layer | Model | Purpose |
|---|---|---|
| Staging | `stg_policies` | Clean freq table - clip exposure, cap BonusMalus at 150 |
| Staging | `stg_claims` | Clean sev table - remove zero amounts, cap at p99 |
| Intermediate | `int_policy_claims` | Join + fix corrupt ClaimNb + aggregate severities |
| Mart | `mart_freq_features` | Modelling-ready frequency table (all 678k policies) |
| Mart | `mart_sev_features` | Modelling-ready severity table (claims-only, ~25k rows) |

29 data tests across all layers including uniqueness, not-null, accepted values, and referential integrity checks.

---

## Modelling

### Frequency (Poisson GLM vs LightGBM)

Target: claim count. Exposure offset: `log(exposure)` - mandatory for varying observation periods.

| Model | Val RMSE | Val Deviance |
|---|---|---|
| Poisson GLM | 0.2103 | 33,283 |
| LightGBM | 0.2163 | 34,606 |

**GLM wins.** Claim frequency follows a log-linear structure well-suited to GLMs. LightGBM adds complexity without adding signal - consistent with actuarial literature.

Top predictors (SHAP): `bonus_malus` (0.24), `veh_age`, `driv_age`, `region`, `veh_brand`.

### Severity (Gamma GLM vs LightGBM)

Target: average claim cost. Claims-only subset (~25k rows, ~4% of policies).

| Model | Val log-RMSE | Val Deviance |
|---|---|---|
| Gamma GLM | 1.2057 | 5,041.5 |
| LightGBM | 1.1971 | 5,001.0 |

**LightGBM wins marginally.** Claim costs have non-linear relationships that trees capture better.

Top predictors (SHAP): `veh_age` (0.040), `bonus_malus`, `region`, `veh_power`, `log_density`.

### Pure premium = Frequency × Severity

Four combinations evaluated on 20% held-out set:

| Combination | Mean PP | RMSE |
|---|---|---|
| GLM x GLM | €125.59 | 6,100.63 |
| **GLM x LGBM** | **€122.48** | **6,100.43** |
| LGBM x GLM | €189.36 | 6,101.53 |
| LGBM x LGBM | €185.77 | 6,100.94 |

**Best: GLM x LGBM.** GLM frequency is more calibrated on this dataset; LightGBM severity captures non-linear cost drivers.

**Note:** RMSE differences are small due to zero-inflation (96% of policies have zero claims). The log-RMSE on claims-only subset shows LGBM frequency better ranks which policies will claim - a genuine signal worth noting.

---

## API

```
GET  /health          liveness + model status
GET  /model/info      model metadata
POST /quote           pure premium for a single policy
POST /quote/batch     pure premium for up to 1,000 policies
```

**Example - high-risk profile (young driver, malus):**
```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/quote" `
  -ContentType "application/json" `
  -Body '{"veh_power":7,"veh_age":3,"veh_brand":"B12","veh_gas":"REGULAR","driv_age":22,"bonus_malus":130,"area":"D","density":1500,"region":"R11"}'
```
```
pure_premium_annual : 904.38
expected_claim_freq : 0.505485
expected_claim_cost : 1789.14
risk_tier           : HIGH
```

**Example - low-risk profile (experienced driver, bonus):**
```
pure_premium_annual : 94.88
expected_claim_freq : 0.060755
expected_claim_cost : 1561.75
risk_tier           : LOW
```

Risk tiers: `LOW < €100`, `MEDIUM €100–200`, `HIGH ≥ €200`.

---

## Key actuarial insights

**Frequency vs severity have different drivers** - a key finding:
- Frequency is dominated by **driver behaviour**: bonus-malus (claims history), driver age
- Severity is dominated by **vehicle characteristics**: vehicle age, power, and **geography** (region, density)

This split is consistent with MTPL pricing literature and has practical implications: a fleet manager can reduce frequency through driver training but cannot easily change severity (vehicle repair costs are market-driven).

**GLM interpretability matters here.** The Poisson GLM coefficients are directly interpretable as multiplicative risk factors - a 22-year-old with BM=130 has ~8x the claim rate of a 45-year-old with BM=55. This is the format regulators and pricing committees expect.

---

## Fintech / insurtech relevance

| Insurance | Fintech analogy |
|---|---|
| Claim frequency model | Default probability (PD) model |
| Claim severity model | Loss given default (LGD) model |
| Pure premium | Expected loss = PD × LGD |
| Exposure offset | Time-at-risk adjustment |
| Bonus-malus | Behavioural credit score |
| Gamma GLM for severity | Gamma regression for loan loss amounts |