# Insurance Pricing Pipeline — Architecture Flow

## Pipeline Orchestration (Prefect)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Prefect Flow: insurance-pricing-pipeline              │
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                          │
│  │  INGEST  │───▶│ DBT RUN  │───▶│ DBT TEST │                          │
│  │          │    │          │    │          │                          │
│  │ CSV →    │    │ staging  │    │ 29 data  │                          │
│  │ DuckDB   │    │ → inter  │    │ quality  │                          │
│  │          │    │ → marts  │    │ tests    │                          │
│  │ retries:2│    │ retries:1│    │ retries:0│◄── fails = pipeline stops│
│  └──────────┘    └──────────┘    └────┬─────┘                          │
│                                       │                                 │
│                                 ┌─────┴─────┐                          │
│                                 ▼           ▼                          │
│                          ┌──────────┐ ┌──────────┐                     │
│                          │  TRAIN   │ │  TRAIN   │  ◄── parallel       │
│                          │   FREQ   │ │   SEV    │                     │
│                          │          │ │          │                     │
│                          │ Poisson  │ │ Gamma    │                     │
│                          │ GLM +    │ │ GLM +    │                     │
│                          │ LightGBM │ │ LightGBM │                     │
│                          │ retries:1│ │ retries:1│                     │
│                          └────┬─────┘ └────┬─────┘                     │
│                               │            │                           │
│                               └─────┬──────┘                          │
│                                     ▼                                  │
│                              ┌──────────┐                              │
│                              │   PURE   │                              │
│                              │ PREMIUM  │                              │
│                              │          │                              │
│                              │ 4 combos │                              │
│                              │ evaluated│                              │
│                              │ retries:0│                              │
│                              └──────────┘                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Architecture (Medallion)

```
 RAW DATA                BRONZE                  SILVER                   GOLD
 ────────               ────────                ────────                 ────────

 freMTPL2freq.csv       ┌─────────────┐        ┌─────────────────┐     ┌──────────────┐
 677,991 policies  ───▶ │  DuckDB     │        │  PySpark        │     │  Model       │
                        │             │        │                 │     │  Training    │
 freMTPL2sev.csv        │  raw_freq   │   dbt  │  Driver flags   │     │              │
 26,444 claims     ───▶ │  raw_sev    │───────▶│  Vehicle flags  │────▶│  Poisson GLM │
                        │             │        │  Interactions   │     │  Gamma GLM   │
                        │  ClaimNb    │        │  Log transforms │     │  LightGBM    │
                        │  corruption │        │  Ordinal encode │     │  SHAP        │
                        │  fixed here │        │                 │     │              │
                        └─────────────┘        │  58 unit tests  │     │  Artifacts:  │
                                               │                 │     │  .pkl models │
                              │                └─────────────────┘     └──────┬───────┘
                              │                                               │
                              ▼                                               ▼
                        ┌─────────────┐                               ┌──────────────┐
                        │  dbt Models │                               │  FastAPI     │
                        │             │                               │  Endpoint    │
                        │  staging:   │                               │              │
                        │  stg_policies (view)                        │  /quote      │
                        │  stg_claims   (view)                        │  /quote/batch│
                        │             │                               │  /health     │
                        │  intermediate:                              │  /model/info │
                        │  int_policy_claims (table)                  │              │
                        │  ↳ ClaimNb fix via CASE/WHEN                │  €95 (low)   │
                        │  ↳ LEFT JOIN severity aggs                  │  to          │
                        │             │                               │  €904 (high) │
                        │  marts:     │                               │              │
                        │  mart_freq_features (table) 678k rows       │  1,000/batch │
                        │  mart_sev_features  (table) ~25k rows       └──────────────┘
                        │             │
                        │  29 quality tests
                        └─────────────┘
```

## Single Responsibility Principle

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Layer          Tool         Responsibility                     │
│   ─────          ────         ──────────────                     │
│                                                                  │
│   Bronze         DuckDB       Ingestion, ClaimNb fix             │
│                  + dbt        Joins, cleaning, quality tests     │
│                                                                  │
│   Silver         PySpark      Feature engineering                │
│                               Flags, interactions, encodings     │
│                               58 unit tests                      │
│                                                                  │
│   Gold           Scikit-learn  Model training + evaluation       │
│                  LightGBM     Artifact saving (.pkl)             │
│                                                                  │
│   Serving        FastAPI      Inference endpoint                 │
│                               Single + batch pricing             │
│                                                                  │
│   Orchestration  Prefect      Task dependencies, retries         │
│                               Logging, observability             │
│                                                                  │
│   CI/CD          GitHub       Automated tests on every push      │
│                  Actions      dbt run + dbt test in CI           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

```
Decision                              Why
────────                              ───

dbt_test retries = 0                  Bad data should stop the pipeline
                                      immediately, not retry

PySpark at 678k records               Demonstrates pattern that scales to
                                      tens of millions (national insurer book)

GLM frequency × LightGBM severity    GLM better calibrated for count data;
                                      LightGBM captures non-linear cost drivers

Exposure as offset, not feature       Actuarial standard: log(exposure) offset
                                      in Poisson GLM ensures correct rate scaling

p99 cap on claim amounts              Extreme tail handled by reinsurance,
                                      not the pricing model

Feature engineering in PySpark,       Clear ownership: dbt = SQL transforms,
not dbt                               Spark = feature engineering, model = training
```
