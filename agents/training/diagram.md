# ML Model Training Agent - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📥  INPUTS                                     │
│  • Goal/Objective (user input)                                              │
│  • Optional linked datasets                                                 │
│  • Optional model type                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1️⃣  MODEL SELECTION                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Select model based on goal                                               │
│  • Generate explanation if not user-specified                               │
│  • User can comment & regenerate (3x max)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2️⃣  DATA COLLECTION AGENT                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Reuses data-retrieval agent from ./data-retrieval                        │
│  • Finds and assembles the right dataset                                    │
│  • Uses provided datasets if available                                      │
│                                                                             │
│  🔧 Tools: SQL, DBs, Vector Stores, CRMs, Web connectors                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3️⃣  CLEANING & STANDARDIZATION AGENT                              🔄 LOOP  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────────┐      ┌──────────────────┐      ┌────────────────┐   │
│    │  Run EDA Report  │ ───▶ │   LLM Reviews    │ ───▶ │    Apply       │   │
│    │  + Validation    │      │    Results       │      │  Transforms    │   │
│    └──────────────────┘      └──────────────────┘      └────────────────┘   │
│            ▲                         │                        │             │
│            │                         │ Ready?                 │             │
│            │                         ▼                        │             │
│            │                  ┌──────────────┐                │             │
│            └──────────────────│   Register   │◀───────────────┘             │
│               (iterate)       │   Dataset    │    (needs more cleaning)     │
│                               └──────────────┘                              │
│                                                                             │
│  🔧 Tools: eda_report, data_validation, clean_ops, row_ops,                 │
│            column_ops, reshape_ops                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3.5️⃣  LABEL & SPLIT DEFINITION                                 👤 HUMAN    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Agent presents schema, user defines:                                       │
│                                                                             │
│    1. Target column          - "Which column is the prediction target?"     │
│    2. Prediction horizon     - "How far into the future?" (30d, 90d...)     │
│    3. Grain                  - "What does one row represent?"               │
│    4. As-of cutoff           - "Which column is the observation timestamp?" │
│    5. Split strategy         - "random / time-based / entity-based"         │
│    6. Forbidden columns      - "Which columns unavailable at predict time?" │
│                                                                             │
│  Mode: [ Manual ] or [ Auto-fill + Review ]                                 │
│                                                                             │
│  🔒 Definitions LOCKED → passed to steps 4, 5, 7                            │
│                                                                             │
│  ⚠️  SKIP IF NOT RELEVANT                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4️⃣  FEATURE SELECTION & SPECIFICATION AGENT                       🔄 LOOP  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Inputs: dataset_summary, goal, label_definition                            │
│                                                                             │
│  • Run statistical analysis to decide which features to use                 │
│  • Exclude forbidden columns (from 3.5) to prevent leakage                  │
│  • Iterate until confident                                                  │
│  • 👤 Human reviews trace, stats, and verifies features                     │
│                                                                             │
│  🔧 Tools: concentration_analysis, correlation_matrix, data_validation,     │
│            distribution_analysis, eda_report, feature_diagnostics,          │
│            group_summary, trend_analysis                                    │
│                                                                             │
│  📄 OUTPUT: feature_spec (JSON contract for step 5)                         │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ { "features": [                                                 │     │
│     │     { "name": "claim_count_90d",                                │     │
│     │       "formula": "count(claims) where claim_date > as_of-90d", │     │
│     │       "source_tables": ["claims", "policies"],                  │     │
│     │       "window": "90 days", "grain": "policy_id",                │     │
│     │       "as_of_constraint": "claim_date < as_of_date" }           │     │
│     │   ] }                                                           │     │
│     └─────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5️⃣  FEATURE ENGINEERING EXECUTOR                            ⚙️ DETERMINISTIC│
├─────────────────────────────────────────────────────────────────────────────┤
│  Inputs: feature_spec, dataset, label_definition                            │
│                                                                             │
│    ┌────────────────┐     ┌────────────────┐     ┌────────────────────┐     │
│    │ Parse          │ ──▶ │ Execute        │ ──▶ │ Run Diagnostics    │     │
│    │ feature_spec   │     │ Transforms     │     │ to Validate        │     │
│    └────────────────┘     └────────────────┘     └────────────────────┘     │
│                                                           │                 │
│                                                    ┌──────┴──────┐          │
│                                                    ▼             ▼          │
│                                              ┌─────────┐   ┌───────────┐    │
│                                              │  PASS   │   │   FAIL    │────┼──▶ Back to 4
│                                              └────┬────┘   └───────────┘    │
│                                                   ▼                         │
│                                          Transformed Dataset                │
│                                                                             │
│  🔧 Tools: feature_ops, agg_ops, row_ops, column_ops                        │
│                                                                             │
│  ⚡ NO LLM reasoning loop - execution follows spec exactly                  │
│  ⚡ Failures = spec failures → blame is clear                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6️⃣  HUMAN CONFIRMATION                                         👤 CHECKPOINT│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  • Show full trace of everything done so far                                │
│  • Human confirms we can proceed to training                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7️⃣  TRAINING SUBAGENT                                             🔄 LOOP  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Inputs: split strategy + indices from 3.5                                  │
│                                                                             │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  i. LLM Analysis: params, architecture, learning config          │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                      │                                      │
│                                      ▼                                      │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  ii. Training:                                                   │     │
│    │      a. Apply split from 3.5                                     │     │
│    │      b. Train on training set                                    │     │
│    │      c. Evaluate on validation                                   │     │
│    │      d. LLM reviews → 👤 Go back to i OR continue                │     │
│    │      e. Run on test set                                          │     │
│    │      f. Review test → 👤 Go back to i OR complete                │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                      │                                      │
│                                      ▼                                      │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  iii. Complete: output weights + audit trace + explanations      │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  🔧 Tools: glm, logistic_regression, random_forest, survival_analysis,      │
│            xgboost_model, model_storage                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8️⃣  REPORT GENERATION                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  • Generate comprehensive report to accompany the weights file              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              📤  OUTPUTS                                    │
│  • Weights file                                                             │
│  • Audit/lineage trace                                                      │
│  • Explanations of what was done                                            │
│  • Final report                                                             │
│  • Follow-up questions (optional)                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| 🔧 | Available Tools |
| 👤 | Human-in-the-Loop |
| 🔄 | Iterative Loop |
| 🔒 | Locked/Finalized |
| ⚙️ | Deterministic (no LLM) |
| ⚠️ | Conditional Step |
