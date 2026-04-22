# Agents

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS AGENT v2                                   │
│                         (gpt-5.1, main orchestrator)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ has 3 tools
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ statistical_      │   │ model_execution_  │   │ data_lookup_      │
│ analysis_tool     │   │ tool              │   │ tool              │
├───────────────────┤   ├───────────────────┤   ├───────────────────┤
│ Find + analyze    │   │ Run pretrained    │   │ Simple data       │
│ data patterns     │   │ models on inputs  │   │ retrieval only    │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
          │                       │                       │
          ▼                       │                       │
┌─────────────────────────┐       │                       │
│ STATISTICAL ANALYSIS    │       │                       │
│ AGENT (gpt-5.1)         │       │                       │
├─────────────────────────┤       │                       │
│ Has both retrieval +    │       │                       │
│ analysis tools directly │       │                       │
└─────────┬───────────────┘       │                       │
          │                       │                       │
          ▼                       │                       │
┌─────────────────────────────────┴───────────────────────┴─────────────────┐
│                              TOOL LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  DATA RETRIEVAL TOOLS          ANALYSIS TOOLS         PRETRAINED MODELS   │
│  ─────────────────────         ──────────────         ─────────────────   │
│  • catalog_search_tool         • eda_report           • credit_risk       │
│  • list_datasets_tool          • data_validation      • loan_default      │
│  • dataset_get_tool            • correlation_matrix   • finbert_tone      │
│  • sql_query_tool              • trend_analysis       • finbert_sentiment │
│  • get_sql_schema_tool         • group_summary        • claim_detection   │
│  • join_merge_tool             • concentration        • chronos2_forecast │
│                                • feature_diagnostics  • timesfm_forecast  │
│                                • distribution                             │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                    │
├────────────────────────────────────────────────────────────────────────────┤
│  datasets/                                                                 │
│  ├── csv/insurance.csv          (health insurance costs)                   │
│  ├── csv/Loan_default.csv       (loan default prediction)                  │
│  ├── csv/Financial Distress.csv (corporate distress)                       │
│  └── sql/*.sql                  (SQL warehouse tables)                     │
│                                                                            │
│  catalog.json                   (dataset metadata & search index)          │
└────────────────────────────────────────────────────────────────────────────┘
```

## Tool Selection Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ANALYSIS AGENT v2                         │
│                                                              │
│  "What data do you have?"  ───────────►  data_lookup_tool   │
│                                                              │
│  "Analyze patterns in..."  ───────────►  statistical_       │
│  "What factors predict..."               analysis_tool      │
│                                                              │
│  "Score this applicant"    ───────────►  model_execution_   │
│  "Analyze sentiment of..." (text input)  tool               │
│                                                              │
│  "Find records to run      ───────────►  data_lookup_tool   │
│   through the model"                     then                │
│                                          model_execution_    │
│                                          tool                │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

| File | Description |
|------|-------------|
| Project root `agent.py` | Jubilee chat orchestrator (`search_datasets`, analysis tools, pretrained tools, training handoff, …) |
| `data-retrieval/agent.py` | Data retrieval utilities (training / catalog flows) |
| `model_index.py` | Registry metadata for pretrained models |
| `prompts.py` | Prompts for planning / intake agents (not the orchestrator prompt, which lives in `agent.py`) |

## Historical note

Nested `analysis_agent_v2` / `statistical_analysis_agent` wrappers were removed; statistical tools live under `tools/data-tools/analysis/` and are registered directly on the orchestrator.

### 1. Analysis tools (`tools/data-tools/analysis/`)

EDA, correlations, grouping, distributions, charts, validation, concentration, trends — see package ``__init__.py``.

### 2. Pretrained tools (`tools/models-tools/pretrained/`)

Credit risk, loan default, sentiment, claims, forecasts — invoked directly by name from the orchestrator when appropriate.
